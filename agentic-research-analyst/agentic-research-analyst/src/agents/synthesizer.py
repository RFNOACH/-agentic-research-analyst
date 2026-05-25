"""
Synthesis Engine — Drafts research memos from collected evidence.

Combines semantic retrieval from the vector store with LLM generation
to produce structured, cited research outputs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.planner import TaskPlan
from src.tools.citation_linker import CitationLinker
from src.tools.vector_store import VectorStoreManager


@dataclass
class Citation:
    """A single source citation linked to a claim in the memo."""

    id: str
    source_url: str
    title: str
    excerpt: str
    claim: str
    confidence: float


@dataclass
class MemoSection:
    """A section of the research memo."""

    heading: str
    content: str
    citations: list[str] = field(default_factory=list)


@dataclass
class ResearchMemo:
    """Complete research memo with executive summary, sections, and citations."""

    title: str
    executive_summary: str
    sections: list[MemoSection]
    citations: list[Citation]
    confidence_score: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_markdown(self) -> str:
        """Render the memo as a Markdown document."""
        lines = [f"# {self.title}", "", self.executive_summary, ""]

        for section in self.sections:
            lines.append(f"## {section.heading}")
            lines.append(section.content)
            lines.append("")

        lines.append("## Sources")
        for c in self.citations:
            lines.append(f"- [{c.title}]({c.source_url}) — {c.excerpt[:100]}...")
        lines.append("")

        return "\n".join(lines)

    def to_json(self) -> dict:
        """Serialize the memo to a JSON-compatible dict."""
        return {
            "title": self.title,
            "executive_summary": self.executive_summary,
            "sections": [
                {"heading": s.heading, "content": s.content, "citations": s.citations}
                for s in self.sections
            ],
            "citations": [
                {
                    "id": c.id,
                    "source_url": c.source_url,
                    "title": c.title,
                    "excerpt": c.excerpt,
                    "claim": c.claim,
                    "confidence": c.confidence,
                }
                for c in self.citations
            ],
            "confidence_score": self.confidence_score,
        }


SYNTHESIS_SYSTEM_PROMPT = """You are a research synthesis engine. Given a research brief,
a task plan, and collected evidence, produce a structured research memo.

Your memo must:
1. Begin with a concise executive summary (2-3 sentences).
2. Organize findings into 3-5 logical sections.
3. Cite sources using [N] notation keyed to the evidence list.
4. Flag low-confidence claims explicitly.
5. Include quantitative data when available.

Respond in JSON:
{
  "title": "Memo title",
  "executive_summary": "2-3 sentence summary",
  "sections": [
    {"heading": "Section Title", "content": "Section body with [1] citations", "citation_ids": ["e1", "e2"]}
  ],
  "confidence_score": 0.85
}
"""


class Synthesizer:
    """Generates cited research memos from evidence."""

    def __init__(self, llm: ChatOpenAI, citation_linker: CitationLinker) -> None:
        self.llm = llm
        self.citation_linker = citation_linker

    def draft(
        self,
        brief: str,
        plan: TaskPlan,
        evidence: list[dict[str, Any]],
        vector_store: VectorStoreManager,
    ) -> ResearchMemo:
        """
        Draft a research memo from collected evidence.

        Parameters
        ----------
        brief : str
            Original research brief.
        plan : TaskPlan
            The execution plan used to gather evidence.
        evidence : list[dict]
            Raw evidence collected by tools.
        vector_store : VectorStoreManager
            Indexed evidence for semantic retrieval.

        Returns
        -------
        ResearchMemo
            Complete research memo with citations.
        """
        # Retrieve most relevant evidence for the brief
        relevant = vector_store.query(brief, top_k=15)

        # Format evidence for the LLM
        evidence_text = "\n\n".join(
            f"[Evidence {i+1}] (source: {e.get('url', 'unknown')})\n{e.get('content', '')[:500]}"
            for i, e in enumerate(relevant)
        )

        messages = [
            SystemMessage(content=SYNTHESIS_SYSTEM_PROMPT),
            HumanMessage(
                content=f"Brief: {brief}\n\nPlan: {plan.reasoning}\n\nEvidence:\n{evidence_text}"
            ),
        ]

        response = self.llm.invoke(messages)
        data = json.loads(response.content)

        # Build citations
        citations = self.citation_linker.link(
            sections=data["sections"],
            evidence=relevant,
        )

        sections = [
            MemoSection(
                heading=s["heading"],
                content=s["content"],
                citations=s.get("citation_ids", []),
            )
            for s in data["sections"]
        ]

        return ResearchMemo(
            title=data["title"],
            executive_summary=data["executive_summary"],
            sections=sections,
            citations=citations,
            confidence_score=data.get("confidence_score", 0.7),
            metadata={"brief": brief, "evidence_count": len(evidence)},
        )
