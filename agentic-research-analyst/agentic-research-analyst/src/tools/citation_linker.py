"""
Citation Linker — Graph-based source attribution engine.

Maps claims in research memos to supporting evidence with
confidence scoring and provenance tracking.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class CitationNode:
    """A node in the citation graph linking a claim to its source."""

    id: str
    claim: str
    source_url: str
    source_title: str
    evidence_excerpt: str
    confidence: float
    section_heading: str


class CitationLinker:
    """
    Links claims in generated text to source evidence.

    Algorithm
    ---------
    1. Extract citation markers [N] from generated sections.
    2. Match markers to evidence items by index.
    3. Score confidence based on semantic overlap.
    4. Build a citation graph for provenance audit.
    """

    def link(
        self,
        sections: list[dict[str, Any]],
        evidence: list[dict[str, Any]],
    ) -> list[CitationNode]:
        """
        Link citation markers in sections to evidence sources.

        Parameters
        ----------
        sections : list[dict]
            Generated memo sections with [N] citation markers.
        evidence : list[dict]
            Evidence pool with source metadata.

        Returns
        -------
        list[CitationNode]
            Resolved citations with source attribution.
        """
        citations = []

        for section in sections:
            content = section.get("content", "")
            heading = section.get("heading", "")

            # Find all [N] citation markers
            markers = re.findall(r"\[(\d+)\]", content)

            for marker in markers:
                idx = int(marker) - 1  # Convert to 0-indexed

                if 0 <= idx < len(evidence):
                    ev = evidence[idx]
                    citation = CitationNode(
                        id=f"c-{uuid.uuid4().hex[:8]}",
                        claim=self._extract_claim(content, marker),
                        source_url=ev.get("url", ""),
                        source_title=ev.get("title", ""),
                        evidence_excerpt=ev.get("content", "")[:300],
                        confidence=ev.get("similarity", 0.5),
                        section_heading=heading,
                    )
                    citations.append(citation)

        return citations

    def _extract_claim(self, text: str, marker: str) -> str:
        """Extract the sentence containing a citation marker."""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        for sentence in sentences:
            if f"[{marker}]" in sentence:
                return sentence.replace(f"[{marker}]", "").strip()
        return ""

    def build_graph(self, citations: list[CitationNode]) -> dict[str, Any]:
        """
        Build a citation provenance graph for audit.

        Returns a dict representation of the citation DAG:
        sources → claims → sections.
        """
        graph: dict[str, Any] = {"sources": {}, "claims": [], "edges": []}

        for c in citations:
            if c.source_url not in graph["sources"]:
                graph["sources"][c.source_url] = {
                    "title": c.source_title,
                    "claims_count": 0,
                }
            graph["sources"][c.source_url]["claims_count"] += 1

            graph["claims"].append({
                "id": c.id,
                "text": c.claim,
                "confidence": c.confidence,
            })

            graph["edges"].append({
                "from": c.source_url,
                "to": c.id,
                "section": c.section_heading,
            })

        return graph
