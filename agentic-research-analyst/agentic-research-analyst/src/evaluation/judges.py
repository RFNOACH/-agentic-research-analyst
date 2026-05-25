"""
LLM-as-Judge Scorers — Automated quality evaluation using GPT-4o.

Scores research memos across completeness, citation accuracy,
and coherence dimensions.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage


class BaseJudge(ABC):
    """Base class for LLM-as-judge evaluators."""

    def __init__(self, model: str = "gpt-4o") -> None:
        self.llm = ChatOpenAI(model=model, temperature=0.0)

    @abstractmethod
    def score(self, brief: str, memo_text: str, **kwargs) -> float:
        """Score a memo on a 0-1 scale."""
        ...


class CompletenessJudge(BaseJudge):
    """Evaluates whether the memo addresses all aspects of the brief."""

    PROMPT = """You are evaluating a research memo for completeness.

Brief: {brief}

Memo:
{memo}

Score the memo from 0.0 to 1.0 on how completely it addresses the brief.
Consider: Are all requested topics covered? Are comparisons made where asked?
Is quantitative data included when appropriate?

Respond with ONLY a JSON object: {{"score": 0.85, "reasoning": "..."}}"""

    def score(self, brief: str, memo_text: str, **kwargs) -> float:
        response = self.llm.invoke([
            HumanMessage(content=self.PROMPT.format(brief=brief, memo=memo_text))
        ])
        data = json.loads(response.content)
        return float(data["score"])


class CitationJudge(BaseJudge):
    """Evaluates citation accuracy — are claims properly sourced?"""

    PROMPT = """You are evaluating citation accuracy in a research memo.

Memo:
{memo}

Evidence sources:
{sources}

Score from 0.0 to 1.0: What fraction of factual claims have valid source attribution?
Flag any claims that appear unsupported (hallucinated).

Respond with ONLY a JSON object: {{"score": 0.90, "unsupported_claims": [], "reasoning": "..."}}"""

    def score(self, brief: str, memo_text: str, **kwargs) -> float:
        sources = kwargs.get("sources", "No sources provided")
        response = self.llm.invoke([
            HumanMessage(content=self.PROMPT.format(memo=memo_text, sources=sources))
        ])
        data = json.loads(response.content)
        return float(data["score"])


class CoherenceJudge(BaseJudge):
    """Evaluates structural coherence and readability."""

    PROMPT = """You are evaluating the coherence of a research memo.

Memo:
{memo}

Score from 0.0 to 1.0 on:
- Logical flow between sections
- Clear executive summary
- Consistent terminology
- Professional tone

Respond with ONLY a JSON object: {{"score": 0.85, "reasoning": "..."}}"""

    def score(self, brief: str, memo_text: str, **kwargs) -> float:
        response = self.llm.invoke([
            HumanMessage(content=self.PROMPT.format(memo=memo_text))
        ])
        data = json.loads(response.content)
        return float(data["score"])
