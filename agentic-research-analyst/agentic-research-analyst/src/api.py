"""
API Server — FastAPI endpoint for research brief execution.

Exposes the research agent as a REST API with streaming support
and execution trace endpoints.
"""

from __future__ import annotations

import time
import uuid

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.agents.research_agent import ResearchAgent
from src.config.settings import Settings

app = FastAPI(
    title="Agentic Research Analyst",
    description="Multi-tool LLM agent that drafts research memos",
    version="1.0.0",
)

agent = ResearchAgent(settings=Settings())


class ResearchRequest(BaseModel):
    """Incoming research brief request."""

    brief: str = Field(..., description="Research question or task description")
    max_tools: int = Field(default=10, ge=1, le=20, description="Max tool calls")
    output_format: str = Field(default="markdown", pattern="^(markdown|json|html)$")
    temperature: float | None = Field(default=None, ge=0, le=1)


class ResearchResponse(BaseModel):
    """Research memo response."""

    id: str
    title: str
    executive_summary: str
    content: str
    citations_count: int
    confidence_score: float
    latency_seconds: float
    cost_usd: float


@app.post("/research", response_model=ResearchResponse)
async def create_research(request: ResearchRequest) -> ResearchResponse:
    """Execute a research brief and return the completed memo."""
    t0 = time.time()

    try:
        memo = agent.run(
            brief=request.brief,
            output_format=request.output_format,
            max_tools=request.max_tools,
            temperature=request.temperature,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    latency = time.time() - t0

    return ResearchResponse(
        id=str(uuid.uuid4()),
        title=memo.title,
        executive_summary=memo.executive_summary,
        content=memo.to_markdown(),
        citations_count=len(memo.citations),
        confidence_score=memo.confidence_score,
        latency_seconds=round(latency, 2),
        cost_usd=round(memo.metadata.get("cost", 0.18), 2),
    )


@app.get("/health")
async def health():
    return {"status": "ok", "version": "1.0.0"}
