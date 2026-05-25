"""
Integration tests for the research agent.

Tests the full agent pipeline with mocked tool responses
to verify state transitions and output structure.
"""

import pytest
from unittest.mock import MagicMock, patch

from src.agents.research_agent import ResearchAgent, AgentState
from src.agents.planner import TaskPlan, SubTask
from src.agents.synthesizer import ResearchMemo, MemoSection, Citation
from src.config.settings import Settings


@pytest.fixture
def agent():
    """Create an agent with mocked LLM."""
    with patch("src.agents.research_agent.ChatOpenAI"):
        return ResearchAgent(settings=Settings())


@pytest.fixture
def sample_plan():
    return TaskPlan(
        tasks=[
            SubTask(id="t1", description="Search market data", tool="web_search", query="AI code assistants market 2025"),
            SubTask(id="t2", description="Search pricing", tool="web_search", query="GitHub Copilot Cursor pricing comparison"),
        ],
        reasoning="Need to gather market data and pricing for comparison",
        brief_summary="Compare AI code assistants",
    )


@pytest.fixture
def sample_memo():
    return ResearchMemo(
        title="AI Code Assistants: Competitive Landscape 2025",
        executive_summary="The AI code assistant market has grown significantly...",
        sections=[
            MemoSection(heading="Market Overview", content="The market is valued at..."),
            MemoSection(heading="Pricing Comparison", content="GitHub Copilot costs..."),
        ],
        citations=[],
        confidence_score=0.87,
    )


class TestAgentState:
    def test_initial_state(self):
        state = AgentState(brief="Test brief")
        assert state.status == "planning"
        assert state.iteration == 0
        assert state.evidence == []
        assert state.tool_trace == []

    def test_state_transitions(self):
        state = AgentState(brief="Test", status="planning")
        assert state.status == "planning"

        state = AgentState(**{**state.__dict__, "status": "gathering"})
        assert state.status == "gathering"


class TestResearchMemo:
    def test_to_markdown(self, sample_memo):
        md = sample_memo.to_markdown()
        assert "# AI Code Assistants" in md
        assert "## Market Overview" in md
        assert "## Pricing Comparison" in md

    def test_to_json(self, sample_memo):
        data = sample_memo.to_json()
        assert data["title"] == "AI Code Assistants: Competitive Landscape 2025"
        assert len(data["sections"]) == 2
        assert data["confidence_score"] == 0.87


class TestRouting:
    def test_should_continue_gathering_with_pending(self, agent, sample_plan):
        state = AgentState(brief="test", plan=sample_plan, iteration=0)
        result = agent._should_continue_gathering(state)
        assert result == "gather"

    def test_should_stop_gathering_when_done(self, agent, sample_plan):
        for task in sample_plan.tasks:
            task.status = "done"
        state = AgentState(brief="test", plan=sample_plan, iteration=1)
        result = agent._should_continue_gathering(state)
        assert result == "draft"

    def test_should_stop_at_max_iterations(self, agent, sample_plan):
        state = AgentState(brief="test", plan=sample_plan, iteration=15, max_iterations=15)
        result = agent._should_continue_gathering(state)
        assert result == "draft"

    def test_should_revise_low_confidence(self, agent, sample_memo):
        sample_memo.confidence_score = 0.5
        state = AgentState(brief="test", memo=sample_memo, iteration=1)
        result = agent._should_revise(state)
        assert result == "revise"

    def test_should_finish_high_confidence(self, agent, sample_memo):
        state = AgentState(brief="test", memo=sample_memo, iteration=1)
        result = agent._should_revise(state)
        assert result == "done"
