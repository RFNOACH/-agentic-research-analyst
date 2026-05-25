"""
Agentic Research Analyst — Core LangGraph Agent

Autonomous multi-tool agent that decomposes research briefs into sub-tasks,
executes tool chains, and synthesizes findings into cited research memos.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

from langgraph.graph import END, StateGraph
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.planner import Planner, TaskPlan
from src.agents.synthesizer import Synthesizer, ResearchMemo
from src.tools.web_search import WebSearchTool
from src.tools.pdf_reader import PDFReaderTool
from src.tools.code_executor import CodeExecutor
from src.tools.vector_store import VectorStoreManager
from src.tools.citation_linker import CitationLinker
from src.config.settings import Settings


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

@dataclass
class AgentState:
    """Immutable snapshot of the agent's working memory at each graph node."""

    brief: str = ""
    plan: TaskPlan | None = None
    evidence: list[dict[str, Any]] = field(default_factory=list)
    tool_trace: list[dict[str, Any]] = field(default_factory=list)
    memo: ResearchMemo | None = None
    iteration: int = 0
    max_iterations: int = 15
    status: Literal["planning", "gathering", "analyzing", "drafting", "done", "failed"] = "planning"


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class ResearchAgent:
    """
    LangGraph-based research agent.

    Workflow
    --------
    1. **Plan** — decompose the brief into scoped sub-tasks.
    2. **Gather** — execute tools (web search, PDF, code) to collect evidence.
    3. **Analyze** — store evidence in vector DB, verify sufficiency.
    4. **Draft** — synthesize a cited research memo.
    5. **Review** — quality-check; loop back if needed.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self.llm = ChatOpenAI(
            model=self.settings.agent.model,
            temperature=self.settings.agent.temperature,
        )

        # Tools
        self.web_search = WebSearchTool(max_results=self.settings.tools.web_search.max_results)
        self.pdf_reader = PDFReaderTool(
            chunk_size=self.settings.tools.pdf_reader.chunk_size,
            overlap=self.settings.tools.pdf_reader.overlap,
        )
        self.code_executor = CodeExecutor(
            timeout=self.settings.tools.code_executor.timeout_seconds,
        )
        self.vector_store = VectorStoreManager(
            embedding_model=self.settings.tools.vector_store.embedding_model,
        )
        self.citation_linker = CitationLinker()

        # Sub-components
        self.planner = Planner(self.llm)
        self.synthesizer = Synthesizer(self.llm, self.citation_linker)

        # Build graph
        self.graph = self._build_graph()

    # ---- Graph construction ------------------------------------------------

    def _build_graph(self) -> StateGraph:
        g = StateGraph(AgentState)

        g.add_node("plan", self._plan_node)
        g.add_node("gather", self._gather_node)
        g.add_node("analyze", self._analyze_node)
        g.add_node("draft", self._draft_node)
        g.add_node("review", self._review_node)

        g.set_entry_point("plan")

        g.add_edge("plan", "gather")
        g.add_edge("gather", "analyze")
        g.add_conditional_edges(
            "analyze",
            self._should_continue_gathering,
            {"gather": "gather", "draft": "draft"},
        )
        g.add_edge("draft", "review")
        g.add_conditional_edges(
            "review",
            self._should_revise,
            {"revise": "draft", "done": END},
        )

        return g.compile()

    # ---- Nodes -------------------------------------------------------------

    def _plan_node(self, state: AgentState) -> AgentState:
        """Decompose brief into a structured task plan."""
        plan = self.planner.decompose(state.brief)
        trace_entry = {
            "step": len(state.tool_trace) + 1,
            "action": "plan",
            "thought": plan.reasoning,
            "sub_tasks": [t.description for t in plan.tasks],
            "timestamp": time.time(),
        }
        return AgentState(
            **{**state.__dict__, "plan": plan, "status": "gathering",
               "tool_trace": [*state.tool_trace, trace_entry]}
        )

    def _gather_node(self, state: AgentState) -> AgentState:
        """Execute the next pending tool call from the plan."""
        new_evidence: list[dict] = []
        new_traces: list[dict] = []

        for task in state.plan.tasks:
            if task.status == "pending":
                tool_name = task.tool
                t0 = time.time()

                if tool_name == "web_search":
                    results = self.web_search.search(task.query)
                    new_evidence.extend(results)
                    task.status = "done"
                    new_traces.append({
                        "step": len(state.tool_trace) + len(new_traces) + 1,
                        "action": "web_search",
                        "query": task.query,
                        "results_count": len(results),
                        "relevant_results": sum(1 for r in results if r.get("relevance", 0) > 0.5),
                        "latency_ms": int((time.time() - t0) * 1000),
                    })

                elif tool_name == "pdf_reader":
                    chunks = self.pdf_reader.extract(task.file_path)
                    new_evidence.extend(chunks)
                    task.status = "done"
                    new_traces.append({
                        "step": len(state.tool_trace) + len(new_traces) + 1,
                        "action": "pdf_reader",
                        "file": task.file_path,
                        "chunks_extracted": len(chunks),
                        "latency_ms": int((time.time() - t0) * 1000),
                    })

                elif tool_name == "code_executor":
                    result = self.code_executor.run(task.code)
                    new_evidence.append({"type": "code_output", "content": result.output})
                    task.status = "done"
                    new_traces.append({
                        "step": len(state.tool_trace) + len(new_traces) + 1,
                        "action": "code_executor",
                        "code": task.code[:200],
                        "output_preview": result.output[:200],
                        "latency_ms": int((time.time() - t0) * 1000),
                    })

        return AgentState(
            **{**state.__dict__,
               "evidence": [*state.evidence, *new_evidence],
               "tool_trace": [*state.tool_trace, *new_traces],
               "iteration": state.iteration + 1}
        )

    def _analyze_node(self, state: AgentState) -> AgentState:
        """Index evidence into vector store for semantic retrieval."""
        self.vector_store.index(state.evidence)
        return AgentState(**{**state.__dict__, "status": "analyzing"})

    def _draft_node(self, state: AgentState) -> AgentState:
        """Synthesize evidence into a research memo with citations."""
        memo = self.synthesizer.draft(
            brief=state.brief,
            plan=state.plan,
            evidence=state.evidence,
            vector_store=self.vector_store,
        )

        trace_entry = {
            "step": len(state.tool_trace) + 1,
            "action": "synthesize",
            "sections_generated": len(memo.sections),
            "citations_linked": len(memo.citations),
            "confidence_score": memo.confidence_score,
        }

        return AgentState(
            **{**state.__dict__, "memo": memo, "status": "drafting",
               "tool_trace": [*state.tool_trace, trace_entry]}
        )

    def _review_node(self, state: AgentState) -> AgentState:
        """Quality check on the draft memo."""
        return AgentState(**{**state.__dict__, "status": "done"})

    # ---- Routing -----------------------------------------------------------

    def _should_continue_gathering(self, state: AgentState) -> str:
        pending = [t for t in state.plan.tasks if t.status == "pending"]
        if pending and state.iteration < state.max_iterations:
            return "gather"
        return "draft"

    def _should_revise(self, state: AgentState) -> str:
        if state.memo and state.memo.confidence_score >= 0.75:
            return "done"
        if state.iteration >= state.max_iterations:
            return "done"
        return "revise"

    # ---- Public API --------------------------------------------------------

    def run(
        self,
        brief: str,
        output_format: str = "markdown",
        max_tools: int = 10,
        temperature: float | None = None,
    ) -> ResearchMemo:
        """
        Execute a full research workflow.

        Parameters
        ----------
        brief : str
            The research question or task description.
        output_format : str
            Output format for the memo ('markdown', 'json', 'html').
        max_tools : int
            Maximum tool invocations per run.
        temperature : float, optional
            Override LLM temperature for this run.

        Returns
        -------
        ResearchMemo
            The completed research memo with citations and trace.
        """
        if temperature is not None:
            self.llm.temperature = temperature

        initial_state = AgentState(
            brief=brief,
            max_iterations=max_tools,
        )

        final_state = self.graph.invoke(initial_state)
        return final_state.memo
