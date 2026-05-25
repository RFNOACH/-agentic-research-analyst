"""
Task Planner — Decomposes research briefs into executable sub-tasks.

Uses structured output from the LLM to generate a dependency-aware
task graph with tool assignments.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Literal

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage


@dataclass
class SubTask:
    """A single executable sub-task within a research plan."""

    id: str
    description: str
    tool: Literal["web_search", "pdf_reader", "code_executor"]
    query: str = ""
    file_path: str = ""
    code: str = ""
    depends_on: list[str] = field(default_factory=list)
    status: Literal["pending", "done", "failed"] = "pending"


@dataclass
class TaskPlan:
    """Structured research plan with sub-tasks and reasoning trace."""

    tasks: list[SubTask]
    reasoning: str
    brief_summary: str


PLANNER_SYSTEM_PROMPT = """You are a research planning agent. Given a research brief,
decompose it into 3-8 concrete sub-tasks that can be executed by tools.

Available tools:
- web_search: Search the web for current information. Provide a search query.
- pdf_reader: Extract text from a PDF document. Provide a file path.
- code_executor: Run Python code for data analysis or computation. Provide code.

Respond in JSON format:
{
  "reasoning": "Your step-by-step reasoning about how to approach this brief",
  "brief_summary": "One-sentence summary of the brief",
  "tasks": [
    {
      "id": "t1",
      "description": "What this task accomplishes",
      "tool": "web_search",
      "query": "search query here",
      "depends_on": []
    }
  ]
}

Rules:
1. Each task should be focused and specific.
2. Use depends_on to express task ordering when needed.
3. Prefer web_search for factual/current information.
4. Use code_executor for comparisons, tables, calculations.
5. Keep total tasks between 3-8 for efficiency.
"""


class Planner:
    """Decomposes a research brief into a structured task plan."""

    def __init__(self, llm: ChatOpenAI) -> None:
        self.llm = llm

    def decompose(self, brief: str) -> TaskPlan:
        """
        Convert a natural-language research brief into an executable task plan.

        Parameters
        ----------
        brief : str
            The research question or task description.

        Returns
        -------
        TaskPlan
            Structured plan with sub-tasks and tool assignments.
        """
        messages = [
            SystemMessage(content=PLANNER_SYSTEM_PROMPT),
            HumanMessage(content=f"Research brief:\n{brief}"),
        ]

        response = self.llm.invoke(messages)
        raw = response.content

        # Parse JSON from response
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code blocks
            if "```json" in raw:
                json_str = raw.split("```json")[1].split("```")[0].strip()
                data = json.loads(json_str)
            elif "```" in raw:
                json_str = raw.split("```")[1].split("```")[0].strip()
                data = json.loads(json_str)
            else:
                raise ValueError(f"Failed to parse planner output: {raw[:200]}")

        tasks = [
            SubTask(
                id=t["id"],
                description=t["description"],
                tool=t["tool"],
                query=t.get("query", ""),
                file_path=t.get("file_path", ""),
                code=t.get("code", ""),
                depends_on=t.get("depends_on", []),
            )
            for t in data["tasks"]
        ]

        return TaskPlan(
            tasks=tasks,
            reasoning=data["reasoning"],
            brief_summary=data["brief_summary"],
        )
