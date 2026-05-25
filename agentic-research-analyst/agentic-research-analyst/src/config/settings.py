"""
Configuration — Typed settings management with environment overrides.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class AgentConfig:
    model: str = "gpt-4o"
    temperature: float = 0.2
    max_iterations: int = 15
    max_tools_per_brief: int = 10


@dataclass
class WebSearchConfig:
    provider: str = "tavily"
    max_results: int = 8


@dataclass
class PDFReaderConfig:
    chunk_size: int = 1000
    overlap: int = 200


@dataclass
class CodeExecutorConfig:
    timeout_seconds: int = 30
    sandbox: bool = True


@dataclass
class VectorStoreConfig:
    provider: str = "chromadb"
    embedding_model: str = "text-embedding-3-small"


@dataclass
class ToolsConfig:
    web_search: WebSearchConfig = field(default_factory=WebSearchConfig)
    pdf_reader: PDFReaderConfig = field(default_factory=PDFReaderConfig)
    code_executor: CodeExecutorConfig = field(default_factory=CodeExecutorConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)


@dataclass
class EvaluationConfig:
    judge_model: str = "gpt-4o"
    num_briefs: int = 50
    domains: list[str] = field(default_factory=lambda: [
        "technology", "finance", "healthcare", "policy"
    ])


@dataclass
class Settings:
    """Application settings with environment variable overrides."""

    agent: AgentConfig = field(default_factory=AgentConfig)
    tools: ToolsConfig = field(default_factory=ToolsConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def __post_init__(self) -> None:
        # Override from environment
        if model := os.getenv("AGENT_MODEL"):
            self.agent.model = model
        if temp := os.getenv("AGENT_TEMPERATURE"):
            self.agent.temperature = float(temp)
        if tavily_results := os.getenv("TAVILY_MAX_RESULTS"):
            self.tools.web_search.max_results = int(tavily_results)
