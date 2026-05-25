"""
Web Search Tool — Tavily-powered real-time information retrieval.

Wraps the Tavily Search API with relevance scoring, deduplication,
and structured output formatting for the agent evidence pool.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from tavily import TavilyClient


@dataclass
class SearchResult:
    """A single web search result with metadata."""

    url: str
    title: str
    content: str
    relevance: float
    published_date: str | None = None


class WebSearchTool:
    """
    Web search tool using Tavily API.

    Features
    --------
    - Relevance-filtered results
    - Automatic deduplication by URL
    - Structured output for evidence pool ingestion
    """

    def __init__(self, max_results: int = 8) -> None:
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            raise ValueError("TAVILY_API_KEY environment variable required")
        self.client = TavilyClient(api_key=api_key)
        self.max_results = max_results

    def search(self, query: str, include_domains: list[str] | None = None) -> list[dict[str, Any]]:
        """
        Execute a web search and return structured results.

        Parameters
        ----------
        query : str
            Search query string.
        include_domains : list[str], optional
            Restrict search to specific domains.

        Returns
        -------
        list[dict]
            Evidence-formatted search results.
        """
        params = {
            "query": query,
            "max_results": self.max_results,
            "search_depth": "advanced",
            "include_answer": True,
        }
        if include_domains:
            params["include_domains"] = include_domains

        response = self.client.search(**params)

        results = []
        seen_urls = set()

        for item in response.get("results", []):
            url = item.get("url", "")
            if url in seen_urls:
                continue
            seen_urls.add(url)

            results.append({
                "type": "web_search",
                "url": url,
                "title": item.get("title", ""),
                "content": item.get("content", ""),
                "relevance": item.get("score", 0.5),
                "published_date": item.get("published_date"),
            })

        return results
