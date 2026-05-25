"""
Metrics — Compute aggregate evaluation metrics from individual brief results.
"""

from __future__ import annotations

from typing import Any


def compute_metrics(results: list[dict[str, Any]]) -> dict[str, float]:
    """
    Compute aggregate metrics from evaluation results.

    Parameters
    ----------
    results : list[dict]
        Individual brief evaluation results.

    Returns
    -------
    dict[str, float]
        Aggregate metrics.
    """
    completed = [r for r in results if r.get("completed", False)]
    total = len(results)

    if not completed:
        return {
            "completion_rate": 0.0,
            "avg_completeness": 0.0,
            "avg_citation_accuracy": 0.0,
            "avg_coherence": 0.0,
            "avg_tool_calls": 0.0,
            "avg_cost_usd": 0.0,
            "avg_latency_s": 0.0,
        }

    return {
        "completion_rate": len(completed) / total,
        "avg_completeness": _mean([r.get("completeness_score", 0) for r in completed]),
        "avg_citation_accuracy": _mean([r.get("citation_accuracy", 0) for r in completed]),
        "avg_coherence": _mean([r.get("coherence_score", 0) for r in completed]),
        "avg_tool_calls": _mean([r.get("tool_calls", 0) for r in completed]),
        "avg_cost_usd": _mean([r.get("cost_usd", 0) for r in completed]),
        "avg_latency_s": _mean([r.get("latency_s", 0) for r in completed]),
    }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
