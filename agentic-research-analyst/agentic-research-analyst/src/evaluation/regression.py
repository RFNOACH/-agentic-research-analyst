"""
Regression Tracker — Compare evaluation results across agent versions.
"""

from __future__ import annotations

import json
from pathlib import Path


def compare(baseline_path: str, current_path: str) -> str:
    """
    Compare two evaluation runs and produce a diff table.

    Parameters
    ----------
    baseline_path : str
        Path to baseline evaluation JSON.
    current_path : str
        Path to current evaluation JSON.

    Returns
    -------
    str
        Formatted comparison table.
    """
    baseline = json.loads(Path(baseline_path).read_text())
    current = json.loads(Path(current_path).read_text())

    metrics = ["completion_rate", "avg_tool_calls", "avg_cost_usd"]

    lines = [
        f"{'Metric':<20} {'Baseline':>10} {'Current':>10} {'Delta':>10}",
        f"{'-'*20} {'-'*10} {'-'*10} {'-'*10}",
    ]

    for m in metrics:
        b_val = baseline.get(m, 0)
        c_val = current.get(m, 0)
        delta = c_val - b_val
        sign = "+" if delta > 0 else ""
        lines.append(f"{m:<20} {b_val:>10.2f} {c_val:>10.2f} {sign}{delta:>9.2f}")

    return "\n".join(lines)
