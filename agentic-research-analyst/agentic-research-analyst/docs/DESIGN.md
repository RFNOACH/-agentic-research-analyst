# Design Document — Agentic Research Analyst

> Architecture decisions, trade-offs, and rationale.

---

## 1. Problem Statement

Manual research workflows are slow, inconsistent, and hard to audit.
Analysts spend 60–70 % of their time on **retrieval** (searching, reading PDFs,
extracting data) rather than **analysis**.  An agentic system should:

| Requirement | Metric |
|---|---|
| Cover a research brief end-to-end | Task completion ≥ 85 % |
| Produce cited, structured output | Citation accuracy ≥ 90 % |
| Stay cost-effective | < $0.25 per brief |
| Be auditable | Full tool-use trace logged |

---

## 2. High-Level Architecture

```
┌─────────────┐
│  User Brief  │
└──────┬──────┘
       ▼
┌──────────────┐     ┌────────────────────────────────────┐
│   Planner    │────▶│  SubTask queue (JSON-structured)   │
└──────┬───────┘     └────────────────────────────────────┘
       ▼
┌──────────────┐     ┌──────┬──────┬──────┬──────────────┐
│   Gatherer   │────▶│ Web  │ PDF  │ Code │ VectorStore  │
│  (tool loop) │     │Search│Reader│ Exec │   (ChromaDB) │
└──────┬───────┘     └──────┴──────┴──────┴──────────────┘
       ▼
┌──────────────┐
│   Analyzer   │  ← cross-references, deduplicates
└──────┬───────┘
       ▼
┌──────────────┐
│   Drafter    │  ← produces ResearchMemo w/ citations
└──────┬───────┘
       ▼
┌──────────────┐
│   Reviewer   │  ← self-critique pass; may loop back
└──────────────┘
```

### Why LangGraph?

We need **conditional edges** (reviewer can loop back to gatherer),
**typed state** (AgentState dataclass flows through every node), and
**streaming** (token-level progress for the API).  LangGraph gives us
a compiled graph with these properties out of the box—no custom
orchestration framework needed.

---

## 3. Node Design

### 3.1 Planner

**Input:** raw brief (str)  
**Output:** `TaskPlan` with ordered `SubTask` list  
**Approach:** Single LLM call with JSON-structured output.  We prompt
for explicit decomposition so the gatherer knows *what* to search vs.
blindly querying the whole brief.

**Trade-off:** One planning call adds ~2 s latency but reduces
irrelevant tool calls by ~40 % (measured in eval suite).

### 3.2 Gatherer

**Input:** `AgentState` with pending sub-tasks  
**Output:** `AgentState` with populated `gathered_content`  
**Approach:** Iterates sub-tasks, picks the right tool per task type,
stores results in the vector store for downstream retrieval.

**Tool routing heuristic:**

| SubTask type | Tool | Fallback |
|---|---|---|
| `web` | `web_search` | — |
| `document` | `pdf_reader` | `web_search` |
| `data` | `code_executor` | — |
| `any` | Vector store similarity search first, then web | — |

### 3.3 Analyzer

Embeds all gathered content, runs similarity-based deduplication
(cosine > 0.92 → drop), and builds a *relevance-ranked* context
window that fits within the LLM's token budget.

### 3.4 Drafter

Single LLM call that receives the ranked context and produces a
`ResearchMemo` with structured sections and inline `[N]` citation
markers.

### 3.5 Reviewer

Self-critique pass.  The LLM scores the draft on three axes
(completeness, citation quality, coherence) on a 1–10 scale.
If **any** axis < 7, the graph routes back to the gatherer with
the reviewer's feedback appended to state.  Max 2 revision loops
to bound cost.

---

## 4. State Management

```python
@dataclass
class AgentState:
    brief: str
    plan: TaskPlan | None
    gathered_content: list[dict]
    analysis: str
    memo: ResearchMemo | None
    review_feedback: str
    revision_count: int          # caps at MAX_REVISIONS (2)
    tool_trace: list[dict]       # full audit log
    metadata: dict               # timing, cost, token counts
```

Every node receives the full state and returns a *delta* dict that
LangGraph merges.  This makes nodes independently testable.

---

## 5. Cost Control

| Lever | Mechanism |
|---|---|
| **Max tool calls** | Hard cap (default 15) in gatherer loop |
| **Token budget** | Analyzer truncates context to 80 % of model window |
| **Revision cap** | Max 2 loops through reviewer → gatherer |
| **Model routing** | Planning uses `gpt-4o-mini`; drafting uses `gpt-4o` |

Average cost per brief: **$0.18** (measured across 50-brief eval set).

---

## 6. Observability

Every tool invocation is logged to `tool_trace` with:

```json
{
  "tool": "web_search",
  "input": {"query": "..."},
  "output_preview": "first 200 chars...",
  "latency_ms": 1230,
  "tokens_used": 450,
  "cost_usd": 0.002
}
```

The trace is returned alongside the memo in both API and CLI modes,
enabling **full reproducibility audits**.

---

## 7. Future Considerations

- **Multi-agent delegation** — split gatherer into parallel sub-agents
  for independent sub-tasks (requires LangGraph `Send` API).
- **Human-in-the-loop** — add an approval gate before the drafter
  node for high-stakes briefs.
- **Streaming citations** — emit citation links as they're discovered
  rather than post-hoc linking.
