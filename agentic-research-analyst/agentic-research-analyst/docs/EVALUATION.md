# Evaluation Framework — Agentic Research Analyst

> How we measure, track, and prevent regressions in a non-deterministic system.

---

## 1. Why Evaluation Matters for Agents

LLM agents are **non-deterministic by design**.  The same brief can
produce different tool-call sequences, different retrieved content,
and therefore different memos.  Traditional unit tests catch code
bugs but miss **behavioural drift** — a prompt tweak that improves
finance briefs might silently degrade healthcare ones.

Our evaluation framework treats the agent like a **statistical system**:
we measure distributions, not point values.

---

## 2. Benchmark Suite

### 2.1 Brief Corpus

13 research briefs across 4 domains, each with human-written reference
memos and expected citation sets:

| Domain | Count | Example Brief |
|---|---|---|
| **Technology** | 4 | "Compare transformer vs. SSM architectures for long-context tasks" |
| **Finance** | 3 | "Analyse the impact of Basel IV on mid-tier European banks" |
| **Healthcare** | 3 | "Summarise recent GLP-1 receptor agonist trial outcomes" |
| **Policy** | 3 | "Evaluate EU AI Act compliance requirements for foundation models" |

### 2.2 Run Modes

```bash
# Smoke test — 3 briefs, ~2 min, good for PR checks
python -m src.evaluation.run --suite smoke

# Full suite — all 13 briefs, ~15 min
python -m src.evaluation.run --suite full

# Single brief for debugging
python -m src.evaluation.run --brief "tech_transformers_vs_ssm"
```

---

## 3. Judges (LLM-as-Judge)

Each memo is scored by three independent judges:

| Judge | What It Measures | Scale |
|---|---|---|
| **CompletenessJudge** | Does the memo address all sub-questions in the brief? | 1–10 |
| **CitationJudge** | Are claims backed by valid, traceable citations? | 1–10 |
| **CoherenceJudge** | Is the memo well-structured, logical, non-repetitive? | 1–10 |

**Judge reliability:**  We measured inter-judge agreement by running
each judge 5× on the same memo.  Krippendorff's α = 0.81 (acceptable).

### Judge Prompting Strategy

Each judge receives:
1. The original brief
2. The generated memo
3. A rubric with anchor examples for scores 2, 5, 8, and 10
4. Instruction to output JSON: `{"score": int, "reasoning": str}`

---

## 4. Aggregate Metrics

```
┌────────────────────────────────────────────────────────────┐
│  Metric              │  Formula                           │
├────────────────────────────────────────────────────────────┤
│  Task Completion     │  % of briefs scoring ≥ 7 on all 3  │
│  Mean Quality Score  │  avg(completeness, citation, coh.) │
│  Cost Efficiency     │  total_cost / n_briefs             │
│  Tool Call Density   │  avg tool calls per brief          │
│  Revision Rate       │  % of briefs requiring ≥ 1 loop   │
└────────────────────────────────────────────────────────────┘
```

Current baselines (v0.3.1, 50-run average):

| Metric | Value | Target |
|---|---|---|
| Task Completion | **87 %** | ≥ 85 % |
| Mean Quality | **8.1 / 10** | ≥ 7.5 |
| Cost / Brief | **$0.18** | ≤ $0.25 |
| Avg Tool Calls | **6.3** | ≤ 10 |
| Revision Rate | **34 %** | — |

---

## 5. Regression Detection

```bash
python -m src.evaluation.regression --baseline v0.3.0 --candidate v0.3.1
```

Produces a diff table:

```
┌──────────────────┬──────────┬───────────┬────────┐
│ Metric           │ Baseline │ Candidate │ Δ      │
├──────────────────┼──────────┼───────────┼────────┤
│ Task Completion  │ 84.6 %   │ 87.0 %    │ +2.4 % │
│ Mean Quality     │ 7.8      │ 8.1       │ +0.3   │
│ Cost / Brief     │ $0.21    │ $0.18     │ -$0.03 │
│ Avg Tool Calls   │ 7.1      │ 6.3       │ -0.8   │
└──────────────────┴──────────┴───────────┴────────┘
```

A regression is flagged when:
- Task completion drops > 3 percentage points, **or**
- Any single domain drops > 5 points, **or**
- Cost per brief increases > 25 %

Flagged regressions block the CI merge (see `.github/workflows/ci.yml`).

---

## 6. Running Evaluations Locally

```bash
# 1. Install eval dependencies
pip install -e ".[eval]"

# 2. Set API keys
export OPENAI_API_KEY=sk-...
export TAVILY_API_KEY=tvly-...

# 3. Run smoke suite
python -m src.evaluation.run --suite smoke --output eval_results/

# 4. View results
cat eval_results/summary.json
```

Results are saved as JSON for programmatic consumption and as
Markdown tables for PR comments.

---

## 7. Known Limitations

- **Judge bias:** LLM judges favour verbose output.  We partially
  mitigate this with explicit rubric anchors penalising filler.
- **Reference set size:** 13 briefs is enough for directional signal
  but not for statistical significance at the domain level.
  Expanding to 30+ briefs is on the roadmap.
- **Cost of eval:** A full suite run costs ~$2.50 in API calls.
  The smoke suite costs ~$0.60.
