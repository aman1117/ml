# Project NN: <name>

One sentence: who uses it and what it does for them.

**Dates:** start YYYY-MM-DD, ship by YYYY-MM-DD (6–8 weeks).
**Modules used:** e.g. `../../rag/`, `../../evals/`, `../../serving/`.
**Run everything:** `python projects/NN-name/run.py` (data → eval → serve → report).

---

## 1. Problem (write before coding)

- The user and the task:
- Why a simpler approach (search, a prompt, a rule) is not enough:
- Out of scope:

## 2. Success metric (write before coding)

- Primary metric and target (e.g. answer correctness ≥ 80% with citation precision ≥ 90%):
- Guardrail metrics (latency p95, cost per request, abstention rate, ...):
- The smallest improvement that matters, and the eval size needed to detect it
  (`../../evals/stats.py`):

## 3. Eval set (write before coding)

- Source of examples (sampled how, from where):
- Size, splits (dev / frozen held-out test), labeling guide:
- Slices that must be reported separately (question types, languages, difficulty):
- Contamination check against any training / retrieval data:

## 4. Baseline

| System | Primary metric (95% CI) | p95 latency | Cost / 1k requests |
|--------|-------------------------|-------------|--------------------|
| Baseline: | | | |

## 5. System

Diagram and a short description of each component, with the file that implements it.

## 6. Experiments (one change at a time)

| # | Change | Metric (95% CI) | Δ vs previous (paired p-value) | Kept? |
|---|--------|-----------------|-------------------------------|-------|
| 1 | | | | |

## 7. Serving and monitoring

- Endpoint, model version, hardware:
- Load test result at target load (throughput, p50/p95/p99, error rate):
- SLOs and alert rules:
- Dashboard / monitoring report location:

## 8. Failure modes

See [`failure_log.md`](failure_log.md). Summary of the top 3 and what fixed them.

## 9. Write-up

- Final numbers (held-out test, run once):
- Trade-offs made (quality vs latency vs cost):
- What surprised me:
- What I would do next with 2× time / 10× budget:
