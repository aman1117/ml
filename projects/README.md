# Projects: ship one serious system every 6–8 weeks

The module folders teach one mechanism each. This folder is where they turn into
**shipped, end-to-end systems**: real data, a real eval set, a baseline, a deployed
endpoint, monitoring, and a written post-mortem of what broke. That is the evidence an
AI-engineer interview at a frontier lab actually looks for: not "I followed a tutorial",
but "here is the system, here is how I measured it, here is the failure I found in the
logs and how I fixed it."

> **Prerequisite:** the module that each project builds on (listed in the roadmap
> below). Every project reuses code you wrote in the modules; don't reach for a wrapper
> framework that hides the stack.

## Files

| File | What it is |
|------|-----------------|
| `README.md` | This page: the rules, the cadence, the roadmap of projects. |
| `_template/README.md` | Copy this to start a project: the one-page spec + the write-up skeleton. |
| `_template/failure_log.md` | The running log of failure modes found in eval runs and production logs. |

### How to start a project

1. `cp -r projects/_template projects/01-<short-name>`
2. Fill in **sections 1–3 of the spec before writing any model code** (problem, success
   metric, eval set). If you can't write the eval, you don't understand the problem yet.
3. Build the dumbest baseline that runs end to end (week 1). Measure it.
4. Iterate: change one thing, re-run the eval, keep the change only if the paired test
   in `../evals/` says it helped.
5. Ship it behind the server in `../serving/`, run `load_test.py`, read the logs with
   `monitor.py`, and add every new failure to `failure_log.md` **and** to the eval set.
6. Write it up (template section 9) and tick it off below.

---

## 1. The rules

- **One project every 6–8 weeks.** Fixed scope, fixed date. Cut features, not the eval
  or the monitoring.
- **Eval first.** At least 100 labeled examples (see `../evals/` section on how many you
  need for the effect size you care about) before the first real model run, with a
  frozen held-out split you never tune on.
- **Baseline first.** Keyword search before dense retrieval, a prompt before a
  fine-tune, a single call before an agent. Every fancier component must beat the
  baseline with a confidence interval that excludes zero.
- **End to end or it didn't happen.** Data pipeline → model/system → HTTP endpoint →
  structured logs → dashboard/report → alerts. Reproducible from one command.
- **No wrappers that hide the stack.** Use the code from the modules (BPE, LoRA, BM25,
  the agent loop, the server). Libraries are fine for plumbing (HTTP, JSON, plotting) and
  for loading open-model weights.
- **Failure modes are the product.** Keep `failure_log.md` current. Each entry becomes a
  regression eval case.
- **Write it up.** A README a stranger can run, with the honest numbers, and a
  "what I would do next" section.

## 2. What "done" means (checklist for every project)

- [ ] Problem statement and success metric written before coding
- [ ] Eval set: ≥ 100 examples, held-out split, labeling guide, contamination check
- [ ] Baseline measured with error bars
- [ ] Final system beats baseline (paired test, $p < 0.05$) on the held-out split
- [ ] Served behind an HTTP endpoint with `/health` and `/metrics`
- [ ] Load test: p50/p95/p99 latency, throughput, error rate at target load
- [ ] Structured request logs + monitoring report + at least two alert rules
- [ ] `failure_log.md` with ≥ 5 real failure modes and what was done about each
- [ ] Regression eval gate in CI (`../evals/harness.py compare`)
- [ ] Write-up with cost/latency/quality trade-offs and next steps

## 3. Roadmap

Each project leans on one or two modules but is a whole system. Suggested order:

| # | Project | Builds on | The hard part you must solve |
|---|---------|-----------|------------------------------|
| 1 | **Docs QA with citations**: RAG over a real documentation set you care about (e.g. PyTorch or a company's public docs), answers with citations and abstention. | `rag/`, `evals/`, `serving/` | Retrieval recall on paraphrased questions; faithfulness; knowing when to say "I don't know". |
| 2 | **Domain fine-tune**: LoRA-fine-tune an open 0.5B–3B model on a narrow domain (e.g. SQL for one schema, log triage, support tickets) and beat prompting + RAG on a held-out set. | `finetuning/`, `evals/`, `tokenizer/` | A clean dataset; measuring forgetting; showing the fine-tune is worth its cost vs. a prompt. |
| 3 | **Tool-using agent with reliability evals**: an agent that completes multi-step tasks against a real API or sandbox, measured with pass^k and a failure taxonomy. | `agents/`, `evals/`, `serving/` | Reliability across repeated runs; prompt injection through tool outputs; budgets and loop detection. |
| 4 | **Pretrain a small LLaMA from scratch** (10M–100M params) on a public corpus with your BPE tokenizer, multi-GPU (rented) with DDP/FSDP, and report loss vs. compute against Chinchilla. | `llama/`, `tokenizer/`, `distributed/` | Throughput (MFU), stable training (no loss spikes), checkpoint/restart. |
| 5 | **Inference service**: serve the project-4 model (or an open model) with continuous batching, quantization and speculative decoding; hit a latency SLO under load. | `serving/`, `llama/` | Tail latency under bursty load; quality loss from quantization measured with evals. |
| 6 | **Production-monitored assistant**: combine 1–3 into one product, run it for real users/traffic for weeks, and drive improvements from the logs. | everything | Drift, new failure modes, closing the loop logs → evals → fixes → regression gate. |

### Status

- [ ] 01 —
- [ ] 02 —
- [ ] 03 —
- [ ] 04 —
- [ ] 05 —
- [ ] 06 —

## 4. Why this matters for an AI engineer

Frontier-lab applied roles (and most strong AI-product teams) hire for the loop
**build → measure → find failures → fix → prevent regressions**, at production quality.
The modules prove you understand the mechanisms; the projects prove you can ship and
measure a system with them. In interviews you will be asked to walk through one project
end to end: the eval design, the baseline, the biggest failure mode you found and how,
and the trade-off you made between quality, latency and cost. Have that story ready
for every project in this folder.
