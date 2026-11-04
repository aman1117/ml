"""
A tiny eval harness from scratch: the part of lm-evaluation-harness / OpenAI evals /
every internal eval stack that actually matters, in one file. (README sections 2-3, 11.)

    Task      name + examples (JSONL) + prompt template + how to call the model
              + scorer (one example -> score) + aggregation (scores -> metrics)
    Model     anything with generate / logprobs / next_char_logits (models.py
              wraps the repo's GPT, RNN and a bigram baseline; an HTTP API model
              would be ~30 lines more)
    run       deterministic per-example seeds, a response cache keyed on the model's
              weight fingerprint, per-example JSONL log (input, output, score,
              latency), and a summary JSON with bootstrap and clustered error bars
    compare   a PAIRED test of a new results file against a stored baseline; exits
              with status 1 on a statistically significant regression (CI gate)

Tasks (evals/tasks/*.jsonl, built from ../rnn/input.txt by `build-tasks`;
"heldout" = paragraphs the models never trained on, "train" = paragraphs they did):
    bpc               bits per character of each sentence given the text before it
    next_word         greedy-decode the next word of the story; exact match
    next_word_sample  sample 20 next words at temperature 1 (held-out words); pass@k, pass^k
    calibration       next-character confidence vs accuracy (ECE, Brier)
    mc                hand-written multiple-choice completions, scored by log-likelihood

Run:
    python evals/harness.py list
    python evals/harness.py run --task next_word --model gpt
    python evals/harness.py run --task next_word bpc --model gpt rnn bigram gpt-full
    python evals/harness.py run --task next_word --model gpt@steps=300
    python evals/harness.py compare evals/runs/next_word/gpt.summary.json \\
                                    evals/runs/next_word/gpt_steps300.summary.json
    python evals/harness.py build-tasks       (regenerates bpc/next_word/calibration.jsonl)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
import warnings
import zlib
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

try:
    from typing import Protocol
except ImportError:  # pragma: no cover
    Protocol = object

import metrics as M
import stats as S
from models import (HELDOUT_PARA, get_model, load_story, paragraph_of, paragraph_spans,
                    split_offset)

warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)

HERE = os.path.dirname(os.path.abspath(__file__))
TASK_DIR = os.path.join(HERE, "tasks")
RUN_DIR = os.path.join(HERE, "runs")
PASS_KS = (1, 2, 5, 10, 20)
# Part of every cache key. The weight fingerprint can't see changes to the
# DECODING code (sampler, stop rule, windowing): bump this when you change them,
# or the cache will happily serve answers from the old code.
CACHE_VERSION = 2


# ----------------------------------------------------------------------------
# The two abstractions
# ----------------------------------------------------------------------------
class Model(Protocol):
    """What the harness needs from a model. Nothing else leaks in."""

    name: str
    fingerprint: str  # changes whenever the weights change (cache key)

    def generate(self, prompt: str, n: int = 1, temperature: float = 0.0,
                 max_chars: int = 20, stop: str = "", seed: int = 0) -> List[str]: ...

    def logprobs(self, prompt: str, continuation: str) -> List[float]: ...

    def next_char_logits(self, prompts: Sequence[str]) -> np.ndarray: ...


@dataclass
class Task:
    name: str
    file: str
    description: str
    template: str  # str.format template over the example's fields
    call: str  # "generate" | "logprobs" | "choices" | "next_char"
    scorer: Callable[[Dict, Any], Dict]  # (example, response) -> {"score": ..., extras}
    aggregate: Callable[[List[Dict]], Dict]  # per-example records -> extra metrics
    metric: str  # what "score" means
    higher_is_better: bool = True
    call_kwargs: Dict = field(default_factory=dict)
    splits: Optional[Sequence[str]] = None  # only these splits (None = all)

    def examples(self) -> List[Dict]:
        with open(os.path.join(TASK_DIR, self.file), encoding="utf-8") as f:
            exs = [json.loads(line) for line in f if line.strip()]
        return [e for e in exs if self.splits is None or e.get("split") in self.splits]

    def file_hash(self) -> str:
        """Hash of the examples AND of the task definition: changing the split
        filter or the sampling parameters must invalidate stored results just
        like editing the JSONL does."""
        h = hashlib.sha1()
        with open(os.path.join(TASK_DIR, self.file), "rb") as f:
            h.update(f.read())
        h.update(json.dumps([self.template, self.call, self.call_kwargs,
                             list(self.splits or [])], sort_keys=True).encode())
        return h.hexdigest()[:12]


# ----------------------------------------------------------------------------
# Scorers: one example + the model's raw response -> a score and extras
# ----------------------------------------------------------------------------
def score_bpc(ex, logprobs):
    nll = -float(np.sum(logprobs))
    n_chars, n_bytes = len(ex["target"]), len(ex["target"].encode("utf-8"))
    return dict(score=M.bits_per_char(nll, n_chars), nll_nats=nll, n_chars=n_chars,
                n_bytes=n_bytes)


def score_next_word(ex, completions):
    out, gold = completions[0], ex["target"]
    return dict(score=M.normalized_exact_match(out, gold, remove_articles=False),
                em_strict=M.exact_match(out, gold),
                em_squad=M.normalized_exact_match(out, gold),
                f1=M.token_f1(out, gold, remove_articles=False),
                normalized=M.normalize_answer(out, remove_articles=False))


def score_samples(ex, completions):
    correct = [M.normalized_exact_match(c, ex["target"], remove_articles=False)
               for c in completions]
    n, c = len(correct), int(sum(correct))
    return dict(score=c / n, n=n, c=c,
                **{f"pass@{k}": M.pass_at_k(n, c, k) for k in PASS_KS},
                **{f"pass^{k}": M.pass_hat_k(n, c, k) for k in PASS_KS})


def score_next_char(ex, probs):
    p = np.asarray(probs)
    t = ex["target_id"]
    pred = int(p.argmax())
    onehot = np.zeros_like(p)
    onehot[t] = 1.0
    return dict(score=float(pred == t), pred_id=pred, conf=float(p.max()), p_target=float(p[t]),
                nll=-math.log(max(p[t], 1e-300)), brier=float(((p - onehot) ** 2).sum()))


def score_choices(ex, choice_logprobs):
    sums = [float(np.sum(lp)) for lp in choice_logprobs]
    means = [float(np.mean(lp)) for lp in choice_logprobs]
    a = ex["answer"]
    return dict(score=float(int(np.argmax(sums)) == a), acc_norm=float(int(np.argmax(means)) == a),
                pred_sum=int(np.argmax(sums)), pred_mean=int(np.argmax(means)),
                sum_logprobs=sums, choice_lens=[len(c) for c in ex["choices"]])


# ----------------------------------------------------------------------------
# Aggregations: extra corpus-level metrics beyond mean(score) ± CI
# ----------------------------------------------------------------------------
def agg_none(recs):
    return {}


def agg_bpc(recs):
    nll = sum(r["nll_nats"] for r in recs)
    chars = sum(r["n_chars"] for r in recs)
    nbytes = sum(r["n_bytes"] for r in recs)
    return dict(bits_per_char_micro=M.bits_per_char(nll, chars),
                bits_per_byte=M.bits_per_byte(nll, nbytes),
                perplexity_per_char=M.perplexity(nll / chars), n_chars=chars)


def agg_next_word(recs):
    return {k: float(np.mean([r[k] for r in recs])) for k in ("em_strict", "em_squad", "f1")}


def agg_samples(recs):
    out = {f"pass@{k}": float(np.mean([r[f"pass@{k}"] for r in recs])) for k in PASS_KS}
    out.update({f"pass^{k}": float(np.mean([r[f"pass^{k}"] for r in recs])) for k in PASS_KS})
    return out


def agg_calibration(recs):
    conf = np.array([r["conf"] for r in recs])
    correct = np.array([r["score"] for r in recs])
    return dict(mean_conf=float(conf.mean()), ece_width=M.ece(conf, correct, 10, "width"),
                ece_mass=M.ece(conf, correct, 10, "mass"),
                brier=float(np.mean([r["brier"] for r in recs])),
                nll=float(np.mean([r["nll"] for r in recs])))


def agg_mc(recs):
    return dict(acc_norm=float(np.mean([r["acc_norm"] for r in recs])))


TASKS = {t.name: t for t in [
    Task("bpc", "bpc.jsonl", "bits per character of each sentence, given the text before it",
         "{context}", "logprobs", score_bpc, agg_bpc, "bits/char (lower is better)",
         higher_is_better=False),
    Task("next_word", "next_word.jsonl", "greedy next word, normalized exact match",
         "{context}", "generate", score_next_word, agg_next_word, "exact match (normalized)",
         call_kwargs=dict(n=1, temperature=0.0, max_chars=20, stop=" \n")),
    Task("next_word_sample", "next_word.jsonl", "20 sampled next words at T = 1: pass@k",
         "{context}", "generate", score_samples, agg_samples, "pass@1 (unbiased, n=20)",
         call_kwargs=dict(n=20, temperature=1.0, max_chars=20, stop=" \n"),
         splits=("heldout",)),  # 20 samples per example: held-out words only, to stay fast
    Task("calibration", "calibration.jsonl", "next-char top-1 accuracy vs confidence",
         "{context}", "next_char", score_next_char, agg_calibration, "top-1 accuracy"),
    Task("mc", "mc.jsonl", "multiple-choice continuation by log-likelihood",
         "{context}", "choices", score_choices, agg_mc, "accuracy (sum of log-probs)"),
]}


# ----------------------------------------------------------------------------
# Running
# ----------------------------------------------------------------------------
class ResponseCache:
    """Append-only JSONL: key -> response. The key includes the model's weight
    fingerprint and every argument of the call (prompt, sampling params, seed),
    so a hit is guaranteed to be the answer the model would give again."""

    def __init__(self, path: Optional[str]):
        self.path, self.data = path, {}
        if path and os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    k, v = json.loads(line)
                    self.data[k] = v

    @staticmethod
    def key(*parts) -> str:
        return hashlib.sha1(json.dumps(parts, sort_keys=True).encode()).hexdigest()

    def get(self, k):
        return self.data.get(k)

    def put(self, k, v):
        self.data[k] = v
        if self.path:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps([k, v]) + "\n")


def example_seed(task: str, ex_id: str, run_seed: int) -> int:
    """Each example gets its own seed derived from (task, id, run seed), so its
    samples do not change when you add, remove or reorder other examples."""
    return zlib.crc32(f"{task}|{ex_id}|{run_seed}".encode())


def call_model(model, task: Task, prompt: str, ex: Dict, seed: int):
    if task.call == "generate":
        return model.generate(prompt, seed=seed, **task.call_kwargs)
    if task.call == "logprobs":
        return model.logprobs(prompt, ex["target"])
    if task.call == "choices":
        return [model.logprobs(prompt, c) for c in ex["choices"]]
    if task.call == "next_char":
        z = model.next_char_logits([prompt])[0]
        p = np.exp(z - z.max())
        return (p / p.sum()).tolist()
    raise ValueError(task.call)


def run_task(task: Task, model, seed: int = 0, use_cache: bool = True, limit: Optional[int] = None,
             out_dir: str = RUN_DIR, verbose: bool = True) -> Dict:
    """Evaluate one model on one task; write <task>/<model>.jsonl + .summary.json."""
    exs = task.examples()[:limit]
    cache = ResponseCache(os.path.join(out_dir, "cache", f"{model.fingerprint}.jsonl")
                          if use_cache else None)
    recs, hits, t0 = [], 0, time.time()
    for ex in exs:
        prompt = task.template.format(**ex)
        s = example_seed(task.name, ex["id"], seed)
        key = ResponseCache.key(CACHE_VERSION, model.fingerprint, task.call, prompt, ex.get("target"),
                                ex.get("choices"), task.call_kwargs, s)
        t1 = time.perf_counter()
        resp = cache.get(key)
        cached = resp is not None
        if not cached:
            resp = call_model(model, task, prompt, ex, s)
            cache.put(key, resp)
        hits += cached
        latency = (time.perf_counter() - t1) * 1000
        sc = task.scorer(ex, resp)
        out = resp if task.call == "generate" else None
        recs.append(dict(id=ex["id"], split=ex.get("split", "all"), cluster=ex.get("para"),
                         input=prompt, target=ex.get("target", ex.get("answer")),
                         output=out[0] if out and len(out) == 1 else out,
                         latency_ms=round(latency, 3), cached=cached, **sc))
    os.makedirs(os.path.join(out_dir, task.name), exist_ok=True)
    base = results_path(task.name, model.name, out_dir)
    with open(base + ".jsonl", "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    summary = dict(task=task.name, model=model.name, model_fingerprint=model.fingerprint,
                   task_file=task.file, task_sha1=task.file_hash(), seed=seed,
                   metric=task.metric, higher_is_better=task.higher_is_better,
                   results_file=os.path.basename(base + ".jsonl"), n=len(recs),
                   cache_hits=hits, elapsed_s=round(time.time() - t0, 2),
                   splits=summarize(recs, task))
    with open(base + ".summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    if verbose:
        print_summary(summary, base + ".summary.json")
    return summary


def results_path(task_name: str, model_name: str, out_dir: str = RUN_DIR) -> str:
    safe = re.sub(r"[^A-Za-z0-9.\-]+", "_", model_name.replace("=", ""))
    return os.path.join(out_dir, task_name, safe)


def load_or_run(task_name: str, spec: str, verbose: bool = False):
    """(summary, per-example records) for one task and model: read the stored run
    if it was made with these exact weights and task file, otherwise run it."""
    task, base = TASKS[task_name], results_path(task_name, spec)
    model = get_model(spec, verbose=verbose)
    if os.path.exists(base + ".summary.json"):
        s, recs = load_results(base + ".summary.json")
        if s["model_fingerprint"] == model.fingerprint and s["task_sha1"] == task.file_hash():
            return s, recs
    s = run_task(task, model, verbose=False)
    if verbose:
        h = s["splits"].get("heldout", s["splits"].get("all"))
        print(f"[harness] ran {task_name} on {spec}: held-out {h['mean']:.3f} "
              f"({s['elapsed_s']} s)", flush=True)
    return load_results(base + ".summary.json")


def summarize(recs: List[Dict], task: Task) -> Dict:
    out = {}
    splits = sorted({r["split"] for r in recs})
    groups = {"all": recs} if len(splits) > 1 else {}
    for split in splits:
        groups[split] = [r for r in recs if r["split"] == split]
    for name, rs in groups.items():
        x = np.array([r["score"] for r in rs], float)
        lo, hi, _ = S.bootstrap_ci(x, n_boot=2000, seed=0)
        clusters = [r["cluster"] for r in rs]
        out[name] = dict(n=len(rs), mean=float(x.mean()), se=S.mean_se(x),
                         ci95_bootstrap=[lo, hi],
                         se_clustered=S.clustered_se(x, clusters) if None not in clusters
                         else None, **task.aggregate(rs))
    return out


def print_summary(s: Dict, path: str):
    print(f"\n{s['task']} | {s['model']} (weights {s['model_fingerprint']}) | {s['n']} examples"
          f" | {s['elapsed_s']} s | cache hits {s['cache_hits']}/{s['n']}")
    print(f"  metric: {s['metric']}")
    for split, r in s["splits"].items():
        lo, hi = r["ci95_bootstrap"]
        extra = "  ".join(f"{k} {v:.3f}" for k, v in r.items()
                          if k not in ("n", "mean", "se", "ci95_bootstrap", "se_clustered", "n_chars")
                          and isinstance(v, float))
        cl = f" clustered SE {r['se_clustered']:.3f}" if r.get("se_clustered") else ""
        print(f"  {split:8s} n={r['n']:4d}  {r['mean']:.3f}  95% CI [{lo:.3f}, {hi:.3f}]  "
              f"SE {r['se']:.3f}{cl}")
        if extra:
            print(f"  {'':8s} {extra}")
    print(f"  -> {os.path.relpath(path)}")


# ----------------------------------------------------------------------------
# Comparing against a baseline: the regression gate
# ----------------------------------------------------------------------------
def load_results(summary_path: str):
    with open(summary_path, encoding="utf-8") as f:
        s = json.load(f)
    path = os.path.join(os.path.dirname(summary_path), s["results_file"])
    with open(path, encoding="utf-8") as f:
        recs = [json.loads(line) for line in f]
    return s, recs


def compare(base_path: str, new_path: str, split: Optional[str] = None, alpha: float = 0.05,
            margin: Optional[float] = None, verbose: bool = True) -> Dict:
    """Pair the two runs by example id and test new - base. Returns a dict with
    'decision' in {REGRESSION, IMPROVEMENT, NO SIGNIFICANT CHANGE, INCONCLUSIVE}."""
    sa, ra = load_results(base_path)
    sb, rb = load_results(new_path)
    if sa["task"] != sb["task"] or sa["task_sha1"] != sb["task_sha1"]:
        raise SystemExit(f"not the same eval: {sa['task']}@{sa['task_sha1']} vs "
                         f"{sb['task']}@{sb['task_sha1']}")
    splits = {r["split"] for r in ra}
    split = split or ("heldout" if "heldout" in splits else "all")
    keep = (lambda r: True) if split == "all" else (lambda r: r["split"] == split)
    a = {r["id"]: r["score"] for r in ra if keep(r)}
    b = {r["id"]: r["score"] for r in rb if keep(r)}
    ids = sorted(set(a) & set(b))
    if len(ids) < len(a) or len(ids) < len(b):
        print(f"warning: only {len(ids)} examples are in both runs", file=sys.stderr)
    xa, xb = np.array([a[i] for i in ids]), np.array([b[i] for i in ids])
    r = S.compare_paired(xa, xb, alpha=alpha)
    sign = 1.0 if sa["higher_is_better"] else -1.0
    gain = sign * r["diff"]  # > 0 means the new model is better
    lo, hi = r["ci"]
    worst_gain = min(sign * lo, sign * hi)
    if r["p"] < alpha:
        decision = "IMPROVEMENT" if gain > 0 else "REGRESSION"
    else:
        decision = "NO SIGNIFICANT CHANGE"
    if margin is not None and decision != "REGRESSION" and worst_gain < -margin:
        decision = "INCONCLUSIVE"  # can't rule out a drop bigger than the margin
    r.update(task=sa["task"], split=split, base=sa["model"], new=sb["model"], decision=decision,
             alpha=alpha, margin=margin, higher_is_better=sa["higher_is_better"],
             mde=S.mde_paired(len(ids), max(np.std(xb - xa, ddof=1), 1e-12), alpha))
    if verbose:
        print_compare(r, sa)
    return r


def print_compare(r: Dict, sa: Dict):
    better = "higher" if r["higher_is_better"] else "lower"
    print(f"\ncompare {r['task']} on split '{r['split']}': {r['n']} paired examples "
          f"(metric: {sa['metric']}; {better} is better)")
    print(f"  base {r['base']:>18s}: {r['mean_a']:.4f}")
    print(f"  new  {r['new']:>18s}: {r['mean_b']:.4f}")
    print(f"  new - base = {r['diff']:+.4f}   paired bootstrap 95% CI [{r['ci'][0]:+.4f}, "
          f"{r['ci'][1]:+.4f}]")
    if "table" in r:
        both, a_only, b_only, neither = r["table"]
        print(f"  both right {both}, only base right {a_only}, only new right {b_only}, "
              f"both wrong {neither}")
    print(f"  {r['test']} p = {r['p']:.3g}   (an unpaired test would give p = "
          f"{r['p_unpaired']:.3g}; score correlation {r['corr']:.2f})")
    print(f"  smallest change this comparison can reliably detect (80% power): "
          f"±{r['mde']:.4f}")
    gate = "FAIL (exit 1)" if r["decision"] in ("REGRESSION", "INCONCLUSIVE") else "PASS (exit 0)"
    note = ""
    if r["decision"] == "INCONCLUSIVE":
        note = f": the CI allows a drop larger than the margin {r['margin']}"
    print(f"  DECISION: {r['decision']} at alpha = {r['alpha']}{note} -> CI gate {gate}")


# ----------------------------------------------------------------------------
# Building the task files from the story
# ----------------------------------------------------------------------------
def build_tasks(verbose: bool = True):
    """Deterministically regenerate bpc / next_word / calibration jsonl. (mc.jsonl
    and judge_pairs.jsonl are hand-written and are not touched.)"""
    text = load_story()
    off = split_offset(text)
    chars = sorted(set(text))
    split_of = (lambda pos: "heldout" if pos >= off else "train")

    def write(name, rows):
        with open(os.path.join(TASK_DIR, name), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        if verbose:
            n_held = sum(r["split"] == "heldout" for r in rows)
            print(f"  wrote tasks/{name}: {len(rows)} examples ({n_held} heldout)")

    sents = []
    for p, (s, e) in enumerate(paragraph_spans(text)):
        for m in re.finditer(r'\S.*?[.!?]"?(?=\s|$)', text[s:e]):
            start = s + m.start()
            if start == 0:
                continue  # the very first sentence has no text before it
            sents.append(dict(id=f"s{len(sents):03d}", split=split_of(start), para=p,
                              context=text[max(0, start - 64):start], target=m.group()))
    write("bpc.jsonl", sents)

    words = []
    for m in re.finditer(r"\S+", text):
        if m.start() < 32:
            continue  # every prompt gets at least the GPT's full 32-char window
        words.append(dict(id=f"w{len(words):03d}", split=split_of(m.start()),
                          para=paragraph_of(text, m.start()),
                          context=text[max(0, m.start() - 64):m.start()], target=m.group()))
    write("next_word.jsonl", words)

    rng = np.random.default_rng(0)
    held = list(range(off, len(text), 2))  # every other held-out character
    train = sorted(rng.choice(np.arange(32, off), size=300, replace=False).tolist())
    rows = [dict(id=f"c{i:04d}", split=split_of(pos), para=paragraph_of(text, pos),
                 context=text[pos - 32:pos], target=text[pos], target_id=chars.index(text[pos]))
            for i, pos in enumerate(train + held)]
    write("calibration.jsonl", rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="evaluate models on tasks")
    r.add_argument("--task", nargs="+", required=True, help=f"one or more of {list(TASKS)}")
    r.add_argument("--model", nargs="+", required=True,
                   help="gpt, gpt-full, rnn, bigram; options like gpt@steps=300,seed=1")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--limit", type=int, default=None, help="only the first N examples")
    r.add_argument("--no-cache", action="store_true")
    r.add_argument("--out-dir", default=RUN_DIR)
    c = sub.add_parser("compare", help="paired test of NEW against BASE; exit 1 on regression")
    c.add_argument("base")
    c.add_argument("new")
    c.add_argument("--split", default=None, help="default: heldout if present, else all")
    c.add_argument("--alpha", type=float, default=0.05)
    c.add_argument("--margin", type=float, default=None,
                   help="also fail unless the CI rules out a drop bigger than this")
    sub.add_parser("list", help="list tasks and models")
    sub.add_parser("build-tasks", help="regenerate the story-derived task files")
    args = ap.parse_args()

    if args.cmd == "list":
        for t in TASKS.values():
            n = len(t.examples())
            print(f"  {t.name:17s} {n:5d} examples  {t.description}")
        print("  models: gpt, gpt-full, rnn, bigram (+ options, e.g. gpt@steps=300,seed=1)")
    elif args.cmd == "build-tasks":
        build_tasks()
    elif args.cmd == "run":
        for name in args.task:
            if name not in TASKS:
                raise SystemExit(f"unknown task {name!r}; choose from {list(TASKS)}")
        for spec in args.model:
            model = get_model(spec)
            for name in args.task:
                run_task(TASKS[name], model, args.seed, not args.no_cache, args.limit, args.out_dir)
    elif args.cmd == "compare":
        res = compare(args.base, args.new, args.split, args.alpha, args.margin)
        raise SystemExit(1 if res["decision"] in ("REGRESSION", "INCONCLUSIVE") else 0)


if __name__ == "__main__":
    main()
