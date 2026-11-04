"""
LLM-as-a-judge from scratch: rubric prompts, pairwise and pointwise grading,
position-swap debiasing, majority voting, and the step everybody skips:
measuring the judge against HUMAN labels before trusting it. (README section 9.)

The labeled set is evals/tasks/judge_pairs.jsonl: 40 questions about the story,
each with a reference answer, two candidate answers (A, B) and a human verdict
(A, B or tie). It deliberately contains the cases that fool judges: long fluent
WRONG answers vs short right ones, and pairs that only differ in length (ties).

The default judge is SIMULATED so the lesson is reproducible offline: a
deterministic function with the three biases Zheng et al. (2023) measured in
GPT-4 / Claude judges, each one a knob you can turn:
    quality     token F1 of the answer against the reference (a reference-guided grader)
    position    + position_bias to whichever answer is shown FIRST
    verbosity   + verbosity_bias per doubling of the answer's length vs the reference
    noise       + noise · N(0, 1), a different draw per sample (like temperature > 0)
A real judge goes through exactly the same protocol code (--real).

Run:
    python evals/judge.py                         simulated judge (default biases)
    python evals/judge.py --position-bias 0 --verbosity-bias 0   an unbiased judge
    python evals/judge.py --real                  OpenAI-compatible endpoint, set
        OPENAI_BASE_URL (e.g. https://api.openai.com/v1), OPENAI_API_KEY, JUDGE_MODEL
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import urllib.error
import urllib.request
import zlib
from typing import Dict, List, Optional

import numpy as np

import metrics as M
import stats as S

HERE = os.path.dirname(os.path.abspath(__file__))
PAIRS = os.path.join(HERE, "tasks", "judge_pairs.jsonl")
LABELS = ("A", "B", "tie")

PAIRWISE_RUBRIC = """\
You are an impartial judge. Compare two answers to a question about a short story.

[Question]
{question}
[Reference answer]
{reference}
[Answer A]
{answer_a}
[Answer B]
{answer_b}

Judge which answer is more CORRECT and more faithful to the reference. An answer
that adds details not in the story is wrong. Do not let the order of the answers,
their length, or their style influence you. If both are equally correct (or equally
wrong), it is a tie. Explain in one or two sentences, then output exactly one of:
[[A]] if A is better, [[B]] if B is better, [[C]] for a tie."""

POINTWISE_RUBRIC = """\
You are an impartial judge. Rate the answer to a question about a short story.

[Question]
{question}
[Reference answer]
{reference}
[Answer]
{answer}

10 = fully correct and faithful to the reference, 5 = partly correct, 1 = wrong or
invented. Length and style do not matter. Explain briefly, then output "Rating: [[n]]"."""


def load_pairs(path: str = PAIRS) -> List[Dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ----------------------------------------------------------------------------
# Judges
# ----------------------------------------------------------------------------
class SimulatedJudge:
    """A deterministic stand-in for an LLM judge with KNOWN biases."""

    def __init__(self, position_bias=0.3, verbosity_bias=0.2, noise=0.15, tie_margin=0.08,
                 seed=0):
        self.position_bias, self.verbosity_bias = position_bias, verbosity_bias
        self.noise, self.tie_margin, self.seed = noise, tie_margin, seed
        self.name = (f"simulated(position {position_bias}, verbosity {verbosity_bias}, "
                     f"noise {noise})")

    def _rng(self, *parts) -> np.random.Generator:
        return np.random.default_rng(zlib.crc32("|".join(map(str, parts + (self.seed,))).encode()))

    def _appeal(self, answer: str, reference: str) -> float:
        """What this judge 'likes': correctness (F1 vs reference) + length."""
        n, n_ref = max(1, len(answer.split())), max(1, len(reference.split()))
        return M.token_f1(answer, reference) + self.verbosity_bias * math.log2(n / n_ref)

    def pairwise(self, question, reference, first, second, sample=0) -> str:
        """'A' (first shown) / 'B' (second shown) / 'tie'."""
        margin = (self._appeal(first, reference) - self._appeal(second, reference)
                  + self.position_bias
                  + self.noise * self._rng("pair", question, first, second, sample).standard_normal())
        if abs(margin) < self.tie_margin:
            return "tie"
        return "A" if margin > 0 else "B"

    def pointwise(self, question, reference, answer, sample=0) -> int:
        """A 1-10 rating; there is no 'first' answer, so no position bias."""
        s = (self._appeal(answer, reference)
             + self.noise * self._rng("point", question, answer, sample).standard_normal())
        return int(np.clip(round(1 + 9 * s), 1, 10))


class HTTPJudge:
    """Any OpenAI-compatible /chat/completions endpoint, via urllib (no SDK)."""

    def __init__(self, base_url: str, api_key: str, model: str, temperature: float = 0.0):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.api_key, self.model, self.temperature = api_key, model, temperature
        self.name = f"http({model})"

    def _ask(self, prompt: str) -> str:
        body = json.dumps({"model": self.model, "temperature": self.temperature,
                           "messages": [{"role": "user", "content": prompt}]}).encode()
        req = urllib.request.Request(self.url, data=body, headers={
            "Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.load(r)["choices"][0]["message"]["content"]

    def pairwise(self, question, reference, first, second, sample=0) -> str:
        text = self._ask(PAIRWISE_RUBRIC.format(question=question, reference=reference,
                                                answer_a=first, answer_b=second))
        m = re.findall(r"\[\[([ABC])\]\]", text)
        return {"A": "A", "B": "B", "C": "tie"}.get(m[-1] if m else "C", "tie")

    def pointwise(self, question, reference, answer, sample=0) -> int:
        text = self._ask(POINTWISE_RUBRIC.format(question=question, reference=reference,
                                                 answer=answer))
        m = re.findall(r"\[\[(\d+)\]\]", text)
        return int(m[-1]) if m else 5


# ----------------------------------------------------------------------------
# Protocols: how you ask the judge, and how you combine its answers
# ----------------------------------------------------------------------------
def flip(v: str) -> str:
    return {"A": "B", "B": "A", "tie": "tie"}[v]


def majority(votes: List[str]) -> str:
    a, b = votes.count("A"), votes.count("B")
    return "A" if a > b else "B" if b > a else "tie"


def verdicts(judge, pairs: List[Dict], k: int = 5) -> Dict[str, List[str]]:
    """Every protocol's verdict on every pair, in the ORIGINAL A/B order."""
    out = {name: [] for name in ("single", "swapped", "swap-debiased", f"majority-{k}",
                                 f"swap + majority-{k}", "pointwise")}
    for p in pairs:
        q, ref, a, b = p["question"], p["reference"], p["answer_a"], p["answer_b"]
        fwd = [judge.pairwise(q, ref, a, b, s) for s in range(k)]
        bwd = [flip(judge.pairwise(q, ref, b, a, s)) for s in range(k)]  # B shown first
        out["single"].append(fwd[0])
        out["swapped"].append(bwd[0])
        # MT-Bench's conservative rule: a win only if it survives the swap.
        out["swap-debiased"].append(fwd[0] if fwd[0] == bwd[0] else "tie")
        out[f"majority-{k}"].append(majority(fwd))
        out[f"swap + majority-{k}"].append(majority(fwd + bwd))
        ra, rb = judge.pointwise(q, ref, a), judge.pointwise(q, ref, b)
        out["pointwise"].append("A" if ra > rb else "B" if rb > ra else "tie")
    return out


def kappa_ci(human: List[str], judged: List[str], n_boot: int = 2000, seed: int = 0):
    """Bootstrap CI of kappa: with 40 items, the uncertainty is NOT small."""
    rng = np.random.default_rng(seed)
    h, j = np.array(human), np.array(judged)
    ks = []
    for _ in range(n_boot):
        i = rng.integers(0, len(h), len(h))
        ks.append(M.cohens_kappa(h[i], j[i], LABELS))
    return tuple(np.quantile(ks, [0.025, 0.975]))


def bias_report(judge, pairs: List[Dict]) -> Dict[str, float]:
    """Position consistency, first-slot win rate, and longer-answer win rate."""
    fwd, bwd = [], []
    longer_judge, longer_human = [], []
    for p in pairs:
        q, ref, a, b = p["question"], p["reference"], p["answer_a"], p["answer_b"]
        v1 = judge.pairwise(q, ref, a, b)
        v2 = judge.pairwise(q, ref, b, a)  # in SHOWN order: 'A' = b
        fwd.append(v1)
        bwd.append(v2)
        if len(a.split()) != len(b.split()) and v1 != "tie":
            long = "A" if len(a.split()) > len(b.split()) else "B"
            longer_judge.append(v1 == long)
            if p["human"] != "tie":
                longer_human.append(p["human"] == long)
    shown = [v for v in fwd + bwd if v != "tie"]
    consistent = np.mean([v1 == flip(v2) for v1, v2 in zip(fwd, bwd)])
    return dict(consistency=float(consistent),
                first_wins=float(np.mean([v == "A" for v in shown])) if shown else float("nan"),
                longer_judge=float(np.mean(longer_judge)), longer_human=float(np.mean(longer_human)))


def evaluate(judge, pairs: List[Dict], k: int = 5, verbose: bool = True) -> Dict:
    human = [p["human"] for p in pairs]
    res = verdicts(judge, pairs, k)
    table = {}
    for name, v in res.items():
        kap = M.cohens_kappa(human, v, LABELS)
        lo, hi = kappa_ci(human, v)
        table[name] = dict(kappa=kap, ci=(lo, hi), agree=M.raw_agreement(human, v),
                           ties=v.count("tie") / len(v))
    bias = bias_report(judge, pairs)
    if verbose:
        print(f"judge: {judge.name}\n{len(pairs)} labeled pairs; human verdicts: "
              f"A {human.count('A')}, B {human.count('B')}, tie {human.count('tie')}\n")
        print(f"  {'protocol':22s} {'kappa':>6s}  {'95% CI (bootstrap)':>18s}  "
              f"{'agreement':>9s}  {'ties':>5s}")
        for name, r in table.items():
            print(f"  {name:22s} {r['kappa']:6.3f}  [{r['ci'][0]:+.2f}, {r['ci'][1]:+.2f}]"
                  f"       {r['agree']:7.1%}  {r['ties']:5.0%}")
        print(f"\n  position consistency (same verdict after swapping A and B): "
              f"{bias['consistency']:.0%}")
        print(f"  first-shown answer wins {bias['first_wins']:.0%} of non-tie verdicts "
              f"(an unbiased judge: ~50%)")
        print(f"  longer answer wins {bias['longer_judge']:.0%} of the judge's decisions vs "
              f"{bias['longer_human']:.0%} of the human's")
    return dict(table=table, bias=bias, verdicts=res, human=human)


def sweep(pairs, k=5, verbosity=0.2, noise=0.15, seeds=20,
          biases=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5)):
    """Mean kappa of each protocol (over `seeds` judge seeds, i.e. different noise
    draws) as the position bias grows, verbosity and noise fixed."""
    human = [p["human"] for p in pairs]
    rows = []
    for pb in biases:
        acc: Dict[str, List[float]] = {}
        for seed in range(seeds):
            res = verdicts(SimulatedJudge(pb, verbosity, noise, seed=seed), pairs, k)
            for name, v in res.items():
                acc.setdefault(name, []).append(M.cohens_kappa(human, v, LABELS))
        rows.append((pb, {n: float(np.mean(v)) for n, v in acc.items()}))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--position-bias", type=float, default=0.3)
    ap.add_argument("--verbosity-bias", type=float, default=0.2)
    ap.add_argument("--noise", type=float, default=0.15)
    ap.add_argument("--k", type=int, default=5, help="samples per order for majority voting")
    ap.add_argument("--real", action="store_true", help="use an OpenAI-compatible endpoint")
    ap.add_argument("--show-prompt", action="store_true", help="print one filled rubric")
    args = ap.parse_args()
    pairs = load_pairs()

    if args.show_prompt:
        p = pairs[0]
        print(PAIRWISE_RUBRIC.format(**{k: p[k] for k in ("question", "reference",
                                                            "answer_a", "answer_b")}) + "\n")

    if args.real:
        need = ("OPENAI_BASE_URL", "OPENAI_API_KEY", "JUDGE_MODEL")
        missing = [v for v in need if not os.environ.get(v)]
        if missing:
            print(f"--real: skipped, environment variable(s) not set: {', '.join(missing)}.\n"
                  "Set e.g. OPENAI_BASE_URL=https://api.openai.com/v1 OPENAI_API_KEY=... "
                  "JUDGE_MODEL=gpt-4o-mini (any OpenAI-compatible server works, e.g. a local "
                  "vLLM or llama.cpp server). The simulated judge needs none of this.")
            return
        judge = HTTPJudge(os.environ["OPENAI_BASE_URL"], os.environ["OPENAI_API_KEY"],
                          os.environ["JUDGE_MODEL"])
        try:
            evaluate(judge, pairs, k=1)
        except (urllib.error.URLError, OSError, KeyError, ValueError) as e:
            print(f"--real: the endpoint failed ({e}); nothing to report.")
        return

    judge = SimulatedJudge(args.position_bias, args.verbosity_bias, args.noise)
    evaluate(judge, pairs, args.k)
    print(f"\n  As the position bias grows (verbosity {args.verbosity_bias}, noise {args.noise}), "
          "kappa vs human:")
    rows = sweep(pairs, args.k, args.verbosity_bias, args.noise)
    names = list(rows[0][1])
    print("  (mean over 20 judge seeds)")
    print("  " + "position bias".rjust(14) + "".join(f"{n:>22s}" for n in names))
    for pb, ks in rows:
        print("  " + f"{pb:14.1f}" + "".join(f"{ks[n]:22.3f}" for n in names))
    by_kind = {}
    v = verdicts(judge, pairs, args.k)["single"]
    for p, x in zip(pairs, v):
        by_kind.setdefault(p["kind"], []).append(x == p["human"])
    print("\n  single-order agreement with the human, by kind of pair:")
    for kind, ok in sorted(by_kind.items(), key=lambda t: -len(t[1])):
        print(f"    {kind:14s} {np.mean(ok):5.0%}  ({len(ok)} pairs)")


if __name__ == "__main__":
    main()
