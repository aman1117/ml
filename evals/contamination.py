"""
Contamination: is the eval set inside the training data? (README section 10.)

Our "held-out" split is the last 7 paragraphs of the story, never trained on. But
the story repeats itself ("Good morning," said the keeper. "Will you come in and
see the light?" appears near the start AND at the end), so some held-out examples
are verbatim copies of training text. This is exactly what happens when a web-scale
pretraining crawl contains a benchmark, or when a fine-tuning set is built from
the same logs as its test set.

Detection (the GPT-3 recipe, Brown et al. 2020, simplified): normalize text to
lowercase words without punctuation, collect every n-gram of the TRAINING text,
and flag an eval example if any of its n-grams is in that set. Examples shorter
than n words are flagged if the whole example appears in the training text, but
only if they have at least MIN_WORDS words: the one-word sentence "Rain." appears
inside the training text ("the rain beat on the windows") without being a copy
of anything. We try n = 13 (GPT-3, conservative) and n = 8 (catches shorter copies).

Then the point of the exercise: score the model separately on clean and
contaminated examples. If the contaminated ones score much higher, your headline
number is inflated by memorization, and only the clean subset measures generalization.

Run:
    python evals/contamination.py
    python evals/contamination.py --model rnn
"""
from __future__ import annotations

import argparse
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np

import stats as S
from harness import load_or_run
from metrics import normalize_answer
from models import load_story, split_offset

NS = (13, 8)
MIN_WORDS = 4  # shorter examples are too short to call contaminated


def words(text: str) -> List[str]:
    return normalize_answer(text, remove_articles=False).split()


def ngram_set(ws: Sequence[str], n: int) -> Set[Tuple[str, ...]]:
    return {tuple(ws[i:i + n]) for i in range(len(ws) - n + 1)}


def example_text(rec: Dict, task: str) -> str:
    """What the model sees for this example: context + answer (the first context
    word may be cut in half by the 64-char window, so it is dropped)."""
    if task == "bpc":
        return rec["target"]
    ctx = rec["input"].split(" ", 1)[1] if " " in rec["input"] else rec["input"]
    return ctx + rec["target"]


def contaminated(ws: List[str], train_words: List[str], train_grams: Set, n: int):
    """(flag, matched n-gram or None)."""
    if len(ws) >= n:
        for g in (tuple(ws[i:i + n]) for i in range(len(ws) - n + 1)):
            if g in train_grams:
                return True, " ".join(g)
        return False, None
    if len(ws) < MIN_WORDS:
        return False, None
    joined, needle = " " + " ".join(train_words) + " ", " " + " ".join(ws) + " "
    return (needle in joined, " ".join(ws) if needle in joined else None)


def flag_examples(recs: List[Dict], task: str, n: int) -> List[Tuple[bool, str]]:
    text = load_story()
    train_words = words(text[:split_offset(text)])
    grams = ngram_set(train_words, n)
    return [contaminated(words(example_text(r, task)), train_words, grams, n) for r in recs]


def gap(recs: List[Dict], flags: List[bool]) -> Dict:
    """Mean score (95% bootstrap CI) on clean vs contaminated examples."""
    out = {}
    for name, want in (("clean", False), ("contaminated", True)):
        x = np.array([r["score"] for r, f in zip(recs, flags) if f == want])
        lo, hi, _ = S.bootstrap_ci(x, n_boot=2000) if len(x) > 1 else (np.nan, np.nan, None)
        out[name] = dict(n=len(x), mean=float(x.mean()) if len(x) else float("nan"), ci=(lo, hi))
    return out


def analyze(model: str = "gpt", verbose: bool = True) -> Dict:
    results = {}
    for task in ("next_word", "bpc"):
        _, recs = load_or_run(task, model)
        held = [r for r in recs if r["split"] == "heldout"]
        train = [r for r in recs if r["split"] == "train"]
        train_mean = float(np.mean([r["score"] for r in train]))
        for n in NS:
            fl = flag_examples(held, task, n)
            flags = [f for f, _ in fl]
            results[(task, n)] = dict(flags=flags, matches=[m for _, m in fl],
                                      gap=gap(held, flags), train_mean=train_mean,
                                      overall=float(np.mean([r["score"] for r in held])),
                                      held=held)
    if verbose:
        report(results, model)
    return results


def report(results: Dict, model: str):
    text = load_story()
    off = split_offset(text)
    print(f"training text: {off} chars ({len(words(text[:off]))} words); held-out "
          f"examples are checked against it.\n")
    units = {"next_word": "exact match, higher is better", "bpc": "bits/char, lower is better"}
    for task in ("next_word", "bpc"):
        print(f"{task} ({units[task]}), model {model}:")
        for n in NS:
            r = results[(task, n)]
            g = r["gap"]
            c, d = g["clean"], g["contaminated"]
            print(f"  {n:2d}-gram: {d['n']:3d} of {len(r['flags'])} held-out examples flagged "
                  f"({d['n'] / len(r['flags']):.0%})")
            print(f"           clean        {c['mean']:.3f} [{c['ci'][0]:.3f}, {c['ci'][1]:.3f}]"
                  f"  (n = {c['n']})")
            print(f"           contaminated {d['mean']:.3f} [{d['ci'][0]:.3f}, {d['ci'][1]:.3f}]"
                  f"  (n = {d['n']})")
        r = results[(task, 13)]
        print(f"  for reference: all held-out {r['overall']:.3f}, in-training examples "
              f"{r['train_mean']:.3f}\n")
    r = results[("bpc", 13)]
    print("Flagged held-out sentences (13-gram), with the training n-gram that matched:")
    for rec, f, m in zip(r["held"], r["flags"], r["matches"]):
        if f:
            print(f"  {rec['score']:.2f} bits/char  {rec['target'][:60]!r:64s} <- '{m[:50]}'")
    short = [rec["target"] for rec in r["held"] if len(words(example_text(rec, "bpc"))) < MIN_WORDS]
    if short:
        print(f"  (too short to judge, counted as clean: {short}; with no minimum length, "
              "'rain' would match 'the rain beat on the windows', a false positive)")
    print("\nThe lesson: the contaminated examples are the 'easy' ones because the model "
          "has seen them.\nReport the clean subset separately, or remove the contaminated "
          "examples from the eval set.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="gpt", help="model spec (see models.py)")
    args = ap.parse_args()
    analyze(args.model)


if __name__ == "__main__":
    main()
