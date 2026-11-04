"""
Watch an eval happen, step by step, in your terminal: one example from prompt to
score, then a whole run with its error bar, the bootstrap behind that error bar,
and finally the paired comparison that a CI regression gate makes.

  STEP 1  one held-out example: the prompt, greedy decoding char by char (with the
          model's probabilities), the raw output, normalization, EM / F1, the JSONL record
  STEP 2  the run accumulates: running accuracy ± 95% CI after 10, 20, 50, ... examples
  STEP 3  the bootstrap: a few resamples by hand, then 10,000 as an ASCII histogram
  STEP 4  A vs B on the SAME examples: the 2x2 table, McNemar, the paired bootstrap
          of the difference, and the CI gate's decision

Models are trained on demand (models.py, ~10 s each the first time).

Run:
    python evals/eval_visual.py                         (press Enter between steps)
    python evals/eval_visual.py --new gpt@seed=1        (a change that is just noise)
    python evals/eval_visual.py --no-pause              (or: < /dev/null)
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np

import metrics as M
import stats as S
from harness import TASKS, compare, load_or_run, results_path
from models import get_model

TTY = sys.stdout.isatty()
PAUSE = True
BAR = 50


def color(code, s):
    return f"\033[{code}m{s}\033[0m" if TTY else s


def cyan(s): return color("36;1", s)      # inputs / prompts
def yellow(s): return color("33;1", s)    # model outputs
def green(s): return color("32;1", s)     # scores, good news
def red(s): return color("31;1", s)       # failures, bad news
def dim(s): return color("2", s)


def pause():
    if PAUSE and sys.stdin.isatty():
        try:
            input(dim("\n  [Enter] next step "))
        except EOFError:
            print()
    print("\n" + "=" * 78 + "\n")


def vis(s):
    return s.replace(" ", "␣").replace("\n", "↵")


def interval_bar(lo, hi, mean, lo_axis=0.0, hi_axis=1.0, width=BAR):
    """A 0..1 axis with the CI as ═══ and the mean as ●."""
    pos = lambda v: int(round((v - lo_axis) / (hi_axis - lo_axis) * (width - 1)))
    cells = ["·"] * width
    for i in range(max(0, pos(lo)), min(width, pos(hi) + 1)):
        cells[i] = "═"
    cells[min(width - 1, max(0, pos(mean)))] = "●"
    return "".join(cells)


def histogram(values, bins=24, height=8, marks=()):
    """Vertical ASCII histogram; `marks` are x values to flag with ▲ underneath."""
    values = np.asarray(values)
    uniq = np.unique(values)
    if 1 < len(uniq) < 1000:  # discrete values (k/n): whole steps per bin, no empty gaps
        step = float(np.min(np.diff(uniq)))
        k = max(1, math.ceil((uniq[-1] - uniq[0]) / step / bins + 1e-9))
        edges = np.arange(uniq[0] - step / 2, uniq[-1] + k * step, k * step)
        counts, edges = np.histogram(values, bins=edges)
    else:
        counts, edges = np.histogram(values, bins=bins)
    bins = len(counts)
    top = counts.max()
    for level in range(height, 0, -1):
        row = "".join("█" if c / top * height >= level - 0.5 else " " for c in counts)
        print("     " + "".join(ch * 2 for ch in row))
    under = [" "] * bins
    for m in marks:
        if edges[0] <= m <= edges[-1]:
            under[min(bins - 1, max(0, int(np.searchsorted(edges, m) - 1)))] = "▲"
    print("     " + "".join(ch * 2 for ch in under))
    print(f"     {edges[0]:<.3f}" + " " * (2 * bins - 12) + f"{edges[-1]:>.3f}")


def step1(model, recs, example_id=None):
    held = [r for r in recs if r["split"] == "heldout"]
    if example_id:
        r = next(x for x in recs if x["id"] == example_id)
    else:  # an example where normalization matters, if there is one
        r = next((x for x in held if x["em_strict"] != x["score"]), held[0])
    print(f"STEP 1: ONE example, from prompt to score (task next_word, id {r['id']}, "
          f"split {r['split']})\n")
    print(f"   prompt  {cyan(repr(r['input']))}")
    print(f"   target  {green(repr(r['target']))}   (the word that really comes next)\n")
    print("   greedy decoding, one character at a time (stop at a space):")
    text = r["input"]
    out = ""
    for _ in range(20):
        z = model.next_char_logits([text + out])[0]
        p = np.exp(z - z.max())
        p /= p.sum()
        top = np.argsort(-p)[:3]
        best = model.chars[top[0]]
        alts = "  ".join(f"{vis(model.chars[i])} {p[i]:4.0%}" for i in top)
        print(f"     after {vis(('…' + out) if out else '(prompt)'):14s} → {alts}")
        if best in " \n":
            break
        out += best
    print(f"\n   raw output        {yellow(repr(out))}")
    norm_o, norm_t = M.normalize_answer(out, False), M.normalize_answer(r["target"], False)
    print(f"   normalized        {yellow(repr(norm_o))} vs {green(repr(norm_t))}   "
          "(lowercase, no punctuation, collapse spaces)")
    print(f"\n   strict EM {M.exact_match(out, r['target']):.0f}   normalized EM "
          f"{M.normalized_exact_match(out, r['target'], False):.0f}   F1 "
          f"{M.token_f1(out, r['target'], False):.2f}   "
          f"SQuAD EM (also drops a/an/the) {M.normalized_exact_match(out, r['target']):.0f}")
    print(dim("\n   The harness logs one JSON line per example (evals/runs/next_word/gpt.jsonl):"))
    rec = {k: r[k] for k in ("id", "split", "output", "target", "score", "em_strict",
                             "latency_ms", "cached")}
    print(dim(f"   {rec}"))
    pause()


def step2(recs):
    held = [r for r in recs if r["split"] == "heldout"]
    x_file = np.array([r["score"] for r in held])
    x = x_file[np.random.default_rng(0).permutation(len(x_file))]  # random order
    print(f"STEP 2: the run accumulates: running accuracy ± 95% CI (mean ± 1.96·SE), "
          f"{len(x)} held-out examples\n   in random order\n")
    print(f"   {'n':>4s}  {'accuracy':>8s}  {'95% CI':>15s}   0{' ' * (BAR - 2)}1")
    for n in (5, 10, 20, 50, 100, 150, len(x)):
        m = x[:n].mean()
        se = M.accuracy_se(m, n)
        lo, hi = max(0.0, m - 1.96 * se), min(1.0, m + 1.96 * se)
        print(f"   {n:4d}  {m:8.3f}  [{lo:.3f}, {hi:.3f}]   {interval_bar(lo, hi, m)}")
    print(dim(f"\n   The CI shrinks like 1/√n: 4× the examples for half the width. With "
              f"{len(x)} examples\n   the accuracy is only known to about ±{1.96 * M.accuracy_se(x.mean(), len(x)):.3f}: "
              "a 3-point 'improvement' would be invisible."))
    print(dim(f"\n   Why random order? In FILE order the running accuracy after 150 examples is "
              f"{x_file[:150].mean():.3f},\n   far outside this CI: the last paragraphs repeat "
              "the start of the story (section 10).\n   Neighbouring examples are not "
              "independent; that is what clustered standard errors are for."))
    pause()
    return x_file


def step3(x, n_boot=10000):
    print(f"STEP 3: where the error bar comes from: the bootstrap\n")
    print("   Pretend the eval set IS the population. Draw n examples from it WITH "
          "replacement,\n   recompute the accuracy, repeat. The spread of those accuracies "
          "is the error bar.\n")
    rng = np.random.default_rng(0)
    for b in range(3):
        idx = rng.integers(0, len(x), len(x))
        counts = np.bincount(idx, minlength=len(x))
        print(f"   resample {b + 1}: picks #{idx[0]}, #{idx[1]}, #{idx[2]}, … "
              f"({(counts == 0).sum()} examples left out, {(counts > 1).sum()} picked twice "
              f"or more) → accuracy {x[idx].mean():.3f}")
    lo, hi, boots = S.bootstrap_ci(x, n_boot=n_boot, seed=0)
    clo, chi = S.clt_ci(x)
    print(f"\n   {n_boot:,} resamples:")
    histogram(boots, marks=(lo, hi))
    print(f"\n   percentile CI: the 2.5% and 97.5% quantiles (▲) = [{lo:.3f}, {hi:.3f}]")
    print(f"   CLT formula    mean ± 1.96·SE              = [{clo:.3f}, {chi:.3f}]")
    print(dim("\n   They agree here. The bootstrap also works for statistics with no formula "
              "(F1, median,\n   bits/char as a ratio of sums), and resampling whole CLUSTERS "
              "handles grouped questions."))
    pause()


def step4(base_spec, new_spec):
    task = TASKS["next_word"]
    load_or_run("next_word", base_spec)
    load_or_run("next_word", new_spec)
    pa, pb = results_path(task.name, base_spec), results_path(task.name, new_spec)
    print(f"STEP 4: is the new model worse? base = {cyan(base_spec)}, new = {yellow(new_spec)}, "
          "same held-out examples\n")
    _, ra = load_or_run("next_word", base_spec)
    _, rb = load_or_run("next_word", new_spec)
    a = np.array([r["score"] for r in ra if r["split"] == "heldout"])
    b = np.array([r["score"] for r in rb if r["split"] == "heldout"])
    both, a_only, b_only, neither = S.mcnemar_table(a, b)
    print(f"   {'':16s}{'new right':>12s}{'new wrong':>12s}")
    print(f"   {'base right':16s}{both:12d}{red(f'{a_only:12d}')}")
    print(f"   {'base wrong':16s}{green(f'{b_only:12d}')}{neither:12d}\n")
    print(f"   Only the {a_only + b_only} DISAGREEMENTS carry information. If the models "
          "were equally good, each one\n   would be a coin flip: McNemar asks how surprising "
          f"{a_only} vs {b_only} is → p = {S.mcnemar(a_only, b_only):.3g}")
    diff, lo, hi, boots = S.paired_bootstrap(a, b, n_boot=10000)
    print(f"\n   paired bootstrap of new - base (resample QUESTIONS, keep each pair together):")
    histogram(boots, marks=(lo, hi, 0.0))
    zero = "" if boots.min() <= 0 <= boots.max() else ", 0 is outside every resample"
    print(f"   difference {diff:+.3f}, 95% CI [{lo:+.3f}, {hi:+.3f}]   (▲ = CI ends and 0{zero})")
    se_unp = math.sqrt(S.mean_se(a) ** 2 + S.mean_se(b) ** 2)
    print(f"   paired SE {S.mean_se(b - a):.4f} vs unpaired SE {se_unp:.4f}: pairing removes "
          "the shared question difficulty")
    print(f"\n   The CI gate: python evals/harness.py compare {os.path.relpath(pa)}.summary.json"
          f" \\\n                                             {os.path.relpath(pb)}.summary.json")
    r = compare(pa + ".summary.json", pb + ".summary.json", verbose=False)
    verdict = r["decision"]
    paint = red if verdict == "REGRESSION" else green
    print(f"   → {paint(verdict)}  (exit code {1 if verdict == 'REGRESSION' else 0})")
    if verdict != "REGRESSION":
        print(dim(f"   'No significant change' is not 'no change': with n = {r['n']} this "
                  f"comparison only reliably\n   detects differences larger than ±{r['mde']:.3f} "
                  "(80% power). Use --margin to demand that."))
    pause()


def main():
    global PAUSE
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="gpt", help="baseline model spec")
    ap.add_argument("--new", default="gpt@steps=300", help="candidate model spec")
    ap.add_argument("--example", default=None, help="example id for step 1, e.g. w600")
    ap.add_argument("--no-pause", action="store_true")
    args = ap.parse_args()
    PAUSE = not args.no_pause

    model = get_model(args.base)
    _, recs = load_or_run("next_word", args.base)
    print("STEP 0: an eval = a fixed set of examples + a way to ask the model + a scorer "
          "+ statistics.\n")
    print(f"   model {cyan(args.base)}: the mini-GPT of ../transformer/, trained on paragraphs "
          "0-16 of the story")
    print(f"   task  {cyan('next_word')}: {TASKS['next_word'].description}; "
          f"{len(recs)} examples, {sum(r['split'] == 'heldout' for r in recs)} of them held out")
    print(dim("   (examples in the 'train' split are text the model was trained on: those "
              "measure memory)"))
    pause()
    step1(model, recs, args.example)
    x = step2(recs)
    step3(x)
    step4(args.base, args.new)
    print("Done. The same machinery runs every task: python evals/harness.py list")


if __name__ == "__main__":
    main()
