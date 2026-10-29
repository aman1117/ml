"""
THE key experiment of this folder: the encoder-decoder bottleneck.

Task: reverse a digit string, e.g. "48213" -> "31284", with lengths 5..40.
Two models, identical in every way (same parameters, same data, same seed, same
number of steps) except for ONE line, the context c_t the decoder sees:

    none       c_t = c  (the encoder's final state: the whole input squeezed into
                         ONE fixed-size vector, the same for every output step)
    attention  c_t = Σ_j α_tj h_j  (a fresh weighted mix of ALL encoder states,
                         recomputed for every output step; scaled dot-product scores,
                         or Bahdanau additive scores with --attention additive)

We then measure exact-match accuracy (every digit right) and per-digit accuracy
for each input length. Expected: without attention, accuracy collapses as inputs
get longer (a fixed vector can't hold 40 digits after a short training run);
with attention it stays high, because output step t simply LOOKS AT input
position S-1-t.

Saves the numbers to bottleneck_results.npz and both models to bottleneck_models.pt
(make_figures.py draws them).

Run (~60 s):
    python attention/bottleneck.py
    python attention/bottleneck.py --steps 3000              (train both longer)
    python attention/bottleneck.py --hidden 128              (a bigger bottleneck vector)
    python attention/bottleneck.py --attention additive      (Bahdanau scores instead)
"""
import argparse
import os
import time

import numpy as np
import torch

from seq2seq_pytorch import exact_match, train
from tasks import DIGIT_VOCAB, sample_reverse

HERE = os.path.dirname(os.path.abspath(__file__))
LENGTHS = [5, 10, 15, 20, 25, 30, 35, 40]
RESULTS_PATH = os.path.join(HERE, "bottleneck_results.npz")
MODELS_PATH = os.path.join(HERE, "bottleneck_models.pt")
CONFIG = dict(task="reverse", steps=800, batch=64, lr=3e-3, emb=32, hidden=64, seed=0)


def per_char_accuracy(preds, pairs):
    """Fraction of output positions that are right (a too-short output counts as wrong there)."""
    right = total = 0
    for p, (_, t) in zip(preds, pairs):
        right += sum(a == b for a, b in zip(p, t))
        total += len(t)
    return right / total


def run(steps=800, hidden=64, attention="dot", n_eval=200, verbose=True):
    cfg = dict(CONFIG, steps=steps, hidden=hidden)
    monitor = sample_reverse(300, np.random.default_rng(5))  # mixed lengths, for the curves
    tests = {L: sample_reverse(n_eval, np.random.default_rng(100 + L), length=L) for L in LENGTHS}
    results = {"lengths": np.array(LENGTHS)}
    models = {}
    for name, kind in [("none", "none"), ("attn", attention)]:
        label = "WITHOUT attention (fixed vector c)" if name == "none" else \
            f"WITH {attention} attention (c_t = Σ α_tj h_j)"
        if verbose:
            print(f"--- training {label}, {cfg['steps']} steps ---")
        t0 = time.time()
        model, hist = train(attention=kind, eval_pairs=monitor, verbose=verbose, **cfg)
        secs = time.time() - t0
        exact, chars = [], []
        for L in LENGTHS:
            acc, preds = exact_match(model, tests[L], DIGIT_VOCAB, 41)
            exact.append(acc)
            chars.append(per_char_accuracy(preds, tests[L]))
        if verbose:
            print(f"    trained + evaluated in {time.time() - t0:.0f} s\n")
        results[f"{name}_exact"] = np.array(exact)
        results[f"{name}_char"] = np.array(chars)
        results[f"{name}_loss"] = np.array([h[1] for h in hist])
        results[f"{name}_eval_steps"] = np.array([h[0] for h in hist if h[2] is not None])
        results[f"{name}_eval_acc"] = np.array([h[2] for h in hist if h[2] is not None])
        results[f"{name}_seconds"] = np.array(secs)
        models[name] = model
    results["n_params"] = np.array(sum(p.numel() for p in models["none"].parameters()))
    return results, models, tests


def print_table(r):
    print("Accuracy vs input length (200 new strings per length)\n")
    print("          |   exact match (all digits right)  |   per-digit accuracy")
    print(" length   |  no attention     with attention  |  no attention   with attention")
    print("----------+-----------------------------------+--------------------------------")
    for i, L in enumerate(r["lengths"]):
        print(f"   {L:3d}    |    {r['none_exact'][i]:6.1%}           {r['attn_exact'][i]:6.1%}"
              f"       |    {r['none_char'][i]:6.1%}          {r['attn_char'][i]:6.1%}")


def show_example(models, length=30, seed=3):
    s, t = sample_reverse(1, np.random.default_rng(seed), length=length)[0]
    print(f"\nOne length-{length} example:")
    print(f"  input           {s}")
    print(f"  target          {t}")
    for name, label in [("none", "no attention"), ("attn", "with attention")]:
        _, (p,) = exact_match(models[name], [(s, t)], DIGIT_VOCAB, 41)
        marks = "".join("^" if i >= len(p) or p[i] != c else " " for i, c in enumerate(t))
        print(f"  {label:<15} {p}")
        print(f"  {'':<15} {marks}   ({marks.count('^')} wrong)")


def save(results, models):
    np.savez(RESULTS_PATH, **results)
    torch.save({"config": CONFIG, "none": models["none"].state_dict(),
                "attn": models["attn"].state_dict()}, MODELS_PATH)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, default=CONFIG["steps"])
    ap.add_argument("--hidden", type=int, default=CONFIG["hidden"],
                    help="encoder size per direction; the bottleneck vector c has 2x this")
    ap.add_argument("--attention", choices=["dot", "additive"], default="dot")
    args = ap.parse_args()

    t0 = time.time()
    print(f"Reverse digit strings of length 5..40. Same model, same data, same {args.steps} "
          f"steps; the bottleneck vector c has {2 * args.hidden} numbers.\n")
    r, models, _ = run(args.steps, args.hidden, args.attention)
    print_table(r)
    show_example(models)
    print(f"\nBoth models have exactly {int(r['n_params']):,} parameters. "
          f"Training time: {float(r['none_seconds']):.0f} s without vs "
          f"{float(r['attn_seconds']):.0f} s with attention.")
    print("Without attention, every digit must survive inside ONE vector; longer input = more to")
    print("cram in = more forgotten. With attention, the decoder re-reads the input at every step.")
    if (args.steps, args.hidden, args.attention) == (CONFIG["steps"], CONFIG["hidden"], "dot"):
        save(r, models)
        print(f"\nSaved the numbers to {os.path.relpath(RESULTS_PATH)} and the models to "
              f"{os.path.relpath(MODELS_PATH)} (used by make_figures.py).")
    print(f"total time {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
