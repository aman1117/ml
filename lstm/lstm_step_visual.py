"""
Watch a trained LSTM read text ONE character at a time, in your terminal.

For every character you see:
  - the one-hot input x_t,
  - the memory coming in: the hidden state h_{t-1} AND the cell state c_{t-1},
  - the four gates, each a vector of H numbers:
        f (forget: how much old memory to KEEP)   i (input: how much to WRITE)
        g (candidate: WHAT to write)              o (output: how much to SHOW)
    as sparklines, with their mean and a one-line interpretation,
  - the new cell c_t = f⊙c_{t-1} + i⊙g split into its "kept" and "written" parts,
  - the new hidden state h_t = o⊙tanh(c_t),
  - the model's top-5 guesses for the NEXT character.

Uses lstm/model.npz if it exists (train it with `python lstm/lstm_numpy.py`),
otherwise trains a quick model first (~8 s).

Run:
    python lstm/lstm_step_visual.py                  (press Enter to go to the next step)
    python lstm/lstm_step_visual.py --prompt "said the keep"
"""
import argparse
import contextlib
import io
import os
import sys

import numpy as np

from lstm_numpy import DEFAULT_TEXT, load_model, softmax, train

HERE = os.path.dirname(os.path.abspath(__file__))
TTY = sys.stdout.isatty()
SHOW = 8  # how many of the H numbers to print
SPARK = "▁▂▃▄▅▆▇█"


def color(code, s):
    return f"\033[{code}m{s}\033[0m" if TTY else s


def cyan(s): return color("36;1", s)      # the input character
def yellow(s): return color("33;1", s)    # memory (h and c)
def green(s): return color("32;1", s)     # results / predictions
def magenta(s): return color("35;1", s)   # gates
def dim(s): return color("2", s)


def pause():
    if sys.stdin.isatty():
        try:
            input(dim("\n  [Enter] next step "))
        except EOFError:  # Ctrl+D: just keep going without pausing
            print()
    print("\n" + "=" * 76 + "\n")


def ch(c):
    """Make spaces and newlines visible."""
    return {" ": "␣", "\n": "\\n"}.get(c, c)


def vec(v, n=SHOW):
    """First n numbers of a column vector, e.g. [+0.12 -0.98 ... ]."""
    v = v.ravel()
    return "[" + " ".join(f"{x:+6.2f}" for x in v[:n]) + (" ...]" if len(v) > n else "]")


def sparkline(v, lo=-1.0, hi=1.0, width=50):
    """All numbers as tiny bars from lo (▁) to hi (█). Wrapped into rows."""
    v = np.clip((v.ravel() - lo) / (hi - lo), 0, 1)
    s = "".join(SPARK[min(7, int(x * 8))] for x in v)
    return [s[i:i + width] for i in range(0, len(s), width)]


def bar(p, width=30):
    n = int(round(p * width))
    return "█" * n + "·" * (width - n)


def name(sym, t):
    return f"{sym}_{{-1}}" if t < 0 else f"{sym}_{t}"


def describe(gate, v):
    """One sentence about a gate vector, in plain words."""
    m = float(v.mean())
    hi, lo = int((v > 0.9).sum()), int((v < 0.1).sum())
    if gate == "f":
        words = ("keeping almost all of the old memory" if m > 0.9 else
                 "keeping most of the old memory" if m > 0.65 else
                 "keeping about half, erasing about half" if m > 0.35 else
                 "erasing most of the old memory")
        return f"forget gate ≈ {m:.2f}: {words} ({hi} units keep > 90%, {lo} erase > 90%)"
    if gate == "i":
        words = ("writing a lot of new content" if m > 0.65 else
                 "writing a moderate amount" if m > 0.35 else "writing very little")
        return f"input gate ≈ {m:.2f}: {words} ({hi} units wide open, {lo} shut)"
    if gate == "o":
        words = ("showing most of the memory to the output" if m > 0.65 else
                 "showing part of the memory" if m > 0.35 else "hiding most of the memory")
        return f"output gate ≈ {m:.2f}: {words} ({hi} units open, {lo} shut)"
    a = float(np.abs(v).mean())
    return (f"candidate g in (-1, 1), mean |g| ≈ {a:.2f}: the new content on offer "
            f"({int((np.abs(v) > 0.9).sum())} units near ±1)")


def get_model():
    path = os.path.join(HERE, "model.npz")
    if os.path.exists(path):
        model, chars = load_model(path)
        if model.cell == "lstm":
            print(dim(f"(loaded the trained LSTM from {os.path.relpath(path)})\n"))
            return model, chars
    print("No trained LSTM at lstm/model.npz yet, so training a quick one first "
          "(3000 iterations, ~8 s).")
    print("For a better model run: python lstm/lstm_numpy.py\n")
    with open(DEFAULT_TEXT, encoding="utf-8") as f:
        text = f.read()
    with contextlib.redirect_stdout(io.StringIO()):  # hide the training printout
        model, _, itos, _ = train(text, iters=3000, sample_every=0, verbose=False)
    return model, [itos[i] for i in range(len(itos))]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("text", nargs="?", help="same as --prompt")
    ap.add_argument("--prompt", default=None, help='text to read (default "keep")')
    args = ap.parse_args()
    prompt = args.prompt or args.text or "keep"

    model, chars = get_model()
    stoi = {c: i for i, c in enumerate(chars)}
    unknown = sorted({c for c in prompt if c not in stoi})
    if unknown:
        print(f"(skipping characters never seen in training: {unknown})\n")
    prompt = "".join(c for c in prompt if c in stoi)
    if not prompt:
        raise SystemExit("The prompt has no characters the model knows.")
    P = model.params
    V, H = model.V, model.H

    print("STEP 0: the setup. Same five parameters as the vanilla RNN, but W_xh, W_hh and b_h")
    print("are FOUR times taller: one (H, ·) block per gate, stacked in the order i, f, g, o.\n")
    print(f"   W_xh {str(P['W_xh'].shape):>10}   input  → 4 gates  ({cyan('what I just read')})")
    print(f"   W_hh {str(P['W_hh'].shape):>10}   h      → 4 gates  ({yellow('what I remember')})")
    print(f"   b_h  {str(P['b_h'].shape):>10}   bias of the 4 gates (forget part started at 1)")
    print(f"   W_hy {str(P['W_hy'].shape):>10}   h      → output   ({green('what I predict next')})")
    print(f"\n   V = {V} different characters, H = {H} memory units.")
    print("   The memory is now TWO vectors: the cell state c (the long-term store, a")
    print("   \"conveyor belt\" that is only scaled and added to) and the hidden state h")
    print("   (what the cell chooses to show: h = o ⊙ tanh(c)).")
    print(f"\n   We will read {cyan(repr(prompt))} one character at a time, starting from "
          "h = c = 0.")
    print(dim(f"   (Only the first {SHOW} of the {H} numbers are printed; the bars show all {H}."
              "\n    Gates: ▁ = 0 (closed) … █ = 1 (open). h, g: ▁ = -1, ▄▅ ≈ 0, █ = +1. "
              "c: ▁ = -5 … █ = +5, clipped.)"))
    pause()

    h, c = model.init_state()
    for t, cc in enumerate(prompt):
        ix = stoi[cc]
        nxt = prompt[t + 1] if t + 1 < len(prompt) else None
        x = np.zeros((V, 1))
        x[ix] = 1.0
        h_new, c_new, (i, f, g, o, tc) = model.cell_step(x, h, c)
        kept, written = f * c, i * g
        p = softmax(P["W_hy"] @ h_new + P["b_y"]).ravel()

        print(f"STEP {t + 1}/{len(prompt)}: read {cyan(repr(cc))}   "
              f"(time step t = {t}, text so far: {cyan(repr(prompt[:t + 1]))})\n")

        print(f"  1) Input x_{t}: one-hot, a 1 at position {cyan(str(ix))}. So W_xh · x_{t} "
              f"just picks column {ix} of W_xh.\n")

        print(f"  2) Memory coming in:   {name('h', t - 1)} {yellow(vec(h))}")
        for row in sparkline(h):
            print(f"{'':27}{yellow(row)}")
        print(f"{'':23}{name('c', t - 1)} {yellow(vec(c))}")
        for row in sparkline(c, -5, 5):
            print(f"{'':27}{yellow(row)}")
        if t == 0:
            print(dim("     (all zeros: nothing has been read yet)"))
        print()

        print(f"  3) The four gates: a = W_xh·x_{t} + W_hh·{name('h', t - 1)} + b_h, "
              "split into 4 blocks of H:\n")
        for sym, v, lo in [("f", f, 0), ("i", i, 0), ("g", g, -1), ("o", o, 0)]:
            fn = "tanh   " if sym == "g" else "sigmoid"
            print(f"     {magenta(sym)} = {fn} {magenta(vec(v))}")
            for row in sparkline(v, lo, 1):
                print(f"{'':17}{magenta(row)}")
            print(f"{'':17}{describe(sym, v)}")
        print()

        print(f"  4) New cell state c_{t} = f ⊙ {name('c', t - 1)}  +  i ⊙ g   "
              "(ADD, don't overwrite):\n")
        print(f"     {'kept    f ⊙ c':<16} = {yellow(vec(kept))}   size {np.linalg.norm(kept):5.2f}")
        print(f"     {'written i ⊙ g':<16} = {cyan(vec(written))}   size {np.linalg.norm(written):5.2f}")
        print(f"     {'':<16}   {'─' * len(vec(kept))}")
        print(f"     {'c_' + str(t):<16} = {green(vec(c_new))}   size {np.linalg.norm(c_new):5.2f}")
        for row in sparkline(c_new, -5, 5):
            print(f"{'':24}{green(row)}")
        share = np.linalg.norm(kept) / (np.linalg.norm(kept) + np.linalg.norm(written) + 1e-12)
        print(dim(f"     ({share:.0%} of the new memory's size is old memory carried over, "
                  f"{1 - share:.0%} is newly written."))
        print(dim(f"      The largest |c| is {np.abs(c_new).max():.2f}: c is NOT squashed "
                  "into (-1, 1) like h is.)\n"))

        print(f"  5) New hidden state h_{t} = o ⊙ tanh(c_{t}): what the cell shows the world.")
        print(f"     {green(vec(h_new))}")
        for row in sparkline(h_new):
            print(f"     {green(row)}")
        print()

        print(f"  6) Prediction: p = softmax(W_hy · h_{t} + b_y). Top 5 guesses for the "
              "NEXT character:\n")
        top = np.argsort(p)[::-1][:5]
        for k in top:
            label = f"'{ch(chars[k])}'"
            line = f"     {label:>5}  {bar(p[k])}  {p[k]:6.1%}"
            if nxt is not None and chars[k] == nxt:
                line = green(line) + "   ← the actual next char in your text"
            print(line)
        if nxt is not None and nxt not in [chars[k] for k in top]:
            print(f"\n     The actual next char {ch(nxt)!r} got only {p[stoi[nxt]]:.1%} "
                  "(not in the top 5).")
        elif nxt is None:
            print("\n     End of your text: generate.py would now SAMPLE one of these, feed")
            print("     it back in as the next input, and repeat. That's text generation.")
        h, c = h_new, c_new
        pause()

    print("RECAP\n")
    print(f"  * We read {len(prompt)} characters with the SAME weights at every step "
          "(inference, not training).")
    print("  * Each step the four gates were computed from (x_t, h_{t-1}) in ONE matrix "
          "multiply, then split:")
    print("      f = erase (how much old c to keep), i = write (how much of g to add),")
    print("      g = the candidate content,          o = read (how much of c to expose as h).")
    print("  * The cell update c_t = f⊙c_{t-1} + i⊙g is a weighted ADD. Backprop through it")
    print("    multiplies the gradient by f only (no W_hh, no tanh'), so when f ≈ 1 the")
    print("    learning signal travels far back. That is the whole trick (README section 4).")
    print("  * h_t = o⊙tanh(c_t) is a filtered view of the memory; c itself is not squashed.")
    print("\nNext: python lstm/long_memory.py shows an RNN failing and an LSTM succeeding at "
          "remembering\nsomething 80 steps back. README sections 3-6 have the pictures "
          "(python lstm/make_figures.py).")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n(stopped)")
