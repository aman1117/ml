"""
Watch a trained RNN read text ONE character at a time, in your terminal.

For every character you see:
  - the one-hot input x_t,
  - the memory coming in, h_{t-1},
  - the two contributions W_xh·x_t ("what I just read") and W_hh·h_{t-1}
    ("what I remember"),
  - the new memory h_t = tanh(...),
  - the model's top-5 guesses for the NEXT character.

Uses rnn/model.npz if it exists (train it with `python rnn/rnn_numpy.py`),
otherwise trains a quick model first (~7 s).

Run:
    python rnn/rnn_step_visual.py                  (press Enter to go to the next step)
    python rnn/rnn_step_visual.py --prompt "The light"
"""
import argparse
import contextlib
import io
import os
import sys

import numpy as np

from rnn_numpy import load_model, softmax, train

HERE = os.path.dirname(os.path.abspath(__file__))
TTY = sys.stdout.isatty()
SHOW = 8  # how many of the H hidden numbers to print
SPARK = "▁▂▃▄▅▆▇█"


def color(code, s):
    return f"\033[{code}m{s}\033[0m" if TTY else s


def cyan(s): return color("36;1", s)      # the input character
def yellow(s): return color("33;1", s)    # memory (hidden state)
def green(s): return color("32;1", s)     # results / predictions
def dim(s): return color("2", s)


def pause():
    if sys.stdin.isatty():
        try:
            input(dim("\n  [Enter] next step "))
        except EOFError:  # Ctrl+D: just keep going without pausing
            print()
    print("\n" + "=" * 72 + "\n")


def ch(c):
    """Make spaces and newlines visible."""
    return {" ": "␣", "\n": "\\n"}.get(c, c)


def vec(v, n=SHOW):
    """First n numbers of a column vector, e.g. [+0.12 -0.98 ... ]."""
    v = v.ravel()
    return "[" + " ".join(f"{x:+6.2f}" for x in v[:n]) + (" ...]" if len(v) > n else "]")


def sparkline(v, width=50):
    """All numbers as tiny bars: ▁ = -1, ▄▅ ≈ 0, █ = +1. Wrapped into rows."""
    v = np.clip(v.ravel(), -1, 1)
    s = "".join(SPARK[min(7, int((x + 1) / 2 * 8))] for x in v)
    return [s[i:i + width] for i in range(0, len(s), width)]


def hname(t):
    return "h_{-1}" if t < 0 else f"h_{t}"


def bar(p, width=30):
    n = int(round(p * width))
    return "█" * n + "·" * (width - n)


def get_model():
    path = os.path.join(HERE, "model.npz")
    if os.path.exists(path):
        model, chars = load_model(path)
        print(dim(f"(loaded the trained model from {os.path.relpath(path)})\n"))
        return model, chars
    print("No rnn/model.npz yet, so training a quick model first (5000 iterations, ~7 s).")
    print("For a better model run: python rnn/rnn_numpy.py\n")
    with open(os.path.join(HERE, "input.txt"), encoding="utf-8") as f:
        text = f.read()
    with contextlib.redirect_stdout(io.StringIO()):  # hide the training printout
        model, _, itos = train(text, iters=5000)
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

    print("STEP 0: the setup. The RNN has just THREE weight matrices (plus two biases):\n")
    print(f"   W_xh {str(P['W_xh'].shape):>10}   input  → memory   ({cyan('what I just read')})")
    print(f"   W_hh {str(P['W_hh'].shape):>10}   memory → memory   ({yellow('what I remember')})")
    print(f"   W_hy {str(P['W_hy'].shape):>10}   memory → output   ({green('what I predict next')})")
    print(f"\n   V = {V} different characters, H = {H} numbers of memory.")
    print(f"\n   We will read {cyan(repr(prompt))} one character at a time.")
    print(f"   The memory h starts as {H} zeros: the model hasn't read anything yet.")
    print(dim(f"   (Only the first {SHOW} of the {H} memory numbers are printed; the bars "
              f"show all {H}: ▁ = -1, ▄▅ ≈ 0, █ = +1.)"))
    pause()

    h = np.zeros((H, 1))
    explained = False
    for t, c in enumerate(prompt):
        ix = stoi[c]
        nxt = prompt[t + 1] if t + 1 < len(prompt) else None
        x = np.zeros((V, 1))
        x[ix] = 1.0
        from_x = P["W_xh"] @ x
        from_h = P["W_hh"] @ h
        a = from_x + from_h + P["b_h"]
        h_new = np.tanh(a)
        p = softmax(P["W_hy"] @ h_new + P["b_y"]).ravel()

        print(f"STEP {t + 1}/{len(prompt)}: read {cyan(repr(c))}   "
              f"(time step t = {t}, text so far: {cyan(repr(prompt[:t + 1]))})\n")

        print(f"  1) Input x_{t}: a one-hot vector of length {V}. All zeros except a 1 at")
        print(f"     position {cyan(str(ix))}, because {cyan(repr(c))} is character #{ix} "
              "in the vocabulary.\n")

        print(f"  2) Memory coming in, {hname(t - 1)}:")
        print(f"     {yellow(vec(h))}")
        for row in sparkline(h):
            print(f"     {yellow(row)}")
        if t == 0:
            print(dim("     (all zeros: nothing has been read yet)"))
        print()

        label = f"W_hh · {hname(t - 1)}"
        print("  3) Mix what I just read with what I remember (first numbers shown):\n")
        print(f"     {'W_xh · x_' + str(t):<14} = {cyan(vec(from_x))}   "
              f"← just column {ix} of W_xh")
        print(f"     {label:<14} = {yellow(vec(from_h))}   ← the memory's vote")
        print(f"     {'+ b_h':<14} = {vec(P['b_h'])}")
        print(f"     {'':<14}   {'─' * len(vec(a))}")
        print(f"     {'sum':<14} = {vec(a)}\n")

        print(f"  4) New memory h_{t} = tanh(sum): squashed into (-1, 1).")
        print(f"     {green(vec(h_new))}")
        for row in sparkline(h_new):
            print(f"     {green(row)}")
        print(dim(f"     (changed by {np.linalg.norm(h_new - h):.2f} in total; "
                  f"{np.mean(np.abs(h_new) > 0.9):.0%} of the {H} numbers are near ±1)\n"))

        print(f"  5) Prediction: p = softmax(W_hy · h_{t} + b_y). Top 5 guesses for the "
              "NEXT character:\n")
        for i in np.argsort(p)[::-1][:5]:
            name = f"'{ch(chars[i])}'"
            line = f"     {name:>5}  {bar(p[i])}  {p[i]:6.1%}"
            if nxt is not None and chars[i] == nxt:
                line = green(line) + "   ← the actual next char in your text"
            print(line)
        if nxt is not None and nxt not in [chars[i] for i in np.argsort(p)[::-1][:5]]:
            print(f"\n     The actual next char {ch(nxt)!r} got only {p[stoi[nxt]]:.1%} "
                  "(not in the top 5).")
        if nxt is not None and p[stoi[nxt]] < 0.1 and not explained:
            explained = True
            print(dim("\n     Why so unsure? In training the memory was almost never empty (it is only"))
            print(dim("     reset at the start of the story), so starting from h = 0 in the middle"))
            print(dim("     of a word confuses it. Give it some context and it does much better:"))
            print(dim('     python rnn/rnn_step_visual.py --prompt "said the keep"'))
        elif nxt is None:
            print("\n     End of your text: generate.py would now SAMPLE one of these, feed")
            print("     it back in as the next input, and repeat. That's text generation.")
        h = h_new
        pause()

    print("RECAP\n")
    print(f"  * We read {len(prompt)} characters and used the SAME three matrices "
          "(W_xh, W_hh, W_hy) at every step.")
    print("    Nothing about the weights changed; this is inference, not training.")
    print(f"  * The only thing that changed was the memory h ({H} numbers). It carried what")
    print("    the model had read so far from one step to the next.")
    print("  * W_xh · x_t just picks one column of W_xh (that's what a one-hot input does).")
    print("  * Each step: new memory = tanh(what I just read + what I remember + bias),")
    print("    then predict the next character from the new memory.")
    print("\nNext: look at the pictures in README sections 2-5 "
          "(python rnn/make_figures.py redraws them).")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n(stopped)")
