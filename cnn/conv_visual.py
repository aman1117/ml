"""
See a convolution computed step by step, as pictures in your terminal.

A 6x6 image and one 3x3 kernel (a vertical-edge detector). The kernel slides over
the image; at every position:
    output cell = Σ (3x3 patch ⊙ kernel) + bias
Then ReLU, then 2x2 max pooling, then what padding and stride do to the output size:
    out = (n + 2p - k) / s + 1
At the end: the 8 filters a REAL trained CNN learned (cnn_numpy.py) and what they
see in a real image.

Run:
    python cnn/conv_visual.py        (press Enter to go to the next step)
"""
import contextlib
import io
import sys

import numpy as np

from cnn_numpy import (CLASSES, Conv2D, ascii_image, conv2d_naive, default_data, get_model,
                       im2col, out_size, side_by_side)

TTY = sys.stdout.isatty()

# The image: a bright block with a dimmer "shoulder" on its left, on a dark background.
X = np.array([[0, 0, 0, 0, 0, 0],
              [0, 0, 1, 1, 0, 0],
              [0, 0, 1, 1, 0, 0],
              [0, 0.5, 1, 1, 1, 0],
              [0, 0.5, 1, 1, 1, 0],
              [0, 0, 0, 0, 0, 0]])
# Sobel vertical-edge kernel: "right side brighter than left side?"
K = np.array([[-1, 0, 1],
              [-2, 0, 2],
              [-1, 0, 1]], dtype=float)
B = -1.0  # bias: needs at least 1 unit of "edge evidence" to survive the ReLU


def color(code, s):
    return f"\033[{code}m{s}\033[0m" if TTY else s


def cyan(s): return color("36;1", s)      # the image / the current window
def yellow(s): return color("33;1", s)    # the kernel
def green(s): return color("32;1", s)     # results
def dim(s): return color("2", s)


def pause():
    if sys.stdin.isatty():
        try:
            input(dim("\n  [Enter] next step "))
        except EOFError:
            print()
    print("\n" + "=" * 72 + "\n")


def n(v):
    return f"{v:g}"


def grid(M, hl=(), paint=cyan, w=6, blank=None):
    """Matrix → text lines. Cells in `hl` are [bracketed] and coloured.
    Cells equal to `blank` (e.g. NaN = not computed yet) are shown as '·'."""
    lines = []
    for i, row in enumerate(M):
        cells = []
        for j, v in enumerate(row):
            s = "·" if (blank is not None and np.isnan(v)) else n(v)
            if (i, j) in hl:
                cells.append(paint(f"[{s:>{w - 2}}]"))
            else:
                cells.append(f" {s:>{w - 2}} ")
        lines.append("".join(cells))
    return lines


def beside(*blocks, titles=None, gap=6):
    """Print text blocks (lists of lines) next to each other. Handles ANSI codes."""
    import re
    vis = lambda s: len(re.sub(r"\033\[[0-9;]*m", "", s))
    widths = [max(vis(line) for line in b) for b in blocks]
    if titles:
        print((" " * gap).join(f"{t:<{w}}" for t, w in zip(titles, widths)).rstrip())
    for i in range(max(len(b) for b in blocks)):
        parts = []
        for b, w in zip(blocks, widths):
            s = b[i] if i < len(b) else ""
            parts.append(s + " " * (w - vis(s)))
        print("   " + (" " * gap).join(parts).rstrip())


def window(i, j, k=3):
    return {(i + a, j + b) for a in range(k) for b in range(k)}


def show_cell(out, i, j):
    patch = X[i:i + 3, j:j + 3]
    prod = patch * K + 0.0  # + 0.0 turns -0 into 0 for printing
    val = prod.sum() + B
    out[i, j] = val
    print(f"  The 3x3 window sits on rows {i}-{i + 2}, columns {j}-{j + 2} of the image:\n")
    beside(grid(X, window(i, j)), grid(patch), grid(K, paint=yellow), grid(prod),
           titles=["   image (window in [ ])", "patch", "kernel", "patch ⊙ kernel"], gap=4)
    terms = " + ".join(f"({n(v)})" if v < 0 else n(v) for v in prod.ravel())
    print(f"\n  sum of the 9 products = {terms}")
    print(f"                        = {n(prod.sum())}")
    print(f"  + bias ({n(B)})           = {green(n(val))}   → output cell ({i}, {j})\n")
    print("  feature map so far:")
    beside(grid(out, {(i, j)}, paint=green, blank=True))
    return val


def main():
    k, Hn = 3, X.shape[0]
    Ho = out_size(Hn, k)

    print("STEP 0: the setup. A 6x6 grayscale image (0 = black, 1 = white) and ONE 3x3 kernel.\n")
    beside(grid(X), [" " + line for line in ascii_image(X)], grid(K, paint=yellow),
           titles=["   image X (numbers)", "  as a picture", "kernel K (weights)"])
    print(f"\n  The kernel's left column is negative and its right column positive: it asks")
    print("  'is it brighter on my RIGHT than on my LEFT?'  That's a vertical-edge detector")
    print("  (the classic Sobel filter). The middle row counts double. Bias b = "
          f"{n(B)}.")
    print(f"\n  Output size: (n + 2p - k) / s + 1 = ({Hn} + 0 - {k}) / 1 + 1 = {Ho}  →  "
          f"a {Ho}x{Ho} feature map.")
    print("  Each output cell = Σ(3x3 patch ⊙ kernel) + bias. Same 9 weights at EVERY position.")
    pause()

    out = np.full((Ho, Ho), np.nan)
    step = 1
    for j in range(Ho):  # first row, one cell at a time, in full detail
        print(f"STEP {step}: output cell (0, {j})\n")
        show_cell(out, 0, j)
        if j == 0:
            print(dim("\n  The block's left edge is inside the window (dark left, bright right):"
                      " a big\n  positive number. The kernel 'found' what it looks for."))
        if j == 2:
            print(dim("\n  Now bright is on the LEFT and dark on the RIGHT: strongly negative."
                      " This kernel\n  fires on dark→bright edges only; it is a pattern detector,"
                      " not a general 'edge' detector."))
        step += 1
        pause()

    print(f"STEP {step}: the remaining rows, same recipe (slide right, then move down one row):\n")
    for i in range(1, Ho):
        for j in range(Ho):
            patch = X[i:i + 3, j:j + 3]
            out[i, j] = (patch * K).sum() + B
            print(f"  cell ({i}, {j}): Σ(patch ⊙ K) = {n((patch * K).sum()):>4}, "
                  f"+ bias → {green(n(out[i, j]))}")
    print("\n  the full feature map:")
    beside(grid(out, paint=green))
    step += 1
    pause()

    print(f"STEP {step}: the same thing, fast. All {Ho * Ho} windows at once as ONE matrix multiply"
          " (im2col):\n")
    cols, _, _ = im2col(X[None, None], k)
    print(f"  im2col cuts every 3x3 patch into a ROW of 9 numbers → a ({cols.shape[0]}, "
          f"{cols.shape[1]}) matrix. The first 3 rows:")
    for r in range(3):
        print("     " + cyan("[" + " ".join(f"{n(v):>3}" for v in cols[r]) + " ]")
              + f"   ← patch at (0, {r})")
    print("     ...")
    print(f"  kernel flattened to a column of 9: {yellow('[' + ' '.join(n(v) for v in K.ravel()) + ']')}")
    print(f"\n  ({cols.shape[0]}, 9) @ (9, 1) + b = ({cols.shape[0]}, 1) → reshape to {Ho}x{Ho}.")
    conv = Conv2D(1, 1, 3)
    conv.params["W"], conv.params["b"] = K[None, None].copy(), np.array([B])
    fast = conv.forward(X[None, None])[0, 0]
    slow = conv2d_naive(X[None, None], K[None, None], np.array([B]))[0, 0]
    assert np.allclose(fast, out) and np.allclose(slow, out)
    print(green("  Same numbers as the cell-by-cell loop ✔") + "  (cnn_numpy.Conv2D and "
          "conv2d_naive agree)")
    print("  GPUs love this: a huge matrix multiply is exactly what they are built for.")
    step += 1
    pause()

    relu = np.maximum(out, 0)
    print(f"STEP {step}: ReLU = max(0, z) on every cell. Negative evidence becomes 0.\n")
    beside(grid(out), grid(relu, {(i, j) for i in range(Ho) for j in range(Ho) if relu[i, j] > 0},
                           paint=green),
           titles=["   feature map", "after ReLU"], gap=8)
    print("\n  Only the block's LEFT edge survives. A second kernel (this one flipped"
          " left↔right)\n  would light up on the right edge: that is why real layers have MANY"
          " kernels (channels).")
    step += 1
    pause()

    print(f"STEP {step}: 2x2 max pooling (stride 2). Keep only the biggest number of each"
          " 2x2 block.\n")
    pooled = np.zeros((Ho // 2, Ho // 2))
    for a in range(Ho // 2):
        for b in range(Ho // 2):
            win = relu[2 * a:2 * a + 2, 2 * b:2 * b + 2]
            r, c = np.unravel_index(win.argmax(), win.shape)  # first max on ties
            pooled[a, b] = win.max()
            cells = {(2 * a + r, 2 * b + c)}
            vals = ", ".join(n(v) for v in win.ravel())
            note = "  (all zero: nothing detected here)" if win.max() == 0 else ""
            print(f"  block ({a}, {b}): values [{vals}] → max {green(n(win.max()))}{note}")
            beside(grid(relu, cells, paint=green))
            print()
    print("  pooled map (4x4 → 2x2):")
    beside(grid(pooled, paint=green))
    print("\n  What pooling buys: 4x fewer numbers for the next layer, and a small shift of the")
    print("  edge inside a block gives the same answer (a bit of translation invariance).")
    print("  Backward: the gradient of each pooled cell goes ONLY to the winning [cell].")
    step += 1
    pause()

    print(f"STEP {step}: padding p and stride s change the output size:"
          "  out = (n + 2p - k) / s + 1\n")
    for p, s in [(0, 1), (1, 1), (0, 2), (1, 2)]:
        o = conv2d_naive(X[None, None], K[None, None], np.array([B]), stride=s, pad=p)[0, 0]
        exact = (Hn + 2 * p - k) / s + 1
        rounding = "" if exact == int(exact) else f" → floor → {int(exact)}"
        print(f"  p={p}, s={s}:  ({Hn} + {2 * p} - {k}) / {s} + 1 = {exact:g}{rounding}"
              f"   →  {o.shape[0]}x{o.shape[1]}")
        beside(grid(o))
        print()
    print("  p=1 with a 3x3 kernel keeps the size ('same' padding): a ring of zeros around the")
    print("  image lets the window be centred on border pixels. s=2 jumps 2 pixels at a time,")
    print("  which halves the size (like pooling, but learned).")
    step += 1
    pause()

    print(f"STEP {step}: a REAL trained CNN. Nobody designed its first-layer kernels; they were"
          " learned.\n")
    with contextlib.redirect_stdout(io.StringIO()) if not TTY else contextlib.nullcontext():
        model = get_model()
    W1, b1 = model.layers[0][1].params["W"], model.layers[0][1].params["b"]
    print(f"  conv1 has {W1.shape[0]} kernels, W shape (C_out, C_in, kH, kW) = {W1.shape}:\n")
    blocks = [grid(np.round(W1[f, 0], 1) + 0.0) for f in range(W1.shape[0])]
    beside(*blocks[:4], titles=[f"   filter {f} (b={b1[f]:+.2f})" for f in range(4)], gap=3)
    print()
    beside(*blocks[4:], titles=[f"   filter {f} (b={b1[f]:+.2f})" for f in range(4, 8)], gap=3)
    _, _, Xval, yval = default_data()
    img = Xval[2:3]
    _, acts = model.forward(img, keep=True)
    fmap = acts["relu1"][0]  # (8, 16, 16) after ReLU
    top = np.argsort(fmap.reshape(len(fmap), -1).max(axis=1))[::-1][:2]
    print(f"\n  A validation image ({CLASSES[yval[2]]}) and the two feature maps (after ReLU)"
          " that respond most:\n")
    vmax = fmap[top].max()
    side_by_side([ascii_image(img[0, 0])] + [ascii_image(fmap[f], 0, vmax) for f in top],
                 ["input image"] + [f"filter {f} feature map" for f in top])
    print("\n  Each feature map is the SAME picture-sized grid, lit up wherever its kernel's")
    print("  pattern appears. Move the shape, and the bright spots move with it (equivariance).")
    step += 1
    pause()

    print("RECAP\n")
    print("  * A conv layer slides a small kernel over the image. Each output cell is a dot")
    print("    product of a patch with the kernel, plus a bias: an MLP neuron that only looks")
    print("    at a 3x3 patch and is REUSED at every position (weight sharing).")
    print("  * Kernel = pattern detector: big positive output where the patch looks like it.")
    print("  * ReLU keeps the positive evidence; max pooling keeps the strongest per block.")
    print("  * Output size = (n + 2p - k) / s + 1. Padding keeps size, stride shrinks it.")
    print("  * In practice all windows are done at once: im2col + one matrix multiply.")
    print("  * A trained CNN learns its own kernels (edges and bars first, shapes later).")
    print("\nNext: README sections 2-6 (python cnn/make_figures.py redraws the pictures).")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n(stopped)")
