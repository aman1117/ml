"""
Draw the pictures used in cnn/README.md and save them to cnn/images/.

Every number shown in a figure is computed here: the script trains the same CNN
as cnn_numpy.py (same data, hyperparameters and seed, so the weights are identical
to what `python cnn/cnn_numpy.py` saves), runs the cnn_vs_mlp.py experiments, and
then looks inside the models.

Run (takes ~15-20 s):
    python cnn/make_figures.py
"""
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

from cnn_numpy import (CLASSES, CNN, Conv2D, conv2d_naive, default_data, draw_shape,  # noqa: E402
                       im2col, out_size, softmax, train)
from cnn_vs_mlp import part1, part2  # noqa: E402
from conv_visual import B as EX_B, K as EX_K, X as EX_X  # noqa: E402  (the terminal example)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "images")
DPI = 130

# Colours used in every figure.
C_IMG, C_KER, C_OUT = "#1f77b4", "#ff7f0e", "#2ca02c"
C_WIN, C_KFILL, C_OFILL = "#dbe9f6", "#ffe6c7", "#e3f4e1"
C_PAD = "#e6e6e6"
C_MLP, C_CNN, C_GAP = "#7f7f7f", "#1f77b4", "#2ca02c"

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "figure.titlesize": 16,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {os.path.relpath(path)}  ({os.path.getsize(path) // 1024} KB)")


def box(ax, x, y, text, fc="white", ec="black", w=2.0, h=1.0, fs=12, lw=1.8, **kw):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.15",
                                fc=fc, ec=ec, lw=lw, zorder=2))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, zorder=3, **kw)


def arrow(ax, a, b, color="black", lw=2.0, style="-|>", ms=18, **kw):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=ms,
                                 color=color, lw=lw, zorder=1, **kw))


def blank(ax, xlim, ylim):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.axis("off")


def fmt_num(v):
    return f"{v:g}" if abs(v - round(v, 2)) < 1e-9 else f"{v:.2f}"


def draw_grid(ax, M, x0, y0, cell=1.0, fill=None, hl=None, fs=12, fmt=fmt_num, lw=1.0,
              shade=False, vmin=0.0, vmax=1.0, text=True):
    """Draw matrix M with its top-left corner at (x0, y0); rows go DOWN.
    fill: default cell colour. hl: {(i, j): colour}. shade: grey level from the value."""
    R, C = M.shape
    hl = hl or {}
    for i in range(R):
        for j in range(C):
            v = M[i, j]
            if (i, j) in hl:
                fc = hl[(i, j)]
            elif shade:
                g = 1 - 0.85 * np.clip((v - vmin) / (vmax - vmin + 1e-12), 0, 1)
                fc = (g, g, g)
            else:
                fc = fill or "white"
            ax.add_patch(Rectangle((x0 + j * cell, y0 - (i + 1) * cell), cell, cell, fc=fc,
                                   ec="#555", lw=lw, zorder=1))
            if text:
                dark = shade and (i, j) not in hl and fc[0] < 0.45
                ax.text(x0 + (j + 0.5) * cell, y0 - (i + 0.5) * cell, fmt(v), ha="center",
                        va="center", fontsize=fs, color="white" if dark else "black", zorder=3)


def outline(ax, x0, y0, rows, cols, cell=1.0, color="red", lw=3.0, ls="-"):
    ax.add_patch(Rectangle((x0, y0 - rows * cell), cols * cell, rows * cell, fill=False,
                           ec=color, lw=lw, ls=ls, zorder=4))


# ---------------------------------------------------------------- model helpers

def train_main():
    """Exactly what `python cnn/cnn_numpy.py` does (same data, seed, hyperparameters)."""
    Xtr, ytr, Xval, yval = default_data()
    model = CNN()
    hist = train(model, Xtr, ytr, Xval, yval, verbose=False)
    return model, hist, (Xtr, ytr, Xval, yval)


def backward_trace(model, x, dlogits):
    """Backward pass that also returns the gradient ARRIVING at every layer's output."""
    d, grads_at = dlogits, {}
    for name, layer in reversed(model.layers):
        grads_at[name] = d
        d = layer.backward(d)
    return grads_at


# ---------------------------------------------------------------- figures

def fig_conv_operation():
    X, K, b = EX_X, EX_K, EX_B
    i, j = 1, 1
    out = conv2d_naive(X[None, None], K[None, None], np.array([b]))[0, 0]
    patch = X[i:i + 3, j:j + 3]
    prod = patch * K + 0.0

    fig, ax = plt.subplots(figsize=(15, 6.6))
    blank(ax, (-0.6, 22.6), (-8.6, 2.2))
    win = {(i + a, j + c): C_WIN for a in range(3) for c in range(3)}
    draw_grid(ax, X, 0, 0, hl=win, shade=True)
    outline(ax, j, -i, 3, 3, color=C_IMG)
    ax.text(3, 0.9, "input image X  (6x6)", ha="center", fontsize=13, weight="bold")
    ax.text(3, 0.35, "0 = black, 1 = white", ha="center", fontsize=10.5, color="#555")
    ax.text(3, -6.3, "blue window = the patch for output[1,1]\n(it slides right, then down)",
            ha="center", va="top", fontsize=10.5, color=C_IMG)

    ax.text(7.6, -3.0, "⊙", fontsize=30, ha="center", va="center")
    draw_grid(ax, K, 8.6, -1.5, fill=C_KFILL)
    outline(ax, 8.6, -1.5, 3, 3, color=C_KER)
    ax.text(10.1, -0.6, "kernel K (3x3)", ha="center", fontsize=13, weight="bold")
    ax.text(10.1, -5.0, f"bias b = {b:g}", ha="center", fontsize=12)

    ax.text(12.4, -3.0, "=", fontsize=30, ha="center", va="center")
    draw_grid(ax, prod, 13.3, -1.5, fill="#f4f4f4")
    ax.text(14.8, -0.6, "patch ⊙ K", ha="center", fontsize=13, weight="bold")
    ax.text(14.8, -5.0, f"sum = {prod.sum():g}", ha="center", fontsize=12)

    ax.text(17.3, -3.0, "→", fontsize=30, ha="center", va="center")
    draw_grid(ax, out, 18.4, -1.0, hl={(i, j): "#9be29b"}, fill=C_OFILL)
    outline(ax, 18.4 + j, -1.0 - i, 1, 1, color=C_OUT)
    ax.text(20.4, -0.1, "output feature map (4x4)", ha="center", fontsize=13, weight="bold")

    terms = " + ".join(f"({v:g})" if v < 0 else f"{v:g}" for v in prod.ravel())
    ax.text(11, -7.9,
            f"output[{i},{j}] = Σ (patch ⊙ K) + b = {terms} + ({b:g}) = {out[i, j]:g}\n"
            "The same 9 weights are used at every one of the 16 positions (weight sharing). "
            f"Output size = (6 - 3)/1 + 1 = {out.shape[0]}.",
            ha="center", va="center", fontsize=12.5,
            bbox=dict(boxstyle="round", fc="#fffbe6", ec="#d4b106"))
    ax.set_title("A convolution: slide the kernel over the image; each output = dot product of "
                 "a patch with the kernel + bias", pad=10)
    save(fig, "conv_operation.png")
    return out


def demo_scene(size=30):
    """A bigger synthetic picture with several shapes, for the hand-made filters."""
    img = np.zeros((size, size))
    draw_shape(img, 5, 2, 2, 11, 2, 1.0)    # square outline
    draw_shape(img, 3, 2, 17, 11, 2, 1.0)   # diagonal /
    draw_shape(img, 4, 16, 2, 11, 1, 1.0)   # plus
    draw_shape(img, 0, 16, 14, 9, 2, 1.0)   # horizontal bar
    draw_shape(img, 1, 17, 21, 11, 1, 0.6)  # dim vertical bar
    return img


def fig_filters():
    img = demo_scene()
    filters = {
        "vertical edge\n(Sobel x)": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], float),
        "horizontal edge\n(Sobel y)": np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], float),
        "diagonal \"/\"": np.array([[-1, -1, 2], [-1, 2, -1], [2, -1, -1]], float),
        "blur\n(box average)": np.full((3, 3), 1 / 9),
        "sharpen": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], float),
    }
    n = len(filters)
    fig, axes = plt.subplots(2, n + 1, figsize=(17, 6.4),
                             gridspec_kw=dict(height_ratios=[1, 2.2]))
    axes[0, 0].axis("off")
    axes[1, 0].imshow(img, cmap="gray", vmin=0, vmax=1)
    axes[1, 0].set_title("input image (30x30)", fontsize=12)
    stats = {}
    for k, (name, Kf) in enumerate(filters.items(), start=1):
        a = axes[0, k]
        a.imshow(Kf, cmap="RdBu_r", vmin=-2.5, vmax=2.5)
        for (r, c), v in np.ndenumerate(Kf):
            a.text(c, r, f"{v:.2g}" if abs(v) < 1 and v != 0 else f"{v:g}", ha="center",
                   va="center", fontsize=11)
        a.set_title(name, fontsize=12)
        out = conv2d_naive(img[None, None], Kf[None, None], np.zeros(1), pad=1)[0, 0]
        m = np.abs(out).max()
        ax = axes[1, k]
        if name.startswith(("blur", "sharpen")):
            ax.imshow(out, cmap="gray", vmin=0, vmax=1)
        else:
            ax.imshow(out, cmap="RdBu_r", vmin=-m, vmax=m)
        ax.set_title(f"output: min {out.min():.1f}, max {out.max():.1f}", fontsize=10.5)
        stats[name.split("\n")[0]] = (out.min(), out.max())
    for a in axes.ravel():
        a.set_xticks([])
        a.set_yticks([])
        for sp in a.spines.values():
            sp.set_visible(False)
    fig.text(0.5, 0.0, "Top: hand-made 3x3 kernels (red = positive weight, blue = negative). "
             "Bottom: the image convolved with each kernel (padding 1).\nEdge kernels give red "
             "(+) on one side of a line and blue (−) on the other and stay ≈ 0 on flat areas. "
             "The diagonal kernel responds strongly only to the \"/\" line.",
             ha="center", va="top", fontsize=12)
    fig.suptitle("Kernels are pattern detectors: each output is large where the image "
                 "locally looks like the kernel", y=1.0)
    fig.tight_layout()
    save(fig, "filters.png")
    return stats


def fig_padding_stride():
    n, k = 5, 3
    cases = [(0, 1), (1, 1), (1, 2)]
    fig, axes = plt.subplots(1, 3, figsize=(16, 7.6))
    for ax, (p, s) in zip(axes, cases):
        o = out_size(n, k, s, p)
        check = conv2d_naive(np.zeros((1, 1, n, n)), np.zeros((1, 1, k, k)), np.zeros(1), s, p)
        assert check.shape[-1] == o
        N = n + 2 * p
        Nmax = n + 2
        xo, yo = (Nmax - N) / 2, -(Nmax - N) / 2  # centre smaller grids in the panel
        blank(ax, (-0.3, Nmax + 0.3), (-Nmax - 1.8 - n - 0.4, 2.0))
        for r in range(N):
            for c in range(N):
                is_pad = r < p or c < p or r >= N - p or c >= N - p
                ax.add_patch(Rectangle((xo + c, yo - r - 1), 1, 1,
                                       fc=C_PAD if is_pad else "white",
                                       ec="#999" if is_pad else "#333",
                                       ls="--" if is_pad else "-", lw=1))
                if is_pad:
                    ax.text(xo + c + 0.5, yo - r - 0.5, "0", ha="center", va="center",
                            color="#999", fontsize=10)
        outline(ax, xo, yo, k, k, color=C_KER, lw=3.5)
        outline(ax, xo + s, yo, k, k, color=C_OUT, lw=2.5, ls="--")
        ax.text(Nmax / 2, 1.3, f"p = {p}, s = {s}", ha="center", fontsize=15, weight="bold")
        ax.text(Nmax / 2, 0.4, "orange: 1st window, green dashed: 2nd window", ha="center",
                fontsize=9.5, color="#555")
        N = Nmax
        x0 = (N - o) / 2
        y0 = -N - 1.8
        for r in range(o):
            for c in range(o):
                ax.add_patch(Rectangle((x0 + c, y0 - r - 1), 1, 1, fc=C_OFILL, ec=C_OUT, lw=1.5))
        ax.add_patch(Rectangle((x0, y0 - 1), 1, 1, fc=C_KER, alpha=0.45, ec=C_OUT, lw=1.5))
        ax.add_patch(Rectangle((x0 + 1, y0 - 1), 1, 1, fc=C_OUT, alpha=0.45, ec=C_OUT, lw=1.5))
        ax.text(N / 2, y0 + 0.75, f"out = ({n} + 2·{p} − {k}) / {s} + 1 = {o}   →   {o}x{o}",
                ha="center", fontsize=12.5,
                bbox=dict(boxstyle="round", fc="#fffbe6", ec="#d4b106"))
    fig.suptitle(f"Padding and stride: a {n}x{n} input (white) and a {k}x{k} kernel.  "
                 "Grey dashed cells are zero padding.", y=0.97)
    fig.text(0.5, 0.06, "p = 0 shrinks the map by k − 1.   p = 1 with a 3x3 kernel keeps "
             "the size ('same' padding).   s = 2 jumps two cells per step and roughly halves "
             "the size.", ha="center", fontsize=12)
    save(fig, "padding_stride.png")


def fig_maxpool(model, Xval, yval):
    """A real 4x4 block of a real ReLU feature map, and the real gradient routing."""
    idx = 0
    x = Xval[idx:idx + 1]
    logits, acts = model.forward(x, keep=True)
    d = np.zeros_like(logits)
    d[0, yval[idx]] = 1.0  # gradient of the true class's score
    g = backward_trace(model, x, d)
    relu1, dpool = acts["relu1"][0], g["pool1"][0]
    dx_pool = model.layers[2][1].backward(g["pool1"])[0]  # what MaxPool2D.backward returns

    # The aligned 4x4 block with the most distinct non-zero values.
    best, where = -1, None
    for ch in range(relu1.shape[0]):
        for r in range(0, 16, 4):
            for c in range(0, 16, 4):
                blk = relu1[ch, r:r + 4, c:c + 4]
                wins = blk.reshape(2, 2, 2, 2).transpose(0, 2, 1, 3).reshape(4, 4)
                ok = all((w == w.max()).sum() == 1 and w.max() > 0 for w in wins)
                score = (blk > 0).sum() + 10 * ok
                if score > best:
                    best, where = score, (ch, r, c)
    ch, r, c = where
    A = relu1[ch, r:r + 4, c:c + 4]
    P = A.reshape(2, 2, 2, 2).max(axis=(1, 3))
    dP = dpool[ch, r // 2:r // 2 + 2, c // 2:c // 2 + 2]
    dA = dx_pool[ch, r:r + 4, c:c + 4]
    colors = ["#fdd0a2", "#c6dbef", "#c7e9c0", "#dadaeb"]
    hlA, hlP, hldA = {}, {}, {}
    for a in range(2):
        for b in range(2):
            col = colors[2 * a + b]
            for u in range(2):
                for v in range(2):
                    hlA[(2 * a + u, 2 * b + v)] = col
            w = A[2 * a:2 * a + 2, 2 * b:2 * b + 2]
            u, v = np.unravel_index(w.argmax(), w.shape)
            hldA[(2 * a + u, 2 * b + v)] = col
            hlP[(a, b)] = col

    fig, ax = plt.subplots(figsize=(15, 7.6))
    blank(ax, (-0.3, 23.6), (-10.0, 2.0))
    cell = 1.4
    draw_grid(ax, A, 0, 0, cell=cell, hl=hlA, fs=12, fmt=lambda v: f"{v:.2f}")
    for a in range(2):
        for b in range(2):
            w = A[2 * a:2 * a + 2, 2 * b:2 * b + 2]
            u, v = np.unravel_index(w.argmax(), w.shape)
            outline(ax, (2 * b + v) * cell, -(2 * a + u) * cell, 1, 1, cell, color="#d62728",
                    lw=3)
    ax.text(2.8, 1.0, "forward: a real 4x4 block of a\nReLU feature map (conv1, ch "
            f"{ch})", ha="center", fontsize=12, weight="bold")
    arrow(ax, (6.0, -2.8), (8.6, -2.8), lw=2.5)
    ax.text(7.3, -2.3, "max of each\n2x2 block", ha="center", va="bottom", fontsize=11)
    draw_grid(ax, P, 9.2, -1.4, cell=cell, hl=hlP, fs=12, fmt=lambda v: f"{v:.2f}")
    ax.text(10.6, 0.0, "pooled (2x2)", ha="center", fontsize=12, weight="bold")

    ax.text(18.2, 1.0, "backward: the gradient goes\nONLY to the winner", ha="center",
            fontsize=12, weight="bold")
    draw_grid(ax, dP, 13.4, -1.4, cell=cell, hl=hlP, fs=11, fmt=lambda v: f"{v:+.3f}")
    ax.text(14.8, 0.0, "dout (2x2)", ha="center", fontsize=12, weight="bold")
    arrow(ax, (16.5, -2.8), (17.3, -2.8), lw=2.5)
    draw_grid(ax, dA, 17.6, 0, cell=cell, hl=hldA, fs=11,
              fmt=lambda v: "0" if v == 0 else f"{v:+.3f}")
    ax.text(20.4, -6.3, "dX (4x4)", ha="center", fontsize=12, weight="bold")
    ax.text(11.6, -8.4,
            "Red boxes = the max of each block. Forward keeps only those 4 numbers. Backward "
            "sends each block's\ngradient to the cell that won and 0 to the other three "
            "(the losers did not affect the output).\nNumbers: a real validation image "
            f"({CLASSES[yval[idx]]}) through the trained CNN; dout = ∂(score of the true "
            "class)/∂(pool1 output).",
            ha="center", va="center", fontsize=12,
            bbox=dict(boxstyle="round", fc="#fffbe6", ec="#d4b106"))
    ax.set_title("Max pooling: keep the strongest response in each 2x2 block; route the "
                 "gradient back to the argmax", pad=12)
    save(fig, "maxpool.png")
    return A, P, dP


def fig_receptive_field():
    """Which input pixels can affect ONE unit at each layer? Computed by propagating a
    one-hot backwards through 3x3 'all-ones' convolutions and 2x2 pooling."""
    conv = Conv2D(1, 1, 3, pad=1)
    conv.params["W"][:] = 1.0

    def back_conv(m):  # every input in the 3x3 window around an active output becomes active
        conv.forward(np.zeros((1, 1) + m.shape))
        return conv.backward(m[None, None])[0, 0] > 0

    def back_pool(m):  # a pooled cell depends on all 4 cells of its 2x2 block
        return np.kron(m, np.ones((2, 2))) > 0

    stages = [("conv1", 16, [back_conv]),
              ("pool1", 8, [back_pool, back_conv]),
              ("conv2", 8, [back_conv, back_pool, back_conv]),
              ("pool2", 4, [back_pool, back_conv, back_pool, back_conv])]
    fig, axes = plt.subplots(2, 4, figsize=(16, 8.6), gridspec_kw=dict(height_ratios=[1, 1.6]))
    sizes = {}
    for k, (name, n, ops) in enumerate(stages):
        u = (n // 2 - 1, n // 2 - 1)  # a unit near the centre
        m = np.zeros((n, n))
        m[u] = 1
        for op in ops:
            m = back_op = op(m.astype(float))
        rf = back_op
        rows, cols = np.where(rf)
        h, w = rows.max() - rows.min() + 1, cols.max() - cols.min() + 1
        sizes[name] = (h, w)
        top = axes[0, k]
        grid_img = np.zeros((n, n))
        grid_img[u] = 1
        top.imshow(grid_img, cmap="Greens", vmin=0, vmax=1.3)
        top.set_xticks(np.arange(-0.5, n, 1), minor=True)
        top.set_yticks(np.arange(-0.5, n, 1), minor=True)
        top.grid(which="minor", color="#999", lw=0.5)
        top.tick_params(which="both", length=0, labelbottom=False, labelleft=False)
        top.set_title(f"after {name}: {n}x{n} map\none unit (green)", fontsize=12)
        bot = axes[1, k]
        bot.imshow(rf, cmap="Blues", vmin=0, vmax=1.4)
        bot.set_xticks(np.arange(-0.5, 16, 1), minor=True)
        bot.set_yticks(np.arange(-0.5, 16, 1), minor=True)
        bot.grid(which="minor", color="#bbb", lw=0.5)
        bot.tick_params(which="both", length=0, labelbottom=False, labelleft=False)
        bot.add_patch(Rectangle((cols.min() - 0.5, rows.min() - 0.5), w, h, fill=False,
                                ec="#d62728", lw=2.5))
        bot.set_title(f"sees {h}x{w} input pixels", fontsize=13, weight="bold", color="#d62728")
    axes[1, 0].set_ylabel("the 16x16 input image", fontsize=12)
    fig.suptitle("Receptive field: how much of the input one unit can see grows with depth "
                 "(3x3 conv: +2, 2x2 pool: x2)", y=0.99)
    fig.text(0.5, 0.02, "conv1 3x3 → pool1 4x4 → conv2 8x8 → pool2 10x10. Each pool2 unit "
             "sees most of the 16x16 image, enough to hold a whole 5-7 px shape.\n"
             "Stacking small kernels grows the view cheaply: two 3x3 convs see 5x5 with 18 "
             "weights instead of 25 (the VGG idea).", ha="center", fontsize=12)
    save(fig, "receptive_field.png")
    return sizes


def fig_im2col():
    Xs = EX_X[1:5, 0:4]  # a 4x4 crop of the terminal example
    Ks = np.stack([EX_K, EX_K.T])  # 2 filters: vertical and horizontal Sobel
    cols, Ho, Wo = im2col(Xs[None, None], 3)  # (4, 9)
    Wm = Ks.reshape(2, -1).T  # (9, 2): one COLUMN per filter
    out = cols @ Wm  # (4, 2)
    ref = conv2d_naive(Xs[None, None], Ks[:, None], np.zeros(2))[0]
    assert np.allclose(out.T.reshape(2, Ho, Wo), ref)
    colors = ["#fdd0a2", "#c6dbef", "#c7e9c0", "#dadaeb"]
    edge = ["#e6550d", "#3182bd", "#31a354", "#756bb1"]

    fig, ax = plt.subplots(figsize=(17, 7.6))
    blank(ax, (-0.7, 27.4), (-11.6, 2.6))
    cm = 0.55  # four small copies of the input, one per window
    for p in range(4):
        r, c = divmod(p, 2)
        gx, gy = -0.4 + c * 2.55, 0.3 - r * 2.9
        win = {(r + a, c + b): colors[p] for a in range(3) for b in range(3)}
        draw_grid(ax, Xs, gx, gy, cell=cm, hl=win, fs=8.5)
        outline(ax, gx + c * cm, gy - r * cm, 3, 3, cm, color=edge[p], lw=2.2)
        ax.text(gx + 2 * cm, gy - 4 * cm - 0.12, f"patch {p}", ha="center", va="top",
                fontsize=9.5, color=edge[p])
    ax.text(2.0, 1.3, "input 4x4: 4 windows", ha="center", fontsize=13, weight="bold")
    arrow(ax, (4.4, -3.0), (5.6, -3.0), lw=2.5)
    ax.text(5.0, -2.5, "im2col", ha="center", fontsize=11, weight="bold")

    hl = {(p, q): colors[p] for p in range(4) for q in range(9)}
    draw_grid(ax, cols, 6.0, -1.0, hl=hl, fs=11)
    ax.text(10.5, 0.5, "cols (4, 9): one ROW per patch", ha="center", fontsize=13, weight="bold")
    for q in range(9):
        ax.text(6.5 + q, -0.6, f"x{q // 3}{q % 3}", ha="center", fontsize=8.5, color="#666")
    ax.text(15.6, -3.0, "@", fontsize=26, ha="center", va="center")
    draw_grid(ax, Wm, 16.4, 1.5, fill=C_KFILL, fs=11)
    ax.text(17.4, 2.0, "W (9, 2)", ha="center", fontsize=13, weight="bold")
    ax.text(19.0, -3.0, "=", fontsize=26, ha="center", va="center")
    hl2 = {(p, q): colors[p] for p in range(4) for q in range(2)}
    draw_grid(ax, out, 19.8, -1.0, hl=hl2, fs=12)
    ax.text(20.8, 0.5, "(4, 2)", ha="center", fontsize=13, weight="bold")
    arrow(ax, (22.1, -3.0), (23.2, -3.0), lw=2.5)
    ax.text(22.65, -2.5, "reshape", ha="center", fontsize=11)
    for f, y0 in enumerate([0.3, -3.5]):
        hl3 = {divmod(p, 2): colors[p] for p in range(4)}
        draw_grid(ax, out[:, f].reshape(2, 2), 23.8, y0, hl=hl3, fs=12)
        ax.text(26.2, y0 - 1.0, ["vertical\nedges", "horizontal\nedges"][f], va="center",
                fontsize=11)
    ax.text(16.9, -7.7, "column 0 = vertical-edge kernel,\ncolumn 1 = horizontal-edge kernel",
            ha="center", va="top", fontsize=10.5, color="#555")
    ax.text(13.3, -10.4,
            "Convolution = (all patches as rows) @ (all filters as columns). Shapes for a real "
            "layer: (N·H_out·W_out, C_in·k·k) @ (C_in·k·k, C_out).\nOne big matrix multiply "
            "is what GPUs (and BLAS on your CPU) do best. The price: every pixel is copied up "
            "to k·k = 9 times.", ha="center", va="center", fontsize=12,
            bbox=dict(boxstyle="round", fc="#fffbe6", ec="#d4b106"))
    ax.set_title("im2col: turn convolution into ONE matrix multiplication", pad=8)
    save(fig, "im2col.png")
    return cols, out


def fig_learned(model, Xval, yval):
    W1 = model.layers[0][1].params["W"][:, 0]  # (8, 3, 3)
    idx = int(np.where(yval == 4)[0][0])  # first 'plus' in the validation set
    x = Xval[idx:idx + 1]
    _, acts = model.forward(x, keep=True)
    m1, m2 = acts["relu1"][0], acts["relu2"][0]
    p = softmax(acts["fc"])[0]

    fig = plt.figure(figsize=(16, 10.2))
    gs = fig.add_gridspec(4, 9, height_ratios=[1, 1.15, 1, 1], hspace=0.45, wspace=0.15)
    m = np.abs(W1).max()
    for f in range(8):
        a = fig.add_subplot(gs[0, f + 1])
        a.imshow(W1[f], cmap="RdBu_r", vmin=-m, vmax=m)
        for (r, c), v in np.ndenumerate(W1[f]):
            a.text(c, r, f"{v:.1f}", ha="center", va="center", fontsize=8.5)
        a.set_title(f"filter {f}", fontsize=10.5)
        a.set_xticks([])
        a.set_yticks([])
    t = fig.add_subplot(gs[0, 0])
    t.axis("off")
    t.text(0.5, 0.5, "conv1 kernels\n(learned 3x3,\nred +, blue −)", ha="center", va="center",
           fontsize=11.5, weight="bold", transform=t.transAxes)

    a = fig.add_subplot(gs[1, 0])
    a.imshow(x[0, 0], cmap="gray", vmin=0, vmax=1)
    a.set_title(f"input ({CLASSES[yval[idx]]})", fontsize=11)
    a.set_xticks([])
    a.set_yticks([])
    v1 = m1.max()
    for f in range(8):
        a = fig.add_subplot(gs[1, f + 1])
        a.imshow(m1[f], cmap="magma", vmin=0, vmax=v1)
        a.set_title(f"conv1 map {f}", fontsize=10)
        a.set_xticks([])
        a.set_yticks([])
    v2 = m2.max()
    for f in range(16):
        a = fig.add_subplot(gs[2 + f // 8, f % 8 + 1])
        a.imshow(m2[f], cmap="magma", vmin=0, vmax=v2)
        a.set_title(f"conv2 map {f}", fontsize=9.5)
        a.set_xticks([])
        a.set_yticks([])
    t = fig.add_subplot(gs[2:, 0])
    t.axis("off")
    t.text(0.5, 0.5, f"conv2 feature\nmaps (8x8)\nafter ReLU\n\nmodel says:\n"
           f"{CLASSES[p.argmax()]} {p.max():.1%}", ha="center", va="center", fontsize=11.5,
           transform=t.transAxes)
    fig.suptitle("Inside the trained CNN: the 8 learned first-layer kernels, and what every "
                 "channel sees in one real image", y=0.955)
    fig.text(0.5, 0.045, "Row 2: conv1 maps are full-size (16x16) and simple: each lights up "
             "where its kernel's small pattern (a bar piece, an edge) appears.\n"
             "Rows 3-4: conv2 maps are coarser (8x8) and combine conv1 channels, so they respond "
             "to bigger parts like crossings or corners. Black = 0 after ReLU.",
             ha="center", va="top", fontsize=12)
    save(fig, "learned_filters.png")
    return idx, p


def fig_training(hist):
    ep = [h["epoch"] for h in hist]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.4))
    a1.plot(ep, [h["train_loss"] for h in hist], "o-", color=C_IMG, lw=2, label="train")
    a1.plot(ep, [h["val_loss"] for h in hist], "s-", color=C_KER, lw=2, label="validation")
    a1.axhline(np.log(len(CLASSES)), color="#d62728", ls="--", lw=1.5)
    a1.text(ep[-1], np.log(len(CLASSES)) * 1.15, f"random guessing: ln(6) = "
            f"{np.log(len(CLASSES)):.2f}", color="#d62728", ha="right", fontsize=11)
    a1.set_yscale("log")
    a1.set_xlabel("epoch")
    a1.set_ylabel("cross-entropy loss (log scale)")
    a1.set_title("loss")
    a1.legend()
    a2.plot(ep, [100 * h["train_acc"] for h in hist], "o-", color=C_IMG, lw=2, label="train")
    a2.plot(ep, [100 * h["val_acc"] for h in hist], "s-", color=C_KER, lw=2, label="validation")
    a2.axhline(100 / len(CLASSES), color="#d62728", ls="--", lw=1.5)
    a2.text(ep[-1], 100 / len(CLASSES) + 2, "chance = 16.7%", color="#d62728", ha="right",
            fontsize=11)
    a2.set_ylim(0, 105)
    a2.set_xlabel("epoch")
    a2.set_ylabel("accuracy (%)")
    a2.set_title("accuracy")
    a2.legend(loc="center right")
    last = hist[-1]
    a2.text(ep[-1], 80, f"final: train {last['train_acc']:.1%}\nval {last['val_acc']:.1%}",
            ha="right", fontsize=11, bbox=dict(boxstyle="round", fc="white", ec="#bbb"))
    for a in (a1, a2):
        a.set_xticks(ep)
    fig.suptitle("Training the NumPy CNN (2,790 parameters, SGD + momentum, 3000 images, "
                 "batch 32): what cnn_numpy.py prints", y=1.0)
    fig.text(0.5, -0.03, "Train numbers are averaged over each epoch while the weights move; "
             "validation is measured at the end of the epoch, so val can beat train early on.",
             ha="center", fontsize=11.5)
    fig.tight_layout()
    save(fig, "training_curves.png")


def fig_cnn_vs_mlp(r1, r2, moves):
    fig = plt.figure(figsize=(16, 6.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 0.8, 1.5], wspace=0.35)
    a = fig.add_subplot(gs[0])
    names = ["MLP", "CNN\n(flatten)", "CNN\n(gap)"]
    params = [r2["MLP"][0], r2["CNN (flatten)"][0], r2["CNN (gap)"][0]]
    bars = a.bar(names, params, color=[C_MLP, C_CNN, C_GAP])
    for b_, v in zip(bars, params):
        a.text(b_.get_x() + b_.get_width() / 2, v * 1.02, f"{v:,}", ha="center", fontsize=11)
    a.set_ylabel("number of parameters")
    a.set_title("model size")
    a.set_ylim(0, max(params) * 1.15)

    a = fig.add_subplot(gs[1])
    accs = [100 * r1["MLP"][2], 100 * r1["CNN"][2]]
    bars = a.bar(["MLP", "CNN"], accs, color=[C_MLP, C_CNN])
    for b_, v in zip(bars, accs):
        a.text(b_.get_x() + b_.get_width() / 2, v + 1, f"{v:.1f}%", ha="center", fontsize=11)
    a.set_ylim(0, 110)
    a.set_ylabel("validation accuracy (%)")
    a.set_title("shapes anywhere\n(same data and budget)")

    a = fig.add_subplot(gs[2])
    keys = ["MLP", "CNN (flatten)", "CNN (gap)"]
    xs = np.arange(3)
    left = [100 * r2[k][1] for k in keys]
    right = [100 * r2[k][2] for k in keys]
    b1 = a.bar(xs - 0.2, left, 0.38, color="#9ecae1", ec="#3182bd", label="left half (seen)")
    b2 = a.bar(xs + 0.2, right, 0.38, color=[C_MLP, C_CNN, C_GAP], ec="black",
               label="right half (never seen)")
    for bb in (b1, b2):
        for b_ in bb:
            h = b_.get_height()
            a.text(b_.get_x() + b_.get_width() / 2, h + 1, f"{h:.0f}%", ha="center", fontsize=11)
    a.axhline(100 / 6, color="#d62728", ls="--", lw=1.5)
    a.text(0.5, 100 / 6 + 2, "chance", color="#d62728", ha="center", fontsize=10)
    a.set_xticks(xs)
    a.set_xticklabels(["MLP", "CNN (flatten)", "CNN (gap)"])
    a.set_ylim(0, 118)
    a.set_ylabel("accuracy (%)")
    a.set_title("train on LEFT-half shapes, test on RIGHT half")
    a.legend(loc="upper left", fontsize=10, ncol=2)
    fig.suptitle("Weight sharing in action: fewer parameters, better accuracy, and (with global "
                 "pooling) generalization to new positions", y=1.02)
    fig.text(0.5, -0.06,
             "MLP: right-half pixels never carried a shape in training; their first-layer "
             f"weights moved {moves[1]:.2f} on average vs {moves[0]:.2f} for left-half pixels. "
             "CNN (flatten): the convs\nstill detect the shape on the right, but the dense layer "
             "has a separate weight per position. CNN (gap): global average pooling forgets "
             "WHERE, keeps WHAT.", ha="center", fontsize=11.5)
    save(fig, "cnn_vs_mlp.png")


def fig_equivariance(cnn_flat, cnn_gap):
    """Same shape, moved 8 pixels right. Both models were trained on LEFT-half shapes only."""
    cls, row, s = 1, 4, 6  # a vertical bar, clean (no noise) so the shift is exact
    A = draw_shape(np.zeros((16, 16)), cls, row, 1, s, 1, 1.0)
    Bimg = draw_shape(np.zeros((16, 16)), cls, row, 9, s, 1, 1.0)
    xs = np.stack([A, Bimg])[:, None]
    _, acts_g = cnn_gap.forward(xs, keep=True)
    pf = softmax(cnn_flat.forward(xs))
    pg = softmax(acts_g["fc"])
    f1, f2 = acts_g["relu1"], acts_g["relu2"]
    ch1 = int(f1[0].reshape(f1.shape[1], -1).max(axis=1).argmax())
    ch2 = int(f2[0].reshape(f2.shape[1], -1).max(axis=1).argmax())
    # Equivariance: B's map == A's map shifted right (8 px at conv1, 4 cells at conv2).
    err1 = np.abs(f1[1][:, :, 8:] - f1[0][:, :, :8]).max()
    # conv2 sees zero PADDING at the image border but a non-zero background activation
    # (relu(bias)) elsewhere, so compare only columns whose 3x3 window avoids the border.
    err2_in = np.abs(f2[1][:, :, 5:7] - f2[0][:, :, 1:3]).max()
    err2_all = np.abs(f2[1][:, :, 4:] - f2[0][:, :, :4]).max()
    g = acts_g["gap"]
    gap_rel = np.abs(g[0] - g[1]).max() / np.abs(g[0]).max()

    fig, axes = plt.subplots(2, 5, figsize=(17, 7.4),
                             gridspec_kw=dict(width_ratios=[1, 1, 1, 1.25, 1.25]))
    for r, label in enumerate(["A: on the left\n(like training)",
                               "B: moved 8 px right\n(never seen)"]):
        axes[r, 0].imshow(xs[r, 0], cmap="gray", vmin=0, vmax=1)
        axes[r, 0].set_ylabel(label, fontsize=12)
        axes[r, 1].imshow(f1[r, ch1], cmap="magma", vmin=0, vmax=f1[:, ch1].max())
        axes[r, 2].imshow(f2[r, ch2], cmap="magma", vmin=0, vmax=f2[:, ch2].max())
        for c, (p, name) in enumerate([(pf[r], "CNN (flatten)"), (pg[r], "CNN (gap)")]):
            a = axes[r, 3 + c]
            k = int(p.argmax())
            colors = ["#bbb"] * 6
            colors[k] = C_GAP if k == cls else "#d62728"
            a.barh(range(6)[::-1], p, color=colors)
            a.set_yticks(range(6)[::-1])
            a.set_yticklabels(CLASSES, fontsize=10)
            a.set_xlim(0, 1.18)
            a.text(p[k] + 0.02, 5 - k, f"{p[k]:.0%}", va="center", fontsize=10.5)
            if r == 0:
                a.set_title(f"{name}\nprediction", fontsize=12)
        for a in axes[r, :3]:
            a.set_xticks([])
            a.set_yticks([])
    axes[0, 0].set_title(f"input (true: {CLASSES[cls]})", fontsize=12)
    axes[0, 1].set_title(f"conv1 map, channel {ch1}\n(16x16)", fontsize=12)
    axes[0, 2].set_title(f"conv2 map, channel {ch2}\n(8x8)", fontsize=12)
    fig.suptitle("Equivariance vs invariance: move the input, and the feature maps move with it",
                 y=1.0)
    fig.text(0.5, -0.01,
             f"Feature maps of B = feature maps of A shifted right: max difference {err1:.0f} "
             f"at conv1 and {err2_in:.0f} at conv2 away from the border → convolution is "
             "translation-EQUIVARIANT.\nFlatten + dense reads each position with its own "
             "weights, so the moved bar looks like something else (red = wrong).\n"
             f"Global average pooling gives almost the same vector for A and B (differs by "
             f"{gap_rel:.0%}, only from zero padding at the border), so the gap model is "
             "translation-INVARIANT.\nBoth models were trained on left-half shapes only "
             "(cnn_vs_mlp.py, part 2).", ha="center", va="top", fontsize=11.5)
    fig.tight_layout()
    save(fig, "equivariance.png")
    return err1, err2_in, err2_all, gap_rel, pf, pg


def fig_confusion(model, Xval, yval):
    p = model.predict_proba(Xval)
    pred = p.argmax(1)
    C = len(CLASSES)
    cm = np.zeros((C, C), int)
    for t, q in zip(yval, pred):
        cm[t, q] += 1
    wrong = np.where(pred != yval)[0]
    ncol = max(2, (min(len(wrong), 10) + 1) // 2)
    fig = plt.figure(figsize=(7.5 + 2.6 * ncol, 6.6))
    gs = fig.add_gridspec(2, 2 + ncol, width_ratios=[3.6, 0.5] + [1] * ncol, wspace=0.25,
                          hspace=0.5)
    a = fig.add_subplot(gs[:, 0])
    a.imshow(cm, cmap="Blues")
    for (r, c), v in np.ndenumerate(cm):
        a.text(c, r, str(v), ha="center", va="center", fontsize=12,
               color="white" if v > cm.max() / 2 else ("#d62728" if v and r != c else "black"))
    a.set_xticks(range(C))
    a.set_xticklabels(CLASSES, rotation=30, ha="right")
    a.set_yticks(range(C))
    a.set_yticklabels(CLASSES)
    a.set_xlabel("predicted")
    a.set_ylabel("true")
    a.set_title(f"confusion matrix: {len(yval)} validation images,\n"
                f"{(pred == yval).mean():.1%} correct", fontsize=13)
    for k, i in enumerate(wrong[:10]):
        ax = fig.add_subplot(gs[k % 2, 2 + k // 2])
        ax.imshow(Xval[i, 0], cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"true: {CLASSES[yval[i]]}\npred: {CLASSES[pred[i]]} {p[i].max():.0%}",
                     fontsize=10.5, color="#d62728")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Where the trained CNN goes wrong (right: all of its validation mistakes)",
                 y=1.0)
    save(fig, "confusion.png")
    return cm, wrong


def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    print("training the CNN (same settings as cnn_numpy.py, ~3 s)...")
    model, hist, (Xtr, ytr, Xval, yval) = train_main()
    print(f"  final: train acc {hist[-1]['train_acc']:.1%}, val loss {hist[-1]['val_loss']:.3f}, "
          f"val acc {hist[-1]['val_acc']:.1%}")
    print("running the cnn_vs_mlp.py experiments (~8 s)...")
    r1 = part1(15, 128, verbose=False)
    r2, moves, (cnn_flat, cnn_gap, _, _) = part2(15, 128, verbose=False)
    print(f"  part 1 val acc: MLP {r1['MLP'][2]:.1%}, CNN {r1['CNN'][2]:.1%}")
    print("  part 2 left/right: " + ", ".join(f"{k} {v[1]:.1%}/{v[2]:.1%}" for k, v in r2.items()))

    out = fig_conv_operation()
    print(f"  conv_operation: output map {out.tolist()}")
    stats = fig_filters()
    print("  filters: " + ", ".join(f"{k} [{a:.1f}, {b:.1f}]" for k, (a, b) in stats.items()))
    fig_padding_stride()
    A, P, dP = fig_maxpool(model, Xval, yval)
    print(f"  maxpool: block max {P.ravel().round(2).tolist()}, dout {dP.ravel().round(3).tolist()}")
    sizes = fig_receptive_field()
    print(f"  receptive fields: {sizes}")
    fig_im2col()
    idx, p = fig_learned(model, Xval, yval)
    print(f"  learned_filters: val image {idx} -> {CLASSES[p.argmax()]} {p.max():.1%}")
    fig_training(hist)
    fig_cnn_vs_mlp(r1, r2, moves)
    e1, e2, e2all, gd, pf, pg = fig_equivariance(cnn_flat, cnn_gap)
    print(f"  equivariance: conv1 err {e1:.1e}, conv2 err {e2:.1e} (incl. border {e2all:.2f}), "
          f"gap rel diff {gd:.1%}; "
          f"flatten says {CLASSES[pf[0].argmax()]} {pf[0].max():.0%} / "
          f"{CLASSES[pf[1].argmax()]} {pf[1].max():.0%}, gap says "
          f"{CLASSES[pg[0].argmax()]} {pg[0].max():.0%} / {CLASSES[pg[1].argmax()]} "
          f"{pg[1].max():.0%}")
    cm, wrong = fig_confusion(model, Xval, yval)
    pred = model.predict_proba(Xval).argmax(1)
    print(f"  confusion: {len(wrong)} mistakes: " + ", ".join(
        f"{CLASSES[yval[i]]} → {CLASSES[pred[i]]}" for i in wrong))
    print(f"total time {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
