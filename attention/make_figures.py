"""
Draw the pictures used in attention/README.md and save them to attention/images/.

Every number in every figure is computed: hand-made example vectors go through the
real attention code (attention_numpy.py), the NumPy key-value model is trained here
(same settings and seed as attention_numpy.py), and the PyTorch models are the ones
saved by seq2seq_pytorch.py (date_model.pt) and bottleneck.py (bottleneck_*.npz/.pt).
If those files don't exist yet, they are trained first with the same settings.

Run (~3 s if the models exist, ~45 s more if they have to be trained):
    python attention/make_figures.py
    python attention/make_figures.py --retrain      (retrain everything from scratch)
"""
import argparse
import datetime as dt
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

import attention_numpy as an  # noqa: E402  (also silences the spurious matmul warnings)
import bottleneck as bn  # noqa: E402
import seq2seq_pytorch as sp  # noqa: E402
from attention_visual import (K, PAD_KEY, PAD_VALUE, QUERIES, TOKENS, V,  # noqa: E402
                              VALUE_FEATURES)
from tasks import DIGIT_VOCAB, format_date, sample_reverse  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "images")
DPI = 130

# One colour per role, used in every figure.
C_Q, C_K, C_V, C_A = "#1f77b4", "#ff7f0e", "#9467bd", "#2ca02c"
F_Q, F_K, F_V, F_A = "#dbe9f6", "#ffe8cc", "#ece2f4", "#e3f4e1"
C_NONE, C_ATT = "#d62728", "#1f77b4"  # without / with attention
C_ENC, C_DEC = "#fdf1c7", "#e3f4e1"

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


def vec(v, fmt="{:g}"):
    return "[" + "  ".join(fmt.format(x) for x in v) + "]"


def show(c):
    return "␣" if c == " " else c


def matrix(ax, M, x0, y0, cell=0.75, fmt="{:.2f}", fc="white", ec="black", shade=None,
           rows=None, cols=None, title=None, shape=True, fs=11, title_color="black"):
    """Draw matrix M with its top-left corner at (x0, y0). shade: colormap name for the cells."""
    n, m = M.shape
    cmap = plt.get_cmap(shade) if shade else None
    for i in range(n):
        for j in range(m):
            c = cmap(0.08 + 0.75 * M[i, j]) if cmap else fc
            ax.add_patch(Rectangle((x0 + j * cell, y0 - (i + 1) * cell), cell, cell, fc=c,
                                   ec=ec, lw=1.0))
            txt_c = "white" if cmap and M[i, j] > 0.6 else "black"
            ax.text(x0 + (j + 0.5) * cell, y0 - (i + 0.5) * cell, fmt.format(M[i, j]),
                    ha="center", va="center", fontsize=fs, color=txt_c)
    if rows:
        for i, r in enumerate(rows):
            ax.text(x0 - 0.12, y0 - (i + 0.5) * cell, r, ha="right", va="center", fontsize=10.5)
    if cols:
        for j, c in enumerate(cols):
            ax.text(x0 + (j + 0.5) * cell, y0 + 0.1, c, ha="center", va="bottom", fontsize=10.5,
                    rotation=0)
    if title:
        ax.text(x0 + m * cell / 2, y0 + (0.55 if cols else 0.2), title, ha="center",
                va="bottom", fontsize=13, weight="bold", color=title_color)
    if shape:
        ax.text(x0 + m * cell / 2, y0 - n * cell - 0.15, f"shape ({n}, {m})", ha="center",
                va="top", fontsize=10.5, color="#555")


# ---------------------------------------------------------------- models / data

def get_date_model(retrain):
    if retrain and os.path.exists(sp.DATE_MODEL_PATH):
        os.remove(sp.DATE_MODEL_PATH)
    return sp.load_or_train_date_model()


def get_bottleneck(retrain):
    if retrain or not (os.path.exists(bn.RESULTS_PATH) and os.path.exists(bn.MODELS_PATH)):
        print("running the bottleneck experiment (same settings as bottleneck.py, ~40 s)...")
        r, models, _ = bn.run(verbose=False)
        bn.save(r, models)
    r = dict(np.load(bn.RESULTS_PATH))
    ckpt = torch.load(bn.MODELS_PATH, map_location="cpu")
    cfg = ckpt["config"]
    models = {}
    for name, kind in [("none", "none"), ("attn", "dot")]:
        m = sp.Seq2Seq(len(DIGIT_VOCAB), cfg["emb"], cfg["hidden"], kind)
        m.load_state_dict(ckpt[name])
        m.eval()
        models[name] = m
    return r, models


@torch.no_grad()
def run_reverse(model, s):
    src, src_len, _, _ = sp.tensors([(s, s[::-1])], DIGIT_VOCAB)
    ids, alpha = model.greedy(src, src_len, 41)
    out = DIGIT_VOCAB.decode(ids[0].tolist())
    _, c, _ = model.encode(src, src_len)
    return out, (None if alpha is None else alpha[0, :len(out) + 1].numpy()), c[0].numpy()


# ---------------------------------------------------------------- figures

def fig_bottleneck_diagram(r, models):
    s = sample_reverse(1, np.random.default_rng(11), length=10)[0][0]
    target = s[::-1]
    out, _, c = run_reverse(models["none"], s)
    n = len(s)

    fig, ax = plt.subplots(figsize=(16, 7.6))
    ax.set_xlim(-0.6, 21.2)
    ax.set_ylim(-2.35, 6.4)
    ax.axis("off")

    # Encoder
    ex = [0.5 + 0.8 * i for i in range(n)]
    for i, (x, ch) in enumerate(zip(ex, s)):
        box(ax, x, 0.4, ch, fc="#dbe9f6", w=0.6, h=0.6, fs=13)
        box(ax, x, 2.2, f"$h_{{{i + 1}}}$", fc=C_ENC, w=0.62, h=0.75, fs=11)
        arrow(ax, (x, 0.72), (x, 1.8), lw=1.3, ms=10)
        if i:
            arrow(ax, (ex[i - 1] + 0.31, 2.2), (x - 0.31, 2.2), lw=1.3, ms=10)
    ax.text(ex[0] - 0.4, 0.4, "input", ha="right", va="center", fontsize=11, color="#444")
    ax.text((ex[0] + ex[-1]) / 2, 3.1, "ENCODER (bidirectional GRU) reads the input", ha="center",
            fontsize=13, weight="bold")

    # The bottleneck vector c, drawn with its real 128 values
    cx0, cx1 = 8.7, 10.3
    ins = ax.inset_axes([cx0, 0.9, cx1 - cx0, 3.7], transform=ax.transData)
    ins.imshow(c.reshape(-1, 8), cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ins.set_xticks([])
    ins.set_yticks([])
    for sp_ in ins.spines.values():
        sp_.set_linewidth(3)
        sp_.set_color(C_NONE)
    arrow(ax, (ex[-1] + 0.31, 2.2), (cx0 - 0.05, 2.2), color=C_NONE, lw=3)
    ax.text((cx0 + cx1) / 2, 4.75, f"c: {c.size} numbers\n(real values)", ha="center",
            va="bottom", fontsize=12, color=C_NONE, weight="bold")
    ax.text((cx0 + cx1) / 2, 0.75, "the ONLY thing\nthe decoder\never sees", ha="center",
            va="top", fontsize=11, color=C_NONE)

    # Decoder
    dx = [11.4 + 0.86 * t for t in range(n + 1)]
    dec_in = (["<s>"] + list(out))[:n + 1]  # greedy decoding: the model's own previous output
    dec_in += [""] * (n + 1 - len(dec_in))
    dec_out = list(out[:n]) + ["<e>"]
    dec_out += [""] * (n + 1 - len(dec_out))
    wrong = 0
    for t, x in enumerate(dx):
        box(ax, x, 0.4, dec_in[t], fc="#f2f2f2", w=0.68, h=0.6, fs=11 if t else 9.5)
        box(ax, x, 2.2, f"$s_{{{t + 1}}}$", fc=C_DEC, w=0.66, h=0.75, fs=11)
        arrow(ax, (x, 0.72), (x, 1.8), lw=1.3, ms=10)
        if t:
            arrow(ax, (dx[t - 1] + 0.33, 2.2), (x - 0.33, 2.2), lw=1.3, ms=10)
        want = target[t] if t < n else "<e>"
        got = dec_out[t]
        ok = got == want
        wrong += (not ok) and t < n
        box(ax, x, 4.1, got if got else "∅", fc="#e3f4e1" if ok else "#fde0dd",
            ec=C_A if ok else C_NONE, w=0.68, h=0.6, fs=11 if t < n else 9.5)
        ax.text(x, 4.65, want, ha="center", va="bottom", fontsize=10, color="#777")
        arrow(ax, (x, 2.58), (x, 3.78), lw=1.3, ms=10)
        ax.plot([x], [3.2], "o", color=C_NONE, ms=5, zorder=4)  # c joins this step's output
    # c is fed into every step's output layer: one dashed "bus" carrying the SAME vector
    ax.plot([cx1 + 0.05, dx[-1]], [3.2, 3.2], color=C_NONE, lw=1.6, ls=(0, (3, 3)), zorder=0)
    arrow(ax, (cx1 + 0.05, 2.2), (dx[0] - 0.33, 2.2), color=C_NONE, lw=3)
    ax.text((dx[0] + dx[-1]) / 2, -0.05, "decoder input = its own previous output "
            "(greedy decoding; <s> = start)", ha="center", va="top", fontsize=10.5, color="#444")
    ax.text(dx[-1] + 0.5, 4.1, "output", ha="left", va="center", fontsize=11, color="#444")
    ax.text(dx[-1] + 0.5, 4.75, "target", ha="left", va="center", fontsize=10, color="#777")
    ax.text((dx[0] + dx[-1]) / 2, 6.05, "DECODER writes the reversed digits", ha="center",
            fontsize=13, weight="bold")
    ax.text((dx[0] + dx[-1]) / 2, 5.35, "c sets the start state s₀ and joins every step's "
            "output (red dots):\nthe SAME vector at every step", ha="center", va="center",
            fontsize=11, color=C_NONE)

    head = "".join(f"{L:>6d}" for L in r["lengths"])
    row_n = "".join(f"{a:>6.0%}" for a in r["none_exact"])
    row_a = "".join(f"{a:>6.0%}" for a in r["attn_exact"])
    ax.text(10.3, -0.75,
            f"Real trained model, NO attention, reversing a length-{n} input: {wrong} of {n} "
            "digits wrong (red).\n\n"
            f"exact match by input length {head}\n"
            f"   without attention        {row_n}\n"
            f"   same model WITH attention{row_a}",
            ha="center", va="top", fontsize=11.5, family="monospace",
            bbox=dict(boxstyle="round", fc="#fffbe6", ec="#d4b106"))
    ax.set_title("Seq2seq WITHOUT attention: the whole input must squeeze through ONE "
                 "fixed-size vector c", fontsize=15)
    save(fig, "bottleneck_diagram.png")
    return s, out, wrong


def fig_attention_mechanism():
    q = np.array(QUERIES["write the MONTH"])
    d = len(q)
    raw = K @ q
    scaled = raw / np.sqrt(d)
    out, A, _ = an.scaled_dot_product_attention(q[None], K, V)
    w, ctx = A[0], out[0]

    fig, ax = plt.subplots(figsize=(16, 8))
    ax.set_xlim(0, 21.7)
    ax.set_ylim(0.9, 10.0)
    ax.axis("off")

    box(ax, 3.5, 9.2, f"query  q = {vec(q)}\n\"I need a MONTH\"", fc=F_Q, ec=C_Q, w=5.2, h=1.2,
        fs=13, lw=2.5)
    ax.text(12.6, 9.25, "score$_j$ = q · k$_j$   →   ÷ √d$_k$   →   softmax   →   "
            "context = Σ$_j$ α$_j$ v$_j$", ha="center", va="center", fontsize=15)

    cols = {"tok": 0.9, "key": 3.5, "raw": 6.4, "scaled": 7.9, "bar": 9.0, "val": 15.9,
            "wv": 19.6}
    hy = 7.7
    heads = [("tok", "input\ntoken"), ("key", "key k$_j$\n(wkd?, day?, mon?, year?)"),
             ("raw", "q · k$_j$"), ("scaled", f"÷ √{d}"),
             ("val", "value v$_j$\n(" + ", ".join(VALUE_FEATURES) + ")"),
             ("wv", "α$_j$ · v$_j$")]
    for k, h in heads:
        ax.text(cols[k], hy, h, ha="center", va="bottom", fontsize=10.5, weight="bold")
    ax.text(cols["bar"] + 1.6, hy, "weight α$_j$ = softmax", ha="center", va="bottom",
            fontsize=11.5, weight="bold", color=C_A)

    ys = [6.8, 5.6, 4.4, 3.2]
    best = int(np.argmax(w))
    for j, (t, y) in enumerate(zip(TOKENS, ys)):
        hl = j == best
        ax.text(cols["tok"], y, t, ha="center", va="center", fontsize=14,
                weight="bold" if hl else "normal")
        box(ax, cols["key"], y, vec(K[j]), fc=F_K, ec=C_K, w=3.9, h=0.75, fs=12)
        ax.text(cols["raw"], y, f"{raw[j]:g}", ha="center", va="center", fontsize=14,
                color=C_K, weight="bold" if hl else "normal")
        ax.text(cols["scaled"], y, f"{scaled[j]:g}", ha="center", va="center", fontsize=14)
        ax.add_patch(Rectangle((cols["bar"], y - 0.28), 3.2 * w[j], 0.56, fc=C_A, ec="none"))
        ax.add_patch(Rectangle((cols["bar"], y - 0.28), 3.2, 0.56, fc="none", ec="#bbb"))
        ax.text(cols["bar"] + 3.35, y, f"{w[j]:.3f}", ha="left", va="center", fontsize=13,
                color=C_A, weight="bold" if hl else "normal")
        box(ax, cols["val"], y, vec(V[j]), fc=F_V, ec=C_V, w=3.2, h=0.75, fs=12)
        ax.text(cols["wv"], y, vec(w[j] * V[j], "{:.2f}"), ha="center", va="center",
                fontsize=11.5)
    ax.plot([cols["wv"] - 2.0, cols["wv"] + 2.0], [2.55, 2.55], color="black", lw=1.5)
    ax.text(cols["wv"] - 2.2, 2.2, "Σ", ha="right", va="center", fontsize=16)
    box(ax, cols["wv"], 1.95, f"context\n{vec(ctx, '{:.2f}')}", fc=F_A, ec=C_A, w=4.0, h=1.05,
        fs=12.5, lw=2.5)
    ax.text(0.4, 1.6,
            f"exp and normalise: e^{scaled[best]:g} = {np.exp(scaled[best]):.1f} out of a total "
            f"{np.exp(scaled).sum():.1f}  →  α = {w[best]:.3f}\n"
            f"The context is MOSTLY march's value {vec(V[best])}: a soft dictionary lookup.\n"
            f"Every key gets SOME weight (never exactly 0), so the whole thing is "
            "differentiable.", ha="left", va="center", fontsize=12,
            bbox=dict(boxstyle="round", fc="#fffbe6", ec="#d4b106"))
    ax.set_title("Attention with real numbers: one query, four keys and values "
                 f"(d$_k$ = {d}, hand-made readable features)", fontsize=15)
    save(fig, "attention_mechanism.png")
    return w, ctx


def fig_date_alignments(model):
    date = dt.date(1987, 11, 24)
    inputs = [format_date(date, f) for f in (3, 2, 1)]
    res = [sp.translate_date(model, s) for s in inputs]
    widths = [len(s) for s in inputs]

    fig = plt.figure(figsize=(17, 6.6))
    gs = fig.add_gridspec(1, 3, width_ratios=widths, wspace=0.12)
    for k, (s, (out, alpha)) in enumerate(zip(inputs, res)):
        ax = fig.add_subplot(gs[k])
        im = ax.imshow(alpha, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(s)))
        ax.set_xticklabels([show(c) for c in s], fontsize=12, family="monospace")
        rows = list(out) + ["<eos>"]
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels(rows, fontsize=12, family="monospace")
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(f"\"{s}\"\n→ {out}", fontsize=13)
        if k == 0:
            ax.set_ylabel("output char (written top → bottom)")
        ax.set_xlabel("input char")
    cb = fig.colorbar(im, ax=fig.axes, fraction=0.015, pad=0.02)
    cb.set_label("attention weight α (each row sums to 1)")
    fig.suptitle("Real alignments from the trained date model: the same date in three formats",
                 y=1.04)
    fig.text(0.45, -0.07,
             "Each ROW is one decoder step: where it looked while writing that output char. "
             "Year digits look at the year, month digits at\nthe month (\"11\" or the letters "
             "of \"nov\"), day digits at the day. Nobody told the model about alignment: it "
             "learned it from (input, output) pairs.", ha="center", fontsize=12)
    save(fig, "date_alignments.png")
    return inputs, [o for o, _ in res]


def fig_accuracy_vs_length(r, models):
    L = r["lengths"]
    s = sample_reverse(1, np.random.default_rng(4), length=20)[0][0]
    out, alpha, _ = run_reverse(models["attn"], s)

    fig = plt.figure(figsize=(17, 5.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.95], wspace=0.28)
    for k, (key, title) in enumerate([("exact", "exact match (ALL digits right)"),
                                      ("char", "per-digit accuracy")]):
        ax = fig.add_subplot(gs[k])
        ax.plot(L, r[f"attn_{key}"], "o-", color=C_ATT, lw=2.8, ms=8, label="with attention")
        ax.plot(L, r[f"none_{key}"], "s-", color=C_NONE, lw=2.8, ms=8,
                label="without attention")
        if key == "char":
            ax.axhline(0.1, color="grey", ls=":", lw=1.5)
            ax.text(L[-1], 0.12, "random guessing (10%)", ha="right", fontsize=10.5,
                    color="grey")
        for x, y in zip(L, r[f"none_{key}"]):
            if x in (5, 10, 15, 40):
                ax.annotate(f"{y:.0%}", (x, y), textcoords="offset points",
                            xytext=(8, -14) if x == 5 else (6, 8),
                            fontsize=10.5, color=C_NONE)
        ax.set_ylim(-0.03, 1.08)
        ax.set_xticks(L)
        ax.set_xlabel("input length (digits)")
        ax.set_title(title)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.grid(alpha=0.3)
        if k == 0:
            ax.set_ylabel("accuracy on 200 new strings per length")
            ax.legend(loc="center right", fontsize=11)
    ax = fig.add_subplot(gs[2])
    ax.imshow(alpha, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(s)))
    ax.set_xticklabels(list(s), fontsize=9, family="monospace")
    rows = list(out) + ["<eos>"]
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=9, family="monospace")
    ax.tick_params(length=0)
    ax.set_xlabel("input digit")
    ax.set_ylabel("output digit")
    ax.set_title("WITH attention: where each output\ndigit looks (length 20)", fontsize=13)
    fig.suptitle("Reversing digit strings: same model, same data, same 800 steps. "
                 "Only difference: attention.", y=1.03)
    fig.text(0.5, -0.06, "Without attention, accuracy collapses as the input grows: everything "
             "must fit in one 128-number vector. With attention, output step t\nsimply looks at "
             "input position S−1−t (the anti-diagonal on the right), so length barely matters.",
             ha="center", fontsize=12)
    save(fig, "accuracy_vs_length.png")
    return s, out


def fig_why_sqrt_dk():
    rng = np.random.default_rng(0)
    ds = np.array([4, 8, 16, 32, 64, 128, 256, 512, 1024])
    m, N = 10, 2000  # 10 keys, 2000 random queries per d
    stats = {}
    samples = {}
    for d in ds:
        q = rng.standard_normal((N, 1, d))
        k = rng.standard_normal((N, m, d))
        s = (q @ np.swapaxes(k, 1, 2))[:, 0, :]  # (N, m) raw dot products
        for name, sc in [("raw", s), ("scaled", s / np.sqrt(d))]:
            a = an.softmax(sc)
            # softmax Jacobian J = diag(a) - a aᵀ; its size = how much gradient gets through
            J = a[:, :, None] * np.eye(m) - a[:, :, None] * a[:, None, :]
            stats[(name, d)] = (a.max(1).mean(), np.linalg.norm(J, axis=(1, 2)).mean(), sc.std())
        samples[d] = s.ravel()

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.4))
    bins = np.linspace(-80, 80, 81)
    ax = axes[0]
    for d, c in zip([4, 64, 512], ["#2ca02c", "#ff7f0e", "#d62728"]):
        ax.hist(samples[d], bins=bins, density=True, histtype="stepfilled", alpha=0.45,
                color=c, label=f"d = {d}: std {samples[d].std():.1f}")
    ax.set_xlim(-80, 80)
    ax.set_xlabel("dot product q · k  (random q, k with N(0, 1) entries)")
    ax.set_ylabel("density")
    ax.set_title("raw scores spread out like √d")
    ax.legend(fontsize=10.5, loc="upper left")
    ax.text(78, ax.get_ylim()[1] * 0.6, "after ÷ √d the std is\n" + ", ".join(
        f"{(samples[d] / np.sqrt(d)).std():.2f}" for d in (4, 64, 512)) + "\nfor all three",
        ha="right", va="center", fontsize=10.5, bbox=dict(boxstyle="round", fc="white", ec="#bbb"))

    for ax, k, title, ylab in [
            (axes[1], 0, "softmax saturates without scaling", "average largest weight (of 10)"),
            (axes[2], 1, "...and then its gradient vanishes",
             "average ‖softmax Jacobian‖ (log)")]:
        raw = [stats[("raw", d)][k] for d in ds]
        sc = [stats[("scaled", d)][k] for d in ds]
        ax.plot(ds, raw, "s-", color=C_NONE, lw=2.5, ms=7, label="softmax(q·k)")
        ax.plot(ds, sc, "o-", color=C_ATT, lw=2.5, ms=7, label="softmax(q·k / √d)")
        ax.set_xscale("log", base=2)
        ax.set_xticks(ds)
        ax.set_xticklabels([str(d) for d in ds], fontsize=10)
        ax.set_xlabel("vector size d")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.grid(alpha=0.3)
        for d in (4, 64, 512):
            i = list(ds).index(d)
            ax.annotate(f"{raw[i]:.2g}" if k else f"{raw[i]:.2f}", (d, raw[i]),
                        textcoords="offset points", xytext=(0, 9), ha="center", fontsize=10,
                        color=C_NONE)
            ax.annotate(f"{sc[i]:.2g}" if k else f"{sc[i]:.2f}", (d, sc[i]),
                        textcoords="offset points", xytext=(0, -16), ha="center", fontsize=10,
                        color=C_ATT)
    axes[1].axhline(1 / m, color="grey", ls=":", lw=1.5)
    axes[1].text(ds[-1], 1 / m + 0.02, "uniform = 0.1", ha="right", fontsize=10, color="grey")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend(loc="center right", fontsize=10.5)
    axes[2].set_yscale("log")
    fig.suptitle("Why divide by √d$_k$: dot products of random vectors grow with the size d",
                 y=1.02)
    fig.text(0.5, -0.05, "Without scaling, big d gives huge scores, softmax becomes ~one-hot "
             "and almost no gradient flows back to Q and K. Dividing by √d keeps the scores'\n"
             "spread at ≈ 1 for every d, so attention starts soft and trainable "
             "(10 keys, 2000 random queries per d).", ha="center", fontsize=12)
    fig.tight_layout()
    save(fig, "why_sqrt_dk.png")
    return stats


def fig_qkv_matrices():
    names = list(QUERIES)
    Q = np.array([QUERIES[n] for n in names])
    d = Q.shape[1]
    S = Q @ K.T / np.sqrt(d)
    Out, A, _ = an.scaled_dot_product_attention(Q, K, V)
    short = ["DAY?", "MONTH?", "YEAR?"]
    kf = ["wkd", "day", "mon", "year"]
    vf = ["day", "mon", "year", "wkd"]

    fig, ax = plt.subplots(figsize=(18, 10.4))
    ax.set_xlim(-0.6, 27.4)
    ax.set_ylim(-0.6, 13.3)
    ax.axis("off")
    c = 1.0
    # Row 1: S = Q Kᵀ / √d, then A = softmax(S) row by row
    y1 = 11.6
    matrix(ax, Q, 1.6, y1, c, "{:g}", fc=F_Q, rows=short, cols=kf, title="Q  (queries)",
           title_color=C_Q)
    ax.text(6.2, y1 - 1.5, "·", fontsize=34, ha="center", va="center")
    matrix(ax, K.T, 7.5, y1, c, "{:g}", fc=F_K, rows=kf, cols=TOKENS,
           title="Kᵀ  (keys, transposed)", title_color=C_K)
    ax.text(12.6, y1 - 1.5, f"÷ √{d}  =", fontsize=20, ha="center", va="center")
    matrix(ax, S, 15.6, y1, c, "{:.2f}", fc="#fff3d6", rows=short, cols=TOKENS,
           title="S = Q Kᵀ / √d  (scores)")
    arrow(ax, (19.8, y1 - 1.5), (21.6, y1 - 1.5), lw=2.5)
    ax.text(20.7, y1 - 1.3, "softmax\neach row", fontsize=12, ha="center", va="bottom")
    matrix(ax, A, 22.0, y1, c, "{:.2f}", shade="Greens", cols=TOKENS,
           title="A  (weights)", title_color=C_A)
    ax.text(24.0, y1 - 3.75, "same rows as S;\neach row sums to 1", ha="center", va="top",
            fontsize=10.5, color="#555")
    # Row 2: Out = A V
    y2 = 5.9
    matrix(ax, A, 1.6, y2, c, "{:.2f}", shade="Greens", rows=short, cols=TOKENS, title="A",
           title_color=C_A)
    ax.text(6.2, y2 - 1.5, "·", fontsize=34, ha="center", va="center")
    matrix(ax, V, 7.6, y2, c, "{:g}", fc=F_V, rows=TOKENS, cols=vf, title="V  (values)",
           title_color=C_V)
    ax.text(12.6, y2 - 1.5, "=", fontsize=30, ha="center", va="center")
    matrix(ax, Out, 14.6, y2, c, "{:.2f}", fc=F_A, rows=short, cols=vf,
           title="Out = A V  (one context per query)", title_color=C_A)

    tbl = [("Q", "(n, d_k)", "(3, 4)"), ("K", "(m, d_k)", "(4, 4)"), ("V", "(m, d_v)", "(4, 4)"),
           ("S = Q Kᵀ / √d_k", "(n, m)", "(3, 4)"), ("A = softmax(S)", "(n, m)", "(3, 4)"),
           ("Out = A V", "(n, d_v)", "(3, 4)")]
    ax.text(19.8, 6.0, "shapes  (n queries, m keys)", fontsize=13, weight="bold")
    for i, (a, b, e) in enumerate(tbl):
        y = 5.2 - i * 0.7
        ax.text(19.8, y, a, fontsize=12, family="monospace")
        ax.text(23.9, y, b, fontsize=12, family="monospace")
        ax.text(25.9, y, e, fontsize=12, color="#555")
    ax.text(13.4, -0.1,
            "Row i of S compares query i with every key. Softmax turns each row into weights. "
            "Row i of Out is query i's blend of the value rows.\n"
            "The inner sizes must match: Q and K share d_k (to dot them), A's m columns match V's "
            "m rows (one weight per value). Batches just add a leading B.",
            ha="center", va="center", fontsize=12,
            bbox=dict(boxstyle="round", fc="#f5f5f5", ec="#bbb"))
    ax.set_title("Attention(Q, K, V) = softmax(Q Kᵀ / √d$_k$) V  as matrices, with the "
                 "same hand-made numbers", fontsize=15)
    save(fig, "qkv_matrices.png")
    return A, Out


def fig_masking():
    q = np.array(QUERIES["write the MONTH"])
    Kp, Vp = np.vstack([K, PAD_KEY]), np.vstack([V, PAD_VALUE])
    toks = TOKENS + ["<pad>"]
    mask = np.array([True, True, True, True, False])
    out_nm, a_nm, _ = an.scaled_dot_product_attention(q[None], Kp, Vp)
    out_m, a_m, _ = an.scaled_dot_product_attention(q[None], Kp, Vp, mask[None])

    batch = ["3 mar 21", "march 3, 2021", "03/03/2021"]
    S = max(len(s) for s in batch)

    fig = plt.figure(figsize=(17, 6.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 0.8, 1.05], wspace=0.3)

    ax = fig.add_subplot(gs[0])
    ax.set_xlim(-2.3, S + 0.2)
    ax.set_ylim(-1.6, len(batch) + 1.4)
    ax.axis("off")
    for b, s in enumerate(batch):
        y = len(batch) - 1 - b
        ax.text(-0.2, y + 0.5, f"example {b + 1}\nlen {len(s)}", ha="right", va="center",
                fontsize=10.5)
        for j in range(S):
            real = j < len(s)
            ax.add_patch(Rectangle((j, y + 0.05), 0.95, 0.9, fc="#dbe9f6" if real else "#d9d9d9",
                                   ec="white"))
            ax.text(j + 0.47, y + 0.5, show(s[j]) if real else "pad", ha="center", va="center",
                    fontsize=12 if real else 7.5, color="black" if real else "#777",
                    family="monospace")
    ax.text(S / 2, len(batch) + 0.5, "a padded batch (B, S) = (3, 13)", ha="center",
            fontsize=13, weight="bold")
    ax.text(S / 2, -0.6, "padding mask: mask[b, j] = j < len[b]\nblue = attend, grey = "
            "score set to −∞ (weight exactly 0)", ha="center", va="top", fontsize=11.5)

    ax = fig.add_subplot(gs[1])
    words = ["the", "cat", "sat", "on", "a", "mat"]
    n = len(words)
    causal = np.tril(np.ones((n, n), dtype=bool))
    ax.imshow(np.where(causal, 1.0, 0.25), cmap="Greens", vmin=0, vmax=1.4)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, "✓" if causal[i, j] else "−∞", ha="center", va="center",
                    fontsize=11 if causal[i, j] else 9, color="black" if causal[i, j] else "#888")
    ax.set_xticks(range(n))
    ax.set_xticklabels(words, fontsize=11)
    ax.set_yticks(range(n))
    ax.set_yticklabels(words, fontsize=11)
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel("key (the token looked AT)")
    ax.set_ylabel("query (the token looking)")
    ax.set_title("preview: causal mask (GPT)\nquery i may only see keys ≤ i", fontsize=13,
                 pad=30)

    ax = fig.add_subplot(gs[2])
    x = np.arange(len(toks))
    ax.bar(x - 0.2, a_nm[0], 0.4, color=C_NONE, alpha=0.8, label="no mask")
    ax.bar(x + 0.2, a_m[0], 0.4, color=C_A, label="with padding mask")
    for xi, (u, v) in enumerate(zip(a_nm[0], a_m[0])):
        ax.text(xi - 0.2, u + 0.015, f"{u:.2f}", ha="center", fontsize=10, color=C_NONE)
        ax.text(xi + 0.2, v + 0.015, f"{v:.2f}", ha="center", fontsize=10, color=C_A)
    ax.set_xticks(x)
    ax.set_xticklabels(toks, fontsize=12)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("attention weight")
    ax.legend(loc="upper left", fontsize=10.5)
    ax.set_title("the \"write the MONTH\" query with a junk <pad>", fontsize=13)
    fig.text(0.5, -0.04, f"Right panel, context = Σ α_j v_j:  without the mask "
             f"{vec(out_nm[0], '{:.2f}')}  (the junk 9s of <pad> leak into every feature);  "
             f"with the mask {vec(out_m[0], '{:.2f}')}  (≈ march's value {vec(V[2])}).",
             ha="center", fontsize=12)
    fig.suptitle("Masking: set forbidden scores to −∞ before the softmax", y=1.04)
    save(fig, "masking.png")
    return a_nm[0], a_m[0], out_nm[0], out_m[0]


def fig_kv_retrieval():
    model, hist = an.train(verbose=False)
    abl_model, hist_abl = an.train(no_attention=True, verbose=False)
    acc, w = an.evaluate(model)
    acc_abl, _ = an.evaluate(abl_model)

    rng = np.random.default_rng(7)
    keys, vals, mask, query, answer, target = an.make_kv_batch(5, rng, min_pairs=3)
    logits, A, _ = model.forward(keys, vals, mask, query)

    fig = plt.figure(figsize=(17, 6.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.25], wspace=0.42)
    ax = fig.add_subplot(gs[0])
    ax.plot(hist[:, 0], hist[:, 2], "o-", color=C_ATT, lw=2.5, ms=5,
            label="accuracy, with attention")
    ax.plot(hist[:, 0], hist[:, 3], "--", color=C_A, lw=2.5,
            label="avg weight on the matching key")
    ax.plot(hist_abl[:, 0], hist_abl[:, 2], "s-", color=C_NONE, lw=2.5, ms=5,
            label="accuracy, q = 0 (uniform average)")
    ax.axhline(0.1, color="grey", ls=":", lw=1.5)
    ax.text(hist[-1, 0], 0.12, "random guess 10%", ha="right", fontsize=10.5, color="grey")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("training step (batch of 64)")
    ax.set_ylabel("on 500 new examples")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.66), fontsize=10.5)
    ax.grid(alpha=0.3)
    ax.set_title("NumPy model, hand-written backprop (~1 s)")

    ax = fig.add_subplot(gs[1])
    P = keys.shape[1]
    M = np.where(mask, A, np.nan)
    ax.imshow(M, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for b in range(len(keys)):
        for j in range(P):
            if not mask[b, j]:
                ax.add_patch(Rectangle((j - 0.5, b - 0.5), 1, 1, fc="#eeeeee", ec="white"))
                ax.text(j, b, "pad", ha="center", va="center", fontsize=9, color="#999")
                continue
            hl = j == target[b]
            ax.text(j, b - 0.12, f"{an.KEYS[keys[b, j]]}→{an.VALUES[vals[b, j]]}", ha="center",
                    va="center", fontsize=12, color="white" if hl else "black",
                    weight="bold" if hl else "normal")
            ax.text(j, b + 0.25, f"{A[b, j]:.2f}", ha="center", va="center", fontsize=9,
                    color="white" if hl else "#555")
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels([f"query '{an.KEYS[query[b]]}'\n→ says {an.VALUES[logits[b].argmax()]}"
                        for b in range(len(keys))], fontsize=11)
    ax.set_xticks(range(P))
    ax.set_xticklabels([f"pair {j + 1}" for j in range(P)], fontsize=10)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("learned attention over the (key → value) pairs on new examples")
    fig.suptitle("Key-value retrieval: attention IS the solution (look up the query's key, "
                 "copy its value)", y=1.02)
    fig.text(0.5, -0.04, f"Final test accuracy on 2000 new examples: {acc:.1%} with attention "
             f"(average weight on the matching key {w:.3f}) vs {acc_abl:.1%} when the query is "
             "zeroed\nso the model can only average the pairs. Padded slots get weight exactly 0 "
             "thanks to the mask.", ha="center", fontsize=12)
    save(fig, "kv_retrieval.png")
    return acc, w, acc_abl


def fig_training_curves(r):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.4))
    win = 25
    for name, label, c in [("none", "without attention", C_NONE),
                           ("attn", "with attention", C_ATT)]:
        loss = r[f"{name}_loss"]
        smooth = np.convolve(loss, np.ones(win) / win, mode="valid")
        steps = np.arange(1, len(loss) + 1)
        axes[0].plot(steps, loss, color=c, alpha=0.18, lw=0.8)
        axes[0].plot(steps[win - 1:], smooth, color=c, lw=2.5, label=label)
        axes[1].plot(r[f"{name}_eval_steps"], r[f"{name}_eval_acc"], "o-", color=c, lw=2.5,
                     ms=7, label=label)
        axes[1].annotate(f"{r[f'{name}_eval_acc'][-1]:.1%}",
                         (r[f"{name}_eval_steps"][-1], r[f"{name}_eval_acc"][-1]),
                         textcoords="offset points", xytext=(-8, 10), ha="right", fontsize=11,
                         color=c)
    axes[0].axhline(np.log(10), color="grey", ls=":", lw=1.5)
    axes[0].text(len(r["none_loss"]), np.log(10) * 1.15, "random guessing: ln(10) = 2.30",
                 ha="right", fontsize=10.5, color="grey")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("training step (batch of 64 random strings, length 5..40)")
    axes[0].set_ylabel("cross-entropy per output digit (log)")
    axes[0].set_title("training loss (light = every step, dark = 25-step average)")
    axes[0].legend(fontsize=11)
    axes[1].set_ylim(-0.03, 1.05)
    axes[1].set_xlabel("training step")
    axes[1].set_ylabel("exact match, 300 test strings of length 5..40")
    axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    axes[1].set_title("test accuracy during training")
    axes[1].legend(loc="center right", fontsize=11)
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle("Training the two reverse models (same data, same seed, same 800 steps)",
                 y=1.02)
    fig.tight_layout()
    save(fig, "training_curves.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--retrain", action="store_true",
                    help="retrain the date model and the bottleneck models first")
    args = ap.parse_args()
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)

    date_model = get_date_model(args.retrain)
    r, models = get_bottleneck(args.retrain)
    print(f"  models ready ({time.time() - t0:.1f} s)")

    s, out, wrong = fig_bottleneck_diagram(r, models)
    print(f"  bottleneck_diagram: {s} -> {out} (want {s[::-1]}), {wrong} wrong")
    w, ctx = fig_attention_mechanism()
    print(f"  attention_mechanism: weights {np.round(w, 3)}, context {np.round(ctx, 2)}")
    inputs, outs = fig_date_alignments(date_model)
    print("  date_alignments: " + ", ".join(f"{i!r} -> {o}" for i, o in zip(inputs, outs)))
    s, out = fig_accuracy_vs_length(r, models)
    print(f"  accuracy_vs_length: none {np.round(r['none_exact'], 3)}, "
          f"attn {np.round(r['attn_exact'], 3)}; example {s} -> {out}")
    stats = fig_why_sqrt_dk()
    for d in (4, 64, 512):
        print(f"  why_sqrt_dk d={d}: raw std {stats[('raw', d)][2]:.1f}, max w raw "
              f"{stats[('raw', d)][0]:.2f} vs scaled {stats[('scaled', d)][0]:.2f}, |J| raw "
              f"{stats[('raw', d)][1]:.2g} vs scaled {stats[('scaled', d)][1]:.2g}")
    A, Out = fig_qkv_matrices()
    print(f"  qkv_matrices: max weight per query {np.round(A.max(1), 3)}")
    a_nm, a_m, o_nm, o_m = fig_masking()
    print(f"  masking: pad weight {a_nm[-1]:.3f} -> {a_m[-1]:.3f}; context {np.round(o_nm, 2)} "
          f"-> {np.round(o_m, 2)}")
    acc, w, acc_abl = fig_kv_retrieval()
    print(f"  kv_retrieval: acc {acc:.1%}, weight on match {w:.3f}, ablation acc {acc_abl:.1%}")
    fig_training_curves(r)
    print(f"total time {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
