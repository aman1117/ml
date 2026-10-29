"""
Draw the pictures used in transformer/README.md and save them to transformer/images/.

Every number in every figure is computed here:
  - the trained mini-GPT: loaded from transformer/model.npz if it was trained with
    the default settings, otherwise trained here with the same settings and seed
    as `python transformer/transformer_numpy.py` (~10 s), so the weights are identical;
  - the char-RNN from ../rnn/rnn_numpy.py, trained on the same text (~15 s);
  - the results of why_transformers.py (loaded from why_transformers.npz, or run, ~25 s);
  - small timing runs of the KV-cache sampler in generate.py.

Run (~35-70 s):
    python transformer/make_figures.py
"""
import contextlib
import io
import math
import os
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

from generate import decode_step, prefill  # noqa: E402
from transformer_numpy import (DEFAULT_CONFIG, DEFAULT_TEXT, GPT, evaluate,  # noqa: E402
                               load_model, load_text, param_count_formula, softmax, train)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "images")
DPI = 130
sys.path.insert(0, os.path.join(HERE, "..", "rnn"))
from rnn_numpy import RNN  # noqa: E402  (the RNN folder's model, for comparison)

# One colour per component, used in every diagram.
C_EMB, C_ATT, C_MLP, C_LN, C_OUT = "#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#d62728"
F_EMB, F_ATT, F_MLP, F_LN, F_OUT, F_RES = ("#dbe9f6", "#ffe8cc", "#e3f4e1", "#ece4f5",
                                           "#fbe3e3", "#fdf1c7")

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "figure.titlesize": 16,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def show(c):
    """Make invisible characters visible in labels."""
    return {" ": "␣", "\n": "↵"}.get(c, c)


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


def plus(ax, x, y, r=0.32, fs=20):
    ax.add_patch(plt.Circle((x, y), r, fc="white", ec="black", lw=1.8, zorder=2))
    ax.text(x, y, "+", fontsize=fs, ha="center", va="center", zorder=3)


# ---------------------------------------------------------------- model helpers

def get_model(data, V):
    """The trained model with the default settings (load if possible, else train)."""
    path = os.path.join(HERE, "model.npz")
    want = dict(vocab_size=V, **DEFAULT_CONFIG)
    if os.path.exists(path):
        model, chars, hist = load_model(path, dtype=np.float64)
        if model.config == want and hist and "time" in hist and len(hist["loss"]) == 2000:
            print(f"loaded {os.path.relpath(path)} (default settings)")
            return model, hist
    print("training the mini-GPT with the default settings (~10 s)...")
    with contextlib.redirect_stdout(io.StringIO()):
        m32, hist = train(data, V, verbose=False)
    model = GPT(dtype=np.float64, **m32.config)
    for (_, p), (_, p32) in zip(model.params().items(), m32.params().items()):
        p[...] = p32
    return model, {k: np.array(v) for k, v in hist.items()}


def attention_of(model, ids):
    """Forward one sequence; returns (logits (T,V), list of (nh,T,T) attention maps)."""
    logits = model.forward(np.array(ids)[None])[0]
    return logits, [a[0] for a in model.attention_maps()]


def head_stats(model, data):
    """Per layer/head, over the whole text: mean weight on the previous token, and the
    mean distance looked back (Σ_j A[t, j]·(t - j)), for queries t >= 8."""
    T = model.config["block_size"]
    starts = np.arange(0, len(data) - T - 1, T)
    model.forward(np.stack([data[i:i + T] for i in starts]))
    back = np.arange(T)[:, None] - np.arange(T)[None, :]  # (T, T): t - j
    prev, dist = [], []
    for A in model.attention_maps():  # (B, nh, T, T)
        t = np.arange(8, T)
        prev.append(A[:, :, t, t - 1].mean(axis=(0, 2)))
        dist.append((A[:, :, 8:, :] * np.maximum(back[8:], 0)).sum(-1).mean(axis=(0, 2)))
    return np.array(prev), np.array(dist)


def train_rnn(text, hidden=100, seq_len=25, lr=0.1, iters=10000, clip=5.0, seed=0):
    """Same loop as ../rnn/rnn_numpy.train, recording the loss per char and the time."""
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    data = [stoi[c] for c in text]
    model = RNN(len(chars), hidden, seed)
    mem = {k: np.zeros_like(v) for k, v in model.params.items()}
    p, h = 0, np.zeros((hidden, 1))
    losses, times = [], []
    t0 = time.time()
    for _ in range(iters + 1):
        if p + seq_len + 1 > len(data):
            p, h = 0, np.zeros((hidden, 1))
        loss, h, cache = model.forward(data[p:p + seq_len], data[p + 1:p + seq_len + 1], h)
        grads = model.backward(cache)
        for k, g in grads.items():
            np.clip(g, -clip, clip, out=g)
            mem[k] += g * g
            model.params[k] -= lr * g / np.sqrt(mem[k] + 1e-8)
        losses.append(loss / seq_len)
        times.append(time.time() - t0)
        p += seq_len
    return np.array(losses), np.array(times)


def smooth(x, window):
    c = np.cumsum(np.insert(np.asarray(x, dtype=float), 0, 0.0))
    return np.array([(c[i + 1] - c[max(0, i - window + 1)]) / (i + 1 - max(0, i - window + 1))
                     for i in range(len(x))])


def sinusoidal(T, d):
    pe = np.zeros((T, d))
    pos = np.arange(T)[:, None]
    i = np.arange(0, d, 2)
    pe[:, 0::2] = np.sin(pos / 10000 ** (i / d))
    pe[:, 1::2] = np.cos(pos / 10000 ** (i / d))
    return pe


def group_params(model):
    """Parameter count per component of our model."""
    groups = {"token embedding": 0, "position embedding": 0, "attention": 0, "MLP": 0,
              "LayerNorms": 0, "output head": 0}
    for name, p in model.params().items():
        if name.startswith("wte"):
            groups["token embedding"] += p.size
        elif name.startswith("wpe"):
            groups["position embedding"] += p.size
        elif ".attn." in name:
            groups["attention"] += p.size
        elif ".mlp." in name:
            groups["MLP"] += p.size
        elif "ln" in name:
            groups["LayerNorms"] += p.size
        else:
            groups["output head"] += p.size
    return groups


# ---------------------------------------------------------------- figures

def fig_architecture(model, chars, prompt="The keeper"):
    cfg = model.config
    V, C, L, nh, T = cfg["vocab_size"], cfg["d_model"], cfg["n_layer"], cfg["n_head"], \
        cfg["block_size"]
    stoi = {c: i for i, c in enumerate(chars)}
    ids = [stoi[c] for c in prompt]
    logits, _ = attention_of(model, ids)
    p = softmax(logits[-1])
    top = np.argsort(p)[::-1][:3]
    g = group_params(model)
    blk = model.blocks[0]
    n_attn = sum(x.size for x in (*blk.attn.c_attn.params.values(),
                                  *blk.attn.c_proj.params.values()))
    n_mlp = sum(x.size for x in (*blk.mlp.c_fc.params.values(), *blk.mlp.c_proj.params.values()))
    B = 16

    fig, ax = plt.subplots(figsize=(16.5, 12.5))
    ax.set_xlim(-0.5, 19.5)
    ax.set_ylim(-0.6, 20.4)
    ax.axis("off")
    sx, bx = 2.6, 7.4  # x of the residual stream and of the sublayer boxes

    box(ax, 4.6, 0.3, f"input text \"{prompt}\"  →  token ids  " +
        " ".join(str(i) for i in ids[:6]) + " …     shape (B, T)", fc="#f5f5f5", w=10.0, h=0.8)
    box(ax, 2.6, 2.0, f"token embedding  wte\n({V}, {C}) = {g['token embedding']:,} params",
        fc=F_EMB, ec=C_EMB, w=4.2, h=1.2, lw=2.5)
    box(ax, 7.2, 2.0, f"position embedding  wpe\n({T}, {C}) = {g['position embedding']:,} params",
        fc=F_EMB, ec=C_EMB, w=4.4, h=1.2, lw=2.5)
    arrow(ax, (2.6, 0.7), (2.6, 1.4))
    arrow(ax, (7.2, 0.7), (7.2, 1.4))
    plus(ax, sx, 3.5)
    arrow(ax, (2.6, 2.6), (sx, 3.18))
    arrow(ax, (7.2, 2.6), (sx + 0.3, 3.6))
    ax.text(sx + 0.45, 3.75, "x = wte[ids] + wpe[0..T-1]", fontsize=11, va="bottom")

    # the block
    ax.add_patch(FancyBboxPatch((0.6, 4.4), 10.6, 9.4, boxstyle="round,pad=0.1,rounding_size=0.3",
                                fc="#fafafa", ec="grey", ls="--", lw=1.5, zorder=0))
    ax.text(11.05, 13.6, f"× {L} blocks\n(n_layer = {L})", ha="right", va="top", fontsize=13,
            weight="bold", color="#555")
    # residual stream
    ax.add_patch(Rectangle((sx - 0.12, 3.85), 0.24, 11.0, fc=F_RES, ec="#d4b106", lw=1.2,
                           zorder=0.5))
    ax.text(sx - 0.75, 9.0, f"residual stream  x   (B, T, C) = (B, T, {C})", rotation=90,
            ha="center", va="center", fontsize=12, color="#8a6d00", weight="bold")
    # attention branch
    arrow(ax, (sx + 0.12, 5.0), (bx - 1.6, 5.0), lw=1.6)
    box(ax, bx, 5.0, "LayerNorm 1", fc=F_LN, ec=C_LN, w=3.2, h=0.7, lw=2)
    box(ax, bx, 6.9, f"masked multi-head self-attention\n{nh} heads × {C // nh} dims"
        f"   q, k, v = x W_q, x W_k, x W_v\nsoftmax(q kᵀ/√{C // nh} + mask) v → concat → W_o"
        f"\n{n_attn:,} params", fc=F_ATT, ec=C_ATT, w=6.9, h=2.0, lw=3, fs=11.5)
    arrow(ax, (bx, 5.35), (bx, 5.9))
    plus(ax, sx, 8.6)
    ax.plot([bx, bx], [7.9, 8.6], color="black", lw=2, zorder=1)
    arrow(ax, (bx, 8.6), (sx + 0.32, 8.6))
    ax.text((sx + bx) / 2 + 0.6, 8.7, "x = x + Attn(LN(x))", fontsize=11, va="bottom")
    # MLP branch
    arrow(ax, (sx + 0.12, 9.6), (bx - 1.6, 9.6), lw=1.6)
    box(ax, bx, 9.6, "LayerNorm 2", fc=F_LN, ec=C_LN, w=3.2, h=0.7, lw=2)
    box(ax, bx, 11.3, f"MLP   Linear {C} → {4 * C}  →  GELU  →  Linear {4 * C} → {C}\n"
        f"(each token on its own)   {n_mlp:,} params", fc=F_MLP, ec=C_MLP, w=6.9, h=1.4,
        lw=3, fs=11.5)
    arrow(ax, (bx, 9.95), (bx, 10.6))
    plus(ax, sx, 12.8)
    ax.plot([bx, bx], [12.0, 12.8], color="black", lw=2, zorder=1)
    arrow(ax, (bx, 12.8), (sx + 0.32, 12.8))
    ax.text((sx + bx) / 2 + 0.6, 12.9, "x = x + MLP(LN(x))", fontsize=11, va="bottom")
    # output
    box(ax, sx, 15.4, f"final LayerNorm", fc=F_LN, ec=C_LN, w=3.2, h=0.7, lw=2)
    box(ax, sx + 1.4, 16.9, f"output head (unembedding)\nlogits = x W_head   ({C}, {V}) = "
        f"{g['output head']:,} params", fc=F_OUT, ec=C_OUT, w=6.0, h=1.1, lw=2.5)
    box(ax, sx + 1.4, 18.6, f"softmax → next-char probabilities   (B, T, V) = (B, T, {V})",
        fc="#f5f5f5", w=7.6, h=0.8)
    arrow(ax, (sx, 14.85), (sx, 15.05))
    arrow(ax, (sx, 15.75), (sx, 16.35))
    arrow(ax, (sx + 1.4, 17.45), (sx + 1.4, 18.2))
    ax.text(sx + 5.0, 19.55, f"real model, after \"{prompt}\":  " + "   ".join(
        f"'{show(chars[i])}' {p[i]:.0%}" for i in top), ha="center", fontsize=12,
        bbox=dict(boxstyle="round", fc="#fffbe6", ec="#d4b106"))

    # shapes + parameter table
    rows = [("tensor", "shape", f"B={B}, T={T}"),
            ("ids", "(B, T)", f"({B}, {T})"),
            ("x (residual stream)", "(B, T, C)", f"({B}, {T}, {C})"),
            ("q, k, v per head", "(B, nh, T, hs)", f"({B}, {nh}, {T}, {C // nh})"),
            ("scores, weights A", "(B, nh, T, T)", f"({B}, {nh}, {T}, {T})"),
            ("MLP hidden", "(B, T, 4C)", f"({B}, {T}, {4 * C})"),
            ("logits", "(B, T, V)", f"({B}, {T}, {V})")]
    y0 = 18.9
    ax.text(15.4, y0 + 0.6, "Shapes (training batch)", ha="center", fontsize=13, weight="bold")
    for k, r in enumerate(rows):
        y = y0 - 0.55 * (k + 0.5)
        wt = "bold" if k == 0 else "normal"
        ax.text(11.7, y, r[0], fontsize=11, weight=wt)
        ax.text(14.75, y, r[1], fontsize=10.5, family="monospace", weight=wt)
        ax.text(19.4, y, r[2], fontsize=10.5, family="monospace", ha="right", weight=wt)
    total = model.n_params()
    prows = [(k, v) for k, v in g.items()]
    y1 = 13.4
    ax.text(15.4, y1 + 0.6, "Parameters", ha="center", fontsize=13, weight="bold")
    for k, (name, n) in enumerate(prows):
        y = y1 - 0.55 * (k + 0.5)
        ax.text(11.7, y, name, fontsize=11)
        ax.text(19.4, y, f"{n:,}  ({n / total:.0%})", fontsize=11, ha="right", family="monospace")
    y = y1 - 0.55 * (len(prows) + 0.6)
    ax.plot([11.7, 19.4], [y + 0.3, y + 0.3], color="black", lw=1)
    ax.text(11.7, y, "total", fontsize=11, weight="bold")
    ax.text(19.4, y, f"{total:,}", fontsize=11, ha="right", family="monospace", weight="bold")
    ax.text(11.7, y - 0.7, f"formula: L·(12d² + 13d) + (V + T)·d + 2d + V·d\n"
            f"= {param_count_formula(V, T, C, L, False):,}  ✔   (12·L·d² alone = "
            f"{12 * L * C * C:,})", fontsize=10.5, va="top", color="#444")

    ax.set_title("A decoder-only Transformer (mini-GPT): the real model in this folder",
                 fontsize=16, pad=10)
    save(fig, "architecture.png")
    return total


def fig_self_attention(model, chars, prompt="The keeper"):
    stoi = {c: i for i, c in enumerate(chars)}
    ids = [stoi[c] for c in prompt]
    T = len(ids)
    _, maps = attention_of(model, ids)
    a = model.blocks[0].attn
    q, k = a.q[0], a.k[0]  # (nh, T, hs)
    A0 = maps[0]
    prev = [np.mean([A0[h, t, t - 1] for t in range(1, T)]) for h in range(A0.shape[0])]
    hd = int(np.argmax(prev))  # same head as attention_step_visual.py shows by default
    S = q[hd] @ k[hd].T * a.scale
    mask = np.tril(np.ones((T, T), dtype=bool))
    A = A0[hd]
    labels = [show(c) for c in prompt]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.6))
    vmax = np.abs(S).max()
    panels = [(S, "1) scores = q·kᵀ / √hs\n(every query with every key)", "RdBu_r", -vmax, vmax),
              (np.where(mask, S, np.nan), "2) + causal mask\n(−∞ above the diagonal)", "RdBu_r",
               -vmax, vmax),
              (np.where(mask, A, np.nan), "3) softmax along each row\n= attention weights",
               "Oranges", 0, 1)]
    for ax, (M, title, cmap, lo, hi) in zip(axes, panels):
        cm = plt.get_cmap(cmap).copy()
        cm.set_bad("#d9d9d9")
        ax.imshow(M, cmap=cm, vmin=lo, vmax=hi)
        ax.set_xticks(range(T))
        ax.set_xticklabels(labels, fontsize=13, family="monospace")
        ax.set_yticks(range(T))
        ax.set_yticklabels([f"{c} (t={i})" for i, c in enumerate(labels)], fontsize=11,
                           family="monospace")
        ax.set_xlabel("key (the token being looked at)")
        ax.xaxis.set_label_position("top")
        ax.xaxis.tick_top()
        ax.set_title(title, fontsize=14, pad=34)
        for i in range(T):
            for j in range(T):
                v = M[i, j]
                if np.isnan(v):
                    ax.text(j, i, "−∞" if "mask" in title else "", ha="center", va="center",
                            fontsize=8.5, color="#888")
                elif "softmax" in title:
                    ax.text(j, i, f"{v:.0%}" if v >= 0.005 else "0", ha="center", va="center",
                            fontsize=8.5, color="white" if v > 0.5 else "black")
                else:
                    ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=8,
                            color="white" if abs(v) > 0.6 * vmax else "black")
        for s in ax.spines.values():
            s.set_visible(False)
    axes[0].set_ylabel("query (the token that is looking)")
    t = T - 1
    j = int(np.argmax(A[t]))
    fig.text(0.5, -0.03,
             f"Layer 0, head {hd} of the trained model reading \"{prompt}\". Row = query, column = "
             "key. Each row of panel 3 sums to 100%.\n"
             f"Last row: '{labels[t]}' takes {A[t, j]:.0%} of its output from '{labels[j]}' "
             f"(t={j}). On average this head puts {prev[hd]:.0%} on the previous token "
             f"(uniform would be {np.mean([1 / (s + 1) for s in range(1, T)]):.0%}).",
             ha="center", fontsize=12.5)
    fig.suptitle("Causal self-attention with real numbers: scores → mask → softmax", y=1.03)
    fig.tight_layout()
    save(fig, "self_attention.png")
    return hd, prev[hd], A[t, j], labels[j]


def fig_attention_heads(model, chars, data, sentence="The light turned and turned."):
    stoi = {c: i for i, c in enumerate(chars)}
    _, maps = attention_of(model, [stoi[c] for c in sentence])
    prev, dist = head_stats(model, data)
    L, nh = prev.shape
    labels = [show(c) for c in sentence]
    T = len(sentence)
    best = np.unravel_index(prev.argmax(), prev.shape)
    far = np.unravel_index(dist.argmax(), dist.shape)

    fig, axes = plt.subplots(L, nh, figsize=(4.6 * nh, 5.4 * L + 0.6))
    for l in range(L):
        for h in range(nh):
            ax = axes[l, h]
            M = np.where(np.tril(np.ones((T, T), dtype=bool)), maps[l][h], np.nan)
            cm = plt.get_cmap("Oranges").copy()
            cm.set_bad("#eeeeee")
            ax.imshow(M, cmap=cm, vmin=0, vmax=1)
            ax.set_xticks(range(T))
            ax.set_xticklabels(labels, fontsize=7.5, family="monospace")
            ax.set_yticks(range(T))
            ax.set_yticklabels(labels, fontsize=7.5, family="monospace")
            ax.tick_params(length=0)
            tag = ""
            if (l, h) == best:
                tag = "  ← most 'previous-token'"
            elif (l, h) == far:
                tag = "  ← looks furthest back"
            ax.set_title(f"layer {l}, head {h}{tag}\nprev token {prev[l, h]:.0%} · "
                         f"looks back {dist[l, h]:.1f} chars", fontsize=11.5,
                         color="#b30000" if tag else "black")
            for s in ax.spines.values():
                s.set_visible(tag != "")
                s.set_color("#b30000")
                s.set_linewidth(2.5)
    fig.suptitle(f"Every head of the trained mini-GPT reading \"{sentence}\"  "
                 "(row = query, column = key; darker = more attention)", y=0.995)
    fig.text(0.5, 0.005,
             "Numbers under each title: averages over the WHOLE training text: the share of "
             "attention on the previous character, and the average distance looked back "
             "(Σ weight × distance).\nLayer 0 heads attend to the last few characters (a soft "
             f"local window, {dist[0].min():.1f}-{dist[0].max():.1f} chars back; the most "
             f"previous-token head reaches {prev[best]:.0%}). Layer 1 heads look further back "
             f"({dist[1].min():.1f}-{dist[1].max():.1f} chars),\npicking positions by what they "
             "contain rather than by a fixed offset. On 4 KB of text no crisp previous-token "
             "or induction head forms.", ha="center", fontsize=12,
             va="bottom")
    fig.tight_layout(rect=(0, 0.07, 1, 0.97), h_pad=5.0)
    save(fig, "attention_heads.png")
    return prev, dist


def fig_positions(model):
    W = model.wpe.params["weight"]  # (T, C)
    T, C = W.shape
    Wn = W / np.linalg.norm(W, axis=1, keepdims=True)
    sim_learned = Wn @ Wn.T
    pe = sinusoidal(T, C)
    pen = pe / np.linalg.norm(pe, axis=1, keepdims=True)
    sim_sin = pen @ pen.T
    near = np.mean([sim_learned[i, i + 1] for i in range(T - 1)])
    far = np.mean([sim_learned[i, j] for i in range(T) for j in range(T) if abs(i - j) > 8])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.9))
    im = axes[0].imshow(sim_learned, cmap="RdBu_r", vmin=-1, vmax=1)
    axes[0].set_title(f"LEARNED wpe of our trained model\ncosine similarity of position vectors",
                      fontsize=13)
    axes[0].set_xlabel("position j")
    axes[0].set_ylabel("position i")
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.03)
    im = axes[1].imshow(pe.T, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    axes[1].set_title("SINUSOIDAL encoding (Vaswani et al. 2017)\nPE[pos, 2i] = sin(pos / "
                      "10000^(2i/d)),  PE[pos, 2i+1] = cos(…)", fontsize=13)
    axes[1].set_xlabel("position")
    axes[1].set_ylabel(f"dimension (d = {C})")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.03)
    im = axes[2].imshow(sim_sin, cmap="RdBu_r", vmin=-1, vmax=1)
    axes[2].set_title("SINUSOIDAL: cosine similarity\n(depends only on the distance |i − j|)",
                      fontsize=13)
    axes[2].set_xlabel("position j")
    axes[2].set_ylabel("position i")
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.03)
    fig.suptitle("Position embeddings: how the model knows WHERE a token is", y=1.02)
    fig.text(0.5, -0.04,
             f"Left: learned from scratch, neighbouring positions ended up similar (cosine "
             f"{near:+.2f} on average for |i − j| = 1) and far-apart positions point in "
             f"opposite directions ({far:+.2f} for |i − j| > 8).\nMiddle: fixed, no parameters: each pair of "
             "dimensions is a sine/cosine of a different wavelength (fast at the top, slow at "
             "the bottom), like the hands of a clock.\nRight: so similarity is a function of "
             "distance alone. RoPE (used in LLaMA) rotates q and k by such angles instead of "
             "adding a vector to x.", ha="center", fontsize=12, va="top")
    fig.tight_layout()
    save(fig, "positional_embeddings.png")
    return near, far


def fig_causal_mask(model, chars, prompt="The keeper"):
    stoi = {c: i for i, c in enumerate(chars)}
    ids = [stoi[c] for c in prompt]
    T = len(ids)
    full = model.forward(np.array(ids)[None])[0]  # (T, V): one parallel pass
    diffs, rows = [], []
    for t in range(T):
        sep = model.forward(np.array(ids[:t + 1])[None])[0, -1]  # a separate run on the prefix
        diffs.append(np.abs(sep - full[t]).max())
        p = softmax(full[t])
        i = int(np.argmax(p))
        rows.append((prompt[:t + 1], chars[i], p[i], prompt[t + 1] if t + 1 < T else None))

    fig = plt.figure(figsize=(16, 7.4))
    ax = fig.add_axes([0.03, 0.12, 0.33, 0.72])
    M = np.tril(np.ones((T, T)))
    ax.imshow(M, cmap=matplotlib.colors.ListedColormap(["#e6e6e6", "#a1d99b"]), vmin=0, vmax=1)
    for i in range(T):
        for j in range(T):
            ax.text(j, i, "✓" if j <= i else "·", ha="center", va="center", fontsize=12,
                    color="#145214" if j <= i else "#999")
    lab = [show(c) for c in prompt]
    ax.set_xticks(range(T))
    ax.set_xticklabels(lab, fontsize=13, family="monospace")
    ax.set_yticks(range(T))
    ax.set_yticklabels(lab, fontsize=13, family="monospace")
    ax.xaxis.tick_top()
    ax.set_xlabel("may attend to (key)")
    ax.xaxis.set_label_position("top")
    ax.set_ylabel("query token (row t)")
    ax.set_title("the causal mask: row t sees only 0..t", fontsize=14, pad=40)
    for s in ax.spines.values():
        s.set_visible(False)

    ax2 = fig.add_axes([0.40, 0.08, 0.58, 0.8])
    ax2.axis("off")
    ax2.set_xlim(0, 10)
    ax2.set_ylim(-0.6, T + 1.6)
    ax2.text(0.0, T + 1.0, "row t of ONE forward pass on the whole text", fontsize=13,
             weight="bold")
    ax2.text(5.9, T + 1.0, "= a separate run on the prefix 0..t", fontsize=13, weight="bold")
    for t, (pre, c, pr, nxt) in enumerate(rows):
        y = T - t - 0.2
        col = "#145214" if nxt is not None and c == nxt else "#333"
        ax2.text(0.0, y, f"t={t}", fontsize=12, family="monospace", color="#666")
        ax2.text(0.9, y, f"\"{pre}\"", fontsize=12.5, family="monospace")
        ax2.text(4.0, y, f"→ '{show(c)}' {pr:4.0%}", fontsize=12.5, family="monospace", color=col)
        ax2.text(5.9, y, f"max |logit difference| = {diffs[t]:.0e}", fontsize=12,
                 family="monospace", color="#555")
    ax2.text(0.0, -0.5,
             "Training: ONE pass over T tokens gives T next-token predictions (and T losses) at "
             "once,\neach one exactly what autoregressive generation would compute after that "
             "prefix.\nWithout the mask, row t could peek at token t+1, the very thing it must "
             "predict.", fontsize=12.5, va="top",
             bbox=dict(boxstyle="round", fc="#fffbe6", ec="#d4b106"))
    fig.suptitle("Why the causal mask makes parallel training = autoregressive generation "
                 "(real model, \"The keeper\")", y=0.99)
    save(fig, "causal_mask.png")
    return max(diffs)


def fig_sqrt_dk(seed=0):
    rng = np.random.default_rng(seed)
    dims = 2 ** np.arange(1, 11)
    n_keys, trials = 32, 2000
    std_raw, std_scaled, max_raw, max_scaled, grad_raw, grad_scaled = [], [], [], [], [], []
    for d in dims:
        q = rng.standard_normal((trials, 1, d))
        k = rng.standard_normal((trials, n_keys, d))
        s = (q @ k.transpose(0, 2, 1))[:, 0]  # (trials, n_keys)
        for scores, std_l, max_l, g_l in ((s, std_raw, max_raw, grad_raw),
                                          (s / np.sqrt(d), std_scaled, max_scaled, grad_scaled)):
            p = softmax(scores)
            std_l.append(scores.std())
            max_l.append(p.max(axis=1).mean())
            # size of the softmax Jacobian diag(p) - p pᵀ: how much gradient can flow
            g_l.append(np.mean(np.sqrt(np.sum((p * (1 - p)) ** 2, axis=1))))
    std_raw, std_scaled = np.array(std_raw), np.array(std_scaled)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.6))
    ax = axes[0]
    ax.loglog(dims, std_raw, "o-", color=C_OUT, lw=2.5, label="q·k  (no scaling)")
    ax.loglog(dims, std_scaled, "o-", color=C_EMB, lw=2.5, label="q·k / √d_k")
    ax.loglog(dims, np.sqrt(dims), ":", color="grey", label="√d_k")
    ax.set_xlabel("head dimension d_k")
    ax.set_ylabel("std of the attention score")
    ax.set_title("1) the spread of q·k grows like √d_k")
    ax.legend(fontsize=11)
    ax = axes[1]
    ax.semilogx(dims, max_raw, "o-", color=C_OUT, lw=2.5, label="no scaling")
    ax.semilogx(dims, max_scaled, "o-", color=C_EMB, lw=2.5, label="÷ √d_k")
    ax.axhline(1 / n_keys, color="grey", ls=":")
    ax.text(dims[0], 1 / n_keys + 0.02, f"uniform = 1/{n_keys}", color="grey", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("head dimension d_k")
    ax.set_ylabel(f"largest softmax weight (over {n_keys} keys)")
    ax.set_title("2) unscaled softmax turns one-hot")
    ax.legend(fontsize=11, loc="center right")
    ax = axes[2]
    ax.loglog(dims, grad_raw, "o-", color=C_OUT, lw=2.5, label="no scaling")
    ax.loglog(dims, grad_scaled, "o-", color=C_EMB, lw=2.5, label="÷ √d_k")
    ax.set_xlabel("head dimension d_k")
    ax.set_ylabel("‖diag(J_softmax)‖  (gradient that gets through)")
    ax.set_title("3) …so its gradient vanishes")
    ax.legend(fontsize=11)
    i64 = list(dims).index(64)
    fig.suptitle("Why divide by √d_k?  Random q, k with N(0, 1) entries (what LayerNorm + "
                 "a fresh init give you)", y=1.02)
    fig.text(0.5, -0.05,
             f"At d_k = 64 the raw scores have std {std_raw[i64]:.1f} (scaled: "
             f"{std_scaled[i64]:.2f}); the top key gets {max_raw[i64]:.0%} of the weight "
             f"(scaled: {max_scaled[i64]:.0%}), and the softmax gradient is "
             f"{grad_scaled[i64] / grad_raw[i64]:.1f}× smaller than with scaling.\n"
             "Dividing by √d_k keeps the scores' variance at 1 for every head size, so "
             "attention starts soft and trainable.", ha="center", fontsize=12.5)
    fig.tight_layout()
    save(fig, "why_sqrt_dk.png")
    return std_raw[i64], max_raw[i64], max_scaled[i64]


def fig_training(hist, rnn_losses, rnn_times, V, final_eval, B=16, T=32, seq_len=25):
    tl = np.asarray(hist["loss"])
    tt = np.asarray(hist["time"])
    t_chars = (np.arange(len(tl)) + 1) * B * T
    r_chars = (np.arange(len(rnn_losses)) + 1) * seq_len
    ts, rs = smooth(tl, 50), smooth(rnn_losses, 400)
    rnn_final = rnn_losses[-200:].mean()

    fig, axes = plt.subplots(1, 2, figsize=(17, 6.2))
    for ax, (xt, xr, xlabel, logx) in zip(axes, [
            (t_chars, r_chars, "characters processed (predictions trained on)", True),
            (tt, rnn_times, "wall-clock seconds (both in NumPy, this laptop)", False)]):
        ax.plot(xr, rnn_losses, color="#f2c9a0", lw=0.4)
        ax.plot(xt, tl, color="#b9cfe5", lw=0.4)
        ax.plot(xr, rs, color=C_ATT, lw=2.5, label="char-RNN  (../rnn/rnn_numpy.py, 25 chars/step)")
        ax.plot(xt, ts, color=C_EMB, lw=2.5, label=f"mini-GPT  ({B}×{T} = {B * T} chars/step)")
        ax.axhline(np.log(V), color=C_OUT, ls="--", lw=1.5)
        ax.text(xt[-1] if not logx else 1e6, np.log(V) + 0.05,
                f"uniform guessing: ln({V}) = {np.log(V):.2f}", color=C_OUT, ha="right",
                va="bottom", fontsize=11)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("training loss per character (nats)")
        ax.set_ylim(0, 4.3)
        if logx:
            ax.set_xscale("log")
            ax.set_xlim(300, 1.2e6)
        ax.legend(loc="lower left", fontsize=11)
    axes[0].set_title("per character seen: similar early on")
    axes[1].set_title("per second: parallel training wins")
    rnn_at_same = rs[np.searchsorted(rnn_times, tt[-1]) - 1] if rnn_times[-1] >= tt[-1] else None
    note = (f"After {tt[-1]:.0f} s: GPT {ts[-1]:.2f}, RNN "
            f"{rnn_at_same:.2f}" if rnn_at_same is not None else "")
    axes[1].text(0.98, 0.6, f"final: mini-GPT {final_eval:.2f} per char on the whole text "
                 f"(perplexity {np.exp(final_eval):.2f})\n         char-RNN {rnn_final:.2f} "
                 f"(last 200 chunks, perplexity {np.exp(rnn_final):.2f})\n{note}",
                 transform=axes[1].transAxes, ha="right", fontsize=11.5, family="monospace",
                 bbox=dict(boxstyle="round", fc="white", ec="#bbb"))
    fig.suptitle("Same text, same units: the mini-GPT vs the char-RNN (training loss per "
                 "character)", y=1.0)
    fig.text(0.5, -0.04, "Faint lines: every step. Thick: moving average. The GPT trains on "
             f"{B * T} positions per step in parallel (one matmul), the RNN on 25 one after "
             "another.\nBoth memorize this tiny 4 KB story; on held-out text the gap is small "
             "(see transformer_pytorch.py and README section 9).", ha="center", fontsize=12)
    fig.tight_layout()
    save(fig, "training_curve.png")
    return ts[-1], rnn_final, rnn_times[-1], tt[-1]


def gpt2_small_groups():
    V, T, d, L = 50257, 1024, 768, 12
    return {"token embedding": V * d, "position embedding": T * d,
            "attention": L * (4 * d * d + 4 * d), "MLP": L * (8 * d * d + 5 * d),
            "LayerNorms": L * 4 * d + 2 * d, "output head": 0}, param_count_formula(V, T, d, L, True)


def fig_params(model):
    ours = group_params(model)
    g2, g2_total = gpt2_small_groups()
    assert sum(g2.values()) == g2_total
    cfg = model.config
    names = list(ours)
    colors = [C_EMB, "#7fb2de", C_ATT, C_MLP, C_LN, C_OUT]

    fig, axes = plt.subplots(1, 2, figsize=(17, 5.8))
    for ax, (groups, total, title) in zip(axes, [
            (ours, model.n_params(), f"our mini-GPT: L={cfg['n_layer']}, d={cfg['d_model']}, "
             f"V={cfg['vocab_size']}, T={cfg['block_size']}"),
            (g2, g2_total, "GPT-2 small: L=12, d=768, V=50,257, T=1,024 (tied head)")]):
        vals = [groups[n] for n in names]
        y = np.arange(len(names))[::-1]
        ax.barh(y, vals, color=colors)
        ax.set_yticks(y)
        ax.set_yticklabels(names)
        for yi, v in zip(y, vals):
            label = f"{v:,}  ({v / total:.1%})" if v else "0 (shares wte)"
            ax.text(v + total * 0.01, yi, label, va="center", fontsize=11)
        ax.set_xlim(0, max(vals) * 1.45)
        ax.set_title(f"{title}\ntotal {total:,} parameters", fontsize=13)
        ax.set_xlabel("parameters")
        big_model = total > 1e6
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
            lambda v, _, big_model=big_model: f"{v / 1e6:.0f}M" if big_model else f"{v / 1e3:.0f}k"))
    L, d = cfg["n_layer"], cfg["d_model"]
    fig.suptitle("Where the parameters live:  per layer 12·d² (attention 4d², MLP 8d²) "
                 "+ embeddings", y=1.02)
    fig.text(0.5, -0.06,
             f"Ours: 12·L·d² = {12 * L * d * d:,} = {12 * L * d * d / model.n_params():.0%} of all "
             "parameters.   GPT-2 small: 12·L·d² = "
             f"{12 * 12 * 768 ** 2 / 1e6:.1f}M = {12 * 12 * 768 ** 2 / g2_total:.0%}, the token "
             f"embedding {g2['token embedding'] / 1e6:.1f}M = "
             f"{g2['token embedding'] / g2_total:.0%} (a 50k-token vocabulary is big).\n"
             "In both, the MLPs hold about twice as many weights as attention. In very large "
             "models (d ≥ 4096) the 12·L·d² term is almost everything.",
             ha="center", fontsize=12)
    fig.tight_layout()
    save(fig, "param_breakdown.png")
    return g2_total


def fig_kv_cache(model):
    cfg = model.config
    # (b) measured time per generated token at position t, with and without the cache
    big = GPT(cfg["vocab_size"], n_layer=2, n_head=4, d_model=128, block_size=256, seed=0)
    rng = np.random.default_rng(0)
    ids = list(rng.integers(0, cfg["vocab_size"], 256))
    positions = [8, 16, 32, 64, 96, 128, 160, 192, 224, 255]
    t_full, t_cache = [], []
    for t in positions:
        x = np.array(ids[:t + 1])[None]
        reps = 5
        t0 = time.perf_counter()
        for _ in range(reps):
            big.forward(x)
        t_full.append((time.perf_counter() - t0) / reps)
        _, cache = prefill(big, ids[:t])
        k0, v0 = list(cache.k), list(cache.v)
        t0 = time.perf_counter()
        for _ in range(reps):
            cache.k, cache.v = list(k0), list(v0)
            decode_step(big, ids[t], cache)
        t_cache.append((time.perf_counter() - t0) / reps)
    t_full, t_cache = np.array(t_full), np.array(t_cache)

    # (c) KV-cache memory: 2 (K and V) × layers × kv_heads × head_dim × bytes, per token
    models = [("GPT-2 small (12 layers, d=768)", 12, 12, 64, "#7f7f7f"),
              ("LLaMA-2-7B (32 layers, 32 KV heads × 128)", 32, 32, 128, C_ATT),
              ("LLaMA-2-70B (80 layers, GQA: 8 KV heads × 128)", 80, 8, 128, C_MLP)]
    ctx = np.array([1024, 2048, 4096, 8192, 16384, 32768])

    fig = plt.figure(figsize=(18, 6.4))
    ax = fig.add_axes([0.0, 0.05, 0.33, 0.85])
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    toks = list("The keepe")
    for i, c in enumerate(toks):
        y = 8.6 - i * 0.8
        box(ax, 1.0, y, f"'{show(c)}'", fc="#f5f5f5", w=1.0, h=0.6, fs=11)
        box(ax, 3.0, y, f"k_{i}", fc=F_ATT, ec=C_ATT, w=1.3, h=0.6, fs=11)
        box(ax, 4.6, y, f"v_{i}", fc=F_MLP, ec=C_MLP, w=1.3, h=0.6, fs=11)
    ax.add_patch(FancyBboxPatch((2.1, 1.9), 3.4, 7.1, boxstyle="round,pad=0.05", fc="none",
                                ec="#d4b106", lw=2.5, ls="--"))
    ax.text(3.8, 9.35, "KV cache (kept from earlier steps)", ha="center", fontsize=12,
            color="#8a6d00", weight="bold")
    y = 0.9
    box(ax, 1.0, y, "'r'", fc=F_RES, w=1.0, h=0.7, fs=12)
    box(ax, 3.0, y, "k_9", fc=F_ATT, ec=C_ATT, w=1.3, h=0.7, fs=11)
    box(ax, 4.6, y, "v_9", fc=F_MLP, ec=C_MLP, w=1.3, h=0.7, fs=11)
    box(ax, 7.6, y, "q_9", fc="#fff", ec=C_OUT, w=1.3, h=0.7, fs=12, lw=2.5)
    ax.text(3.8, 1.45, "new token: append", ha="center", fontsize=10.5, color="#555")
    arrow(ax, (7.6, 1.3), (5.6, 5.0), color=C_OUT, lw=2)
    ax.text(7.7, 3.4, "q_9 attends over\nALL cached k, v\n(t + 1 dot products)",
            fontsize=11, color=C_OUT, ha="left")
    ax.text(5.0, 0.05, "only the newest token goes through the layers", ha="center",
            fontsize=11.5, style="italic")

    ax = fig.add_axes([0.38, 0.12, 0.27, 0.72])
    ax.plot(positions, t_full * 1e3, "o-", color=C_OUT, lw=2.5, label="no cache: re-run all t tokens")
    ax.plot(positions, t_cache * 1e3, "o-", color=C_EMB, lw=2.5, label="KV cache: 1 token")
    ax.set_xlabel("position t of the new token")
    ax.set_ylabel("ms to get its logits (NumPy, measured)")
    ax.set_title("time per generated token\n(2 layers, d = 128, random weights)", fontsize=13)
    ax.legend(fontsize=10.5, loc="upper left")
    ax.text(positions[-1], t_full[-1] * 1e3 * 0.55, f"at t = {positions[-1]}:\n"
            f"{t_full[-1] / t_cache[-1]:.0f}× faster", ha="right", fontsize=11.5)

    ax = fig.add_axes([0.72, 0.12, 0.27, 0.72])
    for name, Ln, kvh, hd, col in models:
        per_tok = 2 * Ln * kvh * hd * 2  # bytes, fp16
        ax.plot(ctx, ctx * per_tok / 2 ** 30, "o-", color=col, lw=2.5, label=name)
    ax.set_xscale("log", base=2)
    ax.set_xticks(ctx)
    ax.set_xticklabels([f"{c // 1024}k" for c in ctx])
    ax.set_xlabel("context length (tokens)")
    ax.set_ylabel("KV cache per sequence (GiB, fp16)")
    ax.set_title("memory = 2 · n_layer · n_kv_heads · d_head\n· T · bytes", fontsize=13)
    ax.legend(fontsize=9.5, loc="upper left")
    ours = 2 * cfg["n_layer"] * cfg["block_size"] * cfg["d_model"] * 4
    fig.suptitle("The KV cache: keep every layer's keys and values, so each new token only "
                 "computes its own", y=1.0)
    fig.text(0.5, -0.04,
             f"Our model's whole cache: 2 × {cfg['n_layer']} layers × {cfg['block_size']} "
             f"tokens × {cfg['d_model']} × 4 bytes = {ours:,} bytes. LLaMA-2-7B needs 0.5 MB "
             "PER TOKEN (2 GiB at 4k context), which is why serving is memory-bound and why "
             "GQA/MQA share K, V across heads.", ha="center", fontsize=12)
    save(fig, "kv_cache.png")
    return positions[-1], t_full[-1], t_cache[-1]


def fig_why_transformers(res):
    names = ["Transformer", "GRU", "RNN"]
    cols = {"Transformer": C_EMB, "GRU": C_ATT, "RNN": "#7f7f7f"}
    n_max = int(res["n_max"])
    fig, axes = plt.subplots(2, 2, figsize=(17, 12))

    ax = axes[0, 0]
    for n in names:
        ax.plot(range(1, n_max + 1), res[f"{n}.by_n"] * 100, "o-", lw=2.5, color=cols[n],
                label=f"{n} ({int(res[f'{n}.params']):,} params): {float(res[f'{n}.acc']):.0%}")
    ax.axhline(100 / 8, color="grey", ls=":")
    ax.text(n_max, 100 / 8 + 2, "chance (1/8)", ha="right", color="grey")
    ax.set_xlabel("number of entries N stored before the questions")
    ax.set_ylabel("test accuracy (%)")
    ax.set_ylim(0, 105)
    ax.set_title("1) the more there is to remember, the more the RNNs forget")
    ax.legend(fontsize=11, loc="lower left", bbox_to_anchor=(0.0, 0.2))

    ax = axes[0, 1]
    bins = res["bins"]
    for n in names:
        ax.plot(bins + 3.5, res[f"{n}.by_d"] * 100, "o-", lw=2.5, color=cols[n], label=n)
    ax.axhline(100 / 8, color="grey", ls=":")
    ax.set_xticks(bins + 3.5)
    ax.set_xticklabels([f"{b}-{b + 7}" for b in bins])
    ax.set_xlabel("distance from the question back to its entry (tokens)")
    ax.set_ylabel("test accuracy (%)")
    ax.set_ylim(0, 105)
    ax.set_title("2) attention: distance is free (one lookup)")
    ax.legend(fontsize=11, loc="center", bbox_to_anchor=(0.5, 0.38))

    ax = axes[1, 0]
    att, x = res["att"], res["ex_x"]
    on = res["on_target"]
    li, hi = np.unravel_index(on.argmax(), on.shape)
    A = att[li, hi]
    n_entries = 8
    q_rows = np.arange(n_entries, n_entries + 12)
    M = A[q_rows][:, :n_entries + 12]
    ax.imshow(M, cmap="Oranges", vmin=0, vmax=1, aspect="auto")

    def name(t):
        t = int(t)
        return f"k{t // 8}={t % 8}" if t < 128 else f"k{t - 128}?"
    ax.set_xticks(range(n_entries + 12))
    ax.set_xticklabels([name(t) for t in x[:n_entries + 12]], rotation=90, fontsize=9.5,
                       family="monospace")
    ax.set_yticks(range(len(q_rows)))
    ax.set_yticklabels([name(x[r]) for r in q_rows], fontsize=10, family="monospace")
    ax.axvline(n_entries - 0.5, color="black", lw=1.5)
    ax.set_xlabel("key position (8 entries, then the questions)")
    ax.set_ylabel("query (a question)")
    ax.set_title(f"3) inside the Transformer: layer {li}, head {hi} looks up\n"
                 f"the matching entry ({on[li, hi]:.0%} of its attention on it)")
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)

    ax = axes[1, 1]
    sp = res["speed"]  # L, seq, par, gru, gpt, share
    ax.loglog(sp[:, 0], sp[:, 1] * 1e3, "o-", color=C_ATT, lw=2.5,
              label="token by token (RNN-style): T small matmuls")
    ax.loglog(sp[:, 0], sp[:, 2] * 1e3, "o-", color=C_EMB, lw=2.5,
              label="all tokens at once (Transformer-style): 1 matmul")
    ax.loglog(sp[:, 0], sp[:, 4] * 1e3, "s--", color="#444", lw=1.8,
              label="2-layer GPT forward (incl. T×T attention)")
    ax.set_xlabel("sequence length T")
    ax.set_ylabel("time (ms, CPU)")
    ax.set_title("4) same arithmetic, parallel is faster;\nattention's T² shows at long T")
    ax.legend(fontsize=10.5, loc="upper left")
    ratio = sp[:, 1] / sp[:, 2]
    i = len(sp) // 2
    ax.annotate(f"{ratio.min():.0f}–{ratio.max():.0f}× faster\nat every T",
                xy=(sp[i, 0], sp[i, 2] * 1e3), xytext=(sp[i, 0] * 1.25, sp[i, 2] * 1e3 * 0.2),
                fontsize=10.5, color=C_EMB, arrowprops=dict(arrowstyle="->", color=C_EMB))
    ax.text(0.98, 0.03, f"GPT: attention is {sp[0, 5]:.0%} of the FLOPs at T = {int(sp[0, 0])},"
            f"\n{sp[-1, 5]:.0%} at T = {int(sp[-1, 0])} (the T² term)", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=10.5, color="#444")
    ax.set_ylim(sp[:, 2].min() * 1e3 * 0.1, sp[:, 4].max() * 1e3 * 4)
    fig.suptitle("Why Transformers: attention looks things up directly, and trains in "
                 "parallel  (why_transformers.py)", y=1.0)
    fig.tight_layout()
    save(fig, "why_transformers.png")


def fig_residual_stream(model, data):
    T = model.config["block_size"]
    starts = np.arange(0, len(data) - T - 1, T)
    X = np.stack([data[i:i + T] for i in starts])
    Y = np.stack([data[i + 1:i + T + 1] for i in starts])
    tn = lambda z: np.linalg.norm(z, axis=-1).mean()  # mean norm per token

    x = model.wte.forward(X) + model.wpe.forward(np.arange(T))
    names, upd, stream = ["embed"], [tn(x)], [tn(x)]
    for l, blk in enumerate(model.blocks):
        a = blk.attn.forward(blk.ln_1.forward(x))
        x = x + a
        names.append(f"attn {l}")
        upd.append(tn(a))
        stream.append(tn(x))
        m = blk.mlp.forward(blk.ln_2.forward(x))
        x = x + m
        names.append(f"MLP {l}")
        upd.append(tn(m))
        stream.append(tn(x))

    model.forward(X, Y)
    dx = model.lm_head.backward(model.dlogits) if model.lm_head else model.dlogits @ \
        model.wte.params["weight"]
    dx = model.ln_f.backward(dx)
    gnames, grads, skip_share = [f"after MLP {len(model.blocks) - 1}"], [tn(dx)], []
    for l in reversed(range(len(model.blocks))):
        blk = model.blocks[l]
        branch = blk.ln_2.backward(blk.mlp.backward(dx))
        skip_share.append(tn(dx) / (tn(dx) + tn(branch)))
        dx = dx + branch
        gnames.append(f"before MLP {l}")
        grads.append(tn(dx))
        branch = blk.ln_1.backward(blk.attn.backward(dx))
        skip_share.append(tn(dx) / (tn(dx) + tn(branch)))
        dx = dx + branch
        gnames.append(f"before attn {l}")
        grads.append(tn(dx))

    fig = plt.figure(figsize=(17, 10.5))
    ax = fig.add_axes([0.02, 0.5, 0.96, 0.44])
    ax.axis("off")
    ax.set_xlim(-0.5, 17)
    ax.set_ylim(0.2, 5.2)
    ax.add_patch(FancyArrowPatch((0.2, 1.6), (16.6, 1.6), arrowstyle="simple,head_width=1.6,"
                                 "head_length=1.0,tail_width=0.9", mutation_scale=22,
                                 fc=F_RES, ec="#d4b106", lw=1.5, zorder=0))
    ax.text(8.4, 0.55, "the residual stream x  (one C-dim vector per token, flowing up the "
            "network)", ha="center", fontsize=13, color="#8a6d00", weight="bold")
    box(ax, 0.9, 3.7, "embed\nwte + wpe", fc=F_EMB, ec=C_EMB, w=1.6, h=1.0, fs=11)
    arrow(ax, (0.9, 3.2), (0.9, 1.9), color=C_EMB)
    xs = [3.4, 6.2, 9.0, 11.8]
    labels = ["LN → attention 0\n(talk to other tokens)", "LN → MLP 0\n(think per token)",
              "LN → attention 1", "LN → MLP 1"]
    for i, (xc, lab) in enumerate(zip(xs, labels)):
        fc, ec = (F_ATT, C_ATT) if i % 2 == 0 else (F_MLP, C_MLP)
        box(ax, xc + 0.6, 3.9, lab, fc=fc, ec=ec, w=2.4, h=1.1, fs=11, lw=2.5)
        arrow(ax, (xc - 0.2, 1.9), (xc - 0.2, 3.3), color="#555", lw=1.6)
        plus(ax, xc + 1.4, 1.6, r=0.28, fs=16)
        arrow(ax, (xc + 1.4, 3.3), (xc + 1.4, 1.9), color=ec, lw=2)
        if i == 0:
            ax.text(xc - 0.3, 2.5, "read", fontsize=10, ha="right", color="#555")
            ax.text(xc + 1.5, 2.5, "write\n(add)", fontsize=10, ha="left", color=ec)
    box(ax, 15.2, 3.7, "LN → head\n→ logits", fc=F_OUT, ec=C_OUT, w=1.8, h=1.0, fs=11)
    arrow(ax, (15.2, 1.9), (15.2, 3.2), color=C_OUT)
    ax.text(8.4, 4.95, "Every sublayer READS the stream (through a LayerNorm) and ADDS its "
            "result back:  x ← x + f(LN(x))", ha="center", fontsize=13.5)

    ax = fig.add_axes([0.06, 0.07, 0.4, 0.34])
    xpos = np.arange(len(names))
    ax.bar(xpos - 0.2, upd, width=0.4, color=[C_EMB, C_ATT, C_MLP, C_ATT, C_MLP],
           label="size of the update it writes ‖f(LN(x))‖")
    ax.bar(xpos + 0.2, stream, width=0.4, color=F_RES, ec="#d4b106",
           label="size of the stream after it ‖x‖")
    ax.set_xticks(xpos)
    ax.set_xticklabels(names)
    ax.set_ylabel("mean norm per token")
    ax.set_title("forward: each sublayer adds an update to x\n(trained model, averaged over "
                 "the whole text)", fontsize=12.5)
    ax.legend(fontsize=10.5, loc="upper left")

    ax = fig.add_axes([0.56, 0.07, 0.4, 0.34])
    g = np.array(grads) / grads[0]
    ax.bar(range(len(g)), g, color="#c6dbef", ec=C_EMB)
    ax.set_xticks(range(len(g)))
    ax.set_xticklabels(gnames, rotation=15, fontsize=10.5)
    ax.set_ylabel("‖dL/dx‖ relative to the top")
    for i, v in enumerate(g):
        ax.text(i, v + 0.03, f"{v:.2f}", ha="center", fontsize=10.5)
    ax.set_ylim(0, max(g) * 1.25)
    ax.set_title("backward: each '+' passes the gradient down untouched AND adds the\n"
                 "branch's share, so it cannot vanish (unlike BPTT in ../rnn/)", fontsize=12.5)
    fig.suptitle("The residual stream view of a Transformer", y=0.995)
    save(fig, "residual_stream.png")
    return names, upd, stream, g


def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    text, chars, data = load_text(DEFAULT_TEXT)
    V = len(chars)
    model, hist = get_model(data, V)
    final_eval = evaluate(model, data)
    print(f"  mini-GPT: {model.n_params():,} params, loss over the whole text {final_eval:.3f}")

    print("training the char-RNN from ../rnn (same settings as rnn_numpy.py, ~15 s)...")
    rnn_losses, rnn_times = train_rnn(text)

    res_path = os.path.join(HERE, "why_transformers.npz")
    if not os.path.exists(res_path):
        print("running why_transformers.py (~25 s)...")
        import why_transformers
        with contextlib.redirect_stdout(io.StringIO()):
            why_transformers.run(save=res_path)
    res = np.load(res_path)

    total = fig_architecture(model, chars)
    hd, prev_hd, w, wc = fig_self_attention(model, chars)
    print(f"  self_attention: layer 0 head {hd}, prev-token {prev_hd:.0%}, last row {w:.0%} "
          f"on '{wc}'")
    prev, dist = fig_attention_heads(model, chars, data)
    print("  heads prev-token: " + " ".join(f"{v:.0%}" for v in prev.ravel()) +
          " | look-back: " + " ".join(f"{v:.1f}" for v in dist.ravel()))
    near, far = fig_positions(model)
    print(f"  positions: learned cosine |i-j|=1 {near:+.2f}, |i-j|>8 {far:+.2f}")
    print(f"  causal_mask: max |full - prefix| = {fig_causal_mask(model, chars):.1e}")
    s64, m_raw, m_sc = fig_sqrt_dk()
    print(f"  sqrt_dk: d=64 raw std {s64:.1f}, max weight raw {m_raw:.0%} scaled {m_sc:.0%}")
    gpt_s, rnn_f, rnn_t, gpt_t = fig_training(hist, rnn_losses, rnn_times, V, final_eval)
    print(f"  training: GPT smoothed {gpt_s:.3f} in {gpt_t:.1f} s; RNN {rnn_f:.3f} in "
          f"{rnn_t:.1f} s")
    g2 = fig_params(model)
    print(f"  params: ours {total:,}; GPT-2 small {g2:,}")
    t, tf, tc = fig_kv_cache(model)
    print(f"  kv_cache: at t={t}: {tf * 1e3:.2f} ms vs {tc * 1e3:.2f} ms ({tf / tc:.0f}x)")
    fig_why_transformers(res)
    names, upd, stream, g = fig_residual_stream(model, data)
    print("  residual: updates " + " ".join(f"{n}={u:.2f}" for n, u in zip(names, upd)) +
          " | grads " + " ".join(f"{v:.2f}" for v in g))
    print(f"total time {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
