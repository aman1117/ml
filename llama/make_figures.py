"""
Draw the pictures used in llama/README.md and save them to llama/images/.

Every number in every figure is computed here or by the scripts of this folder:
  - the trained mini-LLaMA (llama/model.npz, trained here first if missing, ~6 s),
  - the mini-GPT of ../transformer trained with its default settings (loaded from
    ../transformer/model.npz if it is there, else trained in memory, ~10 s; nothing
    is written outside this folder),
  - llama/llama_pytorch.npz (val curves and ablation; runs llama_pytorch.py if missing, ~50 s),
  - llama/moe.npz (expert utilization; runs moe.py if missing, ~45 s),
  - small timing runs of the NumPy layers and of the sliding KV cache in generate.py.

Run (~5 s once the .npz files exist, ~2 min from scratch):
    python llama/make_figures.py
"""
import contextlib
import io
import math
import os
import subprocess
import sys
import time
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

from generate import (REAL_MODELS, context_strategies, decode_step, kv_bytes_per_token,  # noqa: E402
                      prefill)
from llama_numpy import (DEFAULT_TEXT, LLaMA, RMSNorm, get_trained_model, load_text,  # noqa: E402
                         rope_freqs)
import transformer_numpy  # noqa: E402  (../transformer, put on sys.path by llama_numpy)

warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "images")
DPI = 130

# Same colours as ../transformer/make_figures.py, plus one for "what changed".
C_EMB, C_ATT, C_MLP, C_LN, C_OUT = "#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#d62728"
F_EMB, F_ATT, F_MLP, F_LN, F_OUT, F_RES = ("#dbe9f6", "#ffe8cc", "#e3f4e1", "#ece4f5",
                                           "#fbe3e3", "#fdf1c7")
C_GPT, C_LLAMA, C_NEW = "#7f7f7f", "#1f77b4", "#d62728"

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


def smooth(x, window):
    c = np.cumsum(np.insert(np.asarray(x, dtype=float), 0, 0.0))
    return np.array([(c[i + 1] - c[max(0, i - window + 1)]) / (i + 1 - max(0, i - window + 1))
                     for i in range(len(x))])


def silu(x):
    return x / (1 + np.exp(-x))


def gelu(x):
    return 0.5 * x * (1 + np.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))


# ---------------------------------------------------------------- data helpers

def gpt_history(data, V):
    """Training history of ../transformer's GPT with its default settings."""
    path = os.path.join(HERE, "..", "transformer", "model.npz")
    want = dict(vocab_size=V, **transformer_numpy.DEFAULT_CONFIG)
    if os.path.exists(path):
        model, _, hist = transformer_numpy.load_model(path)
        if model.config == want and hist and "time" in hist and len(hist["loss"]) == 2000:
            print("loaded ../transformer/model.npz (default settings)")
            return model, hist
    print("training ../transformer's GPT with its default settings in memory (~10 s)...")
    with contextlib.redirect_stdout(io.StringIO()):
        model, hist = transformer_numpy.train(data, V, verbose=False)
    return model, {k: np.array(v) for k, v in hist.items()}


def results(name, script):
    path = os.path.join(HERE, name)
    if not os.path.exists(path):
        print(f"no llama/{name} yet: running python llama/{script} ...")
        subprocess.run([sys.executable, os.path.join(HERE, script)], check=True,
                       stdout=subprocess.DEVNULL)
    return np.load(path)


# ---------------------------------------------------------------- figures

def fig_architecture(llama, gpt):
    V, C, L = llama.config["vocab_size"], llama.config["d_model"], llama.config["n_layer"]
    nh, nkv, H = llama.config["n_head"], llama.config["n_kv_head"], llama.config["ffn_hidden"]
    fig, ax = plt.subplots(figsize=(17, 12.5))
    ax.set_xlim(0, 20)
    ax.set_ylim(-1.0, 16.4)
    ax.axis("off")
    cols = [(5.0, "GPT-2  (../transformer/)", C_GPT), (15.0, "LLaMA  (this folder)", C_LLAMA)]
    for x, name, col in cols:
        ax.text(x, 16.0, name, ha="center", fontsize=17, weight="bold", color=col)
    W = 7.4
    gx, lx = cols[0][0], cols[1][0]
    # inputs and embeddings
    for x in (gx, lx):
        box(ax, x, 0.3, "token ids (B, T)", fc="#f5f5f5", w=4.0, h=0.7)
        arrow(ax, (x, 0.65), (x, 1.25))
    box(ax, gx, 1.9, f"wte[ids] + wpe[0..T-1]\ntoken ({V}×{C}) + learned position ({32}×{C})",
        fc=F_EMB, ec=C_EMB, w=W, h=1.2, lw=2.5)
    box(ax, lx, 1.9, f"tok_embeddings[ids]   ({V}×{C})\nno position table",
        fc=F_EMB, ec=C_NEW, w=W, h=1.2, lw=3)
    # the block frame
    for x in (gx, lx):
        ax.add_patch(FancyBboxPatch((x - 4.6, 3.0), 9.2, 9.0,
                                    boxstyle="round,pad=0.1,rounding_size=0.3",
                                    fc="#fafafa", ec="grey", ls="--", lw=1.5, zorder=0))
        ax.text(x + 4.5, 11.75, f"× {L} blocks", ha="right", va="top", fontsize=12,
                weight="bold", color="#555")
        arrow(ax, (x, 2.5), (x, 3.55))
    rows = [  # y, gpt text, llama text, face, edge, changed
        (4.0, "LayerNorm  γ ⊙ (x − μ)/σ + β", "RMSNorm  γ ⊙ x / rms(x)", F_LN, C_LN),
        (6.0, f"multi-head attention, {nh} heads\nq, k, v = x W_qkv + b_qkv\n"
              "softmax(q kᵀ/√hs + mask) v → W_o + b",
         f"grouped-query attention: {nh} query heads,\n{nkv} key/value heads, no biases\n"
         "q, k ROTATED by RoPE (angle m·θ_i)", F_ATT, C_ATT),
        (7.75, "x = x + Attn(LN(x))", "x = x + Attn(RMSNorm(x))", F_RES, "#d4b106"),
        (8.9, "LayerNorm  γ, β", "RMSNorm  γ", F_LN, C_LN),
        (10.4, f"MLP: GELU(x W₁ + b₁) W₂ + b₂\nhidden 4·d = {4 * C}, 2 matrices",
         f"SwiGLU: (silu(x w1) ⊙ x w3) w2\nhidden ≈ 8/3·d = {H}, 3 matrices, no biases",
         F_MLP, C_MLP),
        (11.55, "x = x + MLP(LN(x))", "x = x + SwiGLU(RMSNorm(x))", F_RES, "#d4b106"),
    ]
    for i, (y, gt, lt, fc, ec) in enumerate(rows):
        h = 1.55 if "\n" in gt else 0.62
        if "x = x" in gt:
            for x, t in ((gx, gt), (lx, lt)):
                ax.text(x, y, t, ha="center", va="center", fontsize=11.5, style="italic",
                        bbox=dict(boxstyle="round", fc=F_RES, ec="#d4b106"))
            continue
        box(ax, gx, y, gt, fc=fc, ec=ec, w=W, h=h, lw=2, fs=11.5)
        box(ax, lx, y, lt, fc=fc, ec=C_NEW, w=W, h=h, lw=3, fs=11.5)
    for x in (gx, lx):
        for a, b in ((4.3, 5.2), (6.8, 7.5), (8.0, 8.6), (9.2, 9.6), (11.2, 11.35)):
            arrow(ax, (x, a), (x, b), lw=1.5, ms=12)
        arrow(ax, (x, 11.8), (x, 12.65))
    box(ax, gx, 13.0, "final LayerNorm", fc=F_LN, ec=C_LN, w=W, h=0.62, lw=2)
    box(ax, lx, 13.0, "final RMSNorm", fc=F_LN, ec=C_NEW, w=W, h=0.62, lw=3)
    box(ax, gx, 14.2, f"lm_head ({C}×{V}) → logits", fc=F_OUT, ec="#555", w=W, h=0.62, lw=2)
    box(ax, lx, 14.2, f"output ({C}×{V}) → logits", fc=F_OUT, ec="#555", w=W, h=0.62, lw=2)
    for x in (gx, lx):
        arrow(ax, (x, 13.32), (x, 13.88))
    ax.text(gx, 15.2, f"{gpt.n_params():,} parameters", ha="center", fontsize=13)
    ax.text(lx, 15.2, f"{llama.n_params():,} parameters", ha="center", fontsize=13)
    ax.text(10.0, 6.0, "→", fontsize=40, ha="center", va="center", color=C_NEW)
    ax.text(10.0, 10.4, "→", fontsize=40, ha="center", va="center", color=C_NEW)
    ax.text(10.0, -0.8, "Red outline = changed. The skeleton (embeddings → pre-norm residual "
            "blocks → final norm → logits) is identical.", ha="center", fontsize=12.5,
            color=C_NEW)
    ax.set_title("From GPT-2 to LLaMA: same skeleton, four component swaps", fontsize=16)
    save(fig, "architecture_diff.png")


def fig_rmsnorm(model, data):
    rng = np.random.default_rng(0)
    pts = rng.standard_normal((300, 2)) * [1.0, 0.6] + [1.5, 0.4]
    rms_out = pts / np.sqrt((pts ** 2).mean(1, keepdims=True))
    ln_out = (pts - pts.mean(1, keepdims=True)) / pts.std(1, keepdims=True)

    # measured: the inputs of every norm of the trained model, over the whole text
    T = model.config["block_size"]
    starts = np.arange(0, len(data) - T - 1, T)
    model.forward(np.stack([data[i:i + T] for i in starts]))
    names, ratios, cosines = [], [], []
    for l, blk in enumerate(model.blocks):
        for nm, norm in (("attn", blk.attention_norm), ("ffn", blk.ffn_norm)):
            x = norm.xhat / norm.rstd  # recover the input x
            mu = x.mean(-1)
            rms = np.sqrt((x ** 2).mean(-1))
            ratios.append(np.abs(mu / rms).ravel())
            xc = x - x.mean(-1, keepdims=True)
            cos = (x * xc).sum(-1) / (np.linalg.norm(x, axis=-1) * np.linalg.norm(xc, axis=-1))
            cosines.append(cos.ravel())
            names.append(f"layer {l}\n{nm}_norm")

    # timing, our NumPy layers, forward + backward
    x = rng.standard_normal((8192, 512)).astype(np.float32)
    dy = rng.standard_normal(x.shape).astype(np.float32)
    times = {}
    for name, layer in (("LayerNorm", transformer_numpy.LayerNorm(512)), ("RMSNorm", RMSNorm(512))):
        layer.forward(x)
        layer.backward(dy)
        t0 = time.perf_counter()
        for _ in range(20):
            layer.forward(x)
            layer.backward(dy)
        times[name] = (time.perf_counter() - t0) / 20

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.8), gridspec_kw=dict(width_ratios=[1, 1.3, 0.8]))
    ax = axes[0]
    ax.scatter(*pts.T, s=8, color="#bbbbbb", label="inputs x (C = 2)")
    ax.add_patch(plt.Circle((0, 0), math.sqrt(2), fill=False, ls="--", color=C_EMB, lw=1))
    ax.scatter(*rms_out.T, s=14, color=C_EMB, label="RMSNorm(x): on the circle |y| = √C,\n"
               "direction kept")
    ax.scatter(*ln_out.T, s=120, color=C_ATT, marker="X", zorder=5,
               label="LayerNorm(x): ALL inputs land on\n(+1, −1) or (−1, +1)")
    ax.set_aspect("equal")
    ax.set_xlim(-2.5, 4.5)
    ax.set_ylim(-2.5, 3.2)
    ax.axhline(0, color="#ddd", lw=0.8)
    ax.axvline(0, color="#ddd", lw=0.8)
    ax.legend(fontsize=9.5, loc="upper right")
    ax.set_title("what each norm does (C = 2 toy)")

    ax = axes[1]
    ax.boxplot(ratios, showfliers=False, whis=(5, 95))
    ax.set_xticks(range(1, len(names) + 1))
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel("|mean(x)| / rms(x) per token")
    med = [np.median(r) for r in ratios]
    cmed = [np.median(c) for c in cosines]
    top = max(np.percentile(r, 95) for r in ratios)
    ax.set_ylim(0, top * 1.3)
    for i, (m, c) in enumerate(zip(med, cmed)):
        ax.text(i + 1, top * 1.12, f"median {m:.2f}\ncos {c:.3f}", ha="center",
                fontsize=9.5)
    ax.set_title("trained mini-LLaMA: the inputs of its RMSNorms\nhave a small mean "
                 "(box 25–75%, whiskers 5–95%)")

    ax = axes[2]
    ax.bar(list(times), [t * 1e3 for t in times.values()], color=[C_ATT, C_EMB])
    for i, t in enumerate(times.values()):
        ax.text(i, t * 1e3, f"{t * 1e3:.1f} ms", ha="center", va="bottom", fontsize=12)
    ax.set_ylabel("ms, forward + backward")
    ax.set_title(f"cost (our NumPy layers,\n8192 tokens × 512 features)")
    fig.suptitle("RMSNorm = LayerNorm without the mean: keep only the rescaling", y=1.02)
    fig.text(0.5, -0.06,
             "Middle: 'cos' = cosine between x and x − mean(x), i.e. how little re-centering "
             "changes the direction RMSNorm sees.\nRight: one reduction instead of two, and "
             "no β. Zhang & Sennrich (2019) report 7–64% lower running time than LayerNorm "
             "across their models.", ha="center", fontsize=11.5)
    fig.tight_layout()
    save(fig, "rmsnorm_vs_layernorm.png")
    return med, cmed, times


def fig_rope_rotation(hs=16):
    rng = np.random.default_rng(3)
    th = rope_freqs(hs)
    q = rng.standard_normal(hs)
    k = rng.standard_normal(hs)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), gridspec_kw=dict(width_ratios=[1, 1, 1.25]))
    for ax, i in zip(axes[:2], (0, 2)):
        a, b = q[2 * i], q[2 * i + 1]
        cmap = plt.get_cmap("viridis")
        r = math.hypot(a, b)
        ax.add_patch(plt.Circle((0, 0), r, fill=False, ls="--", color="#ccc"))
        for m in range(8):
            c, s = math.cos(m * th[i]), math.sin(m * th[i])
            x, y = a * c - b * s, a * s + b * c
            ax.annotate("", xy=(x, y), xytext=(0, 0),
                        arrowprops=dict(arrowstyle="-|>", color=cmap(m / 7), lw=2.2))
            ax.text(x * 1.13, y * 1.13, f"m={m}", ha="center", va="center", fontsize=10,
                    color=cmap(m / 7))
        ax.set_aspect("equal")
        lim = r * 1.35
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel(f"q[{2 * i}]")
        ax.set_ylabel(f"q[{2 * i + 1}]")
        ax.set_title(f"pair {i}: θ_{i} = {th[i]:.2f} rad/position\n"
                     f"({math.degrees(th[i]):.1f}° per step, one turn every "
                     f"{2 * math.pi / th[i]:.0f} positions)", fontsize=13)
    N = 32
    cos = lambda m: (np.cos(m * th), np.sin(m * th))  # noqa: E731

    def at(v, m):
        c, s = cos(m)
        out = np.empty_like(v)
        out[0::2] = v[0::2] * c - v[1::2] * s
        out[1::2] = v[0::2] * s + v[1::2] * c
        return out

    S = np.array([[at(q, m) @ at(k, n) for n in range(N)] for m in range(N)])
    ax = axes[2]
    im = ax.imshow(S, cmap="RdBu_r", vmin=-np.abs(S).max(), vmax=np.abs(S).max())
    ax.set_xlabel("key position n")
    ax.set_ylabel("query position m")
    ax.set_title("score (R_m q)·(R_n k) for the SAME q, k at every\n(m, n): constant along "
                 "each diagonal m − n", fontsize=13)
    plt.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"RoPE: rotate each pair of dimensions by position × θ_i  "
                 f"(head size {hs}, θ_i = 10000^(−2i/{hs}))", y=1.0)
    fig.tight_layout()
    save(fig, "rope_rotation.png")
    return float(np.abs(np.diff(np.diagonal(S, offset=-3))).max())


def yarn_multiplier(th, L_train, s, alpha=1.0, beta=32.0):
    """YaRN 'NTK-by-parts': pairs that turn many times within the training context keep
    θ; pairs that turn less than once are interpolated (θ/s); a linear ramp in between."""
    r = L_train * th / (2 * math.pi)  # turns per training context
    gamma = np.clip((r - alpha) / (beta - alpha), 0, 1)
    new = (1 - gamma) * th / s + gamma * th
    return th / new


def fig_rope_frequencies(hs=128):
    pairs = np.arange(hs // 2)
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.8))
    ax = axes[0]
    for base, col, name in ((1e4, C_EMB, "base 10,000 (LLaMA 1/2, Mistral v0.1)"),
                            (5e5, C_ATT, "base 500,000 (Llama 3)")):
        ax.plot(pairs, 2 * math.pi / rope_freqs(hs, base), color=col, lw=2.5, label=name)
    for ctx, ls in ((4096, "--"), (8192, ":")):
        ax.axhline(ctx, color=C_OUT, ls=ls, lw=1.3)
        ax.text(1, ctx * 1.25, f"{ctx // 1024}k context", color=C_OUT, fontsize=10.5)
    ax.set_yscale("log")
    ax.set_xlabel("pair index i (dims 2i, 2i+1)")
    ax.set_ylabel("wavelength 2π/θ_i  (positions per full turn)")
    ax.set_title(f"frequency spectrum (head size {hs})")
    ax.legend(fontsize=10, loc="upper left")

    ax = axes[1]
    dist = np.unique(np.round(np.logspace(0, 5.3, 600)).astype(int))
    for base, col in ((1e4, C_EMB), (5e5, C_ATT)):
        th = rope_freqs(hs, base)
        vals = np.array([np.cos(d * th).mean() for d in dist])
        ax.plot(dist, vals, color=col, lw=1.8, label=f"base {base:,.0f}")
    ax.axhline(0, color="#999", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("distance m − n (tokens)")
    ax.set_ylabel("mean_i cos((m − n)·θ_i)")
    ax.set_title("long-range decay: score of q = k\nvs distance (1 at distance 0)")
    ax.legend(fontsize=10)

    ax = axes[2]
    th = rope_freqs(hs, 1e4)
    s, L = 4, 4096
    ntk = rope_freqs(hs, 1e4 * s ** (hs / (hs - 2)))
    ax.plot(pairs, np.full(len(pairs), s), color=C_ATT, lw=2.5,
            label=f"position interpolation: every θ_i / {s}")
    ax.plot(pairs, th / ntk, color=C_MLP, lw=2.5, label="NTK-aware: base × s^(d/(d−2))")
    ax.plot(pairs, yarn_multiplier(th, L, s), color=C_LN, lw=2.5,
            label="YaRN (NTK-by-parts, α=1, β=32)")
    ax.axhline(1, color="#999", lw=0.8)
    ax.set_xlabel("pair index i (fast → slow)")
    ax.set_ylabel("wavelength multiplier vs training")
    ax.set_title(f"context extension {L // 1024}k → {s * L // 1024}k (s = {s}):\n"
                 "which pairs get stretched")
    ax.legend(fontsize=10, loc="center right")
    fig.suptitle("RoPE frequencies: fast pairs resolve neighbours, slow pairs span the "
                 "context; extension rescales them", y=1.02)
    fig.tight_layout()
    save(fig, "rope_frequencies.png")


def fig_swiglu(model, data):
    x = np.linspace(-5, 4, 400)
    T = model.config["block_size"]
    starts = np.arange(0, len(data) - T - 1, T)
    model.forward(np.stack([data[i:i + T] for i in starts]))
    gates = [blk.feed_forward.silu.ravel() for blk in model.blocks]

    fig, axes = plt.subplots(1, 3, figsize=(18.5, 5.6), gridspec_kw=dict(width_ratios=[1, 1, 1.1]))
    ax = axes[0]
    ax.plot(x, np.maximum(x, 0), color="#999", lw=2, label="ReLU")
    ax.plot(x, gelu(x), color=C_ATT, lw=2.5, label="GELU (GPT-2)")
    ax.plot(x, silu(x), color=C_EMB, lw=2.5, label="SiLU = Swish₁ = x·σ(x) (LLaMA's gate)")
    xm = -1.2785
    ax.plot([xm], [silu(xm)], "o", color=C_EMB)
    ax.annotate(f"min {silu(xm):.3f} at x = {xm:.2f}", (xm, silu(xm)), (-4.8, 1.2),
                arrowprops=dict(arrowstyle="->"), fontsize=10.5)
    ax.axhline(0, color="#ddd", lw=0.8)
    ax.set_xlabel("pre-activation")
    ax.legend(fontsize=10, loc="upper left")
    ax.set_title("the activations: smooth, slightly negative")

    ax = axes[1]
    a = np.linspace(-4, 4, 200)
    A, Bv = np.meshgrid(a, a)
    Z = silu(A) * Bv
    lim = np.abs(Z).max()
    im = ax.imshow(Z, extent=[-4, 4, -4, 4], origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim)
    ax.contour(A, Bv, Z, levels=[-2, -0.5, 0.5, 2], colors="k", linewidths=0.6)
    ax.set_xlabel("gate  a = x·w1")
    ax.set_ylabel("value  b = x·w3")
    ax.set_title("one SwiGLU hidden unit: silu(a) · b\n(a ≪ 0 closes the gate whatever b is)")
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[2]
    bins = np.linspace(-0.3, max(g.max() for g in gates) + 0.05, 80)
    stats = []
    for l, (g, col) in enumerate(zip(gates, (C_EMB, C_ATT))):
        neg, big, med = np.mean(g < 0), np.mean(g > 1), np.median(g)
        stats.append((neg, big, med))
        ax.hist(g, bins=bins, histtype="step", lw=2, color=col, density=True,
                label=f"layer {l}: median {med:.2f}, {neg:.0%} < 0, {big:.0%} > 1")
    ax.set_yscale("log")
    ax.set_xlabel("gate value silu(x·w1)")
    ax.set_ylabel("density (log)")
    ax.legend(fontsize=10)
    negs = [st[0] for st in stats]
    ax.set_title(f"trained mini-LLaMA, whole text: most gates sit\nnear 0 ({min(negs):.0%}–"
                 f"{max(negs):.0%} slightly negative), a few open wide")
    fig.suptitle("SwiGLU: (silu(x w1) ⊙ x w3) w2. A learned, input-dependent gate on every "
                 "hidden unit", y=1.02)
    fig.text(0.5, -0.05, f"Parameters: GPT's MLP has 2 matrices of d × 4d = 8d²; SwiGLU has 3 of "
             f"d × (8/3)d = 8d². Same budget, so the hidden size shrinks to 8/3·d "
             f"(ours: 64 → {model.config['ffn_hidden']}).", ha="center", fontsize=12)
    fig.tight_layout()
    save(fig, "swiglu_vs_gelu.png")
    return stats


def fig_gqa():
    fig = plt.figure(figsize=(18, 6.6))
    ax = fig.add_axes([0.0, 0.0, 0.45, 0.92])
    ax.axis("off")
    ax.set_xlim(-0.5, 12.0)
    ax.set_ylim(-0.3, 10.2)
    for row, (name, nkv) in enumerate((("MHA  (GPT-2, LLaMA-2-7B)", 8),
                                       ("GQA  (LLaMA-2-70B, Llama 3, Mistral)", 2),
                                       ("MQA  (PaLM, Falcon-7B)", 1))):
        y = 8.6 - row * 3.4
        ax.text(-0.3, y + 1.05, f"{name}: 8 query heads, {nkv} K/V head{'s' * (nkv > 1)}",
                fontsize=12.5, weight="bold")
        g = 8 // nkv
        for h in range(8):
            box(ax, 0.6 + h * 1.25, y, f"q{h}", fc=F_EMB, ec=C_EMB, w=0.95, h=0.6, fs=11)
        for kv in range(nkv):
            xc = 0.6 + (kv * g + (g - 1) / 2) * 1.25
            w = g * 1.25 - 0.3
            box(ax, xc, y - 1.5, f"K, V head {kv}" if g > 1 else f"kv{kv}", fc=F_ATT, ec=C_ATT,
                w=w, h=0.6, fs=11)
            for h in range(kv * g, (kv + 1) * g):
                ax.plot([0.6 + h * 1.25, xc], [y - 0.3, y - 1.2], color="#999", lw=1)
        ax.text(11.9, y - 1.5, f"cache × {nkv}/8", ha="right", va="center", fontsize=12,
                color=C_OUT)

    ax = fig.add_axes([0.53, 0.1, 0.46, 0.75])
    names, v4, v32 = [], [], []
    for name, V, d, L, nh, nkv, ffn, sw in REAL_MODELS:
        per = kv_bytes_per_token(L, nkv, d // nh)
        names.append(name.replace(" v0.1", "").replace("(70B if it were MHA)", "70B as MHA\n(hypothetical)"))
        v4.append(per * 4096 / 2 ** 30)
        v32.append(per * (32768 if sw is None else min(32768, sw)) / 2 ** 30)
    xs = np.arange(len(names))
    b1 = ax.bar(xs - 0.2, v4, 0.4, color=C_EMB, label="4k tokens")
    b2 = ax.bar(xs + 0.2, v32, 0.4, color=C_ATT, label="32k tokens")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() * 1.08, f"{b.get_height():g}",
                    ha="center", fontsize=10)
    ax.set_yscale("log")
    ax.set_ylim(0.2, 200)
    ax.set_xticks(xs)
    ax.set_xticklabels(names, fontsize=10.5)
    ax.set_ylabel("KV cache per sequence, GiB (fp16, log)")
    ax.legend(fontsize=11, loc="upper left")
    ax.set_title("2 · n_layer · n_kv_head · head_size · T · 2 bytes   (published configs)",
                 fontsize=13)
    ax.text(3, 40, "Mistral: rolling\n4096-token window\n→ capped", ha="center", fontsize=10,
            color="#555")
    fig.suptitle("Grouped-query attention: groups of query heads share one K/V head, "
                 "so the KV cache shrinks n_head / n_kv_head ×", y=1.0)
    save(fig, "gqa.png")


def fig_ablation(res):
    names = [str(n) for n in res["ablation.names"]]
    val, sd, params = res["ablation.val"], res["ablation.spread"], res["ablation.params"]
    steps = int(res["ablation.steps"])
    fig, ax = plt.subplots(figsize=(15, 6))
    xs = np.arange(len(names))
    cols = [C_GPT] + [C_LLAMA] * (len(names) - 1)
    ax.bar(xs, val, yerr=sd, capsize=6, color=cols, alpha=0.85)
    for i, (v, s, p) in enumerate(zip(val, sd, params)):
        ax.text(i, v + s + 0.004, f"{v:.3f}", ha="center", fontsize=12, weight="bold")
        ax.text(i, 0.805, f"{p:,}\nparams", ha="center", fontsize=10.5, color="white")
        if i:
            d = v - val[i - 1]
            ax.text(i - 0.5, max(v, val[i - 1]) + 0.035, f"{d:+.3f}", ha="center",
                    fontsize=11, color=C_MLP if d < 0 else C_OUT)
    ax.axhline(val[0], color=C_GPT, ls="--", lw=1)
    ax.set_xticks(xs)
    ax.set_xticklabels(names, fontsize=11.5)
    ax.set_ylim(0.78, max(val + sd) + 0.07)
    ax.set_ylabel("validation loss per char (lower is better)")
    ax.set_title(f"Ablation, one change at a time ({steps} steps, same batches, mean ± std "
                 f"of 3 seeds, dropout 0.1)")
    fig.text(0.5, -0.03, "Green/red numbers: change from the previous bar. Only RoPE's gain "
             "is clearly larger than the seed-to-seed spread at this tiny scale (3.5k "
             "training characters).", ha="center", fontsize=12)
    fig.tight_layout()
    save(fig, "ablation.png")


def fig_training(gpt_hist, llama_hist, gpt_final, llama_final, pt):
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.8))
    for ax, key, xlabel in ((axes[0], None, "training step (same batches for both)"),
                            (axes[1], "time", "wall-clock seconds (NumPy, this laptop)")):
        for h, col, name in ((gpt_hist, C_GPT, "mini-GPT (../transformer)"),
                             (llama_hist, C_LLAMA, "mini-LLaMA (this folder)")):
            loss = np.asarray(h["loss"])
            xx = np.arange(len(loss)) if key is None else np.asarray(h[key])
            ax.plot(xx, loss, color=col, lw=0.3, alpha=0.5)
            ax.plot(xx, smooth(loss, 50), color=col, lw=2.5, label=name)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("training loss per char")
        ax.set_ylim(0, 3.9)
        ax.legend(fontsize=11)
    axes[0].set_title("NumPy, train loss per step")
    axes[1].set_title("NumPy, train loss per second")
    axes[0].text(0.97, 0.55, f"loss over the whole text after 2000 steps:\n"
                 f"GPT {gpt_final:.3f}   LLaMA {llama_final:.3f}", transform=axes[0].transAxes,
                 ha="right", fontsize=11, bbox=dict(boxstyle="round", fc="white", ec="#bbb"))

    ax = axes[2]
    for name, col in (("GPT", C_GPT), ("LLaMA", C_LLAMA)):
        steps = pt[f"{name}.step"][0]
        va, tr = pt[f"{name}.val"], pt[f"{name}.train"]
        ax.plot(steps, va.mean(0), color=col, lw=2.5, label=f"{name} val")
        ax.fill_between(steps, va.min(0), va.max(0), color=col, alpha=0.2)
        ax.plot(steps, tr.mean(0), color=col, lw=1.5, ls="--", label=f"{name} train")
    ax.set_ylim(0, 2.5)
    ax.set_xlabel("step")
    ax.set_ylabel("loss per char")
    ax.set_title("PyTorch, matched params (~107k), held-out\nlast 10%: mean and range of 3 seeds")
    ax.legend(fontsize=10.5, ncol=2)
    fig.suptitle("mini-GPT vs mini-LLaMA on the same text and the same batches", y=1.02)
    fig.tight_layout()
    save(fig, "training_curves.png")


def fig_moe(res):
    E = res["share_aux"].shape[1]
    fig, axes = plt.subplots(1, 4, figsize=(21, 5.4), gridspec_kw=dict(width_ratios=[1, 1, 1.1, 0.9]))
    for l in range(2):
        ax = axes[l]
        xs = np.arange(E)
        ax.bar(xs - 0.2, res["share_noaux"][l] * 100, 0.4, color=C_OUT, label="no aux loss")
        ax.bar(xs + 0.2, res["share_aux"][l] * 100, 0.4, color=C_MLP, label="+ aux loss (0.01)")
        ax.axhline(100 / E, color="k", ls="--", lw=1)
        ax.text(E - 0.5, 100 / E + 0.5, "uniform", ha="right", fontsize=10)
        ax.set_xticks(xs)
        ax.set_xlabel("expert")
        ax.set_ylabel("% of top-2 assignments")
        cv = [res[k][l].std() / res[k][l].mean() for k in ("share_noaux", "share_aux")]
        ax.set_title(f"layer {l}: utilization\n(CV {cv[0]:.2f} without, {cv[1]:.2f} with)")
        ax.legend(fontsize=10, loc="upper left")
        ax.set_ylim(0, max(res["share_noaux"].max(), res["share_aux"].max()) * 118)
    ax = axes[2]
    for key, col, name in (("maxload_noaux", C_OUT, "no aux loss"),
                           ("maxload_aux", C_MLP, "+ aux loss")):
        ax.plot(smooth(res[key] * 100, 25), color=col, lw=2.2, label=name)
    ax.axhline(100 / E, color="k", ls="--", lw=1)
    ax.set_xlabel("training step")
    ax.set_ylabel("busiest expert's share, layer 0 (%)")
    ax.set_title("load of the busiest expert\nduring training (per batch)")
    ax.legend(fontsize=10)
    ax = axes[3]
    dense, total, active = res["params"]
    bars = ax.bar(["dense\nLLaMA", "MoE\ntotal", "MoE\nactive/token"], [dense, total, active],
                  color=[C_GPT, C_ATT, C_EMB])
    for b, v in zip(bars, (dense, total, active)):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,}", ha="center", va="bottom", fontsize=11)
    vals = res["val"]
    ax.set_title(f"parameters\nval loss: dense {vals[0]:.3f}, MoE {vals[1]:.3f} / {vals[2]:.3f}")
    ax.set_ylim(0, total * 1.15)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1e3:.0f}k"))
    fig.suptitle(f"Mixture of experts: {E} SwiGLU experts, top-2 routing. Without the "
                 "load-balancing loss the router plays favourites", y=1.03)
    fig.tight_layout()
    save(fig, "moe_routing.png")


def fig_sliding(model, data):
    W, Tm = 8, 20
    i, j = np.arange(Tm)[:, None], np.arange(Tm)[None, :]
    causal = (j <= i).astype(float)
    band = ((j <= i) & (i - j < W)).astype(float)
    strat = context_strategies(model, data, model.config["block_size"])

    big = LLaMA(model.config["vocab_size"], n_layer=2, n_head=4, n_kv_head=2, d_model=128,
                block_size=256, dtype=np.float32)
    ids = list(np.random.default_rng(0).integers(0, 44, 768))
    cache = prefill(big, ids[:1], 256)[1]
    t_cache, t_full = [], []
    for p in range(1, 768):
        t0 = time.perf_counter()
        decode_step(big, ids[p], cache)
        t_cache.append(time.perf_counter() - t0)
    for p in range(8, 768, 24):
        x = np.array(ids[max(0, p + 1 - 256):p + 1])[None]
        t0 = time.perf_counter()
        for _ in range(3):
            big.forward(x)
        t_full.append((p, (time.perf_counter() - t0) / 3))

    fig, axes = plt.subplots(1, 3, figsize=(19, 5.9), gridspec_kw=dict(width_ratios=[1, 1.2, 1.1]))
    ax = axes[0]
    img = np.where(band > 0, 2, np.where(causal > 0, 1, 0))
    ax.imshow(img, cmap=matplotlib.colors.ListedColormap(["white", "#e8e8e8", C_EMB]))
    last = Tm - 1
    for p in range(last - W + 1, last + 1):
        ax.text(p, last, str(p % W), ha="center", va="center", fontsize=10, color="white",
                weight="bold")
    ax.set_xticks(range(0, Tm, 2))
    ax.set_yticks(range(0, Tm, 2))
    ax.set_xlabel(f"key position p\n(white numbers: its rolling-buffer slot p mod {W})")
    ax.set_ylabel("query position")
    ax.set_title(f"sliding-window attention, window {W}:\nblue = attended, grey = causal "
                 "but evicted")

    ax = axes[1]
    names = list(strat)
    vals = [strat[n] for n in names]
    cols = [C_EMB, C_GPT, C_ATT, C_OUT]
    bars = ax.barh(np.arange(len(names))[::-1], vals, color=cols)
    for b, v in zip(bars, vals):
        ax.text(v * 1.1, b.get_y() + b.get_height() / 2, f"{v:.3f}", va="center", fontsize=11)
    ax.set_yticks(np.arange(len(names))[::-1])
    ax.set_yticklabels([n.replace(" (", "\n(") for n in names], fontsize=10.5)
    ax.set_xscale("log")
    ax.set_xlim(0.005, 30)
    ax.axvline(math.log(44), color=C_OUT, ls="--", lw=1)
    ax.text(math.log(44) * 1.08, len(names) - 0.75, "uniform\nln 44", fontsize=9.5,
            color=C_OUT)
    ax.set_xlabel("loss per char, targets at position ≥ 32 of the text (log)")
    ax.set_title("trained mini-LLaMA (window 32) reading the\nwhole 3.9k-char text, 4 ways")

    ax = axes[2]
    ax.plot(np.arange(1, 768), np.array(t_cache) * 1e3, color=C_EMB, lw=0.6, alpha=0.6)
    ax.plot(np.arange(1, 768), smooth(np.array(t_cache) * 1e3, 25), color=C_EMB, lw=2.5,
            label="sliding KV cache (decode_step)")
    p, t = zip(*t_full)
    ax.plot(p, np.array(t) * 1e3, "o-", color=C_OUT, lw=2, label="no cache: re-run last ≤256")
    ax.axvline(256, color="#999", ls="--", lw=1)
    ax.text(266, 0.6, "window full:\nslide, never re-prefill", fontsize=10.5)
    ax.set_xlabel("position of the new token")
    ax.set_ylabel("ms per token (NumPy, measured)")
    ax.set_title("random model, window 256, d = 128:\ncost per token stays flat")
    ax.legend(fontsize=10, loc="upper left")
    fig.suptitle("RoPE makes the KV cache slide: keys keep their rotation, the newest token "
                 "overwrites the oldest slot", y=1.03)
    fig.tight_layout()
    save(fig, "sliding_window_cache.png")
    return strat, np.median(t_cache[100:255]), np.median(t_cache[300:]), t_full[-1]


def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    text, chars, data = load_text(DEFAULT_TEXT)
    V = len(chars)
    llama, _, llama_hist = get_trained_model(np.float64)
    gpt, gpt_hist = gpt_history(data, V)
    from llama_numpy import evaluate
    llama_final = evaluate(llama, data)
    gpt_final = transformer_numpy.evaluate(gpt, data)
    print(f"  whole-text loss: GPT {gpt_final:.3f}, LLaMA {llama_final:.3f}; NumPy training "
          f"time GPT {gpt_hist['time'][-1]:.1f} s, LLaMA {llama_hist['time'][-1]:.1f} s")
    pt = results("llama_pytorch.npz", "llama_pytorch.py")
    moe = results("moe.npz", "moe.py")

    fig_architecture(llama, gpt)
    med, cmed, times = fig_rmsnorm(llama, data)
    print("  rmsnorm: median |mean|/rms " + " ".join(f"{m:.2f}" for m in med) + " | cos " +
          " ".join(f"{c:.3f}" for c in cmed) + " | " +
          " ".join(f"{k} {v * 1e3:.1f} ms" for k, v in times.items()))
    print(f"  rope_rotation: max deviation along a diagonal {fig_rope_rotation():.1e}")
    fig_rope_frequencies()
    for l, (neg, big, med) in enumerate(fig_swiglu(llama, data)):
        print(f"  swiglu layer {l}: gate median {med:.2f}, {neg:.0%} negative, {big:.0%} > 1")
    fig_gqa()
    fig_ablation(pt)
    fig_training(gpt_hist, llama_hist, gpt_final, llama_final, pt)
    fig_moe(moe)
    strat, before, after, last_full = fig_sliding(llama, data)
    print("  sliding: " + ", ".join(f"{k}: {v:.3f}" for k, v in strat.items()))
    print(f"  sliding: decode {before * 1e3:.2f} ms (pos 100-255) vs {after * 1e3:.2f} ms "
          f"(pos 300+); no cache at pos {last_full[0]}: {last_full[1] * 1e3:.2f} ms")
    print(f"total time {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
