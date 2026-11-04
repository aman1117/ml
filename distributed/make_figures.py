"""
Draw the pictures used in distributed/README.md and save them to distributed/images/.

Every number in every figure is computed, not typed in:
  - float formats and gradient histograms: precision_numpy.py (its training
    curves and gradient census are loaded from runs/precision.npz, or the
    script's training is re-run here, ~75 s);
  - memory bars: memory.py's formulas;
  - ring all-reduce, ZeRO, tensor and pipeline parallelism: parallel_numpy.py,
    run here (fast);
  - DDP scaling: runs/ddp_scaling.json from ddp_pytorch.py (run here if missing, ~10 s);
  - checkpointing memory vs time: runs/efficient.json from train_efficient.py
    (run here if missing, ~30 s).

Run (~2 s with the runs/ files, ~2 min from scratch):
    python distributed/make_figures.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

import memory as M  # noqa: E402
import parallel_numpy as P  # noqa: E402
import precision_numpy as Q  # noqa: E402

warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "images")
RUNS = os.path.join(HERE, "runs")
DPI = 130

C_SIGN, C_EXP, C_MAN = "#d62728", "#1f77b4", "#2ca02c"
C_PARAM, C_GRAD, C_OPT, C_ACT = "#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"
RANK_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

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


def box(ax, x, y, w, h, text="", fc="white", ec="black", fs=11, lw=1.5, **kw):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.08",
                                fc=fc, ec=ec, lw=lw, zorder=2))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, zorder=3, **kw)


def arrow(ax, a, b, color="black", lw=1.8, ms=15, **kw):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=ms, color=color, lw=lw,
                                 zorder=1, **kw))


# ----------------------------------------------------------------------------
# Data from the other scripts
# ----------------------------------------------------------------------------
def precision_runs():
    path = os.path.join(RUNS, "precision.npz")
    if not os.path.exists(path):
        print("runs/precision.npz missing: running precision_numpy.py (~75 s)")
        subprocess.run([sys.executable, os.path.join(HERE, "precision_numpy.py")], check=True,
                       stdout=subprocess.DEVNULL)
    d = np.load(path)
    return {k: d[k] for k in d.files}


def json_run(name, cmd):
    path = os.path.join(RUNS, name)
    if not os.path.exists(path):
        print(f"runs/{name} missing: running {' '.join(cmd)}")
        subprocess.run([sys.executable] + cmd, check=True, stdout=subprocess.DEVNULL)
    with open(path) as f:
        return json.load(f)


# ----------------------------------------------------------------------------
# 1. Float formats
# ----------------------------------------------------------------------------
def fig_float_formats():
    names = ["fp32", "fp16", "bf16", "fp8_e4m3", "fp8_e5m2"]
    fig, ax = plt.subplots(figsize=(15, 6.2))
    cell = 0.38
    for row, name in enumerate(names):
        f = Q.format_info(name)
        y = (len(names) - 1 - row) * 1.15
        s, e, m = Q.encode_bits(np.pi, name)
        bits = [(b, C_SIGN) for b in s] + [(b, C_EXP) for b in e] + [(b, C_MAN) for b in m]
        for i, (b, c) in enumerate(bits):
            ax.add_patch(Rectangle((i * cell, y), cell * 0.92, 0.6, fc=c, ec="white", alpha=0.85))
            ax.text(i * cell + cell * 0.46, y + 0.3, b, ha="center", va="center", color="white",
                    fontsize=10, fontweight="bold")
        ax.text(-0.3, y + 0.3, name.replace("_", " "), ha="right", va="center", fontsize=13,
                fontweight="bold")
        val = float(Q.round_to_format(np.pi, name))
        ax.text(32 * cell + 0.4, y + 0.3,
                f"max {f['max']:.3g}   min normal {f['min_normal']:.2g}   "
                f"eps {f['eps']:.2g}   π → {val:.7g}",
                ha="left", va="center", fontsize=11, family="monospace")
    ax.text(0, 5 * 1.15 + 0.05, "π in each format:", fontsize=12)
    for x0, c, label in ((0, C_SIGN, "sign"), (1.6, C_EXP, "exponent (range)"),
                         (5.6, C_MAN, "mantissa (precision)")):
        ax.add_patch(Rectangle((x0, -0.75), 0.3, 0.3, fc=c))
        ax.text(x0 + 0.4, -0.6, label, va="center", fontsize=12)
    ax.text(10.5, -0.6, "value = (−1)^sign · 1.mantissa · 2^(exponent − bias)", va="center",
            fontsize=12)
    ax.set_xlim(-2.2, 32 * cell + 9.7)
    ax.set_ylim(-1.0, 6.1)
    ax.axis("off")
    ax.set_title("Floating-point formats: bf16 keeps fp32's 8 exponent bits (range), fp16 keeps "
                 "more mantissa (precision)")
    save(fig, "float_formats.png")


# ----------------------------------------------------------------------------
# 2. fp16 underflow histogram + 3. training curves
# ----------------------------------------------------------------------------
def fig_fp16_underflow(runs):
    acc = int(runs["accum"])
    g = runs["census.activations"].astype(np.float64)
    g = np.abs(g[g != 0]) / acc  # what the fp16 backward sees at a 0.5M-token batch
    S = 2.0 ** 16
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
    bins = np.arange(-50, 6, 1.0)
    for ax, scale, title in ((axes[0], 1.0, "no loss scaling"),
                             (axes[1], S, "loss × 2^16 (shifts every gradient 16 bins right)")):
        lg = np.log2(g * scale)
        zero = np.mean(lg < -25)
        sub = np.mean((lg >= -25) & (lg < -14))
        ax.axvspan(-50, -25, color="#d62728", alpha=0.12)
        ax.axvspan(-25, -14, color="#ff7f0e", alpha=0.12)
        ax.axvspan(-14, 16, color="#2ca02c", alpha=0.08)
        ax.hist(lg, bins=bins, color="#1f77b4", alpha=0.85)
        ax.text(-37.5, 0.97, f"→ 0 in fp16\n{zero:.0%}", transform=ax.get_xaxis_transform(),
                ha="center", va="top", color="#d62728", fontsize=12, fontweight="bold")
        ax.text(-19.5, 0.97, f"subnormal\n{sub:.0%}", transform=ax.get_xaxis_transform(),
                ha="center", va="top", color="#ff7f0e", fontsize=12, fontweight="bold")
        ax.text(-5, 0.97, f"normal fp16\n{1 - zero - sub:.0%}", transform=ax.get_xaxis_transform(),
                ha="center", va="top", color="#2ca02c", fontsize=12, fontweight="bold")
        ax.set_xlim(-50, 5)
        ax.set_xlabel("log2 |activation gradient|")
        ax.set_title(title)
    axes[0].set_ylabel("count")
    fig.suptitle(f"Real activation gradients of the trained mini-GPT, at a 0.5M-token batch "
                 f"(÷{acc}): fp16 flushes half of them to zero", y=1.02)
    save(fig, "fp16_underflow.png")


def fig_precision_training(runs):
    modes = [str(m) for m in runs["modes"]]
    styles = {"fp32": ("black", "-"), "fp16": ("#d62728", "-"), "fp16-static": ("#ff7f0e", "--"),
              "fp16-dynamic": ("#2ca02c", ":"), "bf16": ("#1f77b4", "-"),
              "bf16-nomaster": ("#9467bd", "-"), "bf16-nomaster-sr": ("#8c564b", "--")}
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    k = 25
    for m in modes:
        loss = runs[f"loss.{m}"]
        smooth = np.convolve(loss, np.ones(k) / k, mode="valid")
        c, ls = styles[m]
        ax.plot(np.arange(len(smooth)) + k - 1, smooth, color=c, ls=ls, lw=2,
                label=f"{Q.MODES[m][0]}  ({float(runs[f'whole.{m}']):.3f})")
    ax.set_yscale("log")
    ticks = [0.4, 0.6, 1, 2, 3]
    ax.set_yticks(ticks)
    ax.set_yticklabels([str(t) for t in ticks])
    ax.set_xlabel("step")
    ax.set_ylabel("training loss per char (log scale)")
    ax.set_title("Same init and batches (0.5M-token batch emulated);\nlegend: final loss on the "
                 "whole text", fontsize=13)
    ax.legend(fontsize=10, loc="upper right")
    ax = axes[1]
    sc = runs["scale.fp16-dynamic"]
    sk = runs["skipped.fp16-dynamic"]
    ax.plot(np.log2(sc), color="#2ca02c", lw=2)
    ax.scatter(np.where(sk)[0], np.log2(sc[sk.astype(bool)]), color="#d62728", zorder=3,
               label=f"overflow → step skipped ({int(sk.sum())})")
    ax.set_xlabel("step")
    ax.set_ylabel("log2 loss scale S")
    ax.set_title("fp16 + dynamic scale: S halves on inf,\n×2 after 200 clean steps", fontsize=13)
    ax.legend(fontsize=10)
    save(fig, "precision_training.png")


# ----------------------------------------------------------------------------
# 4. Memory breakdown
# ----------------------------------------------------------------------------
def fig_memory():
    n = 64
    fig, axes = plt.subplots(1, 4, figsize=(16, 5.2))
    for ax, (name, m) in zip(axes, M.MODELS.items()):
        N = M.n_params(m)
        act = M.activation_bytes(m, 1, None, 1, "selective") / M.GB
        labels = ["DP", "ZeRO-1", "ZeRO-2", "ZeRO-3"]
        bottoms = np.zeros(4)
        for key, color, lab in (("weights", C_PARAM, "bf16 params"), ("grads", C_GRAD, "bf16 grads"),
                                ("optimizer", C_OPT, "fp32 master + Adam m, v")):
            vals = np.array([M.zero_bytes_per_gpu(N, n, s)[key] / M.GB for s in range(4)])
            ax.bar(labels, vals, bottom=bottoms, color=color, label=lab)
            bottoms += vals
        ax.bar(labels, [act] * 4, bottom=bottoms, color=C_ACT, label="activations (1 seq)")
        bottoms += act
        top = bottoms.max() * 1.15
        if top > 80:
            ax.axhline(80, color="black", ls="--", lw=1.2)
            ax.text(-0.45, 80, "80 GB GPU", va="bottom", ha="left", fontsize=10)
        ax.set_ylim(0, top)
        for i, b in enumerate(bottoms):
            ax.text(i, b + top * 0.01, f"{b:.0f}" if b >= 10 else f"{b:.1f}", ha="center",
                    va="bottom", fontsize=10)
        ax.set_title(f"{name} ({N / 1e9:.2f}B params)" if N >= 1e9 else f"{name}")
        ax.tick_params(axis="x", rotation=30)
    axes[0].set_ylabel(f"GB per GPU ({n} GPUs)")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=4, fontsize=11, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle("Training memory per GPU: 16 bytes/param of model states, sharded by ZeRO; "
                 "activations (selective recompute, 1 sequence) are not", y=1.02)
    save(fig, "memory_breakdown.png")


def fig_zero_stages():
    """Who holds what: 4 ranks x {params, grads, optimizer} for each stage."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    parts = [("W", C_PARAM, 2), ("G", C_GRAD, 2), ("optimizer", C_OPT, 12)]
    W = 4
    for stage, ax in enumerate(axes):
        for r in range(W):
            x0 = 0
            for j, (label, color, width) in enumerate(parts):
                sharded = (j == 2 and stage >= 1) or (j == 1 and stage >= 2) or (j == 0 and stage >= 3)
                for c in range(W):
                    held = (not sharded) or c == r
                    ax.add_patch(Rectangle((x0 + c * width / W, (W - 1 - r) * 1.2), width / W * 0.95,
                                           0.9, fc=color if held else "white", ec=color,
                                           alpha=0.9 if held else 0.4, lw=1))
                x0 += width + 0.6
            ax.text(-0.4, (W - 1 - r) * 1.2 + 0.45, f"rank {r}", ha="right", va="center", fontsize=11)
        x0 = 0
        for label, color, width in parts:
            ax.text(x0 + width / 2, W * 1.2 + 0.05, label, ha="center", va="bottom", fontsize=10,
                    color=color, fontweight="bold")
            x0 += width + 0.6
        per = M.zero_bytes_per_gpu(1.0, W, stage)["total"]
        ax.set_title(["plain DP", "ZeRO-1: shard optimizer", "ZeRO-2: + gradients",
                      "ZeRO-3 / FSDP: + params"][stage] + f"\n{per:g} bytes/param per rank",
                     fontsize=12)
        ax.set_xlim(-2.2, 17.4)
        ax.set_ylim(-0.3, W * 1.2 + 0.8)
        ax.axis("off")
    fig.suptitle("ZeRO with 4 ranks: each column of a block = 1/4 of the model; filled = this "
                 "rank stores it\nW = bf16 weights (2 bytes/param), G = bf16 gradients (2), "
                 "optimizer = fp32 master + Adam m, v (12)", y=1.13, fontsize=14)
    save(fig, "zero_stages.png")


# ----------------------------------------------------------------------------
# 5. Ring all-reduce
# ----------------------------------------------------------------------------
def fig_ring():
    W = 4
    chunks = [P.split_chunks(np.arange(W, dtype=float), W) for _ in range(W)]
    contrib = [[{r} for _ in range(W)] for r in range(W)]
    states = [("start", [[len(c) for c in row] for row in contrib], [])]
    for phase, s, sends in P.ring_reduce_scatter_steps(chunks):
        for src, dst, c in sends:
            contrib[dst][c] = contrib[dst][c] | contrib[src][c]
        states.append((f"reduce-scatter {s + 1}", [[len(c) for c in row] for row in contrib], sends))
    for phase, s, sends in P.ring_all_gather_steps(chunks):
        for src, dst, c in sends:
            contrib[dst][c] = set(contrib[src][c])
        states.append((f"all-gather {s + 1}", [[len(c) for c in row] for row in contrib], sends))

    fig = plt.figure(figsize=(17, 6.4))
    gs = fig.add_gridspec(2, 7, height_ratios=[1, 1.05], hspace=0.45, wspace=0.25)
    cmap = plt.get_cmap("Greens")
    for k, (title, grid, sends) in enumerate(states):
        ax = fig.add_subplot(gs[0, k])
        g = np.array(grid)
        ax.imshow(g, cmap=cmap, vmin=0, vmax=W + 0.6)
        for r in range(W):
            for c in range(W):
                ax.text(c, r, str(g[r, c]), ha="center", va="center", fontsize=11,
                        color="white" if g[r, c] == W else "black",
                        fontweight="bold" if g[r, c] == W else "normal")
        for src, dst, c in sends:
            ax.add_patch(Rectangle((c - 0.5, dst - 0.5), 1, 1, fill=False, ec="#d62728", lw=2.2))
        ax.set_xticks(range(W))
        ax.set_xticklabels([f"c{c}" for c in range(W)], fontsize=9)
        ax.set_yticks(range(W))
        ax.set_yticklabels([f"W{r}" for r in range(W)] if k == 0 else [], fontsize=10)
        ax.set_title(title, fontsize=11)
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.text(0.5, 0.94, "Ring all-reduce, 4 workers × 4 chunks: number in each cell = how many "
             "workers' gradients are summed into that copy (red = just received)", ha="center",
             fontsize=13)

    ax = fig.add_subplot(gs[1, 0:3])
    ax.set_xlim(-1.7, 1.7)
    ax.set_ylim(-1.45, 1.35)
    ax.set_aspect("equal")
    ax.axis("off")
    R = 1.0
    pos = [(R * np.cos(np.pi / 2 - 2 * np.pi * r / W), R * np.sin(np.pi / 2 - 2 * np.pi * r / W))
           for r in range(W)]
    for r, (x, y) in enumerate(pos):
        ax.add_patch(plt.Circle((x, y), 0.24, fc=RANK_COLORS[r], ec="black", alpha=0.85, zorder=2))
        ax.text(x, y, f"W{r}", ha="center", va="center", color="white", fontsize=13,
                fontweight="bold", zorder=3)
        nx, ny = pos[(r + 1) % W]
        d = np.hypot(nx - x, ny - y)
        ux, uy = (nx - x) / d, (ny - y) / d
        arrow(ax, (x + 0.3 * ux, y + 0.3 * uy), (nx - 0.3 * ux, ny - 0.3 * uy), lw=2.2,
              connectionstyle="arc3,rad=-0.3")
        ax.text((x + nx) / 2 * 1.45, (y + ny) / 2 * 1.45, "n/W", ha="center", va="center",
                fontsize=10, color="#555555")
    ax.text(0, -1.42, "each step: every worker sends 1 chunk to the right",
            ha="center", va="center", fontsize=11)
    ax.set_title("2(W−1) steps; every link busy every step")

    ax = fig.add_subplot(gs[1, 3:7])
    Ws = [2, 4, 8, 16, 32, 64]
    n = 1024
    ring, naive = [], []
    for w in Ws:
        arrays = [np.ones(n) for _ in range(w)]
        log, nlog = P.CommLog(w), P.CommLog(w)
        P.ring_all_reduce(arrays, log)
        P.naive_all_reduce(arrays, nlog)
        ring.append(2 * log.sent.max() / 8 / n)  # each worker receives what its neighbour sends
        naive.append((nlog.sent[0] + nlog.sent[1:].sum()) / 8 / n)  # rank 0 sends + receives
    ax.plot(Ws, ring, "o-", color="#2ca02c", lw=2, label="ring, any worker: 4(W−1)/W · n → 4n")
    ax.plot(Ws, naive, "s-", color="#d62728", lw=2, label="naive, rank 0: 2(W−1) · n")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xlabel("workers W")
    ax.set_ylabel("numbers sent + received\nby the busiest worker (× n)")
    ax.set_xticks(Ws)
    ax.set_xticklabels(Ws)
    ax.legend(fontsize=11)
    ax.set_title("Measured by the simulation (CommLog)")
    save(fig, "ring_allreduce.png")


# ----------------------------------------------------------------------------
# 6. Tensor parallelism
# ----------------------------------------------------------------------------
def fig_tensor_parallel():
    rng = np.random.default_rng(0)
    C, t = 16, 2
    mlp = P.MLP(C, rng, dtype=P.F64)
    for layer in (mlp.c_fc, mlp.c_proj):
        for p in layer.params.values():
            p[...] = 0.3 * rng.standard_normal(p.shape)
    x, dy = rng.standard_normal((2, 8, C)), rng.standard_normal((2, 8, C))
    y_ref = mlp.forward(x)
    dx_ref = mlp.backward(dy)
    y, dx, _ = P.tp_mlp(mlp, x, dy, t)
    err = max(np.abs(y - y_ref).max(), np.abs(dx - dx_ref).max())

    fig, ax = plt.subplots(figsize=(16, 6.2))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 6.9)
    ax.axis("off")
    c0, c1 = RANK_COLORS[0], RANK_COLORS[1]
    for r, (yy, c) in enumerate(((3.6, c0), (0.6, c1))):
        box(ax, 0.2, yy, 1.6, 1.8, "X\n(n, C)\nreplicated", fc="#eeeeee", fs=11)
        cols = ":2C" if r == 0 else "2C:"
        box(ax, 2.5, yy, 1.4, 1.8, f"W1[:, {cols}]\n(C, 4C/t)", fc=c, fs=10, color="white")
        box(ax, 4.6, yy + 0.3, 1.9, 1.2, f"GELU(X W1_{r})\n(n, 4C/t)", fc="white", ec=c, fs=10)
        box(ax, 7.2, yy, 1.4, 1.8, f"W2[{cols}, :]\n(4C/t, C)", fc=c, fs=10, color="white")
        box(ax, 9.3, yy + 0.3, 1.9, 1.2, f"Y_{r} = H_{r} W2_{r}\n(n, C) partial", fc="white", ec=c,
            fs=10)
        arrow(ax, (1.85, yy + 0.9), (2.45, yy + 0.9))
        arrow(ax, (3.95, yy + 0.9), (4.55, yy + 0.9))
        arrow(ax, (6.55, yy + 0.9), (7.15, yy + 0.9))
        arrow(ax, (8.65, yy + 0.9), (9.25, yy + 0.9))
        arrow(ax, (11.25, yy + 0.9), (12.15, 3.3 if r == 0 else 2.7), color=c)
        ax.text(0.2, yy + 1.9, f"GPU {r}", color=c, fontsize=13, fontweight="bold")
    box(ax, 12.2, 2.4, 1.8, 1.2, "all-reduce\nΣ Y_r + b2", fc="#fdf1c7", fs=11)
    arrow(ax, (14.05, 3.0), (14.7, 3.0))
    box(ax, 14.75, 2.5, 1.1, 1.0, "Y\n(n, C)", fc="#eeeeee", fs=11)
    ax.text(3.2, 5.75, "column-parallel W1:\neach GPU gets half the hidden\nunits (GELU needs no talk)",
            ha="center", va="bottom", fontsize=10.5)
    ax.text(7.9, 5.75, "row-parallel W2:\neach GPU's output is\na partial sum of Y",
            ha="center", va="bottom", fontsize=10.5)
    ax.text(8, 0.05, f"Forward: 1 all-reduce. Backward: dX = all-reduce(Σ dX_r), 1 more. "
            f"Simulated with t = {t} in parallel_numpy.py: max |TP − unsharded| = {err:.0e}. "
            f"Attention: same idea, split by heads.", ha="center", fontsize=11)
    ax.set_title("Megatron tensor parallelism for the MLP: Y = GELU(X W1) W2 on 2 GPUs")
    save(fig, "tensor_parallel.png")


# ----------------------------------------------------------------------------
# 7. Pipeline schedules
# ----------------------------------------------------------------------------
def fig_pipeline():
    p, m = 4, 8
    fig, axes = plt.subplots(3, 1, figsize=(16, 9.5), gridspec_kw={"height_ratios": [1, 1, 0.9]})
    cmap = plt.get_cmap("tab10")
    for ax, (name, fn) in zip(axes[:2], (("GPipe", P.schedule_gpipe), ("1F1B", P.schedule_1f1b))):
        tl, span, bubble, peak = P.simulate_schedule(fn(p, m))
        ax.add_patch(Rectangle((0, -0.5), span, p, fc="#dddddd", hatch="///", ec="#bbbbbb", lw=0))
        for s, op, i, start, end in tl:
            color = cmap(i % 10)
            ax.add_patch(Rectangle((start, s - 0.42), end - start, 0.84, fc=color,
                                   alpha=0.55 if op == "F" else 0.95, ec="black", lw=0.8))
            ax.text((start + end) / 2, s, f"{op}{i}", ha="center", va="center", fontsize=9,
                    color="black" if op == "F" else "white", fontweight="bold")
        ax.set_xlim(0, span)
        ax.set_ylim(p - 0.5, -0.5)
        ax.set_yticks(range(p))
        ax.set_yticklabels([f"stage {s}\n(stash ≤ {peak[s]})" for s in range(p)], fontsize=10)
        ax.set_title(f"{name}, p = {p} stages, m = {m} micro-batches: bubble {bubble:.1%} "
                     f"(hatched); peak stashed micro-batches per stage {peak}", fontsize=13)
        ax.set_xlabel("time (forward = 1, backward = 2 units)")
    ax = axes[2]
    ms = np.arange(1, 65)
    for pp, c in ((2, "#1f77b4"), (4, "#ff7f0e"), (8, "#2ca02c"), (16, "#d62728")):
        ax.plot(ms, (pp - 1) / (ms + pp - 1), color=c, lw=2, label=f"p = {pp}: (p−1)/(m+p−1)")
        sims = [P.simulate_schedule(P.schedule_1f1b(pp, mm))[2] for mm in (4, 8, 16, 32, 64)]
        ax.scatter([4, 8, 16, 32, 64], sims, color=c, zorder=3, s=25)
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16, 32, 64])
    ax.set_xticklabels([1, 2, 4, 8, 16, 32, 64])
    ax.set_xlabel("micro-batches m (dots: simulated 1F1B)")
    ax.set_ylabel("bubble fraction")
    ax.legend(fontsize=10, ncol=4)
    ax.set_title("More micro-batches per step shrink the bubble")
    fig.tight_layout()
    save(fig, "pipeline_schedule.png")


# ----------------------------------------------------------------------------
# 8. DDP scaling, 9. checkpointing
# ----------------------------------------------------------------------------
def fig_ddp_scaling():
    d = json_run("ddp_scaling.json", [os.path.join(HERE, "ddp_pytorch.py"), "--parts", "d"])
    rows = d["rows"]
    Ws = np.array([r["W"] for r in rows])
    base = rows[0]["alone_tokens_per_s"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    ax = axes[0]
    ax.plot(Ws, base * Ws / 1e3, "k--", lw=1.5, label="ideal: W × one process")
    ax.plot(Ws, [r["alone_tokens_per_s"] / 1e3 for r in rows], "o-", color="#1f77b4", lw=2,
            label="W processes, no communication")
    ax.plot(Ws, [r["tokens_per_s"] / 1e3 for r in rows], "s-", color="#d62728", lw=2,
            label="DDP (gloo all-reduce)")
    ax.set_xlabel("processes W")
    ax.set_ylabel("tokens / s (thousands)")
    ax.set_xticks(Ws)
    ax.legend(fontsize=11)
    ax.set_title(f"Weak scaling on this CPU ({d['threads']} threads per process)")
    ax = axes[1]
    x = np.arange(len(Ws))
    ax.bar(x - 0.2, [r["alone_step_ms"] for r in rows], 0.4, color="#1f77b4", label="step, no comm")
    ax.bar(x + 0.2, [r["step_ms"] for r in rows], 0.4, color="#d62728", label="step, DDP")
    ax.plot(x + 0.2, [r["allreduce_ms"] for r in rows], "D", color="black", ms=8,
            label=f"one all-reduce of {rows[0]['grad_mb']:.1f} MB")
    ax.set_xticks(x)
    ax.set_xticklabels([f"W={w}" for w in Ws])
    ax.set_ylabel("ms")
    ax.legend(fontsize=11)
    ax.set_title("Where the time goes per step")
    save(fig, "ddp_scaling.png")


def fig_checkpointing():
    d = json_run("efficient.json", [os.path.join(HERE, "train_efficient.py"), "--no-compile"])
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ck = d["checkpointing"]
    ax.plot([r["time"] * 1e3 for r in ck], [r["peak"] / 1e6 for r in ck], "o-", color="#9467bd",
            lw=2, ms=9, label="activation checkpointing (batch 8)")
    for r in ck:
        ax.annotate(r["label"], (r["time"] * 1e3, r["peak"] / 1e6), textcoords="offset points",
                    xytext=(8, 4), fontsize=10)
    ac = d["accumulation"]
    ax.plot([r["time"] * 1e3 for r in ac], [r["peak"] / 1e6 for r in ac], "s--", color="#ff7f0e",
            lw=2, ms=8, label="gradient accumulation (same 8 sequences)")
    for r in ac:
        ax.annotate(f"{8 // r['accum']} × {r['accum']}", (r["time"] * 1e3, r["peak"] / 1e6),
                    textcoords="offset points", xytext=(8, -12), fontsize=10)
    ax.set_xlabel("time per training step (ms)")
    ax.set_ylabel("peak memory of the step (MB, torch.profiler)")
    ax.set_ylim(0, None)
    ax.legend(fontsize=11)
    ax.set_title("Trading compute for memory (6-layer GPT, 8 × 256 tokens, CPU)")
    save(fig, "checkpointing.png")


def main():
    os.makedirs(OUT, exist_ok=True)
    runs = precision_runs()
    fig_float_formats()
    fig_fp16_underflow(runs)
    fig_precision_training(runs)
    fig_memory()
    fig_zero_stages()
    fig_ring()
    fig_tensor_parallel()
    fig_pipeline()
    fig_ddp_scaling()
    fig_checkpointing()


if __name__ == "__main__":
    main()
