"""
Draw the pictures used in finetuning/README.md and save them to finetuning/images/.

Every number in every figure is computed here or by the scripts it calls:
  - finetune_results.npz from finetune_pytorch.py (SFT/DPO curves, accuracy,
    forgetting, singular values of ΔW): loaded, or produced by running the
    experiment now (~70 s, plus ~50 s to pretrain the base model the first time);
  - a LoRA rank sweep (r = 1, 2, 4, 16; r = 8 comes from the experiment), ~70 s;
  - the NumPy base / SFT models of lora_numpy.py for the loss-mask picture
    (loaded from base_numpy.npz + lora_numpy.npz, or trained, ~25 s).

Run (~65 s once finetune_results.npz exists, ~3 min from scratch):
    python finetuning/make_figures.py
"""
import argparse
import contextlib
import copy
import io
import json
import os
import time
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

import domain_data as dd  # noqa: E402
import finetune_pytorch as fp  # noqa: E402
import lora_numpy as ln  # noqa: E402

warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "images")
DPI = 130

C_FROZEN, C_LORA, C_FULL, C_DPO, C_BAD = "#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#d62728"
F_FROZEN, F_LORA = "#dbe9f6", "#ffe8cc"

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "figure.titlesize": 16,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

METHOD_COLORS = {"base, 1-shot prompt": "#7f7f7f", "SFT full": C_FULL,
                 "SFT full + replay": "#98df8a", "SFT LoRA r=8": C_LORA,
                 "LoRA SFT + DPO": C_BAD, "LoRA SFT + DPO+NLL": C_DPO}


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {os.path.relpath(path)}  ({os.path.getsize(path) // 1024} KB)")


def box(ax, x, y, w, h, text, fc, ec, fs=13, **kw):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=fc, ec=ec, lw=2, zorder=2))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, zorder=3, **kw)


def arrow(ax, a, b, color="black", lw=2.0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=18, color=color,
                                 lw=lw, zorder=1))


def smooth(y, k=10):
    y = np.asarray(y, dtype=float)
    if len(y) < k:
        return y
    c = np.cumsum(np.insert(y, 0, 0.0))
    out = (c[k:] - c[:-k]) / k
    return np.concatenate([y[:k - 1], out])


# ---------------------------------------------------------------- data sources

def get_results():
    if not os.path.exists(fp.RESULTS_PATH):
        print("no finetune_results.npz yet: running the finetune_pytorch.py experiment "
              "(~70 s, +50 s to pretrain the base the first time)...")
        import sys
        argv = sys.argv
        sys.argv = [argv[0], "--no-check"]
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                fp.main()
        finally:
            sys.argv = argv
    d = np.load(fp.RESULTS_PATH)
    res = json.loads(str(d["results"]))
    args = json.loads(str(d["args"]))
    full = {k.split(".", 1)[1]: d[k] for k in d.files if k.startswith("spectra_full.")}
    lora = {k.split(".", 1)[1]: d[k] for k in d.files if k.startswith("spectra_lora.")}
    return res, args, full, lora, d["random_svals"]


def rank_sweep(args, ranks=(1, 2, 4, 16)):
    """SFT with LoRA at other ranks, same data/steps/lr as finetune_pytorch.py."""
    env = fp.setup()
    base, _ = fp.get_base(env, log=False)
    data = ln.sft_arrays(env["train"], env["tok"])
    ns = argparse.Namespace(**{**args, "targets": tuple(args["targets"])})
    out = {}
    for r in ranks:
        t0 = time.time()
        torch.manual_seed(0)
        m = fp.apply_lora(copy.deepcopy(base), r, 2.0 * r, ns.targets)
        res = fp.sft_row(f"r={r}", m, env, data, ns, ns.lora_lr, log=False)
        out[r] = res
        sc = res["scores"]
        print(f"  rank sweep r={r:2d}: {res['params']:,} params | exact train "
              f"{sc['train']['exact']:.1%}, paraphrase {sc['paraphrase']['exact']:.1%} | "
              f"out-of-scope refusal {sc['out_of_scope']['exact']:.1%} | story loss "
              f"{res['story_train']:.2f} ({time.time() - t0:.0f} s)")
    return out


# ---------------------------------------------------------------- figures

def fig_lora_diagram():
    fig, ax = plt.subplots(figsize=(12, 5.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")
    box(ax, 1.0, 3.0, 1.2, 0.9, "x\n(N, d_in)", "white", "black")
    box(ax, 4.9, 4.5, 3.8, 1.6, "W0  (d_in × d_out)\nFROZEN\nno gradient, no optimizer state",
        F_FROZEN, C_FROZEN, fs=12)
    box(ax, 3.9, 1.5, 1.5, 1.4, "A\n(d_in × r)\nrandom init", F_LORA, C_LORA, fs=12)
    box(ax, 6.3, 1.5, 1.5, 1.4, "B\n(r × d_out)\nzero init", F_LORA, C_LORA, fs=12)
    ax.text(5.1, 2.0, "h = xA\n(N, r)", ha="center", va="center", fontsize=11, color=C_LORA)
    ax.text(7.8, 1.5, "× α/r", ha="center", va="center", fontsize=14, color=C_LORA,
            fontweight="bold")
    ax.add_patch(plt.Circle((9.1, 3.0), 0.36, fc="white", ec="black", lw=2, zorder=2))
    ax.text(9.1, 3.0, "+", fontsize=22, ha="center", va="center", zorder=3)
    box(ax, 11.0, 3.0, 1.5, 0.9, "y\n(N, d_out)", "white", "black")
    arrow(ax, (1.6, 3.2), (2.95, 4.3))
    arrow(ax, (1.6, 2.8), (3.15, 1.7), C_LORA)
    arrow(ax, (4.65, 1.5), (5.55, 1.5), C_LORA)
    arrow(ax, (6.85, 4.3), (8.8, 3.2))
    arrow(ax, (7.05, 1.5), (7.4, 1.5), C_LORA)
    arrow(ax, (8.2, 1.7), (8.85, 2.7), C_LORA)
    arrow(ax, (9.46, 3.0), (10.25, 3.0))
    ax.text(6.0, 5.75, "y = x W0 + (α/r) · (x A) B          ΔW = (α/r) A B has rank ≤ r",
            ha="center", fontsize=15, fontweight="bold")
    ax.text(6.0, 0.2, "B = 0 at init → ΔW = 0: the model starts exactly at the pretrained one.   "
            "After training: merge W0 ← W0 + (α/r)AB, zero extra inference cost.",
            ha="center", fontsize=11, color="#444")
    ax.text(10.9, 4.6, "trainable per matrix:\nr·(d_in + d_out)\nvs d_in·d_out",
            ha="center", fontsize=11, color=C_LORA)
    save(fig, "lora_diagram.png")


def fig_params_vs_rank(sweep, res):
    env_V = 47
    cfg = fp.BASE_CFG
    d, L = cfg["d_model"], cfg["n_layer"]
    ranks = np.array([1, 2, 4, 8, 16, 32, 64])
    per_block = lambda r: (3 * r * (d + d) + r * (d + d) + r * (d + 4 * d) + r * (4 * d + d))
    ours = np.array([L * per_block(r) for r in ranks])
    n_all = res["SFT full"]["params"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    ax = axes[0]
    ax.plot(ranks, ours, "o-", color=C_LORA, lw=2, label="LoRA on q,k,v,o,fc,proj")
    ax.axhline(n_all, color=C_FULL, ls="--", lw=2, label=f"full fine-tuning ({n_all:,})")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(ranks)
    ax.set_xticklabels(ranks)
    ax.set_xlabel("rank r")
    ax.set_ylabel("trainable parameters")
    ax.set_title(f"Our 4-layer GPT (d = {d}, V = {env_V})")
    for r, n in zip(ranks, ours):
        if r in (1, 8, 64):
            ax.annotate(f"{n:,}\n({n / n_all:.1%})", (r, n), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontsize=10)
    ax.legend(loc="lower right", fontsize=10)
    ax = axes[1]
    D = 4096
    ax.plot(ranks, ranks * 2 * D, "o-", color=C_LORA, lw=2, label="LoRA: r·(d + d)")
    ax.axhline(D * D, color=C_FULL, ls="--", lw=2, label=f"full: d² = {D * D:,}")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(ranks)
    ax.set_xticklabels(ranks)
    ax.set_xlabel("rank r")
    ax.set_title(f"One {D}×{D} matrix (a 7B model's q_proj)")
    for r in (8, 64):
        n = r * 2 * D
        ax.annotate(f"{n:,} ({n / D / D:.2%})", (r, n), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=10)
    ax.legend(loc="lower right", fontsize=10)
    fig.suptitle("LoRA trains r·(d_in + d_out) numbers per matrix instead of d_in·d_out")
    fig.tight_layout()
    save(fig, "params_vs_rank.png")


def fig_loss_mask():
    with contextlib.redirect_stdout(io.StringIO()):
        tok, story_ids, base, sft, _ = ln.load_pipeline()
    row = dd.load_all()[0][0]
    text, mask = dd.render_with_mask(row["messages"])
    ids = np.array([tok.encode(text)])
    tgt = ids[:, 1:]

    def losses(model):
        p = ln.softmax(model.forward(ids[:, :-1]).astype(np.float64))
        return -np.log(np.take_along_axis(p, tgt[..., None], -1)[0, :, 0])

    lb, ls = losses(base), losses(sft)
    m = np.array(mask[1:])
    labels = [{dd.USER: "<u>", dd.ASSISTANT: "<a>", dd.END: "<e>", " ": "␣"}.get(c, c)
              for c in text[1:]]
    T = len(labels)
    fig, axes = plt.subplots(2, 1, figsize=(15, 6.4), gridspec_kw=dict(height_ratios=[1, 2.4]))
    ax = axes[0]
    ax.set_xlim(-0.5, T - 0.5)
    ax.set_ylim(0, 1)
    ax.axis("off")
    for t, (c, k) in enumerate(zip(labels, m)):
        ax.add_patch(Rectangle((t - 0.45, 0.25), 0.9, 0.5, fc="#c7e9c0" if k else "#eeeeee",
                               ec=C_FULL if k else "#bbbbbb", lw=1))
        ax.text(t, 0.5, c, ha="center", va="center", fontsize=9 if len(c) > 1 else 11,
                fontweight="bold" if k else "normal", color="black" if k else "#888")
    ax.text(-0.5, 0.92, "Each box is a TARGET (the character predicted from everything before "
            "it; <u> <a> <e> = <|user|> <|assistant|> <|end|>).\ngreen = mask 1: trained;  "
            "grey = mask 0: context only", fontsize=11, va="bottom")
    ax = axes[1]
    x = np.arange(T)
    ax.bar(x - 0.2, lb, 0.4, color=np.where(m > 0, "#7f7f7f", "#dddddd"), label="base model")
    ax.bar(x + 0.2, ls, 0.4, color=np.where(m > 0, C_FULL, "#c7e9c0"), label="after LoRA SFT")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_xlim(-0.5, T - 0.5)
    ax.set_ylabel("loss on this target (nats)")
    mb, msf = (lb * m).sum() / m.sum(), (ls * m).sum() / m.sum()
    ax.set_title(f"Per-character loss. Masked SFT loss (mean over the {int(m.sum())} green "
                 f"targets): base {mb:.2f} → SFT {msf:.3f}")
    ax.axvspan(-0.5, int(np.argmax(m)) - 0.5, color="#f4f4f4", zorder=0)
    ax.text(2, ax.get_ylim()[1] * 0.92, "prompt: computed, multiplied by 0", fontsize=11,
            color="#666")
    ax.annotate(f"<|end|>: {lb[-1]:.1f} → {ls[-1]:.2f}\n(the base never saw it)",
                (T - 1, lb[-1]), xytext=(T - 12, lb[-1] * 0.95), fontsize=10,
                arrowprops=dict(arrowstyle="->", color="#444"))
    ax.legend(loc="upper center", fontsize=10)
    fig.tight_layout()
    save(fig, "loss_mask.png")


def fig_sft_curves(res, sweep):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    ax.plot(smooth(res["SFT full"]["curve"]), color=C_FULL, lw=2.2,
            label=f"full FT ({res['SFT full']['params']:,} params)")
    cmap = plt.get_cmap("Oranges")
    runs = {**{r: v for r, v in sweep.items()}, 8: res["SFT LoRA r=8"]}
    for i, r in enumerate(sorted(runs)):
        ax.plot(smooth(runs[r]["curve"]), color=cmap(0.35 + 0.6 * i / max(1, len(runs) - 1)),
                lw=1.8, label=f"LoRA r={r} ({runs[r]['params']:,})")
    ax.set_yscale("log")
    ax.set_xlabel("SFT step (16 chats each)")
    ax.set_ylabel("masked training loss (smoothed)")
    ax.set_title("SFT training curves")
    ax.legend(fontsize=10)
    ax = axes[1]
    rs = sorted(runs)
    para = [runs[r]["scores"]["paraphrase"]["exact"] * 100 for r in rs]
    tr = [runs[r]["scores"]["train"]["exact"] * 100 for r in rs]
    oos = [runs[r]["scores"]["out_of_scope"]["exact"] * 100 for r in rs]
    ax.plot(rs, tr, "o-", color="#555", lw=2, label="train exact")
    ax.plot(rs, para, "o-", color=C_LORA, lw=2, label="paraphrase exact")
    ax.plot(rs, oos, "s--", color=C_DPO, lw=2, label="out-of-scope refusal")
    f = res["SFT full"]["scores"]
    ax.axhline(f["paraphrase"]["exact"] * 100, color=C_FULL, ls=":", lw=2,
               label="full FT, paraphrase")
    ax.set_xscale("log", base=2)
    ax.set_xticks(rs)
    ax.set_xticklabels(rs)
    ax.set_ylim(-3, 105)
    ax.set_xlabel("LoRA rank r (alpha = 2r)")
    ax.set_ylabel("% of questions")
    ax.set_title("Held-out accuracy vs rank")
    ax2 = ax.twinx()
    ax2.plot(rs, [runs[r]["story_train"] for r in rs], "^-", color=C_BAD, alpha=0.7, lw=1.5)
    ax2.set_ylabel("story loss after SFT (forgetting)", color=C_BAD)
    ax2.spines["right"].set_visible(True)
    ax2.tick_params(axis="y", colors=C_BAD)
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    save(fig, "sft_curves.png")


def fig_results(res):
    names = list(res)
    groups = [("train", "exact", "train\nexact"), ("paraphrase", "exact", "paraphrase\nexact"),
              ("paraphrase", "format", "paraphrase\nformat"),
              ("unseen_fact", "exact", "unseen fact\nexact"),
              ("unseen_fact", "format", "unseen fact\nformat"),
              ("out_of_scope", "exact", "out of scope\nrefusal"),
              ("unknown_place", "exact", "unknown place\nrefusal")]
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.6), gridspec_kw=dict(width_ratios=[2.6, 1]))
    ax = axes[0]
    w = 0.8 / len(names)
    for i, n in enumerate(names):
        vals = [res[n]["scores"][s][k] * 100 for s, k, _ in groups]
        ax.bar(np.arange(len(groups)) + (i - (len(names) - 1) / 2) * w, vals, w,
               color=METHOD_COLORS.get(n, "#333"), label=n)
    ax.set_xticks(np.arange(len(groups)))
    ax.set_xticklabels([g[2] for g in groups], fontsize=10)
    ax.set_ylabel("% of questions")
    ax.set_ylim(0, 128)
    ax.set_yticks(range(0, 101, 20))
    ax.set_title("Held-out evaluation (greedy decoding; the base model scores 0 everywhere)")
    ax.legend(fontsize=10, ncol=3, loc="upper center")
    ax = axes[1]
    x = np.arange(len(names))
    ax.bar(x - 0.2, [res[n]["story_train"] for n in names], 0.4,
           color=[METHOD_COLORS.get(n, "#333") for n in names], label="story, train part")
    ax.bar(x + 0.2, [res[n]["story_val"] for n in names], 0.4,
           color=[METHOD_COLORS.get(n, "#333") for n in names], alpha=0.5, hatch="//",
           label="story, held-out part")
    ax.axhline(np.log(47), color="#444", ls="--", lw=1)
    ax.text(-0.4, np.log(47) + 0.1, "uniform guessing", fontsize=9, ha="left")
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace(", 1-shot prompt", "").replace("LoRA SFT + ", "+ ")
                        .replace("SFT ", "") for n in names], rotation=30, ha="right",
                       fontsize=10)
    ax.set_ylabel("loss per char on the pretraining text")
    ax.set_title("Catastrophic forgetting")
    ax.set_ylim(0, 8.8)
    ax.legend(fontsize=9, loc="upper left", ncol=2)
    fig.tight_layout()
    save(fig, "results.png")


def fig_dpo(res):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    runs = [("LoRA SFT + DPO", "DPO", C_BAD), ("LoRA SFT + DPO+NLL", "DPO + NLL", C_DPO)]
    for key, label, c in runs:
        h = res[key]["dpo"]
        steps = np.arange(len(h))
        axes[0].plot(steps, smooth([x["margin"] for x in h]), color=c, lw=2, label=label)
        axes[1].plot(steps, smooth([x["chosen"] for x in h]), color=c, lw=2,
                     label=f"{label}: chosen")
        axes[1].plot(steps, smooth([x["rejected"] for x in h]), color=c, lw=2, ls="--",
                     label=f"{label}: rejected")
        axes[2].plot(steps, smooth([x["loss"] for x in h]), color=c, lw=2, label=label)
    axes[0].set_title("Margin z = β[(Δ chosen) − (Δ rejected)]")
    axes[0].set_ylabel("mean over the batch (smoothed)")
    axes[1].set_title("log π − log π_ref of each answer")
    axes[1].axhline(0, color="#444", lw=1)
    axes[2].set_title("DPO loss −log σ(z)")
    axes[2].axhline(np.log(2), color="#444", ls=":", lw=1)
    axes[2].text(len(h) * 0.98, np.log(2) + 0.02, "log 2 (start)", ha="right", fontsize=9)
    for ax in axes:
        ax.set_xlabel("DPO step (16 pairs each)")
        ax.legend(fontsize=9)
    fig.tight_layout()
    save(fig, "dpo_curves.png")


def fig_spectrum(full, lora, rnd, rank):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    kinds = [("attn.c_attn", "q,k,v (c_attn)", C_FROZEN), ("attn.c_proj", "o (attn c_proj)", C_DPO),
             ("mlp.c_fc", "MLP in (c_fc)", C_FULL), ("mlp.c_proj", "MLP out (c_proj)", C_BAD)]
    for kind, label, c in kinds:
        for i, (k, s) in enumerate(sorted((k, s) for k, s in full.items() if kind in k)):
            ax.plot(np.arange(1, len(s) + 1), s / s[0], color=c, lw=1.5, alpha=0.8,
                    label=label if i == 0 else None)
    ax.plot(np.arange(1, len(rnd) + 1), rnd / rnd[0], color="#444", ls="--", lw=2,
            label="random Gaussian matrix")
    ax.axvline(rank, color=C_LORA, ls=":", lw=2)
    ax.text(rank + 1.5, 0.45, f"r = {rank}", color=C_LORA, fontsize=11)
    ax.set_xlabel("singular value index")
    ax.set_ylabel("σ_i / σ_1")
    ax.set_title("Full fine-tuning: singular values of ΔW (all 16 matrices)")
    ax.legend(fontsize=9)
    ax = axes[1]
    k0 = "blocks.0.attn.c_attn.weight"
    for name, spec, c, ls in (("full fine-tuning", full[k0], C_FULL, "-"),
                              (f"LoRA r={rank}", lora[k0], C_LORA, "-"),
                              ("random Gaussian", rnd, "#444", "--")):
        e = np.cumsum(spec ** 2) / np.sum(spec ** 2)
        ax.plot(np.arange(1, len(e) + 1), e, color=c, ls=ls, lw=2.2, label=name)
        k90 = int(np.searchsorted(e, 0.9) + 1)
        ax.plot([k90], [0.9], "o", color=c)
        ax.annotate(f"{k90}", (k90, 0.9), textcoords="offset points", xytext=(4, -16),
                    color=c, fontsize=11)
    ax.axhline(0.9, color="#999", lw=1, ls=":")
    ax.set_xlabel("number of singular directions k")
    ax.set_ylabel("fraction of ‖ΔW‖² in the top k")
    ax.set_title(f"Cumulative energy, {k0.replace('.weight', '')} (384×128)")
    ax.legend(fontsize=10, loc="lower right")
    fig.tight_layout()
    save(fig, "delta_spectrum.png")


def main():
    os.makedirs(OUT, exist_ok=True)
    torch.set_num_threads(fp.THREADS)
    t0 = time.time()
    res, args, full, lora, rnd = get_results()
    print("rank sweep (LoRA SFT at r = 1, 2, 4, 16)...")
    sweep = rank_sweep(args)
    fig_lora_diagram()
    fig_params_vs_rank(sweep, res)
    fig_loss_mask()
    fig_sft_curves(res, sweep)
    fig_results(res)
    fig_dpo(res)
    fig_spectrum(full, lora, rnd, args["rank"])
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
