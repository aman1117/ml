"""
Draw the pictures used in serving/README.md and save them to serving/images/.

Every number in every figure is computed here or read from a real run:
  - batching.py: replayed request traces (static vs continuous), the paged KV
    cache, the max-batch sweep, and this CPU's roofline (measured);
  - quantize_numpy.py: size / quality of every quantization scheme, the outlier test;
  - speculative.py: acceptance rate and speed-up vs γ, and the distribution test;
  - serving_pytorch.py: batched decode throughput of a 53M-parameter GPT;
  - the server's request log serving/logs/requests.jsonl (from load_test.py; run here
    if it does not exist, ~25 s) and monitor.py's analysis of it.

Run (~90 s, or ~65 s when the load-test log exists):
    python serving/make_figures.py
"""
from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from batching import make_trace, measure_roofline, simulate, summarize  # noqa: E402
from common import LOG_DIR, CharTokenizer, get_model, percentile, train_ids  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "images")
DPI = 130
LOG = os.path.join(LOG_DIR, "requests.jsonl")

C_STATIC, C_CONT, C_PAGED = "#d62728", "#2ca02c", "#1f77b4"
C_GREY, C_ORANGE, C_PURPLE = "#7f7f7f", "#ff7f0e", "#9467bd"
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


def box(ax, x, y, text, fc="white", ec="black", w=2.0, h=1.0, fs=11, lw=1.6):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=fc, ec=ec, lw=lw, zorder=2))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, zorder=3)


def arrow(ax, a, b, color="black", lw=1.8, text=None, fs=10, dy=0.18, **kw):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=16, color=color,
                                 lw=lw, zorder=1, **kw))
    if text:
        ax.text((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + dy, text, ha="center", va="bottom",
                fontsize=fs, color=color)


# ----------------------------------------------------------------------------
def load_or_run_logs():
    import monitor
    recs = []
    if os.path.exists(LOG):
        recs, _ = monitor.load_logs(LOG)
    if len(recs) < 500 or len({r["model_version"] for r in recs}) < 2:
        print("no full load-test log yet: running python serving/load_test.py --spawn-server "
              "(~25 s)")
        import subprocess
        subprocess.run([sys.executable, os.path.join(HERE, "load_test.py"), "--spawn-server"],
                       check=True, stdout=subprocess.DEVNULL)
        recs, _ = monitor.load_logs(LOG)
    s, tax, thr, versions, dr, al = monitor.analyze(recs)
    return recs, dict(slis=s, tax=tax, versions=versions, drift=dr, alerts=al)


def fig_lifecycle(recs):
    ok = [r for r in recs if r["status"] == 200 and not r.get("error_type")]
    q = np.median([r["queue_s"] for r in ok]) * 1e3
    ttft = np.median([r["ttft_s"] for r in ok]) * 1e3
    lat = np.median([r["latency_s"] for r in ok]) * 1e3
    tpot = np.median([r["tpot_s"] for r in ok if r.get("tpot_s")]) * 1e3
    n = np.median([r["output_tokens"] for r in ok])
    fig, ax = plt.subplots(figsize=(15, 6.2))
    ax.set_xlim(0, 15.5)
    ax.set_ylim(-1.2, 5.2)
    ax.axis("off")
    y = 3.8
    stages = [(1.0, "client\nPOST /v1/\ncompletions", "#dbe9f6"),
              (3.2, "handler thread\nparse + validate\n(400 if bad)", "#ece4f5"),
              (5.6, "bounded queue\n(429 if full)", "#fdf1c7"),
              (8.0, "engine thread\nadmit: KV blocks\n+ PREFILL", "#ffe8cc"),
              (10.6, "DECODE steps\n(batched with\nother requests)", "#ffe8cc"),
              (13.2, "stream tokens\n(SSE chunks)\nor JSON", "#e3f4e1")]
    for x, t, fc in stages:
        box(ax, x, y, t, fc=fc, w=2.0, h=1.4, fs=10.5)
    for (x1, _, _), (x2, _, _) in zip(stages[:-1], stages[1:]):
        arrow(ax, (x1 + 1.0, y), (x2 - 1.0, y))
    ax.text(10.6, y + 0.85, "↻ one token per step,\nuntil max_tokens or stop", ha="center",
            fontsize=10, color=C_ORANGE)
    box(ax, 13.2, 1.6, "1 JSON log line\n+ /metrics update", fc="#fbe3e3", w=2.4, h=1.0,
        fs=10.5)
    arrow(ax, (13.2, y - 0.7), (13.2, 2.1))
    box(ax, 9.4, 1.6, "monitor.py: SLIs, failure modes,\ndrift, alerts, eval cases", fc="white",
        w=4.0, h=1.0, fs=10.5)
    arrow(ax, (12.0, 1.6), (11.4, 1.6))
    box(ax, 4.4, 1.6, "timeout_s (504), client gone (499),\nslow body (408), shutdown (503)",
        fc="#f2f2f2", w=4.6, h=1.0, fs=10)
    ax.text(4.4, 0.85, "every stage can fail; every failure is logged with a type",
            ha="center", fontsize=9.5, color=C_GREY)
    # median timeline from the load test
    scale = 13.5 / lat
    x0 = 1.0
    ax.add_patch(plt.Rectangle((x0, -0.9), q * scale, 0.45, color="#fdf1c7", ec="black"))
    ax.add_patch(plt.Rectangle((x0 + q * scale, -0.9), (ttft - q) * scale, 0.45,
                               color=C_ORANGE, ec="black"))
    ax.add_patch(plt.Rectangle((x0 + ttft * scale, -0.9), (lat - ttft) * scale, 0.45,
                               color="#2ca02c", alpha=0.6, ec="black"))
    ax.text(x0 + 0.05, -0.68, "queue +\nprefill", fontsize=8, va="center")
    ax.text(x0 + (ttft + (lat - ttft) / 2) * scale, -0.68,
            "decode: one engine step per token, shared with the other running requests",
            fontsize=10, ha="center", va="center")
    ax.text(x0, -0.3, f"median request in the load test: queue {q:.1f} ms | prefill + first "
            f"token -> TTFT {ttft:.1f} ms | {n:.0f} tokens at TPOT {tpot:.2f} ms -> "
            f"latency {lat:.0f} ms", fontsize=10.5)
    ax.set_title("The life of a request in serving/server.py", fontsize=15)
    save(fig, "request_lifecycle.png")


def fig_roofline(rf, pd):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4))
    ax = axes[0]
    ai = np.logspace(-1, 4, 200)
    bw, peak = 2.0, 312.0  # A100: 2 TB/s HBM, 312 TFLOP/s bf16
    ax.loglog(ai, np.minimum(peak, bw * ai), color="black", lw=2)
    ax.axvline(peak / bw, color=C_GREY, ls=":")
    ax.text(peak / bw * 1.1, 1.0, f"ridge ≈ {peak / bw:.0f}\nFLOPs/byte", fontsize=10,
            color=C_GREY)
    for b in (1, 8, 64):
        ax.plot(b, min(peak, bw * b), "o", color=C_PAGED, ms=9)
        ax.annotate(f"decode, batch {b}", (b, min(peak, bw * b)), textcoords="offset points",
                    xytext=(8, -14), fontsize=10, color=C_PAGED)
    ax.plot(2048, peak, "s", color=C_ORANGE, ms=9)
    ax.annotate("prefill, 2048 tokens", (2048, peak), textcoords="offset points",
                xytext=(-60, -22), fontsize=10, color=C_ORANGE)
    ax.text(3, 0.2, "memory-bound:\ntime = bytes / bandwidth", fontsize=10.5)
    ax.text(300, 450, "compute-bound", fontsize=10.5)
    ax.set_xlabel("arithmetic intensity (FLOPs per byte moved)")
    ax.set_ylabel("attainable TFLOP/s")
    ax.set_title("Roofline sketch: 7B model, fp16, A100\n(intensity of a weight matmul ≈ batch "
                 "tokens)", fontsize=12.5)
    ax.set_ylim(0.1, 1000)

    ax = axes[1]
    pts = rf["points"]
    x = np.array([p["ai"] for p in pts])
    y = np.array([p["gflops"] for p in pts])
    xs = np.logspace(-1, 3.2, 200)
    bwc = rf["bandwidth"] / 1e9
    ax.loglog(xs, np.minimum(rf["peak"], bwc * xs), color="black", lw=2,
              label=f"roof: copy bandwidth {bwc:.0f} GB/s, best matmul {rf['peak']:.0f} GFLOP/s")
    ax.loglog(x, y, "o-", color=C_PAGED, ms=7, label="measured y = x @ W (W: 4096², fp32)")
    for p in pts:
        if p["b"] in (1, 2, 16, 128, 2048) and not (p["b"] == 2):
            ax.annotate(f"b={p['b']}", (p["ai"], p["gflops"]), textcoords="offset points",
                        xytext=(6, -14), fontsize=9.5)
    ax.set_xlabel("arithmetic intensity (FLOPs per byte)")
    ax.set_ylabel("GFLOP/s")
    ax.set_title(f"Measured on this CPU: b rows share one read of W\n(b=2..{pts[6]['b']}: "
                 f"~{pts[3]['ms']:.1f} ms whatever b, i.e. memory/kernel bound)", fontsize=12.5)
    ax.annotate("b=2", (pts[1]["ai"], pts[1]["gflops"]), textcoords="offset points",
                xytext=(6, -4), fontsize=9.5)
    ax.legend(fontsize=9.5, loc="lower right")
    fig.suptitle("Why decode is memory-bound and prefill is compute-bound", y=1.02)
    fig.tight_layout()
    save(fig, "roofline.png")


def gantt(ax, reqs, title, t_max, color):
    for i, r in enumerate(sorted(reqs, key=lambda r: r.arrival)):
        ax.barh(i, (r.t_first - r.arrival) * 1e3, left=r.arrival * 1e3, color="#dddddd",
                edgecolor="none", height=0.8)
        ax.barh(i, (r.token_times[-1] - r.t_first) * 1e3, left=r.t_first * 1e3, color=color,
                height=0.8, alpha=0.85)
        if r.t_finish > r.token_times[-1] + 1e-6:  # static: waits for the batch's longest row
            ax.barh(i, (r.t_finish - r.token_times[-1]) * 1e3, left=r.token_times[-1] * 1e3,
                    color=color, alpha=0.25, height=0.8, hatch="///", edgecolor=color)
        ax.plot(r.arrival * 1e3, i, "|", color="black", ms=8)
    ax.set_xlim(0, t_max * 1e3)
    ax.set_ylabel("request (by arrival)")
    ax.set_title(title, fontsize=12.5, loc="left")
    ax.invert_yaxis()


def fig_gantt(model, data):
    trace = make_trace(data, 30, 300.0, seed=1)
    res = {p: simulate(model, trace, p, max_batch=4) for p in ("static", "continuous")}
    t_max = max(r.t_finish for p in res.values() for r in p[0]) * 1.02
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    for ax, p, c in zip(axes, ("static", "continuous"), (C_STATIC, C_CONT)):
        reqs, info = res[p]
        s = summarize(reqs, info["timeline"])
        gantt(ax, reqs, f"{p} batching, max batch 4: p50 latency {s['latency_p50'] * 1e3:.1f} ms, "
                        f"p99 {s['latency_p99'] * 1e3:.1f} ms, {s['throughput']:,.0f} tokens/s",
              t_max, c)
    axes[1].set_xlabel("time (ms; compute is real, idle gaps are simulated)")
    from matplotlib.patches import Patch
    axes[0].legend(handles=[Patch(color="#dddddd", label="waiting (queue + prefill)"),
                            Patch(color=C_STATIC, alpha=0.85, label="generating tokens"),
                            Patch(facecolor=C_STATIC, alpha=0.25, hatch="///",
                                  label="done, waiting for the batch's longest request")],
                   loc="upper right", fontsize=10)
    fig.suptitle("30 Poisson arrivals (300 req/s) replayed on the real mini-GPT", y=1.0)
    fig.tight_layout()
    save(fig, "batching_gantt.png")


def fig_kv(model, data):
    trace = make_trace(data, 300, 300.0)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2), gridspec_kw=dict(width_ratios=[1.4, 1, 1]))
    ax = axes[0]
    reqs_s, info_s = simulate(model, trace, "static", 16)
    reqs_c, info_c = simulate(model, trace, "continuous", 16)
    for info, key, c, label, ls in (
            (info_s, "reserved_tokens", C_STATIC, "static: preallocated rectangles", "-"),
            (info_c, "contiguous_tokens", C_ORANGE, "continuous, contiguous max-length "
                                                    "reservation", "-"),
            (info_c, "reserved_tokens", C_PAGED, "continuous, paged (blocks of 8)", "-"),
            (info_c, "used_tokens", "black", "continuous: tokens actually cached", ":")):
        t = [s["t1"] * 1e3 for s in info["timeline"]]
        ax.step(t, [s[key] for s in info["timeline"]], color=c, lw=1.4, label=label,
                where="post", ls=ls)
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("KV cache (tokens; 1 KB each)")
    ax.set_title("KV memory over the replayed trace", fontsize=12.5)
    ax.legend(fontsize=9.5, loc="upper right", ncol=2)
    ax.set_ylim(0, 720)

    ax = axes[1]
    snap = max(info_c["timeline"], key=lambda s: len(s["tables"]))
    owners = np.full(16 * 4, -1)
    for k, (rid, table) in enumerate(sorted(snap["tables"].items())):
        for b in table:
            owners[b] = k
    grid = owners.reshape(8, 8)
    cmap = plt.get_cmap("tab20")
    img = np.ones((8, 8, 3))
    for (i, j), o in np.ndenumerate(grid):
        if o >= 0:
            img[i, j] = cmap(o % 20)[:3]
    ax.imshow(img)
    for (i, j), o in np.ndenumerate(grid):
        ax.text(j, i, str(i * 8 + j), ha="center", va="center", fontsize=8,
                color="black" if o >= 0 else "#bbbbbb")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"The 64 physical blocks at t = {snap['t1'] * 1e3:.0f} ms:\n"
                 f"{len(snap['tables'])} requests (colors), {np.sum(owners < 0)} free (white)",
                 fontsize=12)

    ax = axes[2]
    sizes = [1, 2, 4, 8, 16, 32]
    waste = []
    for bs in sizes:
        _, info = simulate(model, trace, "continuous", 16, block_size=bs)
        w = np.array([s["t1"] - s["t0"] for s in info["timeline"]])
        used = np.array([s["used_tokens"] for s in info["timeline"]])
        res = np.array([s["reserved_tokens"] for s in info["timeline"]])
        waste.append(1 - (used * w).sum() / (res * w).sum())
    cont = 1 - (np.array([s["used_tokens"] for s in info_c["timeline"]]) *
                np.array([s["t1"] - s["t0"] for s in info_c["timeline"]])).sum() / \
        (np.array([s["contiguous_tokens"] for s in info_c["timeline"]]) *
         np.array([s["t1"] - s["t0"] for s in info_c["timeline"]])).sum()
    stat = 1 - (np.array([s["used_tokens"] for s in info_s["timeline"]]) *
                np.array([s["t1"] - s["t0"] for s in info_s["timeline"]])).sum() / \
        (np.array([s["reserved_tokens"] for s in info_s["timeline"]]) *
         np.array([s["t1"] - s["t0"] for s in info_s["timeline"]])).sum()
    labels = [f"paged\nbs={b}" for b in sizes] + ["contig.\nmax-len", "static\nrect."]
    vals = waste + [cont, stat]
    colors = [C_PAGED] * len(sizes) + [C_ORANGE, C_STATIC]
    bars = ax.bar(range(len(vals)), np.array(vals) * 100, color=colors)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v * 100 + 1, f"{v:.0%}", ha="center", fontsize=9.5)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("reserved but unused KV memory (%)")
    ax.set_title("Waste vs allocation scheme\n(time-averaged)", fontsize=12.5)
    fig.tight_layout()
    save(fig, "kv_paging.png")
    return dict(waste=dict(zip(sizes, waste)), contiguous=cont, static=stat)


def fig_tradeoff(model, data):
    from serving_pytorch import throughput_curve
    sat = make_trace(data, 300, 3000.0)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4))
    ax = axes[0]
    for policy, c in (("static", C_STATIC), ("continuous", C_CONT)):
        xs, ys, bs = [], [], [1, 2, 4, 8, 16, 32, 64]
        for mb in bs:
            reqs, info = simulate(model, sat, policy, mb)
            s = summarize(reqs, info["timeline"])
            xs.append(s["throughput"])
            ys.append(s["tpot_p50"] * 1e3)
        ax.plot(xs, ys, "o-", color=c, label=policy, lw=2)
        for x, y, b in zip(xs, ys, bs):
            ax.annotate(str(b), (x, y), textcoords="offset points", xytext=(5, 4), fontsize=9,
                        color=c)
    ax.set_xlabel("throughput (tokens/s, saturated)")
    ax.set_ylabel("time per output token, p50 (ms)")
    ax.set_title("NumPy mini-GPT: max batch 1 → 64 (labels)", fontsize=12.5)
    ax.legend()
    ax = axes[1]
    tc = throughput_curve()
    B = [r["B"] for r in tc["rows"]]
    ax.plot(B, [r["tok_s"] for r in tc["rows"]], "o-", color=C_PAGED, lw=2, label="tokens/s")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("batch size")
    ax.set_ylabel("decode tokens/s", color=C_PAGED)
    ax2 = ax.twinx()
    ax2.plot(B, [r["step_ms"] for r in tc["rows"]], "s--", color=C_ORANGE, lw=1.5)
    ax2.set_ylabel("ms per decode step", color=C_ORANGE)
    ax2.spines["right"].set_visible(True)
    r = {row["B"]: row for row in tc["rows"]}
    ax.set_title(f"PyTorch, {tc['n_params'] / 1e6:.0f}M-param GPT, fp32, CPU: batch 2 -> 16 "
                 f"steps take\n{r[2]['step_ms']:.1f} -> {r[16]['step_ms']:.1f} ms (8x the tokens), "
                 f"batch 256: {r[256]['step_ms']:.0f} ms", fontsize=12)
    fig.suptitle("The throughput-latency trade-off of the batch size", y=1.02)
    fig.tight_layout()
    save(fig, "throughput_latency.png")


SHORT = {"fp32": "fp32", "fp16": "fp16", "int8 per-tensor": "int8\ntensor",
         "int8 per-channel": "int8\nchan.", "int4 per-channel": "int4\nchan.",
         "int4 group 64": "int4\ng64", "int4 group 32": "int4\ng32"}


def fig_quant(model, tok):
    from quantize_numpy import run_all
    res = run_all(model, tok)
    rows = res["rows"]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    labels = [SHORT[r["label"]] for r in rows]
    x = np.arange(len(rows))
    ax = axes[0]
    ax.bar(x, [r["bytes"] / 1024 for r in rows], color=C_PAGED)
    for i, r in enumerate(rows):
        ax.text(i, r["bytes"] / 1024 + 5, f"{rows[0]['bytes'] / r['bytes']:.1f}x", ha="center",
                fontsize=9.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylabel("model size (KB)")
    ax.set_title("Size (embeddings + LayerNorms stay fp32)", fontsize=12.5)
    ax = axes[1]
    ax.bar(x, [100 * (1 - r["agree"]) for r in rows], 0.6, color=C_PURPLE)
    ax.set_ylabel("% held-out next-char argmax ≠ fp32", color=C_PURPLE)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5)
    ax2 = ax.twinx()
    ax2.plot(x, [r["train_loss"] - rows[0]["train_loss"] for r in rows], "o-", color=C_ORANGE)
    ax2.set_ylabel("Δ loss on the (memorized) training text", color=C_ORANGE)
    ax2.spines["right"].set_visible(True)
    ax.set_title("Quality cost: int8 ≈ free, int4 is not", fontsize=12.5)
    ax = axes[2]
    out = res["outlier"]
    names = [r["label"] for r in out]
    base = {r["label"]: r for r in rows}
    xo = np.arange(len(out))
    ax.bar(xo - 0.2, [100 * (1 - base[n]["agree"]) for n in names], 0.4, color=C_GREY,
           label="original weights")
    ax.bar(xo + 0.2, [100 * (1 - r["agree"]) for r in out], 0.4, color=C_STATIC,
           label="one W_v column x50 (same function)")
    for i, r in enumerate(out):
        if r["label"] != "fp32":
            ax.text(i + 0.2, 100 * (1 - r["agree"]) + 0.3, f"{r['zero_frac']:.0%} of\nother "
                    f"weights\n-> 0", ha="center", fontsize=8.5)
    ax.set_xticks(xo)
    ax.set_xticklabels([SHORT[n] for n in names], fontsize=9.5)
    ax.set_ylabel("% held-out next-char argmax ≠ fp32")
    ax.set_ylim(0, max(100 * (1 - r["agree"]) for r in out) * 1.35)
    ax.set_title("The outlier problem", fontsize=12.5)
    ax.legend(fontsize=9.5, loc="upper left")
    fig.suptitle("Weight quantization of the mini-GPT (quantize_numpy.py)", y=1.02)
    fig.tight_layout()
    save(fig, "quantization.png")


def fig_speculative(target, tok):
    from speculative import (TEST_PROMPT, GPTDraft, NGramDraft, distribution_test,
                             exact_marginals, expected_tokens, get_draft, sweep)
    draft = get_draft(target, tok.chars)
    prompts = [tok.encode(p)[0] for p in ("The keeper", "One morning", "The girl saw",
                                          "\"Good morning,\"", "The sea was")]
    gammas = (1, 2, 3, 4, 5, 6, 8)
    ngram = NGramDraft(train_ids(tok), len(tok))
    res = {"GPT draft (1 layer, d=32)": sweep(target, GPTDraft(draft), prompts, 150, gammas, 0.8),
           "6-gram draft (counts)": sweep(target, ngram, prompts, 150, gammas, 0.8)}
    prompt = tok.encode(TEST_PROMPT)[0]
    N = 20_000
    dist, _ = distribution_test(target, draft, prompt, N)
    exact = exact_marginals(target, prompt, 1.0)
    fig, axes = plt.subplots(1, 4, figsize=(21, 5))
    for (name, (plain, rows)), c in zip(res.items(), (C_PURPLE, C_CONT)):
        g = [r["gamma"] for r in rows]
        axes[0].plot(g, [r["alpha"] for r in rows], "o-", color=c, label=name, lw=2)
        axes[1].plot(g, [r["tokens_per_call"] for r in rows], "o-", color=c, label=name, lw=2)
        axes[1].plot(g, [expected_tokens(r["alpha"], r["gamma"]) for r in rows], ":", color=c,
                     lw=1.5)
        axes[2].plot(g, [r["speedup"] for r in rows], "o-", color=c, label=name, lw=2)
    axes[0].set_title("Acceptance rate α per drafted token", fontsize=12.5)
    axes[0].set_ylim(0, 1)
    axes[1].set_title("Tokens per target forward pass\n(dotted: (1 - α^(γ+1)) / (1 - α))",
                      fontsize=12.5)
    axes[2].axhline(1, color=C_GREY, ls=":")
    axes[2].set_ylabel("wall-clock speed-up vs plain decoding")
    axes[2].set_title("Speed-up (NumPy): the draft must be CHEAP\n(GPT draft = 0.5x a target "
                      "step, n-gram = 0.02x)", fontsize=12.5)
    for ax in axes[:3]:
        ax.set_xlabel("γ (drafted tokens per iteration)")
        ax.legend(fontsize=9.5)
    ax = axes[3]
    e2 = exact[1]
    top = np.argsort(e2)[::-1][:8]
    rng = np.random.default_rng(1)
    from speculative import speculative_generate
    counts = np.zeros(len(e2))
    for _ in range(5000):
        counts[speculative_generate(target, draft, prompt, 2, 3, 1.0, rng=rng)[1]] += 1
    ax.bar(np.arange(8) - 0.2, e2[top], 0.4, color=C_GREY, label="target alone, exact")
    ax.bar(np.arange(8) + 0.2, counts[top] / counts.sum(), 0.4, color=C_CONT,
           label="speculative (γ=3), 5,000 samples")
    ax.set_xticks(range(8))
    ax.set_xticklabels([repr(tok.chars[i]) for i in top])
    ax.set_title(f"2nd generated char after {TEST_PROMPT!r} (temperature 1)\n"
                 f"20k-run test: TV {dist[1]['tv']:.4f}, chi² {dist[1]['chi2']:.0f} < "
                 f"{dist[1]['crit']:.0f}", fontsize=12.5)
    ax.legend(fontsize=9.5)
    fig.tight_layout()
    save(fig, "speculative.png")


def fig_latency_cdf(recs):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    versions = sorted({r["model_version"] for r in recs})
    for ax, key, title in ((axes[0], "latency_s", "end-to-end latency"),
                           (axes[1], "ttft_s", "time to first token")):
        for v, c in zip(versions, (C_PAGED, C_ORANGE)):
            xs = np.sort([r[key] * 1e3 for r in recs if r["model_version"] == v
                          and r["status"] == 200 and not r.get("error_type") and r.get(key)])
            ax.plot(xs, np.arange(1, len(xs) + 1) / len(xs), color=c, lw=2,
                    label=f"{v} ({len(xs)} ok)")
        allx = [r[key] * 1e3 for r in recs if r["status"] == 200 and not r.get("error_type")
                and r.get(key)]
        for q, ls in ((50, "-"), (95, "--"), (99, ":")):
            p = percentile(allx, q)
            ax.axvline(p, color=C_GREY, ls=ls, lw=1)
            ax.text(p, 0.03 + 0.08 * (q == 95) + 0.16 * (q == 99), f" p{q} {p:.0f} ms ",
                    fontsize=9.5, color=C_GREY, ha="right" if q == 95 else "left")
        ax.set_xscale("log")
        ax.set_xlabel("ms (log scale)")
        ax.set_ylabel("fraction of successful requests ≤ x")
        ax.set_title(title, fontsize=12.5)
        ax.legend(fontsize=9.5, loc="upper left")
    fig.suptitle("Latency CDF from the load test (serving/logs/requests.jsonl)", y=1.02)
    fig.tight_layout()
    save(fig, "latency_cdf.png")


def fig_taxonomy(mon):
    import monitor
    tax = mon["tax"]
    n = mon["slis"]["n"]
    names = list(tax)[::-1]
    counts = [tax[m]["count"] for m in names]
    colors = [C_STATIC if monitor.MODES[m][1] else C_PAGED for m in names]
    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.barh(names, counts, color=colors)
    for b, cnt, m in zip(bars, counts, names):
        fs = tax[m]["first_seen"]
        ax.text(b.get_width() + 3, b.get_y() + b.get_height() / 2,
                f"{cnt} ({cnt / n:.1%})" + (f", first at {fs:.1f} s" if fs is not None else ""),
                va="center", fontsize=9.5)
    ax.set_xlabel("requests (a request can have several modes)")
    ax.set_xlim(0, max(counts) * 1.4)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=C_STATIC, label="counts against the availability SLO"),
                       Patch(color=C_PAGED, label="quality / input / client issue")],
              loc="lower right", fontsize=10)
    ax.set_title(f"Failure modes in {n} logged requests (monitor.py)")
    save(fig, "failure_taxonomy.png")


def fig_drift(mon):
    import monitor
    dr = mon["drift"]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    ax = axes[0]
    t = [w["t"] for w in dr["windows"]]
    ax.plot(t, [w["psi_len"] for w in dr["windows"]], "o-", color=C_PAGED, label="prompt length")
    ax.plot(t, [w["psi_chars"] for w in dr["windows"]], "s-", color=C_ORANGE,
            label="character histogram")
    ax.axhline(0.1, color=C_GREY, ls=":", lw=1)
    ax.axhline(0.25, color=C_STATIC, ls="--", lw=1)
    ax.text(t[0], 0.27, "0.25: major shift", color=C_STATIC, fontsize=9.5)
    ax.set_yscale("log")
    ax.set_ylim(0.005, 20)
    ax.set_xlabel("time in the load test (s)")
    ax.set_ylabel("PSI vs the first 30% (reference)")
    ax.set_title("Drift over time windows", fontsize=12.5)
    ax.legend(fontsize=10)
    ax = axes[1]
    k = len(dr["ref_len"])
    r = dr["ref_len"] / dr["ref_len"].sum()
    c = dr["cur_len"] / dr["cur_len"].sum()
    ax.bar(np.arange(k) - 0.2, r, 0.4, color=C_GREY, label="reference")
    ax.bar(np.arange(k) + 0.2, c, 0.4, color=C_PAGED, label="last 10%")
    edges = [0] + [int(e) for e in dr["edges"]]
    ax.set_xticks(range(k))
    ax.set_xticklabels([f">{e}" for e in edges], fontsize=8.5, rotation=45)
    ax.set_xlabel("prompt length bin (chars; reference deciles)")
    ax.set_title(f"Prompt length: PSI {dr['psi_len']:.2f}", fontsize=12.5)
    ax.legend(fontsize=10)
    ax = axes[2]
    g = monitor.VOCAB_GROUPS
    r = dr["ref_groups"] / dr["ref_groups"].sum()
    c = dr["cur_groups"] / dr["cur_groups"].sum()
    ax.bar(np.arange(len(g)) - 0.2, r, 0.4, color=C_GREY, label="reference")
    ax.bar(np.arange(len(g)) + 0.2, c, 0.4, color=C_ORANGE, label="last 10%")
    ax.set_xticks(range(len(g)))
    ax.set_xticklabels(g)
    ax.set_title(f"Character classes: PSI {dr['psi_groups']:.2f}\n(per-char PSI "
                 f"{dr['psi_chars']:.2f}, KL {dr['kl_chars']:.2f})", fontsize=12.5)
    ax.legend(fontsize=10)
    fig.suptitle("Input drift: half the clients start sending long ALL-CAPS prompts at 10 s",
                 y=1.02)
    fig.tight_layout()
    save(fig, "drift_psi.png")


def fig_dashboard(recs, mon):
    import monitor
    t0 = recs[0]["t"]
    t = np.array([r["t"] - t0 for r in recs])
    edges = np.arange(0, t.max() + 1.0, 1.0)
    mid = edges[:-1] + 0.5
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    ax = axes[0, 0]
    groups = [("2xx", lambda r: r["status"] == 200 and not r.get("error_type"), C_CONT),
              ("4xx client", lambda r: r["status"] in (400, 408, 411, 413, 499), C_PAGED),
              ("429", lambda r: r["status"] == 429, C_ORANGE),
              ("5xx / timeout", lambda r: r["status"] >= 500 or r.get("error_type") == "timeout",
               C_STATIC)]
    bottom = np.zeros(len(mid))
    for name, f, c in groups:
        h = np.histogram([tt for tt, r in zip(t, recs) if f(r)], edges)[0]
        ax.bar(mid, h, 1.0, bottom=bottom, color=c, label=name)
        bottom += h
    ax.set_title("requests / s by outcome", fontsize=12.5)
    ax.legend(fontsize=9)
    ax = axes[0, 1]
    for q, ls in ((50, "-"), (95, "--"), (99, ":")):
        ys = []
        for a, b in zip(edges[:-1], edges[1:]):
            xs = [r["latency_s"] * 1e3 for tt, r in zip(t, recs) if a <= tt < b and
                  r["status"] == 200 and not r.get("error_type")]
            ys.append(percentile(xs, q) if xs else np.nan)
        ax.plot(mid, ys, ls, color=C_PAGED, label=f"p{q}")
    ax.axhline(mon["slis"]["latency_slo_s"] * 1e3, color=C_STATIC, lw=1)
    ax.set_yscale("log")
    ax.set_title("latency (ms), SLO line at 500 ms", fontsize=12.5)
    ax.legend(fontsize=9)
    ax = axes[0, 2]
    toks = np.histogram(t, edges, weights=[(r.get("output_tokens") or 0)
                                           if r["status"] == 200 else 0 for r in recs])[0]
    ax.plot(mid, toks, color=C_CONT)
    ax.fill_between(mid, toks, color=C_CONT, alpha=0.2)
    ax.set_title("tokens generated / s", fontsize=12.5)
    ax = axes[1, 0]
    qd = [r.get("queue_depth") or 0 for r in recs]
    ax.plot(t, qd, ".", ms=3, color=C_ORANGE)
    ax.set_title("queue depth seen at arrival (both models)", fontsize=12.5)
    ax = axes[1, 1]
    for (name, lw_, sw, thr), c in zip(monitor.BURN_RULES, (C_STATIC, C_PURPLE)):
        tl, bl = monitor.burn_rates(recs, lw_, mon["slis"]["slo"])
        ax.plot(tl, bl, color=c, label=f"{lw_:g} s window ({name}, threshold {thr})")
        ax.axhline(thr, color=c, ls=":", lw=1)
    ax.set_yscale("log")
    ax.set_ylim(0.3, 100)
    ax.set_title("availability error-budget burn rate (99% SLO)", fontsize=12.5)
    ax.legend(fontsize=8.5)
    ax = axes[1, 2]
    for v, c in zip(sorted({r["model_version"] for r in recs}), (C_PAGED, C_ORANGE)):
        mine = [(tt, "degenerate_repetition" in r["modes"]) for tt, r in zip(t, recs)
                if r["model_version"] == v and r["status"] == 200 and not r.get("error_type")]
        tt = np.array([m[0] for m in mine])
        bad = np.cumsum([m[1] for m in mine])
        ax.step(tt, 100 * bad / np.arange(1, len(mine) + 1), color=c, where="post",
                label=f"{v}: {bad[-1]}/{len(mine)}")
    ax.set_ylabel("%")
    ax.set_title("degenerate outputs so far (cumulative %), by version", fontsize=12.5)
    ax.legend(fontsize=9)
    canary_t = min(r["t"] for r in recs if r["model_version"] != recs[0]["model_version"]) - t0
    for ax in axes.ravel():
        ax.axvline(canary_t, color=C_GREY, ls="--", lw=1)
        ax.set_xlabel("time (s)")
    axes[1, 2].text(canary_t, axes[1, 2].get_ylim()[1] * 0.95, " canary", color=C_GREY,
                    fontsize=9.5, va="top")
    firing = [n for n, f, _ in mon["alerts"] if f]
    fig.suptitle("A monitoring dashboard drawn from the request log  |  firing: "
                 + ", ".join(firing[:4]) + (" ..." if len(firing) > 4 else ""), y=1.0,
                 fontsize=13.5)
    fig.tight_layout()
    save(fig, "dashboard.png")


def main():
    os.makedirs(OUT, exist_ok=True)
    model, chars, _ = get_model()
    tok = CharTokenizer(chars)
    data = train_ids(tok)
    recs, mon = load_or_run_logs()
    fig_lifecycle(recs)
    fig_roofline(measure_roofline(), None)
    fig_gantt(model, data)
    kvres = fig_kv(model, data)
    print("  KV waste: " + ", ".join(f"bs={k}: {v:.0%}" for k, v in kvres["waste"].items())
          + f", contiguous {kvres['contiguous']:.0%}, static {kvres['static']:.0%}")
    fig_tradeoff(model, data)
    fig_quant(model, tok)
    fig_speculative(model, tok)
    fig_latency_cdf(recs)
    fig_taxonomy(mon)
    fig_drift(mon)
    fig_dashboard(recs, mon)


if __name__ == "__main__":
    main()
