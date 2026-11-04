"""
Draw the pictures used in agents/README.md and save them to agents/images/.

Every number in every figure is computed here:
  - the router: loaded from agents/router.npz (or trained, ~1 s) with its training history,
    and evaluated on router_test.jsonl;
  - the agent evals: agent_eval.run_all() is re-run (~12 s), so the figures always match
    the code (it also rewrites agents/runs/eval_results.json and the traces).

Run (~12 s):
    python agents/make_figures.py
"""
import os
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from agent_eval import TAXONOMY, run_all  # noqa: E402
from router_numpy import LABELS, confusion, featurize, load_jsonl, load_router  # noqa: E402

warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "images")
DPI = 130
C_ALL, C_NONE, C_SCRIPT, C_ACCENT = "#2ca02c", "#d62728", "#1f77b4", "#ff7f0e"

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


def box(ax, x, y, text, fc="white", ec="black", w=2.6, h=1.0, fs=11, lw=1.6, **kw):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.15",
                                fc=fc, ec=ec, lw=lw, zorder=2))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, zorder=3, **kw)


def arrow(ax, a, b, color="black", lw=1.8, text=None, tx=0.0, ty=0.0, fs=10, style="-|>", **kw):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=16, color=color, lw=lw, zorder=1, **kw))
    if text:
        ax.text((a[0] + b[0]) / 2 + tx, (a[1] + b[1]) / 2 + ty, text, ha="center", va="center", fontsize=fs,
                color=color, zorder=4, bbox=dict(fc="white", ec="none", pad=1))


# ----------------------------------------------------------------------------
def fig_agent_loop():
    fig, ax = plt.subplots(figsize=(15, 8.2))
    ax.set_xlim(0, 15.5)
    ax.set_ylim(-0.6, 8.6)
    ax.axis("off")
    F_MSG, F_LLM, F_GRD, F_TOOL, F_OBS = "#dbe9f6", "#ffe8cc", "#ece4f5", "#e3f4e1", "#fdf1c7"
    box(ax, 2.3, 7.4, "user request", fc="white", w=2.6, h=0.8)
    box(ax, 2.3, 5.4, "messages\nsystem: tool specs + rules\nuser · assistant · tool …",
        fc=F_MSG, w=3.6, h=1.5)
    box(ax, 7.0, 5.4, "POLICY (the LLM)\nScripted (router-driven) ·\nNoisy · HTTP (real model)",
        fc=F_LLM, w=3.6, h=1.5, ec="#ff7f0e")
    box(ax, 11.9, 5.4, "parse raw text\nAction JSON / Final Answer\n(fences, repairs, stop seq.)",
        fc=F_GRD, w=3.6, h=1.5, ec="#9467bd")
    box(ax, 11.9, 3.0, "validate args (JSON schema)\nloop detection\nconfirm mutation if tainted",
        fc=F_GRD, w=3.6, h=1.5, ec="#9467bd")
    box(ax, 7.0, 1.6, "execute tools  (parallel, timeout, sandbox)\ncalculator · search_docs · read_file\n"
        "unit_convert · kv_memory", fc=F_TOOL, w=4.6, h=1.5, ec="#2ca02c")
    box(ax, 2.3, 2.9, "observation, wrapped:\n<tool_output trust=\"untrusted\">\n… </tool_output>",
        fc=F_OBS, w=3.6, h=1.5)
    box(ax, 14.3, 7.4, "final answer\n→ user", fc="white", w=1.9, h=0.9)
    arrow(ax, (2.3, 7.0), (2.3, 6.15))
    arrow(ax, (4.1, 5.4), (5.2, 5.4), text="compact\ncontext", ty=0.55)
    arrow(ax, (8.8, 5.4), (10.1, 5.4), text="raw text", ty=0.3)
    arrow(ax, (11.9, 4.65), (11.9, 3.75), text="calls", tx=0.5)
    arrow(ax, (10.1, 2.4), (9.3, 1.9))
    arrow(ax, (4.7, 1.8), (4.1, 2.4))
    arrow(ax, (2.3, 3.65), (2.3, 4.65), text="append", tx=0.55)
    arrow(ax, (13.1, 6.15), (14.0, 6.95), text="grounding\ncheck", tx=0.9, ty=-0.15)
    arrow(ax, (11.9, 4.65), (8.8, 4.95), color="#d62728", text="error → feedback (retry)", ty=-0.32,
          connectionstyle="arc3,rad=-0.0", style="-|>", lw=1.4)
    ax.text(7.75, 8.25, "the agent loop:  repeat until Final Answer  or a budget runs out "
            "(max steps · max tokens · wall clock)", ha="center", fontsize=13, weight="bold")
    ax.text(7.0, 7.4, "router (learned, NumPy):\nwhich tool for each clause?", ha="center", va="center",
            fontsize=10, color="#ff7f0e", style="italic")
    arrow(ax, (7.0, 7.0), (7.0, 6.2), color="#ff7f0e", lw=1.2)
    ax.add_patch(FancyBboxPatch((10.0, -0.45), 5.3, 1.45, boxstyle="round,pad=0.02,rounding_size=0.15",
                                fc="#f5f5f5", ec="gray", ls="--", lw=1.2))
    ax.text(12.65, 0.28, "every step → trace.jsonl\n(prompt size, raw output, call, args,\nobservation, "
            "latency, error type)\n→ failure taxonomy → new eval tasks", ha="center", va="center", fontsize=10,
            color="#444")
    save(fig, "agent_loop.png")


def fig_router():
    model, hist = load_router()
    te_text, yte = load_jsonl("router_test.jsonl")
    pred = model.forward(featurize(te_text)).argmax(1)
    C = confusion(yte, pred)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2), gridspec_kw={"width_ratios": [1, 1, 1.25]})
    steps, loss = np.asarray(hist["step"]), np.asarray(hist["loss"])
    per_epoch = len(steps) // len(hist["epoch"])
    ax = axes[0]
    ax.plot(steps, loss, color=C_SCRIPT, alpha=0.35, lw=1, label="train loss (mini-batch)")
    ax.plot(np.asarray(hist["epoch"]) * per_epoch, hist["test_loss"], "o-", color=C_NONE, ms=4,
            label="test loss (held-out phrasings)")
    ax.axhline(np.log(len(LABELS)), color="gray", ls=":", lw=1)
    ax.text(steps[-1], np.log(len(LABELS)) * 1.08, f"ln 6 = {np.log(6):.2f} (guessing)", ha="right",
            color="gray", fontsize=10)
    ax.set_yscale("log")
    ax.set_xlabel("Adam step")
    ax.set_ylabel("cross-entropy")
    ax.set_title("Router training (NumPy, hand-written backward)")
    ax.legend(fontsize=10)
    ax = axes[1]
    ax.plot(hist["epoch"], hist["train_acc"], "o-", color=C_SCRIPT, ms=4, label="train")
    ax.plot(hist["epoch"], hist["test_acc"], "o-", color=C_NONE, ms=4, label="test (held-out phrasings)")
    ax.set_ylim(0.8, 1.01)
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy")
    ax.set_title(f"Accuracy: train {hist['train_acc'][-1]:.3f}, test {hist['test_acc'][-1]:.3f}")
    ax.legend(fontsize=10, loc="lower right")
    ax.text(0.6, 0.45, "the train/test gap is the cost\nof phrasings it never saw", transform=ax.transAxes,
            ha="center", fontsize=10, color="#444")
    ax = axes[2]
    Cn = C / C.sum(1, keepdims=True)
    ax.imshow(Cn, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            if C[i, j]:
                ax.text(j, i, str(C[i, j]), ha="center", va="center", fontsize=11,
                        color="white" if Cn[i, j] > 0.6 else "black")
    short = [l.replace("_", "\n") for l in LABELS]
    ax.set_xticks(range(len(LABELS)))
    ax.set_xticklabels(short, fontsize=9)
    ax.set_yticks(range(len(LABELS)))
    ax.set_yticklabels(LABELS, fontsize=10)
    ax.set_xlabel("predicted tool")
    ax.set_ylabel("true tool")
    ax.set_title(f"Test confusion matrix (accuracy {(pred == yte).mean():.1%})")
    ax.spines[:].set_visible(False)
    fig.tight_layout()
    save(fig, "router.png")


def fig_noise(R):
    noise = R["B"]["noise"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.4), gridspec_kw={"width_ratios": [1, 1.15, 1.15]})
    ax = axes[0]
    for g, c, lab in (("all", C_ALL, "all guardrails"), ("none", C_NONE, "no guardrails")):
        s = np.array([x["success"] for x in R["B"][g]])
        lo = np.array([x["ci"][0] for x in R["B"][g]])
        hi = np.array([x["ci"][1] for x in R["B"][g]])
        ax.plot(noise, s * 100, "o-", color=c, lw=2, label=lab)
        ax.fill_between(noise, lo * 100, hi * 100, color=c, alpha=0.15)
        for x, y in zip(noise, s):
            ax.text(x, y * 100 + (3 if g == "all" else -7), f"{y:.0%}", ha="center", fontsize=9, color=c)
    ax.set_xlabel("noise p (chance per step that the model makes a mistake)")
    ax.set_ylabel("task success (%)")
    ax.set_ylim(0, 108)
    ax.set_title("Success vs model noise (95% bootstrap CI)")
    ax.legend(loc="lower left")
    ax = axes[1]
    lad = R["C"]["ladder"]
    names = ["none"] + [f"+ {n}" for n, _ in lad[1:]]
    s = np.array([x[1]["success"] for x in lad]) * 100
    err = np.array([[x[1]["success"] - x[1]["ci"][0], x[1]["ci"][1] - x[1]["success"]] for x in lad]).T * 100
    cols = [C_NONE] + [C_ACCENT] * (len(lad) - 2) + [C_ALL]
    ax.barh(range(len(lad)), s, xerr=err, color=cols, alpha=0.85, capsize=3)
    for i, v in enumerate(s):
        ax.text(v + err[1][i] + 1.5, i, f"{v:.0f}%", va="center", fontsize=10)
    ax.set_yticks(range(len(lad)))
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlim(0, 112)
    ax.set_xlabel("task success (%) at p = 0.3")
    ax.set_title("Adding guardrails one at a time")
    ax = axes[2]
    loo = R["C"]["leave_one_out"]
    full = R["C"]["ladder"][-1][1]["success"]
    drops = np.array([(full - x[1]["success"]) * 100 for x in loo])
    order = np.argsort(-drops)
    ax.barh(range(len(loo)), drops[order], color=C_NONE, alpha=0.8)
    for i, v in enumerate(drops[order]):
        ax.text(v + 0.6, i, f"−{v:.1f} pts", va="center", fontsize=10)
    ax.set_yticks(range(len(loo)))
    ax.set_yticklabels([f"− {loo[i][0]}" for i in order])
    ax.invert_yaxis()
    ax.set_xlim(0, max(drops) * 1.25 + 1)
    ax.set_xlabel(f"success lost vs all guardrails ({full:.0%}), p = 0.3")
    ax.set_title("Removing one guardrail at a time")
    fig.tight_layout()
    save(fig, "guardrails_vs_noise.png")


def fig_pass_k(R):
    D = R["D"]
    k = D["k"]
    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    for g, c, lab in (("all", C_ALL, "all guardrails"), ("none", C_NONE, "no guardrails")):
        ax.plot(k, np.array(D[g]["pass_at_k"]) * 100, "o--", color=c, lw=1.8, label=f"pass@k, {lab}")
        ax.plot(k, np.array(D[g]["pass_hat_k"]) * 100, "s-", color=c, lw=2.4, label=f"pass^k, {lab}")
        ax.text(k[-1] + 0.15, D[g]["pass_hat_k"][-1] * 100, f"{D[g]['pass_hat_k'][-1]:.0%}", va="center",
                color=c, fontsize=10)
        ax.text(k[-1] + 0.15, D[g]["pass_at_k"][-1] * 100, f"{D[g]['pass_at_k'][-1]:.0%}", va="center",
                color=c, fontsize=10)
    ax.set_xlabel("k (independent attempts at the same task)")
    ax.set_ylabel("% of tasks")
    ax.set_xticks(k)
    ax.set_ylim(-3, 105)
    ax.set_title("pass@k (any of k succeeds) vs pass^k (all k succeed)\nNoisyPolicy p = 0.2, 8 runs per task")
    ax.legend(fontsize=10, loc="center left")
    save(fig, "pass_k.png")


def fig_taxonomy(R):
    none = R["C"]["ladder"][0][1]["taxonomy"]
    allg = R["B"]["all"][R["B"]["noise"].index(0.3)]["taxonomy"]
    n_runs = R["C"]["ladder"][0][1]["n_runs"]
    fig, ax = plt.subplots(figsize=(10, 5.4))
    y = np.arange(len(TAXONOMY))
    ax.barh(y - 0.2, [none[t] for t in TAXONOMY], 0.4, color=C_NONE, alpha=0.85, label="no guardrails")
    ax.barh(y + 0.2, [allg[t] for t in TAXONOMY], 0.4, color=C_ALL, alpha=0.85, label="all guardrails")
    for i, t in enumerate(TAXONOMY):
        ax.text(none[t] + 0.8, i - 0.2, str(none[t]), va="center", fontsize=10)
        ax.text(allg[t] + 0.8, i + 0.2, str(allg[t]), va="center", fontsize=10)
    ax.set_yticks(y)
    ax.set_yticklabels([t.replace("_", " ") for t in TAXONOMY])
    ax.invert_yaxis()
    ax.set_xlabel(f"failed runs (out of {n_runs} runs: 46 tasks × 5, p = 0.3)")
    a, n = R["C"]["ladder"][0][1]["taxonomy_agreement"]
    ax.set_title(f"Failure taxonomy, labeled automatically from the JSONL traces\n"
                 f"(labeler agrees with the injected failure on {a}/{n} = {a / n:.0%} of failed runs)")
    ax.legend(loc="lower right")
    save(fig, "failure_taxonomy.png")


def fig_steps_latency(R):
    i3 = R["B"]["noise"].index(0.3)
    series = [("scripted (p = 0)", R["A"]["steps"], R["A"]["latency"], C_SCRIPT),
              ("noisy p = 0.3, all guardrails", R["B"]["all"][i3]["steps_list"], R["B"]["all"][i3]["latency_list"], C_ALL),
              ("noisy p = 0.3, no guardrails", R["B"]["none"][i3]["steps_list"], R["B"]["none"][i3]["latency_list"], C_NONE)]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    ax = axes[0]
    bins = np.arange(0.5, 11.5, 1)
    for j, (lab, st, _, c) in enumerate(series):
        h, _ = np.histogram(st, bins=bins)
        ax.bar(np.arange(1, 11) + (j - 1) * 0.27, h / len(st) * 100, 0.27, color=c, alpha=0.85,
               label=f"{lab} (mean {np.mean(st):.2f})")
    ax.set_xticks(range(1, 11))
    ax.set_xlabel("steps (model calls) per run")
    ax.set_ylabel("% of runs")
    ax.set_title("Steps per run")
    ax.legend(fontsize=10)
    ax = axes[1]
    for lab, _, lat, c in series:
        lat = np.sort(lat)
        ax.plot(lat, np.arange(1, len(lat) + 1) / len(lat) * 100, color=c, lw=2,
                label=f"{lab}: p50 {np.percentile(lat, 50):.1f} s, p95 {np.percentile(lat, 95):.1f} s")
    ax.set_xlabel("latency per run (s, simulated: LLM latency model + tool latencies)")
    ax.set_ylabel("% of runs finished")
    ax.set_title("Latency distribution (CDF)")
    ax.legend(fontsize=9.5, loc="lower right")
    fig.tight_layout()
    save(fig, "steps_latency.png")


def fig_injection(R):
    E = R["E"]
    defenses = ["none", "delimit only", "confirm only", "delimit + confirm"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2), sharey=True)
    for ax, pol in zip(axes, ["scripted", "noisy p=0.3"]):
        rows = [next(e for e in E if e["defense"] == d and e["policy"] == pol) for d in defenses]
        x = np.arange(len(defenses))
        for j, (key, c, lab) in enumerate((("attempted", C_ACCENT, "attack attempted"),
                                           ("harmed", C_NONE, "attack succeeded (harm done)"),
                                           ("success", C_ALL, "task success"))):
            v = [r[key] * 100 for r in rows]
            ax.bar(x + (j - 1) * 0.26, v, 0.26, color=c, alpha=0.85, label=lab)
            for xi, vi in zip(x, v):
                ax.text(xi + (j - 1) * 0.26, vi + 1.5, f"{vi:.0f}", ha="center", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(defenses)
        ax.set_ylim(0, 108)
        ax.set_title(f"{'ScriptedPolicy' if pol == 'scripted' else 'NoisyPolicy p = 0.3'}: 5 attack tasks × 10 runs")
    axes[0].set_ylabel("% of runs")
    fig.suptitle("Indirect prompt injection: poisoned docs/files try to write or delete memory, read secrets, "
                 "or hijack the answer", fontsize=13, y=1.07)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.01), fontsize=11, frameon=False)
    fig.tight_layout()
    save(fig, "prompt_injection.png")


def main():
    os.makedirs(OUT, exist_ok=True)
    fig_agent_loop()
    fig_router()
    print("running agent_eval.run_all() so the figures match the code ...")
    R = run_all()
    fig_noise(R)
    fig_pass_k(R)
    fig_taxonomy(R)
    fig_steps_latency(R)
    fig_injection(R)


if __name__ == "__main__":
    main()
