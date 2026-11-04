"""
Draw the pictures used in evals/README.md and save them to evals/images/.

Every number in every figure comes from a real computation in this folder:
  - the repo's models (models.py), trained on demand and cached in evals/runs/models/,
  - harness runs (harness.py), cached in evals/runs/<task>/,
  - a small sweep of 12 GPT training runs (4 step counts x 3 seeds) for the
    Goodhart / selection-bias picture,
  - the simulated judge on the 40 human-labeled pairs (judge.py).

Run (~2.5 min from scratch, ~5 s once evals/runs/ is populated):
    python evals/make_figures.py
"""
from __future__ import annotations

import math
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

import judge as J  # noqa: E402
import metrics as M  # noqa: E402
import stats as S  # noqa: E402
from contamination import analyze  # noqa: E402
from harness import TASKS, load_or_run  # noqa: E402
from models import get_model  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "images")
DPI = 130
SWEEP = [f"gpt@steps={s},seed={k}" for s in (600, 1000, 1500, 2000) for k in (0, 1, 2)]

COL = {"gpt": "#1f77b4", "gpt@steps=300": "#ff7f0e", "rnn": "#2ca02c", "bigram": "#7f7f7f",
       "gpt-full": "#d62728", "gpt@seed=1": "#9467bd"}

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


def box(ax, x, y, text, fc="white", ec="black", w=2.0, h=1.0, fs=11, lw=1.6, **kw):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=fc, ec=ec, lw=lw, zorder=2))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, zorder=3, **kw)


def arrow(ax, a, b, color="black", lw=1.8, **kw):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=16, color=color,
                                 lw=lw, zorder=1, **kw))


def heldout_scores(recs):
    return np.array([r["score"] for r in recs if r["split"] == "heldout"])


# ----------------------------------------------------------------------------
# 1. The pipeline
# ----------------------------------------------------------------------------
def fig_pipeline(recs, summary, cmp):
    r = next(x for x in recs if x["id"] == "w695")
    h = summary["splits"]["heldout"]
    fig, ax = plt.subplots(figsize=(15, 6.4))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 6.4)
    ax.axis("off")
    F1, F2, F3, F4 = "#dbe9f6", "#ffe8cc", "#e3f4e1", "#fbe3e3"
    ctx = r["input"][-26:]
    top = [
        (1.6, "eval set\ntasks/next_word.jsonl\n" + f'id {r["id"]}, split {r["split"]}\n'
              f'target "{r["target"]}"', F1),
        (5.2, "prompt template\n\"{context}\"\n\n" + f'"…{ctx}"', F1),
        (8.8, "model call\ngpt (weights " + summary["model_fingerprint"][:6] + "…)\n"
              "greedy, stop at space\nresponse cache + latency", F2),
        (12.6, f'raw output\n"{r["output"]}"\n\nseed per example\n(task, id, run seed)', F2),
    ]
    for x, t, fc in top:
        box(ax, x, 4.9, t, fc=fc, w=3.0, h=2.0, fs=10.5)
    for (x1, _, _), (x2, _, _) in zip(top, top[1:]):
        arrow(ax, (x1 + 1.5, 4.9), (x2 - 1.5, 4.9))
    bottom = [
        (12.6, f'normalize + score\n"{r["output"]}" vs "{r["target"]}"\nstrict EM '
               f'{r["em_strict"]:.0f}, normalized EM {r["score"]:.0f}', F3),
        (8.8, f"per-example log (JSONL)\n{summary['n']} lines: input, output,\nscore, "
              "latency, cached", F3),
        (5.2, f"aggregate + error bars\nheld-out {h['mean']:.3f}\n95% CI [{h['ci95_bootstrap'][0]:.3f}, "
              f"{h['ci95_bootstrap'][1]:.3f}]\nclustered SE {h['se_clustered']:.3f}", F3),
        (1.6, f"compare vs baseline\nnew: {cmp['new']}\ndiff {cmp['diff']:+.3f}, McNemar "
              f"p = {cmp['p']:.0e}\n→ {cmp['decision']}: exit 1", F4),
    ]
    for x, t, fc in bottom:
        box(ax, x, 1.7, t, fc=fc, w=3.0, h=2.0, fs=10.5)
    arrow(ax, (12.6, 3.9), (12.6, 2.7))
    for (x1, _, _), (x2, _, _) in zip(bottom, bottom[1:]):
        arrow(ax, (x1 - 1.5, 1.7), (x2 + 1.5, 1.7))
    ax.text(7.5, 6.25, "One eval example, end to end (python evals/harness.py run --task "
            "next_word --model gpt)", ha="center", fontsize=14, weight="bold")
    ax.text(7.5, 0.25, "blue = data, orange = the model (the only part that changes between "
            "runs), green = deterministic scoring and statistics, red = the CI gate",
            ha="center", fontsize=10.5, color="#555")
    save(fig, "pipeline.png")


# ----------------------------------------------------------------------------
# 2. CI width vs n, and how many examples you need
# ----------------------------------------------------------------------------
def fig_ci_width(x_gpt, disc):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    ax = axes[0]
    ns = np.logspace(1, 4, 200)
    for p, c in ((0.5, "#d62728"), (0.8, "#1f77b4"), (0.95, "#2ca02c")):
        ax.plot(ns, 1.96 * np.sqrt(p * (1 - p) / ns) * 100, color=c, lw=2.2,
                label=f"accuracy {p:.0%}: ±1.96·√(p(1-p)/n)")
    rng = np.random.default_rng(0)
    p_hat = x_gpt.mean()
    sub_n = [25, 50, 100, 216]
    widths = []
    for n in sub_n:
        w = []
        for i in range(200):
            s = x_gpt[rng.choice(len(x_gpt), n, replace=False)] if n < len(x_gpt) else x_gpt
            lo, hi, _ = S.bootstrap_ci(s, n_boot=500, seed=i)
            w.append((hi - lo) / 2)
        widths.append(np.mean(w) * 100)
    ax.plot(sub_n, widths, "o", color="black", ms=8, zorder=5,
            label=f"bootstrap, subsets of our next_word run (acc {p_hat:.2f})")
    for n, lab in ((42, "mc (42)"), (216, "next_word (216)"), (556, "calibration (556)")):
        ax.axvline(n, color="#999", lw=1, ls=":")
        ax.text(n * 1.06, 9.6, lab, rotation=90, va="center", fontsize=9.5, color="#555")
    ax.set_xscale("log")
    ax.set_ylim(0, 18)
    ax.set_xlabel("number of eval examples n (log scale)")
    ax.set_ylabel("95% CI half-width (accuracy points)")
    ax.set_title("Error bars shrink like 1/√n")
    ax.legend(fontsize=9.5, loc="upper right")

    ax = axes[1]
    deltas = np.linspace(0.01, 0.15, 60)
    ax.plot(deltas * 100, [S.required_n_unpaired(0.7, 0.7 + d) for d in deltas], color="black",
            lw=2.4, label="unpaired (two independent samples), per model")
    for psi, c in ((0.05, "#2ca02c"), (0.15, "#1f77b4"), (0.30, "#ff7f0e")):
        ok = deltas < psi
        ax.plot(deltas[ok] * 100, [S.required_n_paired(d, psi) for d in deltas[ok]], color=c,
                lw=2.2, label=f"paired, {psi:.0%} of questions discordant")
    ax.axhline(216, color="#999", ls=":", lw=1)
    ax.text(1.0, 240, "our held-out next_word set (216)", ha="left", fontsize=9, color="#555")
    ax.set_yscale("log")
    ax.set_xlabel("true accuracy gain to detect (points, base accuracy 70%)")
    ax.set_ylabel("examples needed (log scale)")
    ax.set_title("How many examples? (alpha 0.05, power 80%)")
    ax.legend(fontsize=9.5)
    ax.text(0.02, 0.03, f"our real pairs: gpt vs gpt@seed=1 disagree on {disc:.0%} of held-out "
            "questions", transform=ax.transAxes, fontsize=9.5, color="#333")
    fig.tight_layout()
    save(fig, "ci_width.png")
    return dict(zip(sub_n, widths))


# ----------------------------------------------------------------------------
# 3. The bootstrap distribution (iid vs clustered)
# ----------------------------------------------------------------------------
def fig_bootstrap(recs):
    held = [r for r in recs if r["split"] == "heldout"]
    x = np.array([r["score"] for r in held])
    cl = [r["cluster"] for r in held]
    lo, hi, boots = S.bootstrap_ci(x, n_boot=10000)
    cboots = S.cluster_bootstrap_stats(x, cl, n_boot=10000)
    clo, chi = np.quantile(cboots, [0.025, 0.975])
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    ax = axes[0]
    vals, counts = np.unique(boots, return_counts=True)
    ax.bar(vals, counts, width=1 / len(x) * 0.9, color="#9ecae1", edgecolor="none")
    grid = np.linspace(boots.min(), boots.max(), 200)
    se = S.mean_se(x)
    ax.plot(grid, 10000 / len(x) * np.exp(-0.5 * ((grid - x.mean()) / se) ** 2)
            / (se * math.sqrt(2 * math.pi)), color="black", lw=1.8, label="normal (CLT) curve")
    for v in (lo, hi):
        ax.axvline(v, color="#d62728", lw=2)
    ax.axvline(x.mean(), color="#1f77b4", lw=2, ls="--")
    ax.set_title(f"10,000 bootstrap resamples of the {len(x)} held-out questions")
    ax.set_xlabel("accuracy of a resample")
    ax.set_ylabel("count")
    ax.text(0.02, 0.95, f"observed {x.mean():.3f}\npercentile CI\n[{lo:.3f}, {hi:.3f}] (red)",
            transform=ax.transAxes, va="top", fontsize=11)
    ax.legend(loc="upper right", fontsize=10)
    ax = axes[1]
    bins = np.linspace(min(boots.min(), cboots.min()), max(boots.max(), cboots.max()), 50)
    ax.hist(boots, bins=bins, alpha=0.6, color="#1f77b4", label=f"resample questions: "
            f"[{lo:.3f}, {hi:.3f}]")
    ax.hist(cboots, bins=bins, alpha=0.5, color="#ff7f0e", label=f"resample paragraphs: "
            f"[{clo:.3f}, {chi:.3f}]")
    ax.set_title(f"Questions come in {len(set(cl))} paragraphs: resample those")
    ax.set_xlabel("accuracy of a resample")
    ax.legend(fontsize=10, loc="upper right")
    ax.text(0.02, 0.95, f"SE naive {se:.3f}\nSE clustered {S.clustered_se(x, cl):.3f}",
            transform=ax.transAxes, va="top", fontsize=11)
    fig.tight_layout()
    save(fig, "bootstrap.png")
    return dict(ci=(lo, hi), cluster_ci=(clo, chi), se=se, se_cl=S.clustered_se(x, cl))


# ----------------------------------------------------------------------------
# 4. Paired vs unpaired
# ----------------------------------------------------------------------------
def fig_paired(pairs, power_pair):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    names = [b.replace(",", ",\n") + "\nvs " + a for a, b, *_ in pairs]
    sp = [S.mean_se(xb - xa) for _, _, xa, xb in pairs]
    su = [math.sqrt(S.mean_se(xa) ** 2 + S.mean_se(xb) ** 2) for _, _, xa, xb in pairs]
    pos = np.arange(len(pairs))
    ax.bar(pos - 0.2, su, 0.4, color="#bbbbbb", label="unpaired SE of the difference")
    ax.bar(pos + 0.2, sp, 0.4, color="#1f77b4", label="paired SE (same questions)")
    for i, (_, _, xa, xb) in enumerate(pairs):
        ax.text(i, max(su[i], sp[i]) + 0.001, f"corr {np.corrcoef(xa, xb)[0, 1]:.2f}",
                ha="center", fontsize=9.5)
    ax.set_xticks(pos)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylim(0, 0.062)
    ax.set_ylabel("standard error of B - A")
    ax.set_title("Same 216 held-out questions: pairing shrinks the SE")
    ax.legend(fontsize=9.5, loc="upper right", ncol=2)
    ax = axes[1]
    a_name, b_name, xa, xb = power_pair
    rng = np.random.default_rng(0)
    ns = [50, 100, 200, 400, 800, 1600, 3200]
    pw_p, pw_u = [], []
    for n in ns:
        rp = ru = 0
        sims = 600
        for _ in range(sims):
            i = rng.integers(0, len(xa), n)
            _, a_only, b_only, _ = S.mcnemar_table(xa[i], xb[i])
            rp += S.mcnemar(a_only, b_only) < 0.05
            ru += S.two_proportion_ztest(xa[i], xb[i]) < 0.05
        pw_p.append(rp / sims)
        pw_u.append(ru / sims)
    ax.plot(ns, pw_p, "o-", color="#1f77b4", lw=2.4, label="paired (McNemar exact)")
    ax.plot(ns, pw_u, "s-", color="#888", lw=2.4, label="unpaired (two-proportion z-test)")
    ax.axhline(0.8, color="#999", ls=":", lw=1)
    ax.text(52, 0.82, "80% power", fontsize=9.5, color="#555")
    ax.set_xscale("log")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("questions per eval (resampled from the real paired results, log scale)")
    ax.set_ylabel("chance of detecting the difference")
    ax.set_title(f"Power: {b_name} vs {a_name} ({xb.mean() - xa.mean():+.3f})")
    ax.legend(fontsize=10, loc="lower right")
    fig.tight_layout()
    save(fig, "paired_power.png")
    return dict(ns=ns, paired=pw_p, unpaired=pw_u, sp=sp, su=su)


# ----------------------------------------------------------------------------
# 5. pass@k
# ----------------------------------------------------------------------------
def pass_curve(recs, ks, hat=False):
    f = M.pass_hat_k if hat else M.pass_at_k
    return np.array([np.mean([f(r["n"], r["c"], k) for r in recs]) for k in ks])


def fig_pass_at_k(sample_runs):
    ks = np.arange(1, 21)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    for p, c in ((0.05, "#d62728"), (0.2, "#ff7f0e"), (0.5, "#1f77b4"), (0.9, "#2ca02c")):
        ax.plot(ks, 1 - (1 - p) ** ks, color=c, lw=2.4, label=f"per-attempt success p = {p}")
        ax.plot(ks, p ** ks, color=c, lw=1.6, ls="--")
    ax.set_xlabel("k (attempts)")
    ax.set_ylabel("probability")
    ax.set_title("solid: pass@k = 1-(1-p)^k (ANY of k succeeds)\n"
                 "dashed: pass^k = p^k (ALL k succeed)", fontsize=12.5)
    ax.legend(fontsize=9.5, loc="center right")
    ax = axes[1]
    out = {}
    for spec, recs in sample_runs.items():
        pk, hk = pass_curve(recs, ks), pass_curve(recs, ks, hat=True)
        naive = 1 - (1 - pk[0]) ** ks
        c = COL[spec]
        ax.plot(ks, pk, "-", color=c, lw=2.6, label=f"{spec}: pass@k")
        ax.plot(ks, hk, "--", color=c, lw=1.8, label=f"{spec}: pass^k")
        ax.plot(ks, naive, ":", color=c, lw=1.6)
        out[spec] = (pk, hk, naive)
    ax.set_xlabel("k (samples at temperature 1)")
    ax.set_ylabel("rate over 216 held-out words")
    ax.set_title("Real: next_word_sample, 20 samples per word\n"
                 "dotted: 1-(1-pass@1)^k, as if every word were equally hard", fontsize=12.5)
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=9.5, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    save(fig, "pass_at_k.png")
    return out


# ----------------------------------------------------------------------------
# 6. Reliability diagram (+ temperature scaling)
# ----------------------------------------------------------------------------
def calib_logits(spec):
    model = get_model(spec, verbose=False)
    exs = [e for e in TASKS["calibration"].examples() if e["split"] == "heldout"]
    z = model.next_char_logits([e["context"] for e in exs])
    return z, np.array([e["target_id"] for e in exs])


def reliability_panel(ax, axc, conf, correct, title, color):
    rows = M.reliability_data(conf, correct, 10, "width")
    for r in rows:
        if r["count"]:
            mid = (r["lo"] + r["hi"]) / 2
            ax.bar(mid, r["acc"], width=0.095, color=color, alpha=0.85, edgecolor="white")
            over = r["conf"] > r["acc"]
            ax.bar(mid, abs(r["conf"] - r["acc"]), bottom=min(r["acc"], r["conf"]), width=0.095,
                   color="none", edgecolor="#d62728" if over else "#2ca02c", hatch="///", lw=0.8)
    ax.plot([0, 1], [0, 1], "k--", lw=1.2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    e_w, e_m = M.ece(conf, correct, 10, "width"), M.ece(conf, correct, 10, "mass")
    ax.set_title(title, fontsize=12.5)
    ax.text(0.03, 0.97, f"acc {np.mean(correct):.3f}, mean conf {np.mean(conf):.3f}\n"
            f"ECE width {e_w:.3f}, mass {e_m:.3f}\nBrier {M.brier_binary(conf, correct):.3f}",
            transform=ax.transAxes, va="top", fontsize=10)
    axc.bar([(r["lo"] + r["hi"]) / 2 for r in rows], [r["count"] for r in rows], width=0.095,
            color="#999")
    axc.set_xlim(0, 1)
    axc.set_xlabel("confidence (top-1 probability)")
    return e_w, e_m


def fig_reliability():
    z, t = calib_logits("gpt")
    dev, test = np.arange(len(t)) % 2 == 0, np.arange(len(t)) % 2 == 1
    T = M.fit_temperature(z[dev], t[dev])
    out = {"T": T}
    fig, axes = plt.subplots(2, 3, figsize=(16, 6.6), gridspec_kw=dict(height_ratios=[4, 1]),
                             sharex="col")
    panels = []
    p = M.softmax_rows(z[test])
    panels.append(("gpt, held-out (test half)", p, "#1f77b4"))
    p_ts = M.softmax_rows(z[test] / T)
    panels.append((f"gpt + temperature scaling, T = {T:.2f}\n(T fit on the other half)", p_ts,
                   "#9467bd"))
    z3, t3 = calib_logits("gpt@steps=300")
    panels.append(("gpt@steps=300, held-out (test half)", M.softmax_rows(z3[test]), "#ff7f0e"))
    targets = [t[test], t[test], t3[test]]
    for j, ((title, probs, c), tt) in enumerate(zip(panels, targets)):
        conf = probs.max(1)
        correct = (probs.argmax(1) == tt).astype(float)
        out[title.split(",")[0].split("\n")[0]] = reliability_panel(
            axes[0, j], axes[1, j], conf, correct, title, c) + (
            float(np.mean(correct)), float(np.mean(conf)),
            float(-np.mean(np.log(probs[np.arange(len(tt)), tt]))))
    axes[0, 0].set_ylabel("accuracy in the bin")
    axes[1, 0].set_ylabel("count")
    fig.suptitle("Reliability diagrams, next-character prediction on held-out text (bars = "
                 "accuracy; hatched = gap to confidence, red = overconfident)", y=1.0)
    fig.tight_layout()
    save(fig, "reliability.png")
    return out


# ----------------------------------------------------------------------------
# 7. The judge
# ----------------------------------------------------------------------------
def fig_judge(pairs):
    res = J.evaluate(J.SimulatedJudge(), pairs, k=5, verbose=False)
    rows = J.sweep(pairs, k=5)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2))
    ax = axes[0]
    names = list(res["table"])
    colors = ["#bbbbbb", "#999999", "#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]
    for i, n in enumerate(names):
        r = res["table"][n]
        ax.bar(i, r["kappa"], color=colors[i], yerr=[[r["kappa"] - r["ci"][0]],
                                                     [r["ci"][1] - r["kappa"]]], capsize=5)
        ax.text(i, 0.03, f"{r['kappa']:.2f}", ha="center", fontsize=11, color="white",
                weight="bold")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace(" + ", "\n+ ") for n in names], fontsize=9.5)
    ax.set_ylabel("Cohen's kappa vs human labels")
    ax.set_ylim(0, 1)
    b = res["bias"]
    ax.set_title("Default judge (position 0.3, verbosity 0.2, noise 0.15)", fontsize=12.5)
    ax.text(0.02, 0.97, f"40 labeled pairs; bars = 95% bootstrap CI\nposition consistency "
            f"{b['consistency']:.0%}; first shown wins {b['first_wins']:.0%}\nlonger answer wins "
            f"{b['longer_judge']:.0%} (human: {b['longer_human']:.0%})",
            transform=ax.transAxes, va="top", fontsize=10)
    ax = axes[1]
    pbs = [r[0] for r in rows]
    styles = {"single": ("#bbbbbb", "-", "o"), "majority-5": ("#ff7f0e", "-", "s"),
              "swap-debiased": ("#1f77b4", "-", "^"), "swap + majority-5": ("#2ca02c", "-", "D"),
              "pointwise": ("#9467bd", "--", "")}
    for n, (c, ls, mk) in styles.items():
        ax.plot(pbs, [r[1][n] for r in rows], color=c, ls=ls, marker=mk, lw=2.4, label=n)
    ax.set_xlabel("position bias of the judge (bonus for the answer shown first)")
    ax.set_ylabel("mean kappa vs human (20 judge seeds)")
    ax.set_title("Voting averages noise, not bias; swapping removes bias", fontsize=12.5)
    ax.set_ylim(0, 0.8)
    ax.legend(fontsize=10, loc="lower left")
    fig.tight_layout()
    save(fig, "judge.png")
    return res, rows


# ----------------------------------------------------------------------------
# 8. Contamination
# ----------------------------------------------------------------------------
def fig_contamination(by_model, cont):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2))
    ax = axes[0]
    flags = cont[("next_word", 8)]["flags"]
    groups = ["train split\n(trained on)", "held-out,\n8-gram contaminated",
              "held-out,\nclean"]
    specs = list(by_model)
    w = 0.8 / len(specs)
    for j, spec in enumerate(specs):
        recs = by_model[spec]
        held = [r for r in recs if r["split"] == "heldout"]
        sets = [[r["score"] for r in recs if r["split"] == "train"],
                [r["score"] for r, f in zip(held, flags) if f],
                [r["score"] for r, f in zip(held, flags) if not f]]
        for i, s in enumerate(sets):
            s = np.array(s)
            lo, hi, _ = S.bootstrap_ci(s, n_boot=1000)
            ax.bar(i + (j - (len(specs) - 1) / 2) * w, s.mean(), w * 0.95, color=COL[spec],
                   yerr=[[s.mean() - lo], [hi - s.mean()]], capsize=3,
                   label=spec if i == 0 else None)
    ax.set_xticks(range(3))
    ax.set_xticklabels(groups)
    ax.set_ylabel("next_word exact match")
    ax.set_ylim(0, 1.22)
    n_c = sum(flags)
    ax.set_title(f"Same model, three subsets ({n_c} of {len(flags)} held-out words flagged)",
                 fontsize=12.5)
    ax.legend(fontsize=9.5, loc="upper center", ncol=4)
    ax = axes[1]
    r = cont[("bpc", 13)]
    held = r["held"]
    rng = np.random.default_rng(0)
    for f, c, lab in ((False, "#1f77b4", "clean"), (True, "#d62728", "13-gram contaminated")):
        ys = [h["score"] for h, fl in zip(held, r["flags"]) if fl == f]
        xs = (1 if f else 0) + rng.uniform(-0.12, 0.12, len(ys))
        ax.scatter(xs, ys, s=60, color=c, alpha=0.8, label=f"{lab} (n = {len(ys)})")
        ax.hlines(np.mean(ys), (1 if f else 0) - 0.25, (1 if f else 0) + 0.25, color=c, lw=3)
        ax.text((1 if f else 0) + 0.28, np.mean(ys), f"mean {np.mean(ys):.2f}", va="center",
                fontsize=10.5)
    tr = by_model["gpt"]
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["clean", "contaminated"])
    ax.set_xlim(-0.5, 1.8)
    ax.set_ylabel("bits per character (lower = better)")
    ax.set_title("gpt on the 21 held-out sentences (bpc task)", fontsize=12.5)
    ax.legend(fontsize=10, loc="upper right")
    fig.tight_layout()
    save(fig, "contamination.png")


# ----------------------------------------------------------------------------
# 9. Goodhart: picking the best of many runs on a small test set
# ----------------------------------------------------------------------------
def fig_goodhart(sweep_scores, m_show=40, splits=2000):
    """Pick the best of 12 runs on a small dev set, report its dev score: how far is
    that from what the same model scores on fresh data (the test half)?"""
    names = list(sweep_scores)
    X = np.stack([sweep_scores[n] for n in names])  # (configs, 216)
    n = X.shape[1]
    rng = np.random.default_rng(0)
    sizes = [10, 20, 40, 60, 108]
    infl, regret, show = [], [], None
    for m in sizes:
        dv_sel, ts_sel, ts_best, picks = [], [], [], []
        for _ in range(splits):
            p = rng.permutation(n)
            dd, tt = X[:, p[:m]].mean(1), X[:, p[n // 2:]].mean(1)
            b = int(dd.argmax())
            dv_sel.append(dd[b])
            ts_sel.append(tt[b])
            ts_best.append(tt.max())
            picks.append(b)
        dv_sel, ts_sel = np.array(dv_sel), np.array(ts_sel)
        infl.append(float(np.mean(dv_sel - ts_sel)))
        regret.append(float(np.mean(np.array(ts_best) - ts_sel)))
        if m == m_show:
            show = (dv_sel, ts_sel, np.bincount(picks, minlength=len(names)))
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.4))
    ax = axes[0]
    dv_sel, ts_sel, picks = show
    jit = lambda k: rng.uniform(-0.004, 0.004, k)
    ax.scatter(dv_sel + jit(len(dv_sel)), ts_sel + jit(len(ts_sel)), s=6, alpha=0.25,
               color="#1f77b4")
    lo, hi = 0.1, 0.62
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.scatter(dv_sel.mean(), ts_sel.mean(), s=200, marker="X", color="#d62728", zorder=4,
               label=f"mean: dev {dv_sel.mean():.3f}, test {ts_sel.mean():.3f}")
    full = X.mean(1)
    ax.axhline(full.max(), color="#2ca02c", lw=1.5, ls=":")
    ax.text(hi - 0.01, full.max() + 0.006, f"best run on all 216 words: {full.max():.3f}",
            ha="right", fontsize=9.5, color="#2ca02c")
    ax.set_xlim(lo, hi)
    ax.set_ylim(0.1, 0.45)
    ax.set_xlabel(f"dev accuracy of the run picked on {m_show} random held-out words")
    ax.set_ylabel("its accuracy on 108 other held-out words")
    ax.set_title(f"Best of 12 training runs, picked on {m_show} words, {splits:,} times",
                 fontsize=12.5)
    ax.legend(fontsize=10, loc="upper left")
    ax = axes[1]
    ax.plot(sizes, np.array(infl) * 100, "o-", color="#d62728", lw=2.5,
            label="reported (dev) score - true (test) score of the pick")
    ax.plot(sizes, np.array(regret) * 100, "s-", color="#1f77b4", lw=2.5,
            label="best run's test score - the pick's test score")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("dev set size used to pick the best of 12 runs")
    ax.set_ylabel(f"accuracy points (mean over {splits:,} splits)")
    ax.set_title("Selecting on a small set inflates the score you report", fontsize=12.5)
    ax.legend(fontsize=10)
    fig.tight_layout()
    save(fig, "goodhart.png")
    return dict(names=names, full=full, sizes=sizes, infl=infl, regret=regret,
                show=(float(dv_sel.mean()), float(ts_sel.mean())), picks=picks)


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    print("models and harness runs (trained / evaluated on demand, cached in evals/runs/)...")
    specs = ["gpt", "gpt@steps=300", "gpt@seed=1", "rnn", "bigram", "gpt-full"]
    nw = {s: load_or_run("next_word", s, verbose=True) for s in specs}
    sample = {s: load_or_run("next_word_sample", s)[1] for s in ("gpt", "rnn", "gpt@steps=300")}
    from harness import compare, results_path
    cmp = compare(results_path("next_word", "gpt") + ".summary.json",
                  results_path("next_word", "gpt@steps=300") + ".summary.json", verbose=False)

    summary, recs = nw["gpt"]
    fig_pipeline(recs, summary, cmp)

    x_gpt = heldout_scores(recs)
    x_s1 = heldout_scores(nw["gpt@seed=1"][1])
    disc = float(np.mean(x_gpt != x_s1))
    widths = fig_ci_width(x_gpt, disc)
    print(f"  CI half-widths from subsets of the gpt run: "
          + ", ".join(f"n={k}: ±{v:.1f} pts" for k, v in widths.items()))

    b = fig_bootstrap(recs)
    print(f"  bootstrap CI {b['ci'][0]:.3f}-{b['ci'][1]:.3f}; cluster bootstrap "
          f"{b['cluster_ci'][0]:.3f}-{b['cluster_ci'][1]:.3f}; SE {b['se']:.3f} vs clustered "
          f"{b['se_cl']:.3f}")

    print("training the 12-run sweep for the Goodhart picture (cached after the first time)...")
    sweep = {}
    for s in SWEEP:
        sweep[s] = heldout_scores(load_or_run("next_word", s, verbose=True)[1])
    pairs = [("gpt", "gpt@seed=1", x_gpt, x_s1),
             ("gpt", "gpt@steps=1500,seed=1", x_gpt, sweep["gpt@steps=1500,seed=1"]),
             ("gpt", "gpt@steps=300", x_gpt, heldout_scores(nw["gpt@steps=300"][1])),
             ("rnn", "bigram", heldout_scores(nw["rnn"][1]), heldout_scores(nw["bigram"][1]))]
    pw = fig_paired(pairs, ("gpt@steps=1000,seed=1", "gpt", sweep["gpt@steps=1000,seed=1"], x_gpt))
    for (a, bb, xa, xb), sp, su in zip(pairs, pw["sp"], pw["su"]):
        print(f"  {bb} vs {a}: diff {xb.mean() - xa.mean():+.3f}, SE paired {sp:.4f} vs "
              f"unpaired {su:.4f}, corr {np.corrcoef(xa, xb)[0, 1]:.2f}")
    print(f"  power (gpt vs gpt@steps=1000,seed=1, diff "
          f"{x_gpt.mean() - sweep['gpt@steps=1000,seed=1'].mean():+.3f}): "
          + ", ".join(f"n={n}: {p:.2f}/{u:.2f}" for n, p, u in zip(pw["ns"], pw["paired"],
                                                                     pw["unpaired"])))

    pk = fig_pass_at_k(sample)
    for s, (p, h, nv) in pk.items():
        print(f"  {s}: pass@1 {p[0]:.3f} pass@5 {p[4]:.3f} pass@20 {p[19]:.3f} | pass^5 "
              f"{h[4]:.3f} pass^20 {h[19]:.3f} | naive pass@20 {nv[19]:.3f}")

    rel = fig_reliability()
    print(f"  temperature scaling: T = {rel['T']:.2f}; " + "; ".join(
        f"{k}: ECE width {v[0]:.3f} mass {v[1]:.3f} acc {v[2]:.3f} conf {v[3]:.3f} NLL {v[4]:.3f}"
        for k, v in rel.items() if k != "T"))

    pairs_j = J.load_pairs()
    res, rows = fig_judge(pairs_j)
    print("  judge kappas: " + ", ".join(f"{n} {r['kappa']:.3f}" for n, r in res["table"].items()))
    print("  sweep (bias: single / swap+majority): " + ", ".join(
        f"{pb}: {r['single']:.2f}/{r['swap + majority-5']:.2f}" for pb, r in rows))

    cont = analyze("gpt", verbose=False)
    fig_contamination({s: nw[s][1] for s in ("gpt", "rnn", "bigram", "gpt-full")}, cont)

    g = fig_goodhart(sweep)
    print("  sweep, held-out accuracy on all 216 words: " + ", ".join(
        f"{n.split('@')[1]} {v:.3f}" for n, v in zip(g["names"], g["full"])))
    print(f"  picked on 40 dev words: mean dev {g['show'][0]:.3f} vs test {g['show'][1]:.3f}; "
          f"most often picked: {g['names'][int(np.argmax(g['picks']))]}")
    print("  inflation by dev size: " + ", ".join(f"{m}: {v * 100:+.1f} pts (regret "
                                                  f"{r * 100:.1f})" for m, v, r in
                                                  zip(g["sizes"], g["infl"], g["regret"])))
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
