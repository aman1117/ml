"""
Draw the pictures used in rnn/README.md and save them to rnn/images/.

Every number shown in a figure is computed here: the script trains the same
char-RNN as rnn_numpy.py (same hyperparameters, same seed, so the weights are
identical to what `python rnn/rnn_numpy.py` saves) and then looks inside it.

Run (takes ~20-30 s, mostly training):
    python rnn/make_figures.py
"""
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from rnn_numpy import RNN, softmax  # noqa: E402  (also silences matmul warnings)
from vanishing_gradients import jacobian_norms  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "images")
DPI = 130

# One colour per weight matrix, used in every diagram.
C_XH, C_HH, C_HY = "#1f77b4", "#ff7f0e", "#2ca02c"
C_H, C_X, C_P = "#fdf1c7", "#dbe9f6", "#e3f4e1"

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
    return {" ": "␣", "\n": "\\n"}.get(c, c)


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {os.path.relpath(path)}  ({os.path.getsize(path) // 1024} KB)")


# ---------------------------------------------------------------- model helpers

def step(model, h, ix):
    """One RNN time step. Returns (new h, next-char probabilities, pre-tanh a)."""
    P = model.params
    x = np.zeros((model.V, 1))
    x[ix] = 1.0
    a = P["W_xh"] @ x + P["W_hh"] @ h + P["b_h"]
    h = np.tanh(a)
    return h, softmax(P["W_hy"] @ h + P["b_y"]), a


def read(model, stoi, text, h=None):
    """Feed text one char at a time. Returns lists of hidden states and probs."""
    h = np.zeros((model.H, 1)) if h is None else h
    hs, ps = [], []
    for c in text:
        h, p, _ = step(model, h, stoi[c])
        hs.append(h)
        ps.append(p)
    return hs, ps


def sample_text(model, stoi, itos, start="T", n=40, temperature=1.0, seed=0):
    rng = np.random.default_rng(seed)
    P = model.params
    h, _, _ = step(model, np.zeros((model.H, 1)), stoi[start])
    out = start
    for _ in range(n):
        q = softmax((P["W_hy"] @ h + P["b_y"]).ravel() / temperature)
        ix = rng.choice(model.V, p=q)
        out += itos[ix]
        h, _, _ = step(model, h, ix)
    return out


def train_with_history(text, hidden=100, seq_len=25, lr=0.1, iters=10000, clip=5.0,
                       seed=0, snapshot_iters=(0, 500, 2500, 10000)):
    """Same loop as rnn_numpy.train, but records every chunk's loss and some samples."""
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for i, c in enumerate(chars)}
    data = [stoi[c] for c in text]
    model = RNN(len(chars), hidden, seed)
    mem = {k: np.zeros_like(v) for k, v in model.params.items()}
    p, h = 0, np.zeros((hidden, 1))
    losses, snaps = [], {}
    for it in range(iters + 1):
        if p + seq_len + 1 > len(data):
            p, h = 0, np.zeros((hidden, 1))
        inputs, targets = data[p:p + seq_len], data[p + 1:p + seq_len + 1]
        if it in snapshot_iters:
            snaps[it] = sample_text(model, stoi, itos, temperature=0.5, seed=it)
        loss, h, cache = model.forward(inputs, targets, h)
        grads = model.backward(cache)
        for k, g in grads.items():
            np.clip(g, -clip, clip, out=g)
            mem[k] += g * g
            model.params[k] -= lr * g / np.sqrt(mem[k] + 1e-8)
        losses.append(loss / seq_len)
        p += seq_len
    return model, stoi, itos, np.array(losses), snaps


# ---------------------------------------------------------------- drawing helpers

def box(ax, x, y, text, fc="white", ec="black", w=2.0, h=1.0, fs=12, lw=1.8, **kw):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.15",
                                fc=fc, ec=ec, lw=lw, zorder=2))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, zorder=3, **kw)


def arrow(ax, a, b, color="black", lw=2.0, style="-|>", ms=18, **kw):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=ms,
                                 color=color, lw=lw, zorder=1, **kw))


# ---------------------------------------------------------------- figures

def fig_rnn_cell(model, stoi, itos):
    # Real example: memory after reading "said the ", then the cell reads 'k'.
    hs, _ = read(model, stoi, "said the ")
    _, p, _ = step(model, hs[-1], stoi["k"])
    top = np.argsort(p.ravel())[::-1][:4]

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(-0.5, 16.5)
    ax.set_ylim(-0.3, 10.6)
    ax.axis("off")

    ax.add_patch(FancyBboxPatch((2.3, 1.9), 8.4, 4.4, boxstyle="round,pad=0.1,rounding_size=0.3",
                                fc="#f7f7f7", ec="grey", ls="--", lw=1.5, zorder=0))
    ax.text(2.5, 6.1, "the RNN cell (one time step)", fontsize=12, color="grey",
            va="top", style="italic")

    box(ax, 0.9, 4.5, "$h_{t-1}$\nmemory so far\n(H,1) = (100,1)", fc=C_H, w=2.4, h=1.5)
    box(ax, 7.0, 0.6, "$x_t$  =  one-hot of 'k'\n(V,1) = (44,1)", fc=C_X, w=3.4, h=1.1)

    box(ax, 4.2, 4.5, "$W_{hh} \\cdot h_{t-1}$\n(H,H)·(H,1)", ec=C_HH, lw=3, w=2.6, h=1.2)
    box(ax, 7.0, 2.75, "$W_{xh} \\cdot x_t$\n(H,V)·(V,1)", ec=C_XH, lw=3, w=2.6, h=1.1)
    ax.add_patch(plt.Circle((7.0, 4.5), 0.42, fc="white", ec="black", lw=1.8, zorder=2))
    ax.text(7.0, 4.5, "+", fontsize=22, ha="center", va="center", zorder=3)
    ax.text(7.55, 5.55, "+ $b_h$ (H,1)", fontsize=12, ha="left")
    box(ax, 9.3, 4.5, "tanh\nsquash to\n(-1, 1)", w=1.7, h=1.3)

    box(ax, 12.2, 4.5, "$h_t$\nnew memory\n(H,1)", fc=C_H, w=2.0, h=1.4)
    box(ax, 12.2, 6.9, "$y_t = W_{hy} \\cdot h_t + b_y$\n(V,H)·(H,1) → (V,1)", ec=C_HY, lw=3,
        w=3.6, h=1.1)
    box(ax, 12.2, 8.6, "$p_t$ = softmax($y_t$)\nnext-char probabilities (V,1)", fc=C_P,
        w=3.8, h=1.1)

    arrow(ax, (2.1, 4.5), (2.9, 4.5))
    arrow(ax, (5.5, 4.5), (6.58, 4.5), color=C_HH)
    arrow(ax, (7.0, 1.15), (7.0, 2.2))
    arrow(ax, (7.0, 3.3), (7.0, 4.08), color=C_XH)
    arrow(ax, (7.42, 4.5), (8.45, 4.5))
    arrow(ax, (10.15, 4.5), (11.2, 4.5))
    arrow(ax, (12.2, 5.2), (12.2, 6.35))
    arrow(ax, (12.2, 7.45), (12.2, 8.05))
    arrow(ax, (13.2, 4.5), (15.6, 4.5), color="black", lw=3)
    ax.text(14.4, 4.0, "$h_t$ is passed to\nthe next step\n(becomes its $h_{t-1}$)",
            ha="center", va="top", fontsize=11)
    ax.text(14.4, 4.75, "memory →", ha="center", va="bottom", fontsize=12, weight="bold")

    ax.text(6.0, 9.9, "$h_t = \\tanh(W_{xh} x_t + W_{hh} h_{t-1} + b_h)$", fontsize=15, ha="center")
    ax.text(6.0, 9.2, "new memory = squash( what I just read  +  what I remembered )",
            fontsize=12, ha="center", color="#444")

    # Small inset with the trained model's real prediction for this step.
    ins = fig.add_axes([0.835, 0.63, 0.14, 0.15])
    ins.barh(range(4)[::-1], p.ravel()[top], color=C_HY)
    ins.set_yticks(range(4)[::-1])
    ins.set_yticklabels([f"'{show(itos[i])}'" for i in top], fontsize=11)
    ins.set_xlim(0, 1)
    ins.tick_params(axis="x", labelsize=9)
    for k, i in enumerate(top):
        ins.text(p[i, 0] + 0.02, 3 - k, f"{p[i, 0]:.0%}", va="center", fontsize=10)
    ins.set_title("real model, after reading\n\"said the k\"", fontsize=10)
    arrow(ax, (14.15, 8.6), (14.9, 8.6), color="grey", lw=1.5, ms=12)

    ax.set_title("ONE time step of an RNN: read one character, update the memory, "
                 "predict the next character", fontsize=15, pad=12)
    save(fig, "rnn_cell.png")


def fig_unrolled(model, stoi):
    context, inp, tgt = "said the ", "keepe", "eeper"
    h_start = read(model, stoi, context)[0][-1]
    _, ps = read(model, stoi, inp, h=h_start)
    _, ps_empty = read(model, stoi, inp)  # same chars, but starting from h = 0
    p_empty = ps_empty[1][stoi[tgt[1]], 0]
    n = len(inp)
    xs = [2.6 + 2.9 * i for i in range(n)]

    fig, ax = plt.subplots(figsize=(15, 8.4))
    ax.set_xlim(-0.9, xs[-1] + 2.2)
    ax.set_ylim(-1.6, 9.4)
    ax.axis("off")

    box(ax, 0.15, 4.0, "$h_{-1}$\nmemory after\nreading\n\"said the \"", fc=C_H, w=1.7, h=1.9,
        fs=11)
    total = 0.0
    for i, (x, c, t) in enumerate(zip(xs, inp, tgt)):
        pt = ps[i][stoi[t], 0]
        total += -np.log(pt)
        ax.text(x, 9.0, f"t = {i}", ha="center", fontsize=13, weight="bold")
        box(ax, x, 0.4, f"$x_{i}$ = '{c}'", fc=C_X, w=1.7, h=0.8, fs=13)
        box(ax, x, 1.95, "$W_{xh}$", ec=C_XH, fc="#eef4fb", lw=3, w=1.3, h=0.7, fs=13)
        box(ax, x, 4.0, f"$h_{i}$", fc=C_H, w=1.3, h=1.1, fs=15)
        box(ax, x, 5.95, "$W_{hy}$", ec=C_HY, fc="#eef8ec", lw=3, w=1.3, h=0.7, fs=13)
        box(ax, x, 7.65, f"target '{t}'\nP = {pt:.2f}\nloss = {-np.log(pt):.2f}",
            fc=C_P, w=2.0, h=1.4, fs=11)
        arrow(ax, (x, 0.8), (x, 1.6))
        arrow(ax, (x, 2.3), (x, 3.45), color=C_XH)
        arrow(ax, (x, 4.55), (x, 5.6))
        arrow(ax, (x, 6.3), (x, 6.95), color=C_HY)
        left = xs[i - 1] + 0.65 if i else 1.05
        arrow(ax, (left, 4.0), (x - 0.65, 4.0), color=C_HH, lw=3)
        ax.text((left + x - 0.65) / 2, 4.15, "$W_{hh}$", color=C_HH, ha="center",
                va="bottom", fontsize=13, weight="bold")
    arrow(ax, (xs[-1] + 0.65, 4.0), (xs[-1] + 1.9, 4.0), color=C_HH, lw=3)
    ax.text(xs[-1] + 1.3, 4.15, "...", ha="center", va="bottom", fontsize=14)

    ax.text(-0.8, 7.65, "predict\nthe next\nchar", fontsize=11, va="center", color="#444")
    ax.text(-0.8, 0.4, "read", fontsize=11, va="center", color="#444")
    ax.text((xs[0] + xs[-1]) / 2, -0.75,
            "Same colour = the SAME matrix. $W_{xh}$ (blue), $W_{hh}$ (orange) and $W_{hy}$ (green) "
            "are reused at every step;\nonly the memory $h_t$ changes. "
            f"Total loss for this chunk = sum of the {n} losses = {total:.2f}\n"
            f"Memory matters: start from an empty memory ($h_{{-1}}$ = 0) and the same model gives "
            f"'e' after \"ke\" only P = {p_empty:.2%} (vs {ps[1][stoi[tgt[1]], 0]:.0%} above).",
            ha="center", va="top", fontsize=12.5,
            bbox=dict(boxstyle="round", fc="#fffbe6", ec="#d4b106"))
    ax.set_title("The RNN \"unrolled\" in time: reading 'keepe' and learning to predict "
                 "'eeper' (the next char each time)", fontsize=15)
    save(fig, "unrolled.png")


def fig_hidden_state(model, stoi, text):
    sentence = "The light turned and turned."
    hs, _ = read(model, stoi, sentence)
    Hm = np.hstack(hs)  # (H, T)

    # Find the unit that best tracks "I just read a space" over the whole story.
    story = np.hstack(read(model, stoi, text)[0])
    is_space = np.array([c == " " for c in text])
    corr = np.array([abs(np.corrcoef(u, is_space)[0, 1]) for u in story])
    su = int(np.nanargmax(corr))
    m_sp, m_other = story[su, is_space].mean(), story[su, ~is_space].mean()

    by_var = [u for u in np.argsort(Hm.var(axis=1))[::-1] if u != su]
    units = np.sort(np.array([su] + by_var[:29]))
    row = int(np.where(units == su)[0][0])
    saturated = float((np.abs(story) > 0.9).mean())

    fig, ax = plt.subplots(figsize=(14, 8.4))
    im = ax.imshow(Hm[units], cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(sentence)))
    ax.set_xticklabels([show(c) for c in sentence], fontsize=14, family="monospace")
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(units)))
    ax.set_yticklabels([f"unit {u}" for u in units], fontsize=8.5)
    ax.get_yticklabels()[row].set_color("#d62728")
    ax.get_yticklabels()[row].set_fontweight("bold")
    ax.set_ylabel(f"30 of the {model.H} hidden units (the ones that change most)")
    ax.add_patch(plt.Rectangle((-0.5, row - 0.5), len(sentence), 1, fill=False, ec="#ffd400",
                               lw=3))
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("value of the unit  (tanh keeps it in -1 … +1)")
    ax.set_xlabel(
        "Each COLUMN is the whole memory $h_t$ right after reading that character "
        "(read left → right). It is all the RNN knows about the past.\n"
        f"Yellow row: unit {su} is a \"just saw a space\" detector. Over the whole story it "
        f"averages {m_sp:+.2f} right after a space vs {m_other:+.2f} otherwise.\n"
        f"Nobody programmed that: it was learned. Also note: {saturated:.0%} of all values are "
        "near ±1 (dark) = saturated tanh (see section 5).", fontsize=12, labelpad=12)
    ax.set_title(f"Inside a trained RNN: the hidden state while reading \"{sentence}\"",
                 fontsize=15, pad=36)
    save(fig, "hidden_state.png")
    return su, m_sp, m_other, saturated


def fig_next_char_probs(model, stoi, itos):
    prompt = "The keepe"
    hs, _ = read(model, stoi, prompt)
    P = model.params
    y = (P["W_hy"] @ hs[-1] + P["b_y"]).ravel()
    temps = [0.5, 1.0, 2.0]
    probs = {T: softmax(y / T) for T in temps}
    top = np.argsort(probs[1.0])[::-1][:10]
    labels = [f"'{show(itos[i])}'" for i in top]
    notes = {0.5: "low T: sharper, safer,\nmore repetitive",
             1.0: "T = 1: the model's\nreal probabilities",
             2.0: "high T: flatter, more\nrandom, more mistakes"}

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.6), sharey=True)
    for ax, T in zip(axes, temps):
        pr = probs[T][top]
        bars = ax.bar(range(10), pr, color=["#d62728"] + ["#7f7f7f"] * 9)
        ax.set_xticks(range(10))
        ax.set_xticklabels(labels, fontsize=12, family="monospace")
        ax.set_ylim(0, 1.08)
        ax.set_title(f"temperature T = {T}", fontsize=14, weight="bold")
        for b, v in zip(bars[:3], pr[:3]):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.0%}", ha="center", fontsize=11)
        rest = 1 - pr.sum()
        ax.text(9.4, 1.0, notes[T] + f"\n\ntop-1: {pr[0]:.0%}\nother {model.V - 10} chars: "
                f"{rest:.1%}", ha="right", va="top", fontsize=11,
                bbox=dict(boxstyle="round", fc="#f5f5f5", ec="#bbb"))
    axes[0].set_ylabel("probability of next char")
    fig.suptitle(f"After reading \"{prompt}\", what comes next?   p = softmax(y / T)   "
                 "(top 10 chars, ␣ = space)", y=1.0)
    fig.text(0.5, -0.04, "Same logits y every time. Dividing by T < 1 stretches the gaps "
             "(the favourite wins more often); T > 1 shrinks them (rare chars get picked more).",
             ha="center", fontsize=12)
    fig.tight_layout()
    save(fig, "next_char_probs.png")
    return {T: float(probs[T][top[0]]) for T in temps}, itos[top[0]]


def fig_bptt(model, stoi):
    text = open(os.path.join(HERE, "input.txt"), encoding="utf-8").read()
    start = text.index("The light turned and turned")
    chunk = text[start:start + 26]
    inp, tgt = chunk[:-1], chunk[1:]
    hs, ps = read(model, stoi, inp)
    P = model.params
    T = len(inp)

    # Gradient of ONLY the last step's loss, sent backwards by hand (README section 3).
    dy = ps[-1].copy()
    dy[stoi[tgt[-1]]] -= 1.0
    dh = P["W_hy"].T @ dy
    norms = np.zeros(T)
    for k in range(T - 1, -1, -1):
        norms[k] = np.linalg.norm(dh)
        dh = P["W_hh"].T @ ((1 - hs[k] ** 2) * dh)

    fig, ax = plt.subplots(figsize=(14, 6.4))
    colors = plt.cm.Reds(0.35 + 0.65 * np.arange(T) / (T - 1))
    ax.bar(range(T), norms, color=colors, ec="#7a0000")
    ax.set_yscale("log")
    ax.set_xticks(range(T))
    ax.set_xticklabels([show(c) for c in inp], fontsize=14, family="monospace")
    ax.set_xlabel(f"input character at step k  (the chunk reads \"{inp}\")")
    ax.set_ylabel("‖ dL_last / dh_k ‖   (log scale)")
    ax.set_ylim(norms.min() / 30, norms.max() * 300)
    for k in (T - 1, T - 6, T - 11, 0):
        ax.text(k, norms[k] * 1.4, f"{norms[k]:.1e}", ha="center", fontsize=10)

    ymax = norms.max() * 60
    ax.annotate("", xy=(0, ymax), xytext=(T - 1, ymax),
                arrowprops=dict(arrowstyle="-|>", color="#d62728", lw=2.5, mutation_scale=20))
    ax.text((T - 1) / 2, ymax * 1.25, "backward pass: the learning signal flows right → left "
            "and fades as it goes", color="#d62728", ha="center", fontsize=12)
    drop10 = norms[T - 1] / norms[T - 11]
    ax.text(-0.4, norms.max() * 15,
            f"The last prediction is: after \"…{inp[-6:]}\" the next char should be '{tgt[-1]}'.\n"
            f"Its signal is {drop10:,.0f}× weaker 10 steps back and "
            f"{norms[T - 1] / norms[0]:,.0f}× weaker at the start of the chunk.\n"
            "Each step back multiplies by $W_{hh}^T$ and the tanh slope (≤ 1). "
            "That's the vanishing gradient.", va="top", fontsize=11.5,
            bbox=dict(boxstyle="round", fc="white", ec="#bbb"))
    ax.set_title("Backprop through time: how far back can the LAST prediction send "
                 "learning signal?", pad=12)
    save(fig, "bptt.png")
    return norms, drop10


def fig_vanishing():
    radii = [0.5, 0.9, 1.0, 1.1, 1.5]
    colors = ["#2166ac", "#67a9cf", "#333333", "#ef8a62", "#b2182b"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))
    results = {}
    for ax, act in zip(axes, ["linear", "tanh"]):
        for r, c in zip(radii, colors):
            norms = jacobian_norms(r, act)
            d = sorted(norms)
            v = [norms[k] for k in d]
            results[(act, r)] = norms
            ax.plot(d, v, color=c, lw=2.5, label=f"radius {r}")
        ax.set_yscale("log")
        ax.axhline(1, color="grey", ls=":", lw=1.5)
        ax.set_xlabel("distance back in time  (t − k steps)")
        ax.set_title(f"{act} activation" + ("  (just W_hh, no squashing)" if act == "linear"
                                            else "  (what a real RNN uses)"))
        ax.set_xlim(0, 60)
        ax.set_ylim(1e-20, 1e12)
        ax.axhspan(1, 1e12, color="#fde0dd", alpha=0.5, zorder=0)
        ax.axhspan(1e-20, 1, color="#deebf7", alpha=0.5, zorder=0)
        ax.text(58, 1e10, "EXPLODING (> 1)", ha="right", color="#b2182b", fontsize=13, weight="bold")
        ax.text(58, 1e-18, "VANISHING (< 1)", ha="right", color="#2166ac", fontsize=13, weight="bold")
    axes[0].set_ylabel("‖ dh_t / dh_k ‖   (log scale)")
    axes[0].legend(title="spectral radius of W_hh", loc="lower left", fontsize=10.5,
                   title_fontsize=10.5, bbox_to_anchor=(0.0, 0.08))
    fig.suptitle("The gradient from d steps back is a product of d matrices, so it shrinks or "
                 "grows like radius^d", y=1.0)
    fig.text(0.5, -0.03,
             f"linear, radius 0.5: {results[('linear', 0.5)][20]:.0e} after 20 steps;   "
             f"radius 1.5: {results[('linear', 1.5)][20]:.0e}.   "
             f"With tanh, even radius 1.5 ends at {results[('tanh', 1.5)][59]:.0e} after 59 "
             "steps: saturated units have slope ≈ 0.", ha="center", fontsize=12)
    fig.tight_layout()
    save(fig, "vanishing_gradients.png")
    return results


def fig_training(losses, snaps, V):
    window = 200
    smooth = np.array([losses[max(0, i - window + 1):i + 1].mean() for i in range(len(losses))])
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_axes([0.07, 0.42, 0.9, 0.5])
    ax.plot(losses, color="#c7c7c7", lw=0.6, label="loss on each 25-char chunk")
    ax.plot(smooth, color=C_XH, lw=2.5, label=f"average of the last {window} chunks")
    ax.axhline(np.log(V), color="#d62728", ls="--", lw=1.5)
    ax.text(len(losses) * 0.99, np.log(V) + 0.06, f"random guessing: ln({V}) = {np.log(V):.2f}",
            color="#d62728", ha="right", va="bottom")
    ax.set_xlabel("training iteration (one 25-char chunk each)")
    ax.set_ylabel("loss per character (nats)")
    ax.set_xlim(-150, len(losses) + 150)
    ax.set_ylim(0, 4.6)
    n_spike = int(np.max(np.where(losses > 4.6)[0])) + 1
    ax.text(len(losses) * 0.03, 4.35, f"(off the top: the first {n_spike} chunks spike up to "
            f"{losses.max():.1f} while the first big Adagrad steps settle)", fontsize=10.5,
            color="#666")
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.76), fontsize=11)
    ax.set_title("Training the char-RNN: loss goes down, samples go from gibberish to words",
                 fontsize=15)

    letters = "ABCD"
    lines = []
    for L, (it, s) in zip(letters, sorted(snaps.items())):
        ax.plot(it, smooth[it], "o", ms=13, color="black", zorder=5)
        ax.text(it, smooth[it], L, color="white", ha="center", va="center", fontsize=9,
                weight="bold", zorder=6)
        s = s.replace("\n", "\\n").replace("$", r"\$")
        lines.append((L, it, smooth[it], s))
    fig.text(0.07, 0.32, "Samples from the model at those moments (start with 'T', "
             "40 chars, temperature 0.5):", fontsize=12.5, weight="bold")
    for k, (L, it, sl, s) in enumerate(lines):
        y = 0.25 - k * 0.065
        fig.text(0.07, y, f"{L}  iter {it:>5}  loss {sl:.2f}", fontsize=12, family="monospace")
        fig.text(0.31, y, f"\"{s}\"", fontsize=12.5, family="monospace",
                 bbox=dict(boxstyle="round", fc="#f5f5f5", ec="#ddd"))
    save(fig, "training_progress.png")
    return smooth, lines


def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(HERE, "input.txt"), encoding="utf-8") as f:
        text = f.read()

    print("training the char-RNN (same settings as rnn_numpy.py, ~15 s)...")
    model, stoi, itos, losses, snaps = train_with_history(text)
    print(f"  done in {time.time() - t0:.1f} s, final avg loss/char "
          f"{losses[-200:].mean():.3f}")

    fig_rnn_cell(model, stoi, itos)
    fig_unrolled(model, stoi)
    su, m_sp, m_other, sat = fig_hidden_state(model, stoi, text)
    print(f"  hidden_state: unit {su} after space {m_sp:+.2f} vs {m_other:+.2f}; "
          f"{sat:.0%} saturated")
    tp, best = fig_next_char_probs(model, stoi, itos)
    print(f"  next_char_probs: top char {best!r}: " +
          ", ".join(f"T={T}: {v:.1%}" for T, v in tp.items()))
    norms, drop10 = fig_bptt(model, stoi)
    print(f"  bptt: last {norms[-1]:.2e}, 10 back {norms[-11]:.2e} ({drop10:.0f}x), "
          f"first {norms[0]:.2e}")
    fig_vanishing()
    _, lines = fig_training(losses, snaps, model.V)
    for L, it, sl, s in lines:
        print(f"  sample @ {it:>5} (loss {sl:.2f}): {s}")
    print(f"total time {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
