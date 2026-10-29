"""
Draw the pictures used in lstm/README.md and save them to lstm/images/.

Every number shown in a figure is computed here, from the real models:
  - the char-level LSTM and GRU of lstm_numpy.py (default settings, seed 0),
  - the vanilla RNN of ../rnn/rnn_numpy.py trained the same way on the same text,
  - the long-memory experiment of long_memory.py.

To stay fast it REUSES results you already have: lstm/model.npz and
lstm/model_gru.npz (if they were trained with the default settings),
lstm/long_memory.npz, and the cached RNN baselines lstm/rnn_baseline*.npz.
Anything missing is trained/run first (identical results, same seeds).

Run (2-3 min from scratch; ~45 s if you already ran lstm_numpy.py (both cells)
and long_memory.py; a few seconds once the RNN baselines are cached too):
    python lstm/make_figures.py
"""
import os
import sys
import time

sys.dont_write_bytecode = True  # importing ../rnn must not leave files in that folder

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import ticker  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from lstm_numpy import (DEFAULT_TEXT, GRU, LSTM, load_model, onehot, run_config,  # noqa: E402
                        save_model, softmax, train)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, "..", "rnn"))
from rnn_numpy import RNN  # noqa: E402

OUT = os.path.join(HERE, "images")
DPI = 130

# Same palette as ../rnn/make_figures.py, plus one colour per gate.
C_XH, C_HH, C_HY = "#1f77b4", "#ff7f0e", "#2ca02c"
C_H, C_X, C_P = "#fdf1c7", "#dbe9f6", "#e3f4e1"
C_C = "#fde0dd"  # the cell state
C_F, C_I, C_G, C_O = "#d62728", "#1f77b4", "#9467bd", "#2ca02c"
COL = {"RNN": "#7f7f7f", "RNN (H=210)": "#bcbd22", "LSTM": "#d62728", "GRU": "#1f77b4",
       "LSTM (b_f=1)": "#ff9896", "LSTM (chrono)": "#d62728", "GRU (chrono)": "#1f77b4"}

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


def op(ax, x, y, sym, r=0.36, fc="white"):
    """A small circle with an operator inside (×, +)."""
    ax.add_patch(plt.Circle((x, y), r, fc=fc, ec="black", lw=1.8, zorder=2))
    ax.text(x, y, sym, fontsize=20, ha="center", va="center", zorder=3)


# ---------------------------------------------------------------- models

def get_char_model(text, cell):
    """The default-settings LSTM/GRU: reuse lstm/model*.npz if it matches, else train it."""
    path = os.path.join(HERE, "model.npz" if cell == "lstm" else "model_gru.npz")
    config = run_config(cell, 100, 25, 0.1, 10000, text)
    if os.path.exists(path):
        d = np.load(path)
        if "config" in d and str(d["config"]) == config:
            model, chars = load_model(path)
            print(f"  reusing {os.path.relpath(path)} ({config})")
            return model, chars, d["losses"]
    print(f"  training the {cell.upper()} (same as `python lstm/lstm_numpy.py"
          f"{'' if cell == 'lstm' else ' --cell gru'}`)...")
    model, _, itos, losses = train(text, cell, sample_every=0, verbose=False)
    chars = [itos[i] for i in range(len(itos))]
    if not os.path.exists(path):  # never overwrite a model trained with other settings
        save_model(path, model, chars, losses=losses, config=np.array(config))
    return model, chars, losses


def get_rnn_baseline(text, hidden, lr):
    """../rnn/rnn_numpy.py's RNN, trained with its own loop and settings (cached here)."""
    path = os.path.join(HERE, f"rnn_baseline_h{hidden}.npz")
    config = run_config("rnn", hidden, 25, lr, 10000, text)
    if os.path.exists(path):
        d = np.load(path)
        if str(d["config"]) == config:
            model = RNN(len(d["chars"]), hidden)
            for k in model.params:
                model.params[k] = d[k]
            print(f"  reusing {os.path.relpath(path)} ({config})")
            return model, d["losses"]
    print(f"  training the vanilla RNN, H={hidden}, lr={lr} (rnn_numpy.py's loop)...")
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    data = [stoi[c] for c in text]
    model = RNN(len(chars), hidden, 0)
    mem = {k: np.zeros_like(v) for k, v in model.params.items()}
    p, h, losses = 0, np.zeros((hidden, 1)), []
    for _ in range(10001):
        if p + 26 > len(data):
            p, h = 0, np.zeros((hidden, 1))
        loss, h, cache = model.forward(data[p:p + 25], data[p + 1:p + 26], h)
        for k, g in model.backward(cache).items():
            np.clip(g, -5.0, 5.0, out=g)
            mem[k] += g * g
            model.params[k] -= lr * g / np.sqrt(mem[k] + 1e-8)
        losses.append(loss / 25)
        p += 25
    np.savez(path, chars=np.array(chars), losses=np.array(losses), config=np.array(config),
             **model.params)
    return model, np.array(losses)


def get_long_memory():
    path = os.path.join(HERE, "long_memory.npz")
    if not os.path.exists(path):
        print("  running long_memory.py first (~45 s)...")
        import long_memory
        argv, sys.argv = sys.argv, ["long_memory.py"]
        long_memory.main()
        sys.argv = argv
    else:
        print(f"  reusing {os.path.relpath(path)}")
    return np.load(path)


def run_lstm(model, stoi, text):
    """Read text from h = c = 0. Returns per-step arrays (T, H) of h, c, f, i, g, o and probs."""
    h, c = model.init_state()
    rec = {k: [] for k in "hcfigop"}
    for ch in text:
        h, c, (i, f, g, o, _) = model.cell_step(onehot(stoi[ch], model.V), h, c)
        p = softmax(model.params["W_hy"] @ h + model.params["b_y"])
        for k, v in zip("hcfigop", (h, c, f, i, g, o, p)):
            rec[k].append(v.ravel())
    return {k: np.array(v) for k, v in rec.items()}


def run_gru(model, stoi, text):
    (h,) = model.init_state()
    rec = {k: [] for k in "hrznp"}
    for ch in text:
        h, (r, z, n, _) = model.cell_step(onehot(stoi[ch], model.V), h)
        p = softmax(model.params["W_hy"] @ h + model.params["b_y"])
        for k, v in zip("hrznp", (h, r, z, n, p)):
            rec[k].append(v.ravel())
    return {k: np.array(v) for k, v in rec.items()}


# ---------------------------------------------------------------- figures

def fig_lstm_cell(lstm, stoi, itos):
    # Real example: the LSTM has read "said the " and now reads 'k'.
    ctx = "said the "
    rec = run_lstm(lstm, stoi, ctx + "k")
    f, i, g, o = rec["f"][-1], rec["i"][-1], rec["g"][-1], rec["o"][-1]
    c_prev, c_new, h_new = rec["c"][-2], rec["c"][-1], rec["h"][-1]
    p = rec["p"][-1]
    top = np.argsort(p)[::-1][:4]
    kept, written = np.linalg.norm(f * c_prev), np.linalg.norm(i * g)

    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(-0.6, 18.2)
    ax.set_ylim(-1.6, 10.7)
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((2.1, 0.8), 10.3, 8.0, boxstyle="round,pad=0.1,rounding_size=0.3",
                                fc="#f7f7f7", ec="grey", ls="--", lw=1.5, zorder=0))
    ax.text(12.3, 1.05, "the LSTM cell (one time step)", fontsize=12, color="grey",
            style="italic", ha="right")

    belt = 7.6
    # The conveyor belt: c_{t-1} -> ×f -> +i⊙g -> c_t
    box(ax, 0.8, belt, "$c_{t-1}$\ncell state\n(H,1)", fc=C_C, w=2.0, h=1.4)
    box(ax, 15.0, belt, "$c_t$\nnew cell state\n(H,1)", fc=C_C, w=2.4, h=1.4)
    ax.plot([1.8, 13.8], [belt, belt], color="#b2182b", lw=6, alpha=0.35, zorder=0,
            solid_capstyle="round")
    arrow(ax, (1.8, belt), (3.54, belt), lw=3)
    op(ax, 3.9, belt, "×")
    arrow(ax, (4.26, belt), (6.64, belt), lw=3)
    op(ax, 7.0, belt, "+")
    arrow(ax, (7.36, belt), (13.8, belt), lw=3)
    ax.text(7.6, belt + 0.95, "the cell-state \"conveyor belt\": only × f and + i⊙g, "
            "no matrix, no squashing", fontsize=12.5, color="#b2182b", ha="center")
    ax.text(3.9, belt + 0.45, "erase", fontsize=11, ha="center", va="bottom", color=C_F,
            weight="bold")
    ax.text(7.0, belt + 0.45, "write", fontsize=11, ha="center", va="bottom", color=C_I,
            weight="bold")
    ax.text(4.5, belt - 0.75, f"kept  ‖f⊙c‖ = {kept:.1f}", fontsize=10.5, color="#555")
    ax.text(7.5, belt - 0.75, f"written  ‖i⊙g‖ = {written:.1f}", fontsize=10.5, color="#555")

    # The input bus: [h_{t-1}; x_t] feeds all four gates.
    bus = 1.9
    box(ax, 0.8, bus, "$h_{t-1}$\nhidden state\n(H,1)", fc=C_H, w=2.0, h=1.4)
    box(ax, 2.95, -0.35, "$x_t$ = one-hot of 'k'  (V,1)", fc=C_X, w=3.6, h=0.75, fs=11.5)
    ax.plot([1.8, 10.3], [bus, bus], color="black", lw=2.2, zorder=1)
    arrow(ax, (2.95, 0.03), (2.95, bus - 0.05), lw=2.2)
    ax.text(7.6, -1.05, "every gate reads the same input:  "
            "$a = W_{xh} x_t + W_{hh} h_{t-1} + b_h$   (4H,1), split into 4 blocks of H",
            fontsize=12, ha="center", va="top")

    gy = 4.0
    gates = [(3.9, "f", "σ  forget", f, C_F), (5.75, "i", "σ  input", i, C_I),
             (7.65, "g", "tanh  cand.", g, C_G), (9.6, "o", "σ  output", o, C_O)]
    for x, sym, lab, v, col in gates:
        mean = f"mean |g| {np.abs(v).mean():.2f}" if sym == "g" else f"mean {v.mean():.2f}"
        box(ax, x, gy, f"${sym}_t$\n{lab}\n{mean}", ec=col, lw=3, w=1.75, h=1.4, fs=11)
        arrow(ax, (x, bus), (x, gy - 0.72), color=col)
    # f -> × on the belt
    arrow(ax, (3.9, gy + 0.72), (3.9, belt - 0.38), color=C_F)
    # i, g -> × -> + on the belt
    op(ax, 7.0, 5.95, "×")
    arrow(ax, (5.75, gy + 0.72), (6.72, 5.75), color=C_I)
    arrow(ax, (7.65, gy + 0.72), (7.2, 5.66), color=C_G)
    arrow(ax, (7.0, 6.31), (7.0, belt - 0.38), color="black")
    # c_t -> tanh -> × o -> h_t
    box(ax, 11.4, 5.95, "tanh", w=1.1, h=0.6, fs=11.5)
    arrow(ax, (11.4, belt), (11.4, 6.27))
    op(ax, 11.4, gy, "×")
    arrow(ax, (11.4, 5.64), (11.4, gy + 0.37))
    arrow(ax, (10.43, gy), (11.03, gy), color=C_O)
    ax.text(11.65, 5.0, "read", fontsize=11, color=C_O, weight="bold")
    box(ax, 15.0, gy, "$h_t = o_t ⊙ \\tanh(c_t)$\nnew hidden state (H,1)\n"
        f"all in (-1, 1): max |h| {np.abs(h_new).max():.2f}", fc=C_H, w=3.3, h=1.4, fs=11)
    arrow(ax, (11.77, gy), (13.33, gy), lw=2.5)
    arrow(ax, (15.0, gy - 0.72), (15.0, 2.75), color=C_HY)
    ax.text(15.0, 2.6, "→ prediction  $p_t$ = softmax($W_{hy} h_t + b_y$)\n"
            "and → the next step's $h_{t-1}$", ha="center", va="top", fontsize=11)
    ax.text(15.0, belt - 0.95, f"max |c| = {np.abs(c_new).max():.2f}\n(not squashed)",
            ha="center", va="top", fontsize=10.5, color="#555")

    ax.text(7.2, 10.25, "$c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t$        "
            "$h_t = o_t ⊙ \\tanh(c_t)$", fontsize=16, ha="center")
    ax.text(7.2, 9.65, "new memory = (keep some old memory) + (write some new content);  "
            "then show part of it", fontsize=12, ha="center", color="#444")

    ins = ax.inset_axes([14.3, -1.0, 2.6, 1.75], transform=ax.transData)
    ins.barh(range(4)[::-1], p[top], color=C_HY)
    ins.set_yticks(range(4)[::-1])
    ins.set_yticklabels([f"'{show(itos[k])}'" for k in top], fontsize=11)
    ins.set_xlim(0, 1)
    ins.tick_params(axis="x", labelsize=9)
    for k, ix in enumerate(top):
        ins.text(p[ix] + 0.03, 3 - k, f"{p[ix]:.0%}", va="center", fontsize=10)
    ins.set_title("real model, after reading \"said the k\":", fontsize=10)
    ax.set_title("ONE time step of an LSTM: four gates decide what to erase, write and show "
                 "(real values for reading 'k')", fontsize=15, pad=6)
    save(fig, "lstm_cell.png")
    return dict(f=f.mean(), i=i.mean(), o=o.mean(), g=np.abs(g).mean(), kept=kept,
                written=written, top=itos[top[0]], p=p[top[0]])


def fig_gru_cell(gru, stoi, itos):
    rec = run_gru(gru, stoi, "said the k")
    r, z, n = rec["r"][-1], rec["z"][-1], rec["n"][-1]
    p = rec["p"][-1]
    top = np.argsort(p)[::-1][:4]

    fig, ax = plt.subplots(figsize=(16, 9.8))
    ax.set_xlim(-0.6, 18.2)
    ax.set_ylim(-1.6, 10.3)
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((2.1, 0.8), 10.3, 8.0, boxstyle="round,pad=0.1,rounding_size=0.3",
                                fc="#f7f7f7", ec="grey", ls="--", lw=1.5, zorder=0))
    ax.text(12.3, 1.05, "the GRU cell (one time step)", fontsize=12, color="grey", style="italic",
            ha="right")

    belt = 7.6
    box(ax, 0.8, belt, "$h_{t-1}$\nhidden state\n(H,1)", fc=C_H, w=2.0, h=1.4)
    box(ax, 15.0, belt, "$h_t$\nnew hidden state\n(H,1)", fc=C_H, w=2.4, h=1.4)
    ax.plot([1.8, 13.8], [belt, belt], color="#b2182b", lw=6, alpha=0.35, zorder=0,
            solid_capstyle="round")
    arrow(ax, (1.8, belt), (4.64, belt), lw=3)
    op(ax, 5.0, belt, "×")
    arrow(ax, (5.36, belt), (8.64, belt), lw=3)
    op(ax, 9.0, belt, "+")
    arrow(ax, (9.36, belt), (13.8, belt), lw=3)
    ax.text(7.6, belt + 0.95, "the GRU's highway is h itself: keep z of the old h, "
            "fill the rest (1 − z) with new content", fontsize=12.5, color="#b2182b",
            ha="center")
    ax.text(5.0, belt + 0.45, "keep", fontsize=11, ha="center", va="bottom", color=C_I,
            weight="bold")
    ax.text(9.0, belt + 0.45, "blend in", fontsize=11, ha="center", va="bottom", color=C_G,
            weight="bold")

    bus = 1.9
    # h_{t-1} also feeds the gates: drop it from the belt to the bus.
    ax.plot([2.4, 2.4], [belt, bus], color="black", lw=2.2, zorder=1)
    ax.plot([2.4, 10.0], [bus, bus], color="black", lw=2.2, zorder=1)
    box(ax, 4.6, -0.35, "$x_t$ = one-hot of 'k'  (V,1)", fc=C_X, w=3.6, h=0.75, fs=11.5)
    arrow(ax, (4.6, 0.03), (4.6, bus - 0.05), lw=2.2)
    ax.text(6.9, -1.05, "the gates read $x_t$ and $h_{t-1}$:   "
            "$W_{xh} x_t + b_{xh}$  and  $W_{hh} h_{t-1} + b_{hh}$   (3H,1) each, "
            "split into 3 blocks: r, z, n", fontsize=12, ha="center", va="top")

    gy = 4.0
    box(ax, 3.6, gy, f"$r_t$\nσ  reset\nmean {r.mean():.2f}", ec=C_F, lw=3, w=1.65, h=1.4, fs=11)
    box(ax, 5.6, gy, f"$z_t$\nσ  update\nmean {z.mean():.2f}", ec=C_I, lw=3, w=1.65, h=1.4,
        fs=11)
    box(ax, 9.0, gy, f"$n_t$  tanh  candidate\n$\\tanh(W_{{xn}} x_t + b_{{xn}}$\n"
        f"$+\\ r_t ⊙ (W_{{hn}} h_{{t-1}} + b_{{hn}}))$\nmean |n| {np.abs(n).mean():.2f}",
        ec=C_G, lw=3, w=3.3, h=1.75, fs=10.5)
    arrow(ax, (3.6, bus), (3.6, gy - 0.72), color=C_F)
    arrow(ax, (5.6, bus), (5.6, gy - 0.72), color=C_I)
    arrow(ax, (9.0, bus), (9.0, gy - 0.9), color=C_G)
    # r scales the memory's vote inside n
    ax.annotate("", xy=(7.55, gy - 0.88), xytext=(4.1, gy - 0.7),
                arrowprops=dict(arrowstyle="-|>", color=C_F, lw=2, mutation_scale=16,
                                connectionstyle="arc3,rad=0.3"))
    ax.text(7.3, 2.62, "r: how much of the old memory\nthe candidate may look at",
            fontsize=9.5, color=C_F, ha="center", va="top")
    # z -> × on the belt; 1 - z -> × with n -> + on the belt
    arrow(ax, (5.6, gy + 0.72), (5.0, belt - 0.38), color=C_I)
    box(ax, 7.05, 5.95, "1 − z", ec=C_I, w=0.95, h=0.55, fs=11)
    arrow(ax, (6.0, gy + 0.72), (6.85, 5.66), color=C_I)
    op(ax, 9.0, 5.95, "×")
    arrow(ax, (7.53, 5.95), (8.64, 5.95), color=C_I)
    arrow(ax, (9.0, gy + 0.9), (9.0, 5.59), color=C_G)
    arrow(ax, (9.0, 6.31), (9.0, belt - 0.38))
    arrow(ax, (15.0, belt - 0.72), (15.0, 5.0), color=C_HY)
    ax.text(15.0, 4.8, "→ prediction  $p_t$ = softmax($W_{hy} h_t + b_y$)\n"
            "and → the next step's $h_{t-1}$\n\nno separate cell state c,\n"
            "no output gate: h IS the memory", ha="center", va="top", fontsize=11)

    ax.text(7.2, 9.85, "$h_t = (1 - z_t) ⊙ n_t + z_t ⊙ h_{t-1}$", fontsize=16, ha="center")
    ax.text(7.2, 9.25, "new memory = blend of (new candidate) and (old memory); z decides "
            "the mix per unit", fontsize=12, ha="center", color="#444")

    ins = ax.inset_axes([14.3, -1.0, 2.6, 1.75], transform=ax.transData)
    ins.barh(range(4)[::-1], p[top], color=C_HY)
    ins.set_yticks(range(4)[::-1])
    ins.set_yticklabels([f"'{show(itos[k])}'" for k in top], fontsize=11)
    ins.set_xlim(0, 1)
    ins.tick_params(axis="x", labelsize=9)
    for k, ix in enumerate(top):
        ins.text(p[ix] + 0.03, 3 - k, f"{p[ix]:.0%}", va="center", fontsize=10)
    ins.set_title("real GRU, after reading \"said the k\":", fontsize=10)
    ax.set_title("ONE time step of a GRU: two gates (reset r, update z), one state h "
                 "(real values for reading 'k')", fontsize=15, pad=6)
    save(fig, "gru_cell.png")
    return dict(r=r.mean(), z=z.mean(), top=itos[top[0]], p=p[top[0]])


def fig_gate_heatmap(lstm, stoi, text):
    sentence = "The light turned and turned."
    start = text.index(sentence)
    rec = run_lstm(lstm, stoi, text[:start + len(sentence)])  # read the story up to here
    sl = slice(start, start + len(sentence))
    f, i, o = rec["f"][sl], rec["i"][sl], rec["o"][sl]
    story = run_lstm(lstm, stoi, text)
    # 20 units spread over the ranking of their average forget gate on the whole story,
    # from the most "long-term" unit (f ≈ 1 always) down to the most "short-term" one.
    fbar = story["f"].mean(0)
    order = np.argsort(fbar)[::-1]
    units = order[np.linspace(0, lstm.H - 1, 20).round().astype(int)]

    fig, axes = plt.subplots(3, 1, figsize=(14, 12.5), sharex=True)
    for ax, (sym, M, col, words) in zip(axes, [
            ("f", f, "Reds", "FORGET gate f: 1 = keep the old memory, 0 = erase it"),
            ("i", i, "Blues", "INPUT gate i: 1 = write the candidate g, 0 = write nothing"),
            ("o", o, "Greens", "OUTPUT gate o: 1 = show this memory unit in h, 0 = hide it")]):
        im = ax.imshow(M[:, units].T, cmap=col, vmin=0, vmax=1, aspect="auto")
        ax.set_yticks(range(len(units)))
        ax.set_yticklabels([f"unit {u}  (avg f {fbar[u]:.2f})" for u in units], fontsize=8)
        whole = story[sym]
        ax.set_title(f"{words}     (whole story: mean {whole.mean():.2f}, "
                     f"{(whole > 0.9).mean():.0%} of values > 0.9, "
                     f"{(whole < 0.1).mean():.0%} < 0.1)", fontsize=12.5, loc="left")
        cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
        cb.set_label(sym, rotation=0, fontsize=14)
    axes[-1].set_xticks(range(len(sentence)))
    axes[-1].set_xticklabels([show(c) for c in sentence], fontsize=14, family="monospace")
    fbar_o = np.corrcoef(fbar, story["o"].mean(0))[0, 1]
    axes[-1].set_xlabel("character just read (the model has read the story up to this "
                        "sentence, so its memory is in context)\n"
                        f"Across all {lstm.H} units, the more a unit remembers (avg f) the less "
                        f"it shows (avg o): correlation {fbar_o:+.2f}", labelpad=8)
    fig.suptitle(f"Inside the trained LSTM: gate values while reading \"{sentence}\"\n"
                 f"(20 of the {lstm.H} units, sorted from long-term memory (top: f ≈ 1 almost "
                 "always) to short-term (bottom))", y=0.995)
    fig.tight_layout()
    save(fig, "gate_heatmap.png")
    out = {s: (story[s].mean(), (story[s] > 0.9).mean(), (story[s] < 0.1).mean())
           for s in "fio"}
    out["fbar"] = (fbar.max(), fbar.min(), int((fbar > 0.9).sum()), fbar_o)
    return out


def fig_cell_vs_hidden(lstm, stoi, text):
    story = run_lstm(lstm, stoi, text)
    # Find the unit whose c best tracks "characters since the last . ! ?".
    since, n = np.zeros(len(text)), 0
    for k, ch in enumerate(text):
        n = 0 if ch in ".!?" else n + 1
        since[k] = n
    corr = np.nan_to_num([np.corrcoef(u, since)[0, 1] for u in story["c"].T])
    u = int(np.argmax(np.abs(corr)))

    start = text.index("At night I light the lamp")
    seg = slice(start, start + 120)
    c, h, o = story["c"][seg, u], story["h"][seg, u], story["o"][seg, u]
    passage = text[seg]
    T = len(passage)
    cmax, hmax = np.abs(story["c"]).max(), np.abs(story["h"]).max()
    big = np.abs(story["c"]).max(0)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8.4), sharex=True,
                                   gridspec_kw=dict(height_ratios=[1.3, 1]))
    for ax in (ax1, ax2):
        for k, ch in enumerate(passage):
            if ch in ".!?":
                ax.axvline(k, color="#999", ls=":", lw=1.3)
    ax1.plot(c, color="#b2182b", lw=2.5, marker="o", ms=3.5)
    ax1.set_ylabel(f"cell state $c_t$\n(unit {u})")
    ax1.set_title(f"Unit {u}'s cell state drifts down during a sentence and jumps back up "
                  f"after each . ! ?  (correlation with \"chars since the last . ! ?\": "
                  f"{corr[u]:+.2f} over the whole story)", fontsize=12.5, loc="left")
    ax1.text(0.99, 0.95, f"c is never squashed: over the story this unit reaches "
             f"|c| = {np.abs(story['c'][:, u]).max():.1f};\nthe largest |c| of any unit is "
             f"{cmax:.1f} ({int((big > 5).sum())} of {lstm.H} units go beyond ±5)",
             transform=ax1.transAxes, ha="right", va="top", fontsize=11,
             bbox=dict(boxstyle="round", fc="white", ec="#bbb"))
    ax2.plot(o, color=C_O, lw=1.5, alpha=0.7, label=f"output gate $o_t$ (unit {u})")
    ax2.plot(h, color="#d4a017", lw=2.5, marker="o", ms=3.5,
             label=f"$h_t$ = $o_t$ ⊙ tanh($c_t$) ≈ $-o_t$ here, because c ≪ 0")
    ax2.legend(loc="upper right", fontsize=10.5, ncol=2)
    ax2.axhspan(-1, 1, color="#fdf1c7", alpha=0.5, zorder=0)
    ax2.set_ylim(-1.15, 1.6)
    ax2.set_ylabel(f"hidden state $h_t$\n= $o_t$ ⊙ tanh($c_t$)")
    ax2.set_title(f"The same unit's hidden state is squashed into (-1, 1) (max |h| of any "
                  f"unit: {hmax:.3f}), and its output gate keeps it hidden most of the time "
                  f"(avg o = {story['o'][:, u].mean():.2f})",
                  fontsize=12.5, loc="left")
    ax2.set_xticks(range(T))
    ax2.set_xticklabels([show(ch) for ch in passage], fontsize=7.5, family="monospace")
    ax2.set_xlim(-1, T)
    ax2.set_xlabel("characters read (dotted lines: . ! ?). The text: "
                   f"\"{passage[:60].strip()}…\"", labelpad=8)
    fig.suptitle("Cell state vs hidden state: c is the long-term store (any size), h is "
                 "the bounded view the rest of the network sees", y=1.0)
    fig.tight_layout()
    save(fig, "cell_vs_hidden.png")
    return u, corr[u], cmax, hmax, int((big > 5).sum())


def char_grad_flow(rnn, lstm, stoi, text, T=100):
    """||dL_last/dh_k|| (RNN) and ||dL_last/dc_k|| (LSTM) for a real T-char chunk."""
    start = text.index("The light turned and turned")
    chunk = text[start - T + 27:start + 28]
    inp, tgt = chunk[:-1], chunk[1:]
    # vanilla RNN
    P = rnn.params
    h, hs = np.zeros((rnn.H, 1)), []
    for ch in inp:
        h = np.tanh(P["W_xh"] @ onehot(stoi[ch], rnn.V) + P["W_hh"] @ h + P["b_h"])
        hs.append(h)
    dy = softmax(P["W_hy"] @ hs[-1] + P["b_y"])
    dy[stoi[tgt[-1]]] -= 1
    dh, rnn_n = P["W_hy"].T @ dy, []
    for k in range(T - 1, -1, -1):
        rnn_n.append(np.linalg.norm(dh))
        dh = P["W_hh"].T @ ((1 - hs[k] ** 2) * dh)
    # LSTM: same backward as LSTM.backward, but only for the LAST step's loss.
    P = lstm.params
    H = lstm.H
    h, c = lstm.init_state()
    cs, gates = [c], []
    for ch in inp:
        h, c, gt = lstm.cell_step(onehot(stoi[ch], lstm.V), h, c)
        cs.append(c)
        gates.append(gt)
    dy = softmax(P["W_hy"] @ h + P["b_y"])
    dy[stoi[tgt[-1]]] -= 1
    dh, dc_next, lstm_n = P["W_hy"].T @ dy, np.zeros((H, 1)), []
    for k in range(T - 1, -1, -1):
        i, f, g, o, tc = gates[k]
        dc = dh * o * (1 - tc ** 2) + dc_next
        lstm_n.append(np.linalg.norm(dc))
        da = np.vstack([dc * g * i * (1 - i), dc * cs[k] * f * (1 - f),
                        dc * i * (1 - g ** 2), dh * tc * o * (1 - o)])
        dh, dc_next = P["W_hh"].T @ da, dc * f
    return np.array(rnn_n), np.array(lstm_n), inp


def fig_gradient_flow(lm, rnn, lstm, stoi, text):
    models = [str(m) for m in lm["models"]]
    D = int(lm["flow_delay"])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5, 6.4))
    d = np.arange(D + 2)
    for k, m in enumerate(models):
        ax1.plot(d, lm[f"flow_{k}_trained"], color=COL[m], lw=2.8,
                 label=f"{m}{', ||dL/dc_k||' if m.startswith('LSTM') else ', ||dL/dh_k||'}")
        ax1.plot(d, lm[f"flow_{k}_init"], color=COL[m], lw=1.5, ls="--", alpha=0.8)
    ax1.set_yscale("log")
    ax1.set_ylim(1e-23, 5)
    ax1.set_xlim(0, D + 1)
    ax1.axhline(1, color="grey", ls=":", lw=1)
    ax1.set_xlabel(f"distance back from the last step (d steps)")
    ax1.set_ylabel("gradient reaching step k, relative to the last step")
    ax1.set_title(f"\"Remember the first token\" task, delay D = {D}\n"
                  "solid = after training, dashed = at initialization", fontsize=13)
    ax1.legend(loc="lower left", fontsize=10)
    rnn_tr = lm[f"flow_{models.index('RNN')}_trained"][D + 1]
    chrono_tr = lm[f"flow_{models.index('LSTM (chrono)')}_trained"][D + 1]
    ax1.text(D + 0.5, 1e-2, f"at the first token ({D + 1} steps back):\n"
             f"LSTM (chrono) {chrono_tr:.1e}\nRNN {rnn_tr:.1e}", ha="right", va="top",
             fontsize=10.5,
             bbox=dict(boxstyle="round", fc="white", ec="#bbb"))

    rnn_n, lstm_n, inp = char_grad_flow(rnn, lstm, stoi, text)
    T = len(inp)
    ax2.plot(np.arange(T), rnn_n / rnn_n[0], color=COL["RNN"], lw=2.8,
             label="vanilla RNN, ||dL/dh_k||")
    ax2.plot(np.arange(T), lstm_n / lstm_n[0], color=COL["LSTM"], lw=2.8,
             label="LSTM, ||dL/dc_k||")
    ax2.set_yscale("log")
    ax2.axhline(1, color="grey", ls=":", lw=1)
    ax2.set_xlim(0, T - 1)
    ax2.set_xlabel("distance back from the last character (d steps)")
    ax2.set_title(f"Char models trained on input.txt: the last char's loss,\n"
                  f"sent back over a real {T}-char chunk (\"…{inp[-24:]}\")", fontsize=13)
    ax2.legend(loc="lower left", fontsize=10.5)
    r25, l25 = rnn_n[25] / rnn_n[0], lstm_n[25] / lstm_n[0]
    r99, l99 = rnn_n[-1] / rnn_n[0], lstm_n[-1] / lstm_n[0]
    ax2.text(T - 2, 0.5, f"25 steps back: RNN {r25:.1e}, LSTM {l25:.1e}\n"
             f"{T - 1} steps back: RNN {r99:.1e}, LSTM {l99:.1e}", ha="right", va="top",
             fontsize=10.5, bbox=dict(boxstyle="round", fc="white", ec="#bbb"))
    fig.suptitle("The key insight: through the cell state the gradient is multiplied by f "
                 "(≈ 1), not by $W_{hh}$·tanh′ (< 1)", y=1.01)
    fig.tight_layout()
    save(fig, "gradient_flow.png")
    return dict(rnn_tr=rnn_tr, chrono_tr=chrono_tr, r25=r25, l25=l25, r99=r99, l99=l99, T=T)


def fig_long_memory(lm):
    models = [str(m) for m in lm["models"]]
    delays, acc, seeds = lm["delays"], lm["acc"], lm["seeds"]
    fig, ax = plt.subplots(figsize=(12, 6.2))
    offs = np.linspace(-0.05, 0.05, len(models))
    for k, m in enumerate(models):
        mean = acc[k].mean(axis=1)
        wide = m == "LSTM (chrono)"  # drawn wide so it shows under the GRU line
        ax.plot(delays, mean, color=COL[m], lw=7 if wide else 2.8, marker="o",
                ms=13 if wide else 8, label=m, ls="--" if m == "LSTM (b_f=1)" else "-",
                alpha=0.9, zorder=2 if wide else 3)
        for j, D in enumerate(delays):
            ax.scatter(D * np.exp(np.full(len(seeds), offs[k])), acc[k, j], color=COL[m],
                       s=18, alpha=0.6, zorder=3)
    ax.axhline(0.25, color="black", ls=":", lw=1.5)
    ax.text(delays[0], 0.27, "chance (1 of 4 signals) = 25%", fontsize=11)
    ax.set_xscale("log", base=2)
    ax.set_xticks(delays)
    ax.set_xticklabels([str(D) for D in delays])
    ax.set_ylim(0, 1.08)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
    ax.set_xlabel("delay D = number of random distractor tokens between the signal and the "
                  "question")
    ax.set_ylabel("test accuracy (2000 new sequences)")
    ax.legend(loc="center left", fontsize=11, bbox_to_anchor=(0.0, 0.47))
    ax.set_title("\"Remember the first token\": who can carry information across D steps?\n"
                 f"(line = mean of {len(seeds)} seed{'s' if len(seeds) > 1 else ''}, dots = each seed; H = 32, Adam, "
                 "≤ 800 steps)", fontsize=13.5)
    save(fig, "long_memory.png")


def fig_training(curves, V):
    window = 200
    fig, ax = plt.subplots(figsize=(13, 6.6))
    for name, (losses, n_params) in curves.items():
        smooth = np.array([losses[max(0, k - window + 1):k + 1].mean()
                           for k in range(len(losses))])
        ax.plot(losses, color=COL[name], lw=0.4, alpha=0.25)
        ls = "--" if name == "RNN (H=210)" else "-"
        ax.plot(smooth, color=COL[name], lw=2.6, ls=ls,
                label=f"{name}: {n_params:,} params, final {smooth[-1]:.3f}")
    ax.axhline(np.log(V), color="black", ls=":", lw=1.5)
    ax.text(len(losses) * 0.99, np.log(V) * 1.08, f"random guessing: ln({V}) = {np.log(V):.2f}",
            ha="right", fontsize=11)
    ax.set_yscale("log")
    ax.set_ylim(0.01, 8)
    ax.set_xlim(0, len(losses))
    ax.set_xlabel("training iteration (one 25-char chunk each, same text, same Adagrad loop)")
    ax.set_ylabel("loss per character (nats, log scale)")
    ax.legend(loc="lower left", fontsize=11, title=f"average of the last {window} chunks",
              title_fontsize=11)
    ax.set_title("Same text, same training loop, 10,000 iterations: the gated models "
                 "learn faster and go much lower", fontsize=14)
    save(fig, "training_curves.png")


def fig_forget_decay(lm):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.8),
                                   gridspec_kw=dict(width_ratios=[1, 1.35]))
    z = np.linspace(-6, 6, 300)
    sig, th = 1 / (1 + np.exp(-z)), np.tanh(z)
    ax1.plot(z, sig, color=C_F, lw=3, label="sigmoid σ(z) ∈ (0, 1): a GATE")
    ax1.plot(z, th, color=C_G, lw=3, label="tanh(z) ∈ (-1, 1): CONTENT")
    ax1.axhline(0, color="grey", lw=0.8)
    ax1.axhline(1, color="grey", lw=0.8, ls=":")
    ax1.plot([1], [1 / (1 + np.exp(-1))], "o", color="black", ms=8, zorder=5)
    ax1.annotate(f"forget bias b_f = 1\n→ f starts at σ(1) = {1 / (1 + np.exp(-1)):.2f}",
                 xy=(1, 0.731), xytext=(-5.8, 0.72), fontsize=10.5,
                 arrowprops=dict(arrowstyle="->", color="black"))
    ax1.set_xlabel("pre-activation z")
    ax1.set_ylim(-1.15, 1.15)
    ax1.legend(loc="lower right", fontsize=10.5)
    ax1.set_title("Gates multiply, so they live in (0, 1): 0 = closed, 1 = open.\n"
                  "Content can be + or −, so it uses tanh.", fontsize=12.5)

    k = np.arange(0, 101)
    sig1 = 1 / (1 + np.exp(-1))
    D = int(lm["flow_delay"])
    models = [str(m) for m in lm["models"]]
    rnn_flow = lm[f"flow_{models.index('RNN')}_init"]
    per_step = rnn_flow[D + 1] ** (1 / (D + 1))  # geometric mean shrink factor per step
    lines = [(0.5, "#2166ac"), (sig1, "#ff9896"), (0.9, "#ef8a62"), (0.99, "#d62728"),
             (1.0, "black")]
    for fv, col in lines:
        y = fv ** k
        lab = f"f = {fv:.2f}" + ("  (= σ(1), our init)" if fv == sig1 else "")
        ax2.plot(k, y, color=col, lw=2.6, label=lab)
        nudge = {1.0: 4.0, 0.99: 0.25}.get(fv, 1.0)  # keep the two top labels apart
        ax2.text(101, fv ** 100 * nudge, f"{fv ** 100:.0e}" if fv < 1 else "1",
                 va="center", fontsize=10, color=col)
    ax2.plot(k, per_step ** k, color="#7f7f7f", lw=2, ls="--",
             label=f"vanilla RNN at init, measured: ×{per_step:.2f} per step")
    ax2.set_yscale("log")
    ax2.set_ylim(1e-33, 20)
    ax2.set_xlim(0, 110)
    ax2.set_xlabel("steps k later")
    ax2.set_ylabel("Π f  = how much of $c_0$ (and of its gradient) survives")
    ax2.legend(loc="lower left", fontsize=10.5)
    ax2.set_title("Along the cell state, memory AND gradient are multiplied by f each step:\n"
                  "$c_0$'s share of $c_k$ = $f^k c_0$,   $\\partial L/\\partial c_0$ = "
                  "$f^k\\, \\partial L/\\partial c_k$   (constant f, path through c)",
                  fontsize=12.5)
    fig.tight_layout()
    save(fig, "forget_decay.png")
    return per_step


def fig_param_counts(V, H):
    parts = {
        "RNN": RNN(V, H).params,
        "GRU": GRU(V, H).params,
        "LSTM": LSTM(V, H).params,
    }
    groups = [("W_xh (input → gates)", ["W_xh"], C_XH), ("W_hh (memory → gates)", ["W_hh"], C_HH),
              ("gate biases", ["b_h", "b_xh", "b_hh"], "#9467bd"),
              ("W_hy, b_y (output layer)", ["W_hy", "b_y"], C_HY)]
    fig, ax = plt.subplots(figsize=(13, 5.2))
    names = list(parts)
    totals = {}
    for row, nm in enumerate(names):
        left = 0
        P = parts[nm]
        for label, keys, col in groups:
            n = sum(P[k].size for k in keys if k in P)
            ax.barh(row, n, left=left, color=col, ec="white", label=label if row == 0 else None)
            if n > 4000:
                ax.text(left + n / 2, row, f"{n:,}", ha="center", va="center", color="white",
                        fontsize=10.5, weight="bold")
            left += n
        totals[nm] = left
        blocks = {"RNN": "1 block", "GRU": "3 gate blocks (r, z, n)",
                  "LSTM": "4 gate blocks (i, f, g, o)"}[nm]
        ax.text(left + 800, row, f"{left:,}   ({blocks}: ×{left / totals['RNN']:.1f})",
                va="center", fontsize=11.5)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=13)
    ax.set_xlim(0, totals["LSTM"] * 1.45)
    ax.set_xlabel("number of parameters")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.legend(loc="upper center", ncol=4, fontsize=10.5, bbox_to_anchor=(0.5, -0.17))
    ax.set_title(f"Parameter count for our char models (V = {V}, H = {H}): the recurrent "
                 "part is ×3 (GRU) and ×4 (LSTM)", fontsize=13.5)
    save(fig, "param_counts.png")
    return totals


def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    with open(DEFAULT_TEXT, encoding="utf-8") as f:
        text = f.read()

    print("getting the trained models...")
    lstm, chars, lstm_losses = get_char_model(text, "lstm")
    gru, _, gru_losses = get_char_model(text, "gru")
    rnn, rnn_losses = get_rnn_baseline(text, 100, 0.1)
    rnn210, rnn210_losses = get_rnn_baseline(text, 210, 0.05)
    lm = get_long_memory()
    stoi = {c: i for i, c in enumerate(chars)}
    itos = dict(enumerate(chars))
    print(f"  ready after {time.time() - t0:.1f} s")

    cell = fig_lstm_cell(lstm, stoi, itos)
    print(f"  lstm_cell: f {cell['f']:.2f} i {cell['i']:.2f} o {cell['o']:.2f} "
          f"|g| {cell['g']:.2f}; kept {cell['kept']:.2f} written {cell['written']:.2f}; "
          f"next {cell['top']!r} {cell['p']:.1%}")
    gc = fig_gru_cell(gru, stoi, itos)
    print(f"  gru_cell: r {gc['r']:.2f} z {gc['z']:.2f}; next {gc['top']!r} {gc['p']:.1%}")
    gs = fig_gate_heatmap(lstm, stoi, text)
    for s in "fio":
        m, hi, lo = gs[s]
        print(f"  gate_heatmap: {s} mean {m:.2f}, >0.9 {hi:.0%}, <0.1 {lo:.0%}")
    print("  gate_heatmap: per-unit average f from {:.2f} to {:.2f}; {} units above 0.9; "
          "corr(avg f, avg o) {:+.2f}".format(*gs["fbar"]))
    u, cu, cmax, hmax, nbig = fig_cell_vs_hidden(lstm, stoi, text)
    print(f"  cell_vs_hidden: unit {u} corr {cu:+.2f}; max|c| {cmax:.1f}, max|h| {hmax:.3f}, "
          f"{nbig} units beyond ±5")
    gf = fig_gradient_flow(lm, rnn, lstm, stoi, text)
    print(f"  gradient_flow: task first token RNN {gf['rnn_tr']:.1e} vs LSTM(chrono) "
          f"{gf['chrono_tr']:.1e}; char 25 back RNN {gf['r25']:.1e} LSTM {gf['l25']:.1e}; "
          f"{gf['T'] - 1} back RNN {gf['r99']:.1e} LSTM {gf['l99']:.1e}")
    fig_long_memory(lm)
    n = lambda m: sum(v.size for v in m.params.values())  # noqa: E731
    curves = {"RNN": (rnn_losses, n(rnn)), "RNN (H=210)": (rnn210_losses, n(rnn210)),
              "GRU": (gru_losses, n(gru)), "LSTM": (lstm_losses, n(lstm))}
    fig_training(curves, lstm.V)
    for name, (losses, k) in curves.items():
        print(f"  training: {name:<12} {k:>6,} params, avg loss/char of the last 200 "
              f"chunks {losses[-200:].mean():.3f}, at iter 2000 {losses[1801:2001].mean():.3f}")
    per_step = fig_forget_decay(lm)
    print(f"  forget_decay: RNN at init shrinks ×{per_step:.2f}/step; "
          f"0.73^80 = {(1 / (1 + np.exp(-1))) ** 80:.1e}")
    totals = fig_param_counts(lstm.V, lstm.H)
    print("  param_counts: " + ", ".join(f"{k} {v:,}" for k, v in totals.items()))
    print(f"total time {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
