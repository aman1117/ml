"""
Draw the pictures used in tokenizer/README.md and save them to tokenizer/images/.

Every number in every figure is computed here, with the code of this folder:
  - utf8_bytes.png       characters -> code points -> UTF-8 bytes
  - bpe_merges.png       the training loop on the toy corpus of bpe_visual.py
  - token_boundaries.png the same text cut by bytes, BPE-512 and BPE-4096
  - compression.png      bytes/token, embedding cost and data per token vs vocabulary size
  - zipf.png             token frequencies of the V=4096 tokenizer on its training text
  - bpb_curves.png       char vs BPE mini-GPTs in bits per byte (bpe_pytorch.npz, or runs
                         bpe_pytorch.py first, ~2 min)
  - failure_modes.png    letters, digits, other languages, leading/trailing spaces

Run (~10 s once bpe_pytorch.npz exists):
    python tokenizer/make_figures.py
"""
import os
import subprocess
import sys
import warnings
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.font_manager import fontManager  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

from bpe import (GPT2_PATTERN, Tokenizer, best_pair, get_stats, get_tokenizer,  # noqa: E402
                 load_corpus, merge, show, split_text, train_bpe)
from bpe_visual import TOY  # noqa: E402
from why_tokenization import (compression_table, digit_splits, language_costs,  # noqa: E402
                              zipf_stats)

warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="Glyph .* missing from")  # if no CJK font is installed

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "images")
DPI = 130
PALETTE = ["#cfe2f3", "#fde2c4", "#d5ecd0", "#f6d5e5", "#fff2b3", "#e1d8f0"]
C_NEW, C_TR, C_VA = "#7bd389", "#1f77b4", "#d62728"
RUN_COLORS = {"chars": "#7f7f7f", "BPE-512": "#1f77b4", "BPE-4096": "#d62728"}

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "figure.titlesize": 16,
    "axes.spines.top": False,
    "axes.spines.right": False,
})
# DejaVu has no CJK glyphs; fall back to a font that does, when one is installed (macOS).
_fallback = [f for f in ("Arial Unicode MS", "Noto Sans CJK SC", "PingFang SC")
             if f in {x.name for x in fontManager.ttflist}]
plt.rcParams["font.family"] = ["DejaVu Sans"] + _fallback
MONO = ["DejaVu Sans Mono"] + _fallback


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {os.path.relpath(path)}  ({os.path.getsize(path) // 1024} KB)")


def token_row(ax, x, y, pieces, h=0.8, unit=0.21, fs=11, colors=None, highlight=(), gap=0.04):
    """Draw tokens as adjacent coloured boxes, width proportional to their text length."""
    for k, p in enumerate(pieces):
        w = max(1, len(p)) * unit + 0.12
        fc = C_NEW if k in highlight else (colors[k] if colors else PALETTE[k % len(PALETTE)])
        ax.add_patch(FancyBboxPatch((x, y - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.08",
                                    fc=fc, ec="#555555", lw=0.8))
        ax.text(x + w / 2, y, p, ha="center", va="center", fontsize=fs, family=MONO)
        x += w + gap
    return x


# ---------------------------------------------------------------- figures

def fig_utf8():
    text = "héllo 世界 🙂"
    fig, ax = plt.subplots(figsize=(14, 3.6))
    ax.set_xlim(-0.2, 18.6)
    ax.set_ylim(-0.6, 3.4)
    ax.axis("off")
    colors = {1: "#cfe2f3", 2: "#fde2c4", 3: "#d5ecd0", 4: "#f6d5e5"}
    x, bw = 0.0, 0.95
    for c in text:
        b = c.encode("utf-8")
        w = len(b) * bw
        glyph = c if ord(c) < 0x1F000 else "☺"  # matplotlib can't draw colour emoji
        ax.add_patch(FancyBboxPatch((x, 2.05), w - 0.06, 0.9, boxstyle="round,pad=0,rounding_size=0.08",
                                    fc="white", ec="#333333", lw=1.2))
        ax.text(x + (w - 0.06) / 2, 2.62, glyph if c != " " else "␣", ha="center", va="center",
                fontsize=17)
        ax.text(x + (w - 0.06) / 2, 2.22, f"U+{ord(c):X}", ha="center", va="center", fontsize=8,
                color="#555555")
        for k, byte in enumerate(b):
            ax.add_patch(FancyBboxPatch((x + k * bw, 0.9), bw - 0.06, 0.8,
                                        boxstyle="round,pad=0,rounding_size=0.06",
                                        fc=colors[len(b)], ec="#555555", lw=0.8))
            ax.text(x + k * bw + (bw - 0.06) / 2, 1.42, f"{byte:02X}", ha="center", va="center",
                    fontsize=11, family=MONO)
            ax.text(x + k * bw + (bw - 0.06) / 2, 1.08, str(byte), ha="center", va="center",
                    fontsize=8, color="#555555")
        x += w + 0.12
    ax.text(-0.15, 2.5, "characters\n(code points)", ha="right", va="center", fontsize=11)
    ax.text(-0.15, 1.3, "UTF-8 bytes\n(hex, decimal)", ha="right", va="center", fontsize=11)
    n_c, n_b = len(text), len(text.encode())
    ax.text(x / 2, 0.25, f"{n_c} characters → {n_b} bytes.   1 byte: ASCII   2: accents, Greek, "
            "Cyrillic, Arabic   3: most CJK, Devanagari   4: emoji", ha="center", fontsize=11)
    ax.text(x / 2, -0.3, "Byte-level BPE starts from the 256 byte values: any string is "
            "encodable, nothing is ever <unk>.", ha="center", fontsize=11, style="italic")
    fig.suptitle("Unicode text is stored as UTF-8 bytes (1 to 4 per character)", y=1.0)
    save(fig, "utf8_bytes.png")


def fig_merges(n_merges=6):
    chunks = split_text(TOY)
    freq = Counter(chunks)
    words = [list(c.encode()) for c in freq]
    weights = list(freq.values())
    vocab = {i: bytes([i]) for i in range(256)}
    shown = TOY[:44]  # draw the start of the text, but learn from all of it
    shown_chunks = split_text(shown)
    rows = [([show(bytes([b])) for c in shown_chunks for b in c.encode()], set(), "bytes", None)]
    merges = []
    for k in range(n_merges):
        stats = get_stats(words, weights)
        pair = best_pair(stats)
        idx = 256 + k
        vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
        merges.append((pair, idx))
        words = [merge(w, pair, idx) for w in words]
        seq = []
        for c in shown_chunks:
            ids = list(c.encode())
            for p, i in merges:
                ids = merge(ids, p, i)
            seq += ids
        label = (f"merge {k + 1}: {show(vocab[pair[0]])!r} + {show(vocab[pair[1]])!r}\n"
                 f"→ {idx} {show(vocab[idx])!r}   (count {stats[pair]})")
        rows.append(([show(vocab[i]) for i in seq], {j for j, i in enumerate(seq) if i == idx},
                     label, len(seq)))
    total = sum(len(w) for c in chunks for w in [c.encode()])
    fig, ax = plt.subplots(figsize=(15, 0.95 * len(rows) + 1.2))
    ax.set_xlim(-6.2, 15.2)
    ax.set_ylim(-len(rows) + 0.4, 1.0)
    ax.axis("off")
    for r, (pieces, hl, label, _) in enumerate(rows):
        y = -r
        x_end = token_row(ax, 0, y, pieces, h=0.62, unit=0.16, fs=10, highlight=hl, gap=0.03)
        ax.text(-0.25, y, label, ha="right", va="center", fontsize=10.5,
                family=MONO if r else None)
        ax.text(x_end + 0.2, y, f"{len(pieces)} tokens", va="center", fontsize=10.5,
                color="#555555")
    ax.text(0, 0.75, f"Training text: {TOY!r}  ({total} bytes, {len(chunks)} regex chunks)",
            fontsize=10.5, family=MONO, va="bottom")
    fig.subplots_adjust(top=0.97, bottom=0.02)
    fig.suptitle("BPE training: merge the most frequent adjacent pair, repeat "
                 "(green = the new token; first 44 characters shown)", y=1.02)
    save(fig, "bpe_merges.png")


def fig_boundaries(tok):
    lines = ["The keeper lit the lamp. Backpropagation computes",
             "the gradient: dW = x.T @ dy  (12345 tokens!)",
             "Le gardien alluma la lampe. 灯塔"]
    setups = [("bytes (V=256)", tok.truncated(256)), ("BPE-512", tok.truncated(512)),
              ("BPE-4096", tok)]
    fig, axes = plt.subplots(len(setups), 1, figsize=(15, 2.0 * len(setups) + 0.4))
    for ax, (name, t) in zip(axes, setups):
        ax.set_xlim(0, 20.5)
        ax.set_ylim(-len(lines) + 0.4, 0.8)
        ax.axis("off")
        n_tok = n_b = 0
        for r, line in enumerate(lines):
            ids = t.encode_ordinary(line)
            n_tok, n_b = n_tok + len(ids), n_b + len(line.encode())
            token_row(ax, 0, -r, [t.token_str(i) for i in ids], h=0.7, unit=0.13, fs=9.5, gap=0.02)
        ax.set_title(f"{name}: {n_tok} tokens for {n_b} bytes ({n_b / n_tok:.2f} bytes/token)",
                     loc="left", fontsize=13)
    fig.suptitle("Token boundaries (each box is one token; ␣ = space, \\xNN = one byte of a "
                 "multi-byte character)", y=1.0)
    fig.tight_layout()
    save(fig, "token_boundaries.png")


def fig_compression(rows):
    bpe = [r for r in rows if r["name"].startswith(("BPE", "bytes"))]
    V = np.array([r["V"] for r in bpe])
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    ax = axes[0]
    ax.plot(V, [r["bpt_tr"] for r in bpe], "o-", color=C_TR, lw=2, label="BPE, train text")
    ax.plot(V, [r["bpt_va"] for r in bpe], "s-", color=C_VA, lw=2, label="BPE, held-out text")
    w, c = rows[0], rows[1]
    ax.scatter([w["V"]], [w["bpt_va"]], marker="*", s=180, color="#9467bd", zorder=3,
               label=f"words: {w['bpt_va']:.2f}, but {w['unk_frac']:.0%} of val bytes <unk>")
    ax.scatter([c["V"]], [c["bpt_va"]], marker="D", s=60, color="#7f7f7f", zorder=3,
               label=f"characters (V={c['V']})")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("vocabulary size V")
    ax.set_ylabel("bytes per token (compression)")
    ax.set_title("Bigger V → fewer, longer tokens")
    ax.legend(fontsize=9.5, loc="upper left")
    for r in bpe:
        ax.annotate(f"{r['bpt_va']:.2f}", (r["V"], r["bpt_va"]), textcoords="offset points",
                    xytext=(4, -13), fontsize=9, color=C_VA)

    ax = axes[1]
    Vs = np.array([128, 256, 512, 1024, 2048, 4096, 8192, 32768, 50257, 100277, 200000])
    for d, body, col, lab in [(64, 98_304, "#2ca02c", "d=64 (our mini-GPT, 2 blocks: 98k)"),
                              (768, 84_934_656, "#ff7f0e", "d=768 (GPT-2 small, 12 blocks: 85M)")]:
        share = 2 * Vs * d / (2 * Vs * d + body)
        ax.plot(Vs, 100 * share, "o-", color=col, lw=2, label=lab, ms=4)
    for v, name in [(50257, "GPT-2"), (100277, "GPT-4"), (200000, "GPT-4o")]:
        ax.axvline(v, color="#bbbbbb", lw=1, ls=":")
        ax.text(v, 3, name, rotation=90, fontsize=9, color="#777777", ha="right")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("vocabulary size V")
    ax.set_ylabel("% of parameters in wte + lm_head (2·V·d)")
    ax.set_title("…but the embedding tables grow like V·d")
    ax.legend(fontsize=9.5, loc="upper left")
    ax.set_ylim(0, 100)

    ax = axes[2]
    ax.plot(V, [r["per_entry"] for r in bpe], "o-", color="#8c564b", lw=2)
    for r in bpe:
        ax.annotate(f"{r['per_entry']:.0f}", (r["V"], r["per_entry"]), textcoords="offset points",
                    xytext=(5, 4), fontsize=9)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("vocabulary size V")
    ax.set_ylabel("training tokens per vocabulary entry")
    ax.set_title("…and each token gets less training data")
    fig.suptitle("Vocabulary size trade-offs, measured on our 178 KB corpus "
                 "(why_tokenization.py)", y=1.02)
    fig.tight_layout()
    save(fig, "compression.png")


def fig_zipf(tok, z):
    counts = z["counts"][:256 + len(tok.merges)]
    s = z["sorted"]
    ranks = np.arange(1, len(s) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    ax = axes[0]
    ax.loglog(ranks, s, ".", ms=3, color=C_TR, label="token count (V = 4096, training text)")
    c = s[9] * 10.0  # a slope -1 reference line through rank 10
    ax.loglog(ranks, c / ranks, "--", color="#999999", lw=1.5, label="Zipf: count ∝ 1/rank")
    ax.set_xlabel("rank")
    ax.set_ylabel("occurrences in the training text")
    ax.set_title(f"Zipf's law: fitted slope {z['slope']:.2f} (ranks 10-1000)")
    top = "\n".join(f"{k + 1}. {tok.token_str(i)!r:8s} {c:5d}" for k, (i, c) in
                    enumerate(z["top10"][:6]))
    ax.text(0.03, 0.03, "most frequent:\n" + top, transform=ax.transAxes, fontsize=9.5,
            family=MONO, va="bottom", bbox=dict(fc="white", ec="#bbbbbb", lw=0.8))
    ax.legend(fontsize=10)
    ax = axes[1]
    learned = counts[256:]
    bins = [0, 1, 2, 3, 6, 11, 21, 51, 101, 10_000]
    labels = ["0", "1", "2", "3-5", "6-10", "11-20", "21-50", "51-100", ">100"]
    h = np.histogram(learned, bins=bins)[0]
    bars = ax.bar(range(len(h)), h, color=["#d62728"] + ["#1f77b4"] * (len(h) - 1))
    for b, v in zip(bars, h):
        ax.text(b.get_x() + b.get_width() / 2, v + 10, str(v), ha="center", fontsize=9.5)
    ax.set_xticks(range(len(h)))
    ax.set_xticklabels(labels)
    ax.set_xlabel("occurrences of a learned token in the encoded training text")
    ax.set_ylabel("number of learned tokens (of 3,840)")
    ax.set_ylim(0, h.max() * 1.45)
    ex = ", ".join(repr(tok.token_str(i)) for i in z["zero"][:4])
    ax.set_title(f"{len(z['zero'])} learned tokens never appear as themselves")
    ax.text(0.97, 0.95, f"zero-count examples: {ex}\n(stepping stones to longer tokens: their\n"
            "input embeddings never get a gradient → glitch-token risk)", transform=ax.transAxes,
            ha="right", va="top", fontsize=9.5, family=MONO,
            bbox=dict(fc="#fbe3e3", ec="#d62728", lw=0.8))
    fig.tight_layout()
    save(fig, "zipf.png")


def fig_bpb():
    path = os.path.join(HERE, "bpe_pytorch.npz")
    if not os.path.exists(path):
        print("no tokenizer/bpe_pytorch.npz yet: running bpe_pytorch.py (~2 min)...")
        subprocess.run([sys.executable, os.path.join(HERE, "bpe_pytorch.py")], check=True,
                       stdout=subprocess.DEVNULL)
    d = np.load(path)
    labels = [str(x) for x in d["labels"]]
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    for lab in labels:
        h = d[f"hist_{lab}"]  # step, bytes seen, seconds, train bpb, val bpb, val nats/token
        col = RUN_COLORS.get(lab, None)
        axes[0].plot(h[:, 0], h[:, 5], "o-", ms=3, color=col, lw=2, label=lab)
        axes[1].plot(h[:, 0], h[:, 4], "o-", ms=3, color=col, lw=2, label=f"{lab} val")
        axes[1].plot(h[:, 0], h[:, 3], "--", color=col, lw=1.3, label=f"{lab} train")
        axes[2].plot(h[:, 1] / 1e6, h[:, 4], "o-", ms=3, color=col, lw=2, label=lab)
        best = int(np.argmin(h[:, 4]))
        axes[1].annotate(f"{h[-1, 4]:.2f}", (h[-1, 0], h[-1, 4]), textcoords="offset points",
                         xytext=(4, 0), fontsize=9, color=col)
        if best != len(h) - 1:
            axes[1].annotate(f"best {h[best, 4]:.2f}", (h[best, 0], h[best, 4]),
                             textcoords="offset points", xytext=(-10, -16), fontsize=9, color=col)
    axes[0].set_title("Per-token val loss: NOT comparable")
    axes[0].set_ylabel("val loss (nats per token)")
    axes[1].set_title("Bits per byte: comparable (lower = better)")
    axes[1].set_ylabel("bits per byte")
    axes[1].set_ylim(0, 5.5)
    axes[2].set_title("Val bits per byte vs text seen")
    axes[2].set_ylabel("val bits per byte")
    axes[2].set_xlabel("MB of training text seen")
    axes[2].set_ylim(2.2, 5.5)
    for ax in axes[:2]:
        ax.set_xlabel("training step (same model, batch, context in tokens)")
    for ax in axes:
        ax.legend(fontsize=9)
    fig.suptitle("The same mini-GPT on characters vs BPE tokens (bpe_pytorch.py)", y=1.02)
    fig.tight_layout()
    save(fig, "bpb_curves.png")


def fig_failures(tok, train):
    fig = plt.figure(figsize=(17, 9.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.1], hspace=0.35, wspace=0.25)

    ax = fig.add_subplot(gs[0, 0])
    ax.set_xlim(0, 10)
    ax.set_ylim(-4.6, 0.9)
    ax.axis("off")
    ax.set_title("a) The model sees ids, not letters (or case, or spaces)", loc="left")
    items = [" strawberry", " Mississippi", "the", " the", " The", " THE",
             "The answer is", "The answer is "]
    for r, w in enumerate(items):
        ids = tok.encode_ordinary(w)
        y = -0.58 * r
        x = token_row(ax, 2.6, y, [tok.token_str(i) for i in ids], h=0.46, unit=0.17, fs=10)
        ax.text(2.45, y, repr(w), ha="right", va="center", fontsize=10, family=MONO)
        if len(ids) <= 4:
            ax.text(x + 0.1, y, str(ids), va="center", fontsize=9, color="#555555", family=MONO)

    ax = fig.add_subplot(gs[0, 1])
    tok2 = Tokenizer(train_bpe(train, 4096, GPT2_PATTERN), GPT2_PATTERN)
    splits = digit_splits({"GPT-2 regex": tok2, "GPT-4 regex": tok})
    shapes = sorted(set(splits["GPT-2 regex"][0]) | set(splits["GPT-4 regex"][0]),
                    key=lambda s: -(splits["GPT-2 regex"][0][s] + splits["GPT-4 regex"][0][s]))
    x = np.arange(len(shapes))
    for k, (name, col) in enumerate([("GPT-2 regex", "#ff7f0e"), ("GPT-4 regex", "#1f77b4")]):
        vals = [splits[name][0].get(s, 0) for s in shapes]
        ax.bar(x + (k - 0.5) * 0.4, vals, 0.4, color=col,
               label=f"{name}: {len(splits[name][0])} different shapes")
    ax.set_xticks(x)
    ax.set_xticklabels(shapes, rotation=30, fontsize=10)
    ax.set_xlabel("how a 4-digit number is split (digits per token)")
    ax.set_ylabel("numbers (of 9,000)")
    ax.set_title("b) 1000-9999 with our V=4096 tokenizers: inconsistent digit chunks", loc="left")
    ax.legend(fontsize=10)

    ax = fig.add_subplot(gs[1, :])
    rows = language_costs(tok)
    names = [r["lang"] for r in rows]
    toks = [r["tokens"] for r in rows]
    cols = ["#2ca02c" if r["lang"] == "English" else "#9467bd" if "*" in r["lang"] else "#1f77b4"
            for r in rows]
    bars = ax.bar(names, toks, color=cols)
    for b, r in zip(bars, rows):
        ax.text(b.get_x() + b.get_width() / 2, r["tokens"] + 2,
                f"{r['tokens']}\n({r['ratio']:.1f}×)", ha="center", fontsize=9.5)
    ax.set_ylabel("tokens")
    ax.set_ylim(0, max(toks) * 1.25)
    ax.set_title("c) \"The weather is nice today, so I am going for a walk in the park.\" in other "
                 "languages: tokens with our English-trained V=4096 tokenizer (* = not a translation)",
                 loc="left")
    fig.suptitle("Tokenization failure modes, measured (why_tokenization.py)", y=0.98)
    save(fig, "failure_modes.png")


def main():
    os.makedirs(OUT, exist_ok=True)
    train, val = load_corpus("train"), load_corpus("val")
    tok = get_tokenizer(4096)
    tok_full = Tokenizer(train_bpe(train, 100_000))
    fig_utf8()
    fig_merges()
    fig_boundaries(tok)
    fig_compression(compression_table(tok_full, train, val))
    fig_zipf(tok, zipf_stats(tok, train))
    fig_failures(tok, train)
    fig_bpb()


if __name__ == "__main__":
    main()
