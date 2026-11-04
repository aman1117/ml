"""
Draw the pictures used in rag/README.md and save them to rag/images/.

Every number in every figure is computed here, from the same code the scripts run:
  - the corpus, the chunkers and the eval set (corpus.py, chunking.py, qa.jsonl),
  - the trained bi-encoder (rag/embedder.npz; trained first if missing, ~30 s),
  - every retriever on every chunking (retrieval_eval.py),
  - the end-to-end pipeline metrics and the abstention sweep (rag_pipeline.py),
  - the vector-index benchmark (rag/index_results.npz; `python rag/index.py` is run
    first if it is missing, ~20 s).

Run (~10 s once embedder.npz and index_results.npz exist, ~60 s from scratch):
    python rag/make_figures.py
"""
import os
import sys
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

from bm25 import BM25, tokenize  # noqa: E402
from chunking import CHUNKERS, chunk_stats, get_chunks  # noqa: E402
from corpus import QTYPES, load_corpus, load_qa  # noqa: E402
from embed_numpy import Encoder, corpus_units, get_embedder, make_pairs, split_groups  # noqa: E402
from rag_pipeline import ABSTAIN_THRESHOLD, RAG, evaluate, summarize, threshold_sweep  # noqa: E402
from retrieval_eval import (METHODS, Retriever, bootstrap_ci, load_eval, metric_values,  # noqa: E402
                            run_all)

warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "images")
DPI = 130

DOC_COLORS = {"mlp": "#1f77b4", "cnn": "#ff7f0e", "rnn": "#2ca02c", "lstm": "#d62728",
              "attention": "#9467bd", "transformer": "#8c564b"}
METHOD_COLORS = {"tfidf": "#bbbbbb", "bm25": "#1f77b4", "dense": "#ff7f0e", "hybrid": "#2ca02c",
                 "hybrid+rerank": "#9467bd", "hybrid+mmr": "#8c564b"}
F_IDX, F_QRY, F_GEN, F_EVAL = "#dbe9f6", "#ffe8cc", "#e3f4e1", "#ece4f5"

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
                                boxstyle="round,pad=0.02,rounding_size=0.15",
                                fc=fc, ec=ec, lw=lw, zorder=2))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, zorder=3, **kw)


def arrow(ax, a, b, color="black", lw=1.8, style="-|>", ms=16, **kw):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=ms,
                                 color=color, lw=lw, zorder=1, **kw))


# ----------------------------------------------------------------------------
def fig_pipeline(ret, docs, summ):
    fig, ax = plt.subplots(figsize=(16, 7.4))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 7.4)
    ax.axis("off")
    ax.add_patch(Rectangle((0.1, 4.25), 15.8, 3.0, fc="#f7fbff", ec="#9ecae1", lw=1, zorder=0))
    ax.add_patch(Rectangle((0.1, 0.1), 15.8, 4.0, fc="#fffaf2", ec="#fdd0a2", lw=1, zorder=0))
    ax.text(0.25, 7.05, "INDEX TIME (offline, re-run when the docs change)", fontsize=12,
            color="#3182bd", weight="bold", va="center")
    ax.text(0.25, 3.85, "QUERY TIME (per question)", fontsize=12, color="#e6550d", weight="bold",
            va="center")
    nchar = sum(len(d.text) for d in docs)
    box(ax, 1.5, 5.6, f"6 READMEs\n{nchar:,} chars", F_IDX, w=2.2, h=1.1)
    box(ax, 4.7, 5.6, f"chunk by heading\n+ contextual header\n{len(ret.chunks)} chunks", F_IDX, w=2.8, h=1.3)
    box(ax, 8.6, 6.4, f"inverted index (BM25)\n{len(ret.bm25.index.vocab):,} terms, "
        f"{ret.bm25.index.n_postings():,} postings", F_IDX, w=3.6, h=1.0)
    box(ax, 8.6, 4.95, f"bi-encoder ({ret.enc.n_params() / 1e6:.1f}M params)\n"
        f"-> {ret.emb.shape[0]} x {ret.emb.shape[1]} unit vectors", F_IDX, w=3.6, h=1.0)
    arrow(ax, (2.6, 5.6), (3.3, 5.6))
    arrow(ax, (6.1, 5.9), (6.8, 6.3))
    arrow(ax, (6.1, 5.3), (6.8, 5.0))
    box(ax, 12.9, 5.65, "vector index\nexact here; IVF / PQ / HNSW\nwhen n is large", F_IDX, w=3.2, h=1.2)
    arrow(ax, (10.4, 4.95), (11.3, 5.4))

    box(ax, 1.3, 2.3, "question", F_QRY, w=1.8, h=0.9, weight="bold")
    box(ax, 4.0, 3.15, "BM25 top 100\n(inverted index)", F_QRY, w=2.6, h=0.95)
    box(ax, 4.0, 1.45, "dense top 100\n(vector index, cosine)", F_QRY, w=2.6, h=0.95)
    arrow(ax, (2.2, 2.5), (2.8, 3.05))
    arrow(ax, (2.2, 2.1), (2.8, 1.55))
    box(ax, 6.75, 2.3, "RRF fusion\nΣ 1/(60+rank)", F_QRY, w=2.2, h=1.0)
    arrow(ax, (5.3, 3.05), (5.7, 2.6))
    arrow(ax, (5.3, 1.55), (5.7, 2.0))
    box(ax, 9.25, 2.3, "rerank top 20\nlogistic reg. on\npair features", F_QRY, w=2.3, h=1.3)
    arrow(ax, (7.85, 2.3), (8.1, 2.3))
    box(ax, 11.6, 2.3, "prompt:\nsources [1]..[4]\n+ question", F_GEN, w=2.1, h=1.3)
    arrow(ax, (10.4, 2.3), (10.55, 2.3))
    box(ax, 14.25, 2.95, "generate + cite\n(extractive or LLM)", F_GEN, w=2.8, h=1.0)
    box(ax, 14.25, 1.45, f"or ABSTAIN if\nmax P(relevant) < {ABSTAIN_THRESHOLD}", F_GEN, w=2.8, h=1.0)
    arrow(ax, (12.65, 2.55), (12.85, 2.85))
    arrow(ax, (12.65, 2.05), (12.85, 1.6))
    ax.text(8.0, 0.45,
            f"evaluate: retrieval recall@k / MRR / nDCG (retrieval_eval.py)   |   answers: evidence in "
            f"answer {summ['evidence_hit']:.2f}, citation precision {summ['cite_precision']:.2f}, "
            f"faithfulness {summ['faithfulness']:.2f}, abstention accuracy {summ['abstention_accuracy']:.0%}",
            ha="center", fontsize=10.5, color="#555", bbox=dict(fc=F_EVAL, ec="#9e9ac8", boxstyle="round"))
    ax.set_title("The RAG pipeline in this folder (all numbers from the real system)", pad=4)
    save(fig, "pipeline.png")


# ----------------------------------------------------------------------------
def fig_chunking(docs, qs):
    d = next(x for x in docs if x.name == "lstm/README.md")
    secs = [s for s in d.sections if s.level >= 3 and s.path[1].startswith("3.")]
    lo, hi = secs[0].start - 600, secs[3].start + 300
    names = list(CHUNKERS)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(16, 5.6), gridspec_kw=dict(width_ratios=[2.6, 1]))
    for row, name in enumerate(names):
        y = len(names) - 1 - row
        cs = [c for c in get_chunks(name, docs) if c.doc == d.name and c.end > lo and c.start < hi]
        for j, c in enumerate(cs):
            off = 0.18 if j % 2 else -0.18
            ax.add_patch(Rectangle((max(c.start, lo), y + off - 0.15), min(c.end, hi) - max(c.start, lo),
                                   0.3, fc=["#6baed6", "#fd8d3c"][j % 2], ec="black", lw=0.8, alpha=0.85))
    k = 0
    for s in d.sections:
        if lo < s.start < hi:
            ax.axvline(s.start, color="#555", ls="--", lw=1)
            ax.text(s.start + 25, len(names) - 0.45 + 0.32 * (k % 2), s.title[:34], fontsize=8.5,
                    color="#333", va="bottom", bbox=dict(fc="white", ec="none", pad=0.5), clip_on=True)
            k += 1
    for q in qs:
        for slot in q.slots:
            for e in slot:
                if e.file == d.name and lo < e.span[0] < hi:
                    ax.add_patch(Rectangle((e.span[0], -0.75), e.span[1] - e.span[0], 0.25,
                                           fc="#d62728", ec="none"))
                    ax.text(e.span[0], -0.95, q.id, fontsize=8.5, color="#d62728", va="top")
    ax.text(lo, -1.35, "red = gold evidence spans of qa.jsonl questions", fontsize=9.5, color="#d62728")
    ax.set_xlim(lo, hi)
    ax.set_ylim(-1.5, len(names) + 0.3)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names[::-1])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("character offset in lstm/README.md (dashed lines = section headings)")
    ax.set_title(f"The same {hi - lo:,} characters cut by each strategy\n"
                 "(alternating colours = consecutive chunks; overlapping ones are offset)")

    st = {n: chunk_stats(get_chunks(n, docs), docs, qs) for n in names}
    x = np.arange(len(names))
    b1 = ax2.bar(x - 0.2, [st[n]["cross"] / st[n]["n"] * 100 for n in names], 0.4,
                 label="span two sections", color="#9467bd")
    b2 = ax2.bar(x + 0.2, [st[n]["mid"] / st[n]["n"] * 100 for n in names], 0.4,
                 label="end mid-sentence", color="#8c564b")
    for bars in (b1, b2):
        for r in bars:
            ax2.text(r.get_x() + r.get_width() / 2, r.get_height() + 1, f"{r.get_height():.0f}",
                     ha="center", fontsize=9)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{n} ({st[n]['n']})" for n in names], fontsize=9.5, rotation=30, ha="right")
    ax2.set_ylabel("% of chunks")
    ax2.set_ylim(0, 105)
    ax2.legend(fontsize=9, loc="upper right")
    ax2.set_title("Bad cuts, whole corpus (number of chunks)")
    save(fig, "chunking.png")


# ----------------------------------------------------------------------------
def fig_bm25(ret):
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    tf = np.linspace(0, 12, 200)
    ax = axes[0]
    for k1, c in [(0.5, "#9ecae1"), (1.2, "#1f77b4"), (2.0, "#08519c"), (5.0, "#08306b")]:
        ax.plot(tf, tf * (k1 + 1) / (tf + k1), color=c, lw=2.2, label=f"BM25 k1 = {k1}")
    ax.plot(tf[tf >= 1], 1 + np.log(tf[tf >= 1]), color="#ff7f0e", lw=2, ls="--", label="TF-IDF: 1 + ln tf")
    ax.plot(tf, tf, color="#bbbbbb", lw=1.5, ls=":", label="raw tf")
    ax.set_ylim(0, 7)
    ax.set_xlabel("term frequency in the chunk (tf)")
    ax.set_ylabel("contribution / idf")
    ax.set_title("Saturation: the 10th hit adds little\n(average-length chunk, b irrelevant)")
    ax.legend(fontsize=9)

    ax = axes[1]
    L = np.linspace(0.1, 4, 200)
    for b, c in [(0.0, "#bbbbbb"), (0.25, "#9ecae1"), (0.75, "#1f77b4"), (1.0, "#08306b")]:
        ax.plot(L, 3 * 2.2 / (3 + 1.2 * (1 - b + b * L)), color=c, lw=2.2, label=f"b = {b}")
    lens = ret.bm25.index.doc_len / ret.bm25.index.avgdl
    ax2 = ax.twinx()
    ax2.hist(lens, bins=30, range=(0, 4), color="#fdd0a2", alpha=0.6, zorder=0)
    ax2.set_yticks([])
    ax2.spines["right"].set_visible(False)
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)
    ax.set_xlabel("chunk length / average length (|d| / avgdl)")
    ax.set_ylabel("contribution / idf  (tf = 3, k1 = 1.2)")
    ax.set_title("Length normalisation: long chunks are\ndiscounted (orange: our chunk lengths)")
    ax.legend(fontsize=9)

    ax = axes[2]
    ix = ret.bm25.index
    N = ix.N
    df = np.arange(1, N + 1)
    ax.plot(df, np.log(1 + (N - df + 0.5) / (df + 0.5)), color="#1f77b4", lw=2.2, label="BM25 idf")
    ax.plot(df, np.log((1 + N) / (1 + df)) + 1, color="#ff7f0e", lw=2, ls="--", label="TF-IDF idf (smoothed)")
    for t in ["attention", "lstm", "gradient", "im2col", "alibi", "softmax", "mask", "chinchilla"]:
        st = tokenize(t)[0]
        if st in ix.vocab:
            k = ix.vocab[st]
            ax.plot(ix.df[k], ret.bm25.idf[k], "o", color="black", ms=5)
            ax.annotate(t, (ix.df[k], ret.bm25.idf[k]), xytext=(4, 3), textcoords="offset points", fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel(f"document frequency df (out of N = {N} chunks)")
    ax.set_ylabel("idf")
    ax.set_title("Rare terms count more\n(real terms from our index)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    save(fig, "bm25_curves.png")


# ----------------------------------------------------------------------------
def pca2(X):
    Xc = X - X.mean(0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:2].T, S ** 2 / (S ** 2).sum()


def fig_embeddings(ret, enc):
    init = Encoder(enc.config["n_buckets"], enc.config["dim"], 0, idf=enc.idf, tau=enc.tau,
                   trigram_weight=enc.trigram_weight)
    texts = [c.text for c in ret.chunks]
    docs = np.array([c.doc.split("/")[0] for c in ret.chunks])
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2))
    for ax, (name, e) in zip(axes, [("random init (hashed bag of n-grams)", init), ("trained bi-encoder", enc)]):
        X = e.encode(texts)
        P, var = pca2(X)
        S = X @ X.T
        same = docs[:, None] == docs[None, :]
        off = ~np.eye(len(X), dtype=bool)
        for dname, col in DOC_COLORS.items():
            m = docs == dname
            ax.scatter(P[m, 0], P[m, 1], s=22, color=col, alpha=0.75, label=dname, edgecolor="none")
        ax.set_title(f"{name}\nmean cosine: same README {S[same & off].mean():.2f}, "
                     f"different README {S[~same].mean():.2f}", fontsize=12.5)
        print(f"  {name}: mean cosine same README {S[same & off].mean():.3f}, "
              f"different README {S[~same].mean():.3f}, PCA variance {var[0]:.1%} + {var[1]:.1%}")
        ax.set_xlabel(f"PC 1 ({var[0]:.0%} of variance)")
        ax.set_ylabel(f"PC 2 ({var[1]:.0%})")
    axes[1].legend(fontsize=10, markerscale=1.6, loc="best")
    fig.suptitle(f"{len(texts)} chunk embeddings projected to 2D with PCA, coloured by source README")
    fig.tight_layout()
    save(fig, "embedding_pca.png")


def fig_infonce(enc, hist):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.4), gridspec_kw=dict(width_ratios=[1.15, 1.25, 1]))
    ax = axes[0]
    loss = hist["loss"]
    ax.plot(hist["step"], loss, color="#c6dbef", lw=0.8, label="per batch")
    sm = np.convolve(loss, np.ones(25) / 25, mode="valid")
    ax.plot(hist["step"][24:], sm, color="#1f77b4", lw=2.2, label="moving average (25)")
    ax.axhline(np.log(128), color="#999", ls=":", lw=1.2)
    ax.text(hist["step"][-1], np.log(128) * 0.93, "chance: ln 128 = 4.85", ha="right", fontsize=9, color="#666")
    ax.set_yscale("log")
    ax.set_xlabel("step (batch of 128 pairs)")
    ax.set_ylabel("InfoNCE loss (log scale)")
    ax2 = ax.twinx()
    ax2.plot(hist["val_step"], hist["val_acc"], "o-", color="#d62728", lw=1.8, ms=4,
             label="held-out sections: top-1")
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("val top-1 accuracy", color="#d62728")
    ax2.spines["right"].set_visible(True)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=9, loc="upper right", bbox_to_anchor=(1, 0.9))
    ax.set_title(f"Training: loss {sm[0]:.2f} -> {sm[-1]:.2f}, val top-1 "
                 f"{hist['val_acc'][0]:.2f} -> {hist['val_acc'][-1]:.2f}", fontsize=12.5)

    _, val_units = split_groups(corpus_units(), 0.1, 0)  # sections never trained on
    rng = np.random.default_rng(3)
    pairs = [p for p in make_pairs(val_units, rng, ("heading",))]
    seen, chosen = set(), []
    for p in pairs:
        if p[2] not in seen and p[0] not in seen:  # one paragraph per section, distinct titles
            seen |= {p[2], p[0]}
            chosen.append(p)
    chosen = [chosen[i] for i in np.linspace(0, len(chosen) - 1, 8).astype(int)]
    for ax, (name, e) in zip(axes[1:], [("random init", Encoder(enc.config["n_buckets"], enc.config["dim"], 0,
                                                                  idf=enc.idf, trigram_weight=enc.trigram_weight)),
                                         ("trained", enc)]):
        Q = e.encode([p[0] for p in chosen])
        D = e.encode([p[1] for p in chosen])
        S = Q @ D.T
        im = ax.imshow(S, cmap="viridis", vmin=-0.1, vmax=1, aspect="auto")
        for i in range(len(S)):
            for j in range(len(S)):
                ax.text(j, i, f"{S[i, j]:.2f}", ha="center", va="center", fontsize=7.5,
                        color="white" if S[i, j] < 0.6 else "black")
        ax.set_xticks(range(len(S)))
        ax.set_xticklabels([f"para {i}" for i in range(len(S))], rotation=60, fontsize=8)
        if name == "random init":
            ax.set_yticks(range(len(S)))
            ax.set_yticklabels([p[0][:32] for p in chosen], fontsize=8.5)
        else:
            ax.set_yticks([])
        acc = np.mean(S.argmax(1) == np.arange(len(S)))
        print(f"  {name}: held-out heading -> paragraph, diagonal argmax {acc:.0%}, "
              f"mean diagonal cosine {np.diag(S).mean():.2f}, mean off-diagonal "
              f"{S[~np.eye(len(S), dtype=bool)].mean():.2f}")
        ax.set_title(f"{name}: cos(heading, paragraph), held-out\nsections; row argmax on the diagonal: "
                     f"{acc:.0%}", fontsize=11.5)
    fig.colorbar(im, ax=axes[2], fraction=0.046)
    fig.tight_layout()
    save(fig, "infonce.png")


# ----------------------------------------------------------------------------
def fig_retrievers(res, qs):
    fig, axes = plt.subplots(1, 2, figsize=(17, 5.4), gridspec_kw=dict(width_ratios=[2.2, 1]))
    ax = axes[0]
    groups = QTYPES[:3] + ["all"]
    w = 0.8 / len(METHODS)
    for i, m in enumerate(METHODS):
        vals, los, his = [], [], []
        for g in groups:
            v, lo, hi = bootstrap_ci(metric_values(res[m], qs, "recall@5", None if g == "all" else g))
            vals.append(v)
            los.append(v - lo)
            his.append(hi - v)
        x = np.arange(len(groups)) + (i - len(METHODS) / 2 + 0.5) * w
        ax.bar(x, vals, w, yerr=[los, his], capsize=2.5, color=METHOD_COLORS[m], label=m,
               error_kw=dict(lw=1, ecolor="#444"))
    n = {g: sum(q.type == g for q in qs) for g in QTYPES}
    n["all"] = sum(q.answerable for q in qs)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([f"{g} ({n[g]} q)" for g in groups])
    ax.set_ylabel("recall@5 (95% bootstrap CI)")
    ax.set_ylim(0, 1.1)
    ax.legend(ncol=3, fontsize=9.5, loc="upper right")
    ax.set_title("Which retriever finds the gold evidence in its top 5?")
    ax = axes[1]
    for m in METHODS:
        ks = [1, 3, 5, 10]
        ax.plot(ks, [np.mean(metric_values(res[m], qs, f"recall@{k}")) for k in ks], "o-",
                color=METHOD_COLORS[m], lw=2, label=m)
    ax.set_xscale("log")
    ax.set_xticks([1, 3, 5, 10])
    ax.set_xticklabels(["1", "3", "5", "10"])
    ax.set_xlabel("k (chunks retrieved)")
    ax.set_ylabel("recall@k (all 57 answerable)")
    ax.set_title("Recall grows with k")
    ax.legend(fontsize=9)
    fig.tight_layout()
    save(fig, "retrievers.png")


def fig_ablation(abl, names, methods):
    M = np.array([[abl[n][m] for m in methods] for n in names])
    fig, ax = plt.subplots(figsize=(10.5, 5))
    im = ax.imshow(M, cmap="Greens", vmin=M.min() - 0.1, vmax=M.max() + 0.02)
    for i in range(len(names)):
        for j in range(len(methods)):
            best = M[i, j] == M[:, j].max()
            ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center", fontsize=11,
                    weight="bold" if best else "normal")
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([m.replace("+", "\n+") for m in methods])
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel("retriever")
    ax.set_ylabel("chunking strategy")
    fig.colorbar(im, ax=ax, fraction=0.04, label="recall@5")
    ax.set_title("Chunking x retriever: recall@5 on 57 questions (bold = best per column)\n"
                 "95% CIs are about ±0.1 wide: most of these differences are noise", fontsize=12.5)
    save(fig, "ablation.png")


# ----------------------------------------------------------------------------
def fig_ann(r):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.4))
    ax = axes[0]
    for kind, col in [("clustered", "#1f77b4"), ("uniform", "#bbbbbb")]:
        a = r[f"{kind}_ivf"]
        ax.plot(a[:, 2], a[:, 1], "o-", color=col, lw=2, label=f"IVF on {kind} vectors")
        if kind == "clustered":
            for p, rec, t, _ in a:
                ax.annotate(f"nprobe {int(p)}", (t, rec), xytext=(5, -12), textcoords="offset points",
                            fontsize=8.5, color=col)
    ax.plot([r["clustered_exact_ms"]], [1.0], "s", color="black", ms=9, label="exact (brute force)")
    pq = r["pq"]
    ax.plot(pq[:, 2], pq[:, 1], "^", color="#ff7f0e", ms=9, label="PQ (codes only)")
    ax.plot(pq[:, 4], pq[:, 3], "v", color="#d62728", ms=9, label="PQ + exact re-score of top 100")
    for row in pq:
        ax.annotate(f"m={int(row[0])}", (row[2], row[1]), xytext=(6, -3), textcoords="offset points",
                    fontsize=8.5, color="#ff7f0e")
    ax.set_xscale("log")
    ax.set_xlabel("milliseconds per query (one query at a time, NumPy, this laptop)")
    ax.set_ylabel("recall@10 vs exact search")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9, loc="lower right")
    ax.set_title(f"n = {int(r['n']):,} vectors x D = {int(r['dim'])}: speed vs recall")

    ax = axes[1]
    D = int(r["dim"])
    mem = [4 * D] + [int(m) for m in pq[:, 0]]
    rec = [1.0] + list(pq[:, 1])
    labels = ["float32"] + [f"PQ m={int(m)}" for m in pq[:, 0]]
    bars = ax.bar(range(len(mem)), mem, color=["black"] + ["#ff7f0e"] * len(pq))
    for i, (b, rc) in enumerate(zip(bars, rec)):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() * 1.15,
                f"{mem[i]} B\nrecall {rc:.2f}" + ("" if i == 0 else f"\n(+re-score {pq[i - 1, 3]:.2f})"),
                ha="center", fontsize=9)
    ax.set_yscale("log")
    ax.set_ylim(4, 5000)
    ax.set_xticks(range(len(mem)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("bytes per vector (log scale)")
    gb = 100e6 * 4 * D / 1e9
    ax.set_title(f"Memory: 100M vectors = {gb:.0f} GB in float32, {100e6 * 16 / 1e9:.1f} GB as PQ m=16")
    fig.tight_layout()
    save(fig, "ann_tradeoff.png")


def fig_end_to_end(rows, qs, sweep, conf):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.4), gridspec_kw=dict(width_ratios=[1.5, 1]))
    ax = axes[0]
    metrics = [("evidence_hit", "evidence\nin answer"), ("gold_recall", "gold-answer\nrecall"),
               ("f1", "token F1"), ("cite_precision", "citation\nprecision"),
               ("cite_recall", "citation\nrecall"), ("faithfulness", "faithfulness")]
    cols = {"keyword": "#1f77b4", "semantic": "#ff7f0e", "multihop": "#2ca02c"}
    w = 0.26
    for i, t in enumerate(QTYPES[:3]):
        vals, errs = [], [[], []]
        for k, _ in metrics:
            v = [rows[q.id][k] for q in qs if q.type == t and not np.isnan(rows[q.id][k])]
            m, lo, hi = bootstrap_ci(v)
            vals.append(m)
            errs[0].append(m - lo)
            errs[1].append(hi - m)
        ax.bar(np.arange(len(metrics)) + (i - 1) * w, vals, w, yerr=errs, capsize=2, color=cols[t],
               label=f"{t} ({sum(q.type == t for q in qs)} q)", error_kw=dict(lw=1, ecolor="#444"))
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels([m[1] for m in metrics], fontsize=10.5)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("mean over questions (95% CI)")
    ax.legend(fontsize=9.5, loc="upper left")
    ax.set_title("Extractive generator, end to end (answerable questions)")
    ax.text(5, 1.04, "copying is faithful\nby construction", ha="center", fontsize=8.5, color="#555")

    ax = axes[1]
    ts = np.linspace(0, 0.6, 61)
    un = [np.mean([conf[q.id] < t for q in qs if not q.answerable]) for t in ts]
    an = [np.mean([conf[q.id] < t for q in qs if q.answerable]) for t in ts]
    ax.plot(ts, un, color="#d62728", lw=2.4, label="unanswerable (6 q): abstained  ↑ good")
    ax.plot(ts, an, color="#1f77b4", lw=2.4, label="answerable (57 q): abstained  ↓ good")
    ax.axvline(ABSTAIN_THRESHOLD, color="black", ls="--", lw=1.2)
    s = dict((t, (u, a)) for t, u, a in sweep)
    u0, a0 = s[ABSTAIN_THRESHOLD]
    ax.text(ABSTAIN_THRESHOLD - 0.01, 0.78, f"threshold {ABSTAIN_THRESHOLD}:\n{u0:.0%} of unanswerable,\n"
            f"{a0:.0%} of answerable", fontsize=9.5, ha="right")
    for q in qs:
        if not q.answerable:
            ax.plot(conf[q.id], -0.04, "x", color="#d62728", ms=7)
    ax.set_xlabel("threshold on max P(relevant) of the retrieved chunks")
    ax.set_ylabel("fraction abstained")
    ax.set_ylim(-0.08, 1.05)
    ax.legend(fontsize=9, loc="lower right")
    ax.set_title("Abstention trade-off (x = confidence of each\nunanswerable question)")
    fig.tight_layout()
    save(fig, "end_to_end.png")


def main():
    os.makedirs(OUT, exist_ok=True)
    docs = load_corpus()
    qs = load_qa(docs=docs)
    enc, hist = get_embedder()

    ret = Retriever(embedder=enc)
    qs, rel = load_eval()
    res = run_all(ret, qs, rel)
    rag = RAG(ret)
    rows = evaluate(rag, qs)
    summ = summarize(rows, qs)
    sweep, conf = threshold_sweep(rag, qs, [0.1, 0.15, ABSTAIN_THRESHOLD, 0.25, 0.3])

    fig_pipeline(ret, docs, summ)
    fig_chunking(docs, qs)
    fig_bm25(ret)
    fig_embeddings(ret, enc)
    fig_infonce(enc, hist)
    fig_retrievers(res, qs)

    names, methods = list(CHUNKERS), METHODS[:5]
    abl = {}
    for n in names:
        qs2, rel2 = load_eval(n)
        r2 = run_all(Retriever(n, embedder=enc), qs2, rel2, methods)
        abl[n] = {m: np.mean(metric_values(r2[m], qs2, "recall@5")) for m in methods}
    fig_ablation(abl, names, methods)

    path = os.path.join(HERE, "index_results.npz")
    if not os.path.exists(path):
        print("No rag/index_results.npz yet: running python rag/index.py first (~20 s)")
        import index
        sys.argv = [sys.argv[0]]
        index.main()
    fig_ann(np.load(path))
    fig_end_to_end(rows, qs, sweep, conf)


if __name__ == "__main__":
    main()
