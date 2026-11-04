"""
A dense bi-encoder trained FROM SCRATCH in NumPy, with a hand-written backward pass.

BM25 only matches words that are literally shared. A dense retriever maps the query
and every chunk to a vector, so that "texts about the same thing" end up close even
when they use different words. The famous ones (DPR, E5, OpenAI embeddings) are
Transformers pretrained on billions of tokens. Ours must train in a minute on 180 KB of
text, so every part is the smallest thing that still teaches the real mechanism:

  text -> features   lowercase words (stopwords dropped) + their character trigrams
                     ("<gate>" -> "<ga", "gat", "ate", "te>"), each HASHED into one of
                     B = 2^15 buckets (the hashing trick: no vocabulary, no OOV; an
                     unseen word like "gating" still shares trigrams with "gate")
  features -> x      x = Σ_j w_j E[id_j]   a weighted mean of embedding rows (an
                     "embedding bag"). w_j = idf of the bucket (rare features count
                     more), trigrams down-weighted by 0.5, normalised to sum to 1
  x -> z             z = x + tanh(x W1 + b1) W2         a residual MLP
  z -> e             e = z / ‖z‖                        unit length: dot = cosine

  E: (B, D)  W1: (D, D)  b1: (D,)  W2: (D, D)    D = 128, 4.2M parameters (99.6% in E)

The SAME encoder embeds queries and chunks (a "siamese" / shared bi-encoder).

Training: InfoNCE with in-batch negatives (Oord et al. 2018; DPR, Karpukhin et al.
2020). A batch has N (query, positive passage) pairs. Score every query against every
passage, S = Q Pᵀ / τ  (N, N), and do softmax cross-entropy with the diagonal as the
target: each query must pick its own passage out of the N in the batch. The other
N - 1 passages are free negatives. τ = 0.05 sharpens the softmax (cosines live in
[-1, 1]; /τ stretches them to [-20, 20]).

Where do (query, passage) pairs come from with NO labelled questions? We mine them
from the corpus itself (self-supervision), three kinds, regenerated every epoch:
  heading   section title -> a paragraph of that section ("Why divide by √d_k?" -> the
            paragraph explaining it). Closest to a real question -> answer pair.
  ict       one sentence -> the rest of its paragraph (the Inverse Cloze Task, Lee et
            al. 2019): learn which statements belong together, with the sentence REMOVED
            so the model can't just match identical words.
  drop      30-50% of a paragraph's content words, shuffled -> the paragraph. A
            keyword-style query; teaches robustness to missing words and order.
Pairs from the same section are "false negatives" for each other (two paragraphs under
one heading both answer the heading query), so their off-diagonal scores are masked.
With only ~900 pairs per epoch and 4.2M parameters the model memorises fast, so every
training text also goes through FEATURE DROPOUT: half of its hashed features are
dropped at random, a fresh subset each time (a cheap augmentation, like word dropout).
10% of the sections are held out to measure generalisation to unseen text. The eval
questions in qa.jsonl are NEVER used for training.

gradient_check.py checks every backward() here against finite differences.

Run (~45 s; saves rag/embedder.npz):
    python rag/embed_numpy.py
    python rag/embed_numpy.py --feat-drop 0 --tau 0.1
    python rag/embed_numpy.py --seeds 0 1 2 3     (seed-to-seed spread, ~2 min)
"""
from __future__ import annotations

import argparse
import os
import re
import time
import warnings
import zlib
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from bm25 import STOPWORDS
from chunking import blocks, own_regions, sentence_spans
from corpus import load_corpus

warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PATH = os.path.join(HERE, "embedder.npz")
N_BUCKETS = 2 ** 15
WORD_RE = re.compile(r"[a-z0-9]+")


# ----------------------------------------------------------------------------
# Text -> hashed bag of features
# ----------------------------------------------------------------------------
def bucket(s: str, n_buckets: int) -> int:
    """A stable hash (Python's hash() is randomised per process, so we can't use it)."""
    return zlib.crc32(s.encode("utf-8")) % n_buckets


_TOKEN_FEATS: Dict[Tuple[str, int], List[Tuple[int, bool]]] = {}


def token_features(tok: str, n_buckets: int) -> List[Tuple[int, bool]]:
    key = (tok, n_buckets)
    if key not in _TOKEN_FEATS:
        w = f"<{tok}>"
        feats = [(bucket("w:" + tok, n_buckets), False)]
        feats += [(bucket("t:" + w[i:i + 3], n_buckets), True) for i in range(len(w) - 2)]
        _TOKEN_FEATS[key] = feats
    return _TOKEN_FEATS[key]


def text_tokens(text: str) -> List[str]:
    return [t for t in WORD_RE.findall(text.lower()) if t not in STOPWORDS] or ["<empty>"]


def raw_features(text: str, n_buckets: int):
    """(bucket ids, is_trigram flags) with repeats: one entry per occurrence."""
    ids, tri = [], []
    for t in text_tokens(text):
        for b, is_tri in token_features(t, n_buckets):
            ids.append(b)
            tri.append(is_tri)
    return np.array(ids, dtype=np.int64), np.array(tri, dtype=bool)


def feature_idf(texts: Sequence[str], n_buckets: int) -> np.ndarray:
    """idf per bucket over a set of texts (paragraphs of the corpus). Buckets never seen
    get the maximum idf: an unknown feature is as informative as the rarest one."""
    df = np.zeros(n_buckets)
    for t in texts:
        df[np.unique(raw_features(t, n_buckets)[0])] += 1
    return (np.log((1 + len(texts)) / (1 + df)) + 1).astype(np.float32)


class Bags:
    """A batch of texts as a ragged list of (feature id, weight), CSR style:
    entries of text i are ids[offsets[i]:offsets[i+1]], with weights summing to 1."""

    def __init__(self, ids, w, offsets):
        self.ids, self.w, self.offsets = ids, w, offsets
        self.n = len(offsets) - 1
        self.seg = np.repeat(np.arange(self.n), np.diff(offsets))  # which text each entry is in

    @staticmethod
    def from_texts(texts: Sequence[str], idf: np.ndarray, trigram_weight: float,
                   drop: float = 0.0, rng=None) -> "Bags":
        all_ids, all_w, offsets = [], [], [0]
        for t in texts:
            ids, tri = raw_features(t, len(idf))
            if drop > 0:  # feature dropout (training only): a fresh random subset each time
                keep = rng.random(len(ids)) >= drop
                keep[rng.integers(len(ids))] = True
                ids, tri = ids[keep], tri[keep]
            w = idf[ids] * np.where(tri, trigram_weight, 1.0)
            u, inv = np.unique(ids, return_inverse=True)  # merge repeated features
            wu = np.bincount(inv, weights=w)
            all_ids.append(u)
            all_w.append(wu / wu.sum())
            offsets.append(offsets[-1] + len(u))
        return Bags(np.concatenate(all_ids), np.concatenate(all_w), np.array(offsets))

    @staticmethod
    def concat(bags: Sequence["Bags"]) -> "Bags":
        offs, base = [np.zeros(1, dtype=np.int64)], 0
        for b in bags:
            offs.append(b.offsets[1:] + base)
            base += b.offsets[-1]
        return Bags(np.concatenate([b.ids for b in bags]), np.concatenate([b.w for b in bags]),
                    np.concatenate(offs))

    def select(self, idx: Sequence[int]) -> "Bags":
        parts = [np.arange(self.offsets[i], self.offsets[i + 1]) for i in idx]
        sel = np.concatenate(parts)
        offs = np.concatenate([[0], np.cumsum([len(p) for p in parts])])
        return Bags(self.ids[sel], self.w[sel], offs)


# ----------------------------------------------------------------------------
# The encoder: embedding bag -> residual MLP -> L2 normalise
# ----------------------------------------------------------------------------
def l2_normalize(z):
    n = np.linalg.norm(z, axis=-1, keepdims=True)
    return z / n, n


def l2_normalize_backward(de, e, n):
    """e = z / ‖z‖. The Jacobian (I − e eᵀ) / ‖z‖ removes the part of the gradient along
    e: making z longer doesn't change e, only rotating it does."""
    return (de - e * np.sum(de * e, axis=-1, keepdims=True)) / n


class Encoder:
    def __init__(self, n_buckets=N_BUCKETS, dim=128, seed=0, dtype=np.float32, idf=None,
                 tau=0.05, trigram_weight=0.5):
        rng = np.random.default_rng(seed)
        D = dim
        self.config = dict(n_buckets=n_buckets, dim=dim, tau=tau, trigram_weight=trigram_weight)
        self.params = {
            "E": (rng.standard_normal((n_buckets, D)) / np.sqrt(D)).astype(dtype),
            "W1": (rng.standard_normal((D, D)) / np.sqrt(D)).astype(dtype),
            "b1": np.zeros(D, dtype=dtype),
            # W2 starts small, so at init z ≈ x: a random projection of the bag of
            # features, i.e. a fuzzy lexical matcher. Training bends it from there.
            "W2": (rng.standard_normal((D, D)) * 0.1 / np.sqrt(D)).astype(dtype),
        }
        self.idf = idf if idf is not None else np.ones(n_buckets, dtype=np.float32)
        self.tau, self.trigram_weight = tau, trigram_weight

    def bags(self, texts: Sequence[str], drop: float = 0.0, rng=None) -> Bags:
        return Bags.from_texts(texts, self.idf, self.trigram_weight, drop, rng)

    def forward(self, bags: Bags):
        p = self.params
        contrib = bags.w[:, None].astype(p["E"].dtype) * p["E"][bags.ids]
        x = np.add.reduceat(contrib, bags.offsets[:-1], axis=0)  # (n, D) weighted means
        h = np.tanh(x @ p["W1"] + p["b1"])
        z = x + h @ p["W2"]
        e, n = l2_normalize(z)
        self.cache = (bags, x, h, e, n)
        return e

    def backward(self, de, dense_E=False):
        """de = dL/de (n, D) -> grads. dE is returned SPARSE as (rows, values), because a
        batch touches a few thousand of the 32,768 rows (dense_E=True for the checks)."""
        bags, x, h, e, n = self.cache
        p = self.params
        dz = l2_normalize_backward(de, e, n)
        g = {"W2": h.T @ dz}
        du = (dz @ p["W2"].T) * (1 - h ** 2)  # through tanh
        g["W1"] = x.T @ du
        g["b1"] = du.sum(0)
        dx = dz + du @ p["W1"].T  # residual: the gradient also skips the MLP
        # embedding bag: x_i = Σ_j w_ij E[id_ij]  ->  dE[id_ij] += w_ij dx_i
        contrib = bags.w[:, None].astype(dx.dtype) * dx[bags.seg]
        rows, inv = np.unique(bags.ids, return_inverse=True)
        order = np.argsort(inv, kind="stable")
        starts = np.searchsorted(inv[order], np.arange(len(rows)))
        vals = np.add.reduceat(contrib[order], starts, axis=0)
        if dense_E:
            dE = np.zeros_like(p["E"])
            dE[rows] = vals
            g["E"] = dE
        else:
            g["E"] = (rows, vals)
        return g

    def encode(self, texts: Sequence[str], batch: int = 512) -> np.ndarray:
        out = [self.forward(self.bags(texts[i:i + batch])) for i in range(0, len(texts), batch)]
        return np.concatenate(out) if out else np.zeros((0, self.config["dim"]), np.float32)

    def n_params(self):
        return sum(v.size for v in self.params.values())


# ----------------------------------------------------------------------------
# InfoNCE with in-batch negatives
# ----------------------------------------------------------------------------
def info_nce(q, d, tau, mask=None):
    """q: (N, D) queries, d: (M, D) passages with M >= N; the positive of q_i is d_i,
    every other row of d is a negative (rows N..M-1 are extra / hard negatives).
    mask: (N, M) bool, True = ignore this pair (a known false negative).

    L = -(1/N) Σ_i log softmax(S_i)_i,   S = q dᵀ / τ                      (N, M)
    dS = (softmax(S) - onehot) / N,  dq = dS d / τ,  dd = dSᵀ q / τ"""
    N = q.shape[0]
    S = q @ d.T / tau
    if mask is not None:
        S = np.where(mask, -1e9, S)
    S = S - S.max(axis=1, keepdims=True)
    P = np.exp(S)
    P /= P.sum(axis=1, keepdims=True)
    r = np.arange(N)
    loss = float(-np.mean(np.log(P[r, r] + 1e-30)))
    dS = P.copy()
    dS[r, r] -= 1
    dS /= N
    return loss, dS @ d / tau, dS.T @ q / tau, P


# ----------------------------------------------------------------------------
# Self-supervised training pairs mined from the corpus
# ----------------------------------------------------------------------------
def clean_title(t: str) -> str:
    return re.sub(r"^\d+\.\s*", "", t).replace("`", "").replace("*", "")


def corpus_units(docs=None):
    """Prose paragraphs of every section: [(group, section title, paragraph, sentences)]."""
    docs = docs or load_corpus()
    units = []
    for d in docs:
        for a, b in own_regions(d):
            sec = d.section_at(a)
            if clean_title(sec.title) in ("Files", "Resources", "Suggested order"):
                continue  # tables of file names and reading lists: not content
            for s, e, kind in blocks(d.text, a, b):
                txt = d.text[s:e]
                if kind == "code" or txt.startswith(("#", "![")) or len(txt) < 60:
                    continue
                sents = [d.text[x:y] for x, y in sentence_spans(d.text, s, e, kind)]
                units.append((f"{d.name}#{sec.start}", clean_title(sec.title), txt, sents))
    return units


def make_pairs(units, rng, kinds=("heading", "ict", "drop")):
    """One fresh sample of (query, passage, group, kind) pairs."""
    pairs = []
    for group, title, para, sents in units:
        if "heading" in kinds:
            pairs.append((title, para, group, "heading"))
        if "ict" in kinds and len(sents) >= 2:
            k = int(rng.integers(len(sents)))
            pairs.append((sents[k], " ".join(sents[:k] + sents[k + 1:]), group, "ict"))
        if "drop" in kinds:
            words = text_tokens(para)
            keep = rng.random(len(words)) < rng.uniform(0.3, 0.5)
            q = [w for w, k in zip(words, keep) if k] or words[:3]
            rng.shuffle(q)
            pairs.append((" ".join(q[:24]), para, group, "drop"))
    return pairs


def split_groups(units, val_frac, seed=0):
    groups = sorted({u[0] for u in units})
    rng = np.random.default_rng(seed)
    val = set(rng.choice(groups, int(round(val_frac * len(groups))), replace=False)) if val_frac else set()
    return [u for u in units if u[0] not in val], [u for u in units if u[0] in val]


def group_mask(groups_q, groups_d):
    """True where query i and passage j come from the same section but j != i."""
    gq = np.array(groups_q)[:, None]
    gd = np.array(groups_d)[None, :]
    m = gq == gd
    np.fill_diagonal(m[:, :len(groups_q)], False)
    return m


class Adam:
    """Adam; the embedding table gets LAZY updates (only the rows in the batch), like
    torch.optim.SparseAdam. A dense update would decay all 32k rows' moments every step."""

    def __init__(self, params, lr=2e-3, betas=(0.9, 0.999), eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, betas[0], betas[1], eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0
        self.tE = np.zeros(params["E"].shape[0], dtype=np.int64)  # per-row step counts

    def step(self, params, grads, lr=None):
        lr = lr or self.lr
        self.t += 1
        for k, g in grads.items():
            if k == "E":
                rows, g = g
                self.tE[rows] += 1
                t = self.tE[rows][:, None]
                m, v = self.m[k], self.v[k]
                m[rows] = self.b1 * m[rows] + (1 - self.b1) * g
                v[rows] = self.b2 * v[rows] + (1 - self.b2) * g * g
                mh = m[rows] / (1 - self.b1 ** t)
                vh = v[rows] / (1 - self.b2 ** t)
                params[k][rows] -= (lr * mh / (np.sqrt(vh) + self.eps)).astype(params[k].dtype)
            else:
                m, v = self.m[k], self.v[k]
                m[...] = self.b1 * m + (1 - self.b1) * g
                v[...] = self.b2 * v + (1 - self.b2) * g * g
                mh = m / (1 - self.b1 ** self.t)
                vh = v / (1 - self.b2 ** self.t)
                params[k] -= (lr * mh / (np.sqrt(vh) + self.eps)).astype(params[k].dtype)


def in_batch_accuracy(enc, pairs):
    """Top-1 accuracy of picking each query's own passage among ALL passages of `pairs`."""
    if not pairs:
        return float("nan")
    q = enc.encode([p[0] for p in pairs])
    d = enc.encode([p[1] for p in pairs])
    S = q @ d.T
    same = group_mask([p[2] for p in pairs], [p[2] for p in pairs])
    S = np.where(same, -np.inf, S)  # a same-section passage is not counted as a mistake
    return float(np.mean(S.argmax(1) == np.arange(len(pairs))))


def train(steps=1200, batch=128, lr=3e-3, tau=0.05, dim=128, seed=0, val_frac=0.1,
          kinds=("heading", "ict", "drop"), verbose=True, n_buckets=N_BUCKETS,
          trigram_weight=0.5, feat_drop=0.5):
    rng = np.random.default_rng(seed)
    units = corpus_units()
    train_units, val_units = split_groups(units, val_frac, seed)
    idf = feature_idf([u[2] for u in units], n_buckets)
    enc = Encoder(n_buckets, dim, seed, idf=idf, tau=tau, trigram_weight=trigram_weight)
    opt = Adam(enc.params, lr)
    val_pairs = make_pairs(val_units, np.random.default_rng(seed + 1), kinds)
    if verbose:
        n_pairs = len(make_pairs(train_units, np.random.default_rng(seed), kinds))
        print(f"self-supervised pairs per epoch: {n_pairs} "
              f"from {len(train_units)} paragraphs in "
              f"{len({u[0] for u in train_units})} sections "
              f"(+{len(val_pairs)} val pairs from {len({u[0] for u in val_units})} held-out sections)")
        print(f"encoder: {n_buckets:,} buckets x D={dim} + residual MLP = {enc.n_params():,} params, "
              f"tau={tau}, batch={batch} (in-batch negatives: {batch - 1} per query)")
    hist = dict(step=[], loss=[], val_acc=[], val_step=[])
    acc0 = in_batch_accuracy(enc, val_pairs)
    hist["val_step"].append(0)
    hist["val_acc"].append(acc0)
    if verbose:
        print(f"step     0 | val top-1 among {len(val_pairs)} passages: {acc0:.3f} (random init)")
    pairs, pos, t0, smooth = [], 0, time.time(), None
    for step in range(1, steps + 1):
        if pos + batch > len(pairs):  # new epoch: fresh ict/drop samples, new order
            pairs = make_pairs(train_units, rng, kinds)
            pairs = [pairs[i] for i in rng.permutation(len(pairs))]
            pos = 0
        bp = pairs[pos:pos + batch]
        pos += batch
        bags = enc.bags([p[0] for p in bp] + [p[1] for p in bp], feat_drop, rng)
        e = enc.forward(bags)
        qe, de = e[:len(bp)], e[len(bp):]
        mask = group_mask([p[2] for p in bp], [p[2] for p in bp])
        loss, dq, dd, _ = info_nce(qe, de, tau, mask)
        grads = enc.backward(np.concatenate([dq, dd]).astype(e.dtype))
        cur_lr = lr * 0.5 * (1 + np.cos(np.pi * step / steps))  # cosine decay
        opt.step(enc.params, grads, max(cur_lr, lr * 0.05))
        smooth = loss if smooth is None else 0.95 * smooth + 0.05 * loss
        hist["step"].append(step)
        hist["loss"].append(loss)
        if step % 100 == 0 or step == steps:
            acc = in_batch_accuracy(enc, val_pairs)
            hist["val_step"].append(step)
            hist["val_acc"].append(acc)
            if verbose:
                print(f"step {step:5d} | InfoNCE loss {smooth:.3f} (chance ln {batch} = "
                      f"{np.log(batch):.2f}) | val top-1 {acc:.3f} | {time.time() - t0:5.1f} s")
    return enc, {k: np.array(v) for k, v in hist.items()}


def save_embedder(enc, hist, path=DEFAULT_PATH):
    np.savez(path, idf=enc.idf, **enc.params, **{f"hist_{k}": v for k, v in hist.items()},
             **{f"cfg_{k}": v for k, v in enc.config.items()})


def load_embedder(path=DEFAULT_PATH, dtype=np.float32):
    z = np.load(path)
    cfg = {k[4:]: z[k].item() for k in z.files if k.startswith("cfg_")}
    enc = Encoder(int(cfg["n_buckets"]), int(cfg["dim"]), dtype=dtype, idf=z["idf"],
                  tau=float(cfg["tau"]), trigram_weight=float(cfg["trigram_weight"]))
    for k in enc.params:
        enc.params[k] = z[k].astype(dtype)
    hist = {k[5:]: z[k] for k in z.files if k.startswith("hist_")}
    return enc, hist


def get_embedder(path=DEFAULT_PATH, verbose=True):
    """The trained embedder: load rag/embedder.npz, or train it first (~45 s)."""
    if os.path.exists(path):
        return load_embedder(path)
    if verbose:
        print("No rag/embedder.npz yet, so training the bi-encoder first (~45 s).")
        print("To keep it, it is saved; to retrain: python rag/embed_numpy.py\n")
    enc, hist = train(verbose=False)
    save_embedder(enc, hist, path)
    return enc, hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--feat-drop", type=float, default=0.5, help="feature dropout (0 = off)")
    ap.add_argument("--trigram-weight", type=float, default=0.5)
    ap.add_argument("--kinds", default="heading,ict,drop",
                    help="which self-supervised pair types to train on")
    ap.add_argument("--out", default=DEFAULT_PATH)
    ap.add_argument("--seeds", type=int, nargs="+",
                    help="train one model per seed (init, pair sampling AND val split) and report "
                         "the spread of the qa.jsonl metrics; nothing is saved")
    args = ap.parse_args()
    kinds = tuple(args.kinds.split(","))
    if args.seeds:
        from retrieval_eval import Retriever, load_eval, metric_values, run_all
        qs, rel = load_eval()
        print(f"{'seed':>4s} {'val top-1':>9s} {'dense R@5':>9s} {'dense MRR':>9s} {'hybrid R@5':>10s}")
        rows = []
        for seed in args.seeds:
            enc, hist = train(args.steps, args.batch, args.lr, args.tau, args.dim, seed=seed,
                              val_frac=args.val_frac, kinds=kinds, verbose=False,
                              trigram_weight=args.trigram_weight, feat_drop=args.feat_drop)
            res = run_all(Retriever(embedder=enc), qs, rel, ["dense", "hybrid"])
            row = [hist["val_acc"][-1]] + [np.mean(metric_values(res[m], qs, k)) for m, k in
                                           [("dense", "recall@5"), ("dense", "mrr"), ("hybrid", "recall@5")]]
            rows.append(row)
            print(f"{seed:4d} {row[0]:9.3f} {row[1]:9.3f} {row[2]:9.3f} {row[3]:10.3f}", flush=True)
        r = np.array(rows)
        print(f"mean {r[:, 0].mean():9.3f} {r[:, 1].mean():9.3f} {r[:, 2].mean():9.3f} {r[:, 3].mean():10.3f}")
        print(f" std {r[:, 0].std():9.3f} {r[:, 1].std():9.3f} {r[:, 2].std():9.3f} {r[:, 3].std():10.3f}")
        return

    t0 = time.time()
    enc, hist = train(args.steps, args.batch, args.lr, args.tau, args.dim, val_frac=args.val_frac,
                      kinds=kinds, trigram_weight=args.trigram_weight, feat_drop=args.feat_drop)
    print(f"trained in {time.time() - t0:.1f} s")
    save_embedder(enc, hist, args.out)
    print(f"saved {os.path.relpath(args.out)}")

    # The real test: the hand-written questions, never seen in training.
    from retrieval_eval import Retriever, load_eval, mean_metrics
    qs, rel = load_eval()
    init = Encoder(enc.config["n_buckets"], enc.config["dim"], 0, idf=enc.idf, tau=args.tau,
                   trigram_weight=enc.trigram_weight)
    print("\nretrieval on qa.jsonl (markdown+ctx chunks, exact cosine search):")
    for name, e in [("random init (hashed bag of n-grams)", init), ("trained bi-encoder", enc)]:
        m = mean_metrics(Retriever(embedder=e), "dense", qs, rel)
        print(f"  {name:36s} recall@5 {m['recall@5']:.3f}  MRR {m['mrr']:.3f}  "
              f"nDCG@10 {m['ndcg@10']:.3f}")


if __name__ == "__main__":
    main()
