"""
Retrieval evaluation from scratch: which retriever finds the evidence, for which kind
of question, and how sure can we be with 57 questions?

Retrievers compared on qa.jsonl (all over the same chunks):
  tfidf          cosine of (1 + ln tf)·idf vectors                         (bm25.py)
  bm25           Okapi BM25, k1 = 1.2, b = 0.75                            (bm25.py)
  dense          our NumPy bi-encoder, exact cosine search                 (embed_numpy.py)
  hybrid         Reciprocal Rank Fusion of bm25 and dense (Cormack et al. 2009):
                     RRF(d) = Σ_r 1 / (60 + rank_r(d))
                 uses only RANKS, so a BM25 score of 14.2 and a cosine of 0.61 never
                 have to be put on the same scale
  hybrid+rerank  re-order hybrid's top 20 with a small learned "cross-feature" reranker:
                 logistic regression on features of the (question, chunk) PAIR (BM25,
                 cosine, idf-weighted term coverage, query-bigram phrase matches, terms in
                 the heading path, RRF score). Trained with 5-fold cross-validation over
                 questions, so no question is ever scored by a model that saw its labels.
  hybrid+mmr     Maximal Marginal Relevance on hybrid's top 20: trade relevance for
                 diversity (λ = 0.7), which can help multi-hop questions

Metrics (implemented below), per answerable question; a question has 1-2 evidence
SLOTS, and a chunk satisfies a slot if it covers at least half of the gold evidence span:
  recall@k     fraction of slots satisfied by the top k            (did we find it?)
  precision@k  fraction of the top k chunks that satisfy any slot  (how much noise?)
  MRR          1 / rank of the first relevant chunk                (how high?)
  nDCG@k       Σ gain_i / log2(i + 1) over the top k, normalised by the best possible
               ranking; gain_i = number of NEW slots chunk i satisfies
Uncertainty: 95% bootstrap confidence intervals (resample the questions with
replacement 2,000 times), and a paired bootstrap for "is A better than B?".

Run (~25 s):
    python rag/retrieval_eval.py
    python rag/retrieval_eval.py --no-ablation
"""
from __future__ import annotations

import argparse
import math
import time
import warnings
from typing import Dict, List, Optional, Sequence

import numpy as np

from bm25 import BM25, TFIDF, tokenize
from chunking import CHUNKERS, DEFAULT_CHUNKER, Chunk, get_chunks
from corpus import QTYPES, Question, load_corpus, load_qa
from embed_numpy import get_embedder
from index import ExactIndex

warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)

KS = (1, 3, 5, 10)
METHODS = ["tfidf", "bm25", "dense", "hybrid", "hybrid+rerank", "hybrid+mmr"]
RRF_K = 60
N_CAND = 20
FEATURES = ["bm25/max", "cosine", "idf coverage", "bigram match", "heading coverage", "rrf"]


# ----------------------------------------------------------------------------
# Relevance labels: gold evidence spans -> which chunks satisfy which slot
# ----------------------------------------------------------------------------
def covers(chunk: Chunk, file: str, span, min_frac: float = 0.5) -> bool:
    if chunk.doc != file or span is None:
        return False
    overlap = min(chunk.end, span[1]) - max(chunk.start, span[0])
    return overlap >= min_frac * (span[1] - span[0])


def relevance(q: Question, chunks: Sequence[Chunk]) -> List[set]:
    """For each evidence slot, the set of chunk ids that satisfy it."""
    return [{c.id for c in chunks if any(covers(c, e.file, e.span) for e in slot)}
            for slot in q.slots]


def load_eval(chunker: str = DEFAULT_CHUNKER):
    docs = load_corpus()
    qs = load_qa(docs=docs)
    chunks = get_chunks(chunker, docs)
    return qs, {q.id: relevance(q, chunks) for q in qs}


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def dcg(gains):
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def evaluate_ranking(ranking: Sequence[int], slots: List[set], q: Optional[Question] = None,
                     ks=KS) -> Dict[str, float]:
    """All metrics for one ranked list of chunk ids against the slot relevance sets."""
    any_rel = set().union(*slots) if slots else set()
    ranking = list(ranking)
    out = {}
    for k in ks:
        top = ranking[:k]
        out[f"recall@{k}"] = sum(bool(s & set(top)) for s in slots) / len(slots)
        out[f"precision@{k}"] = sum(c in any_rel for c in top) / k
    first = next((i for i, c in enumerate(ranking) if c in any_rel), None)
    out["mrr"] = 0.0 if first is None else 1.0 / (first + 1)
    # nDCG@10 with gain = number of slots a chunk satisfies for the first time
    done, gains = set(), []
    for c in ranking[:10]:
        new = {j for j, s in enumerate(slots) if c in s} - done
        gains.append(len(new))
        done |= new
    # ideal: greedily pick chunks that satisfy the most unsatisfied slots
    ideal, left = [], set(range(len(slots)))
    while left and len(ideal) < 10:
        best = max(any_rel, key=lambda c: len({j for j in left if c in slots[j]}), default=None)
        got = {j for j in left if best is not None and best in slots[j]}
        if not got:
            break
        ideal.append(len(got))
        left -= got
    out["ndcg@10"] = dcg(gains) / dcg(ideal) if ideal else 0.0
    return out


def bootstrap_ci(values, n_boot=2000, seed=0, alpha=0.05):
    """Mean and percentile bootstrap CI: resample the questions with replacement."""
    v = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), (n_boot, len(v)))].mean(1)
    return float(v.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_bootstrap(a, b, n_boot=2000, seed=0):
    """CI of mean(a - b) when a and b are scores on the SAME questions, and the fraction
    of resamples where a does not beat b (a one-sided bootstrap p-value)."""
    d = np.asarray(a, np.float64) - np.asarray(b, np.float64)
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(1)
    return float(d.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975)), \
        float(np.mean(means <= 0))


# ----------------------------------------------------------------------------
# The retrieval system: sparse + dense + fusion + rerank
# ----------------------------------------------------------------------------
def rrf(rankings: Sequence[Sequence[int]], n: int, k: int = RRF_K) -> np.ndarray:
    """Reciprocal Rank Fusion scores for n items from several rankings (best first)."""
    s = np.zeros(n)
    for r in rankings:
        s[np.asarray(r)] += 1.0 / (k + 1 + np.arange(len(r)))
    return s


def bigrams(toks):
    return set(zip(toks, toks[1:]))


class Retriever:
    """Everything needed to search one chunking of the corpus."""

    def __init__(self, chunker: str = DEFAULT_CHUNKER, embedder=None):
        self.chunker = chunker
        self.chunks = get_chunks(chunker)
        self.toks = [tokenize(c.text) for c in self.chunks]
        self.head_toks = [set(tokenize(c.heading)) for c in self.chunks]
        self.tok_sets = [set(t) for t in self.toks]
        self.bigram_sets = [bigrams(t) for t in self.toks]
        self.bm25 = BM25(self.toks)
        self.tfidf = TFIDF(self.toks)
        self.enc = embedder if embedder is not None else get_embedder()[0]
        self.emb = self.enc.encode([c.text for c in self.chunks])
        self.index = ExactIndex(self.emb)
        self.reranker = None  # weights, set by fit_reranker()
        self._cache = {}

    def query(self, question: str):
        """Everything about a query that several stages reuse, computed once."""
        if question not in self._cache:
            qt = tokenize(question)
            qv = self.enc.encode([question])[0]
            bm = self.bm25.score(qt)
            dn = self.index.scores(qv)
            r_b, r_d = np.argsort(-bm, kind="stable"), np.argsort(-dn, kind="stable")
            fused = rrf([r_b[:100], r_d[:100]], len(self.chunks))
            self._cache[question] = dict(toks=qt, vec=qv, bm25=bm, dense=dn, rank_bm25=r_b,
                                         rank_dense=r_d, rrf=fused,
                                         rank_hybrid=np.argsort(-fused, kind="stable"))
        return self._cache[question]

    def features(self, question: str, cand: Sequence[int]) -> np.ndarray:
        """(len(cand), 6) features of each (question, chunk) pair, see FEATURES."""
        Q = self.query(question)
        qt = list(dict.fromkeys(Q["toks"]))
        idf = {t: (self.bm25.idf[self.bm25.index.vocab[t]] if t in self.bm25.index.vocab
                   else self.bm25.idf.max()) for t in qt}
        total = sum(idf.values()) or 1.0
        qb = bigrams(Q["toks"])
        bm = Q["bm25"][cand]
        F = np.zeros((len(cand), len(FEATURES)))
        F[:, 0] = bm / (bm.max() + 1e-9)
        F[:, 1] = Q["dense"][cand]
        for r, c in enumerate(cand):
            F[r, 2] = sum(idf[t] for t in qt if t in self.tok_sets[c]) / total
            F[r, 3] = len(qb & self.bigram_sets[c]) / max(len(qb), 1)
            F[r, 4] = sum(idf[t] for t in qt if t in self.head_toks[c]) / total
        F[:, 5] = Q["rrf"][cand] * RRF_K / 2  # 1.0 = ranked first by both retrievers
        return F

    def rerank_scores(self, question, cand, w=None):
        w = self.reranker if w is None else w
        F = self.features(question, cand)
        return 1 / (1 + np.exp(-(np.hstack([F, np.ones((len(F), 1))]) @ w)))

    def mmr(self, question, cand, lam=0.7, k=10):
        """Maximal Marginal Relevance: pick the next chunk maximising
        λ·sim(q, d) - (1 - λ)·max_{already picked p} sim(d, p)."""
        Q = self.query(question)
        cand = list(cand)
        rel = Q["dense"][cand]
        rel = (rel - rel.min()) / (rel.max() - rel.min() + 1e-9) * 0.5 + \
            0.5 * Q["rrf"][cand] / (Q["rrf"][cand].max() + 1e-12)
        E = self.emb[cand]
        picked = []
        while len(picked) < min(k, len(cand)):
            red = (E @ E[picked].T).max(1) if picked else np.zeros(len(cand))
            score = lam * rel - (1 - lam) * red
            score[picked] = -np.inf
            picked.append(int(score.argmax()))
        return [cand[i] for i in picked]

    def search(self, method: str, question: str, k: int = 10, reranker=None) -> List[int]:
        Q = self.query(question)
        if method == "bm25":
            return list(Q["rank_bm25"][:k])
        if method == "tfidf":
            return list(np.argsort(-self.tfidf.score(Q["toks"]), kind="stable")[:k])
        if method == "dense":
            return list(Q["rank_dense"][:k])
        hyb = list(Q["rank_hybrid"])
        if method == "hybrid":
            return hyb[:k]
        cand = hyb[:N_CAND]
        if method == "hybrid+rerank":
            s = self.rerank_scores(question, cand, reranker)
            return [cand[i] for i in np.argsort(-s, kind="stable")] [:k]
        if method == "hybrid+mmr":
            return self.mmr(question, cand, k=k)
        raise ValueError(method)


# ----------------------------------------------------------------------------
# The reranker: logistic regression on pair features, cross-validated
# ----------------------------------------------------------------------------
def train_logreg(X, y, l2=1e-3, steps=2000, lr=1.0):
    """Plain batch gradient descent on the mean logistic loss + L2 (not on the bias).
    No class re-weighting, on purpose: then the output is a CALIBRATED probability that
    a chunk is relevant (~1 in 12 candidates is), which rag_pipeline.py uses to abstain."""
    Xb = np.hstack([X, np.ones((len(X), 1))])
    w = np.zeros(Xb.shape[1])
    for _ in range(steps):
        p = 1 / (1 + np.exp(-(Xb @ w)))
        g = Xb.T @ (p - y) / len(y) + l2 * np.r_[w[:-1], 0]
        w -= lr * g
    return w


def reranker_data(ret: Retriever, qs, rel):
    X, y = [], []
    for q in qs:
        cand = list(ret.query(q.question)["rank_hybrid"][:N_CAND])
        any_rel = set().union(*rel[q.id])
        X.append(ret.features(q.question, cand))
        y.append(np.array([c in any_rel for c in cand], float))
    return np.vstack(X), np.concatenate(y)


def fold_of(qs, n_folds=5):
    ans = sorted(q.id for q in qs if q.answerable)
    f = {qid: i % n_folds for i, qid in enumerate(ans)}
    unans = sorted(q.id for q in qs if not q.answerable)
    f.update({qid: i % n_folds for i, qid in enumerate(unans)})
    return f


def cv_rerankers(ret: Retriever, qs, rel, n_folds=5):
    """One reranker per fold, trained on the answerable questions of the OTHER folds.
    Returns {question id: weights to use for it} and the all-data weights (for new
    questions, e.g. the interactive pipeline)."""
    folds = fold_of(qs, n_folds)
    ans = [q for q in qs if q.answerable]
    ws = []
    for f in range(n_folds):
        X, y = reranker_data(ret, [q for q in ans if folds[q.id] != f], rel)
        ws.append(train_logreg(X, y))
    X, y = reranker_data(ret, ans, rel)
    w_all = train_logreg(X, y)
    return {q.id: ws[folds[q.id]] for q in qs}, w_all


# ----------------------------------------------------------------------------
# Running the comparison
# ----------------------------------------------------------------------------
def run_all(ret: Retriever, qs, rel, methods=METHODS, k=10, per_q_w=None):
    """{method: {question id: metrics}} for the answerable questions."""
    if per_q_w is None and any("rerank" in m for m in methods):
        per_q_w, ret.reranker = cv_rerankers(ret, qs, rel)
    out = {}
    for m in methods:
        out[m] = {q.id: evaluate_ranking(ret.search(m, q.question, k, per_q_w[q.id] if per_q_w else None),
                                         rel[q.id], q)
                  for q in qs if q.answerable}
    return out


def mean_metrics(ret, method, qs, rel):
    res = run_all(ret, qs, rel, [method])[method]
    return {k: float(np.mean([r[k] for r in res.values()])) for k in next(iter(res.values()))}


def metric_values(res, qs, metric, qtype=None):
    return [res[q.id][metric] for q in qs if q.answerable and (qtype is None or q.type == qtype)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-ablation", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    qs, rel = load_eval()
    ans = [q for q in qs if q.answerable]
    ret = Retriever()
    res = run_all(ret, qs, rel)
    n_by = {t: sum(q.type == t for q in ans) for t in QTYPES}
    print(f"qa.jsonl: {len(ans)} answerable questions ({n_by['keyword']} keyword, "
          f"{n_by['semantic']} semantic, {n_by['multihop']} multi-hop), "
          f"{len(qs) - len(ans)} unanswerable (used in rag_pipeline.py)")
    print(f"chunks: {DEFAULT_CHUNKER}, {len(ret.chunks)} chunks; 95% bootstrap CIs over questions\n")

    print(f"  {'retriever':14s} {'recall@1':>9s} {'recall@5':>17s} {'recall@10':>9s} {'P@5':>6s} "
          f"{'MRR':>17s} {'nDCG@10':>8s}")
    for m in METHODS:
        r5 = bootstrap_ci(metric_values(res[m], qs, "recall@5"))
        mrr = bootstrap_ci(metric_values(res[m], qs, "mrr"))
        mean = lambda k: np.mean(metric_values(res[m], qs, k))
        print(f"  {m:14s} {mean('recall@1'):9.3f} {r5[0]:6.3f} [{r5[1]:.2f},{r5[2]:.2f}] "
              f"{mean('recall@10'):9.3f} {mean('precision@5'):6.3f} "
              f"{mrr[0]:6.3f} [{mrr[1]:.2f},{mrr[2]:.2f}] {mean('ndcg@10'):8.3f}")

    print("\nrecall@5 by question type [95% CI]:")
    print(f"  {'retriever':14s}" + "".join(f"{t + ' (' + str(n_by[t]) + ')':>22s}" for t in QTYPES[:3]))
    for m in METHODS:
        cells = []
        for t in QTYPES[:3]:
            v, lo, hi = bootstrap_ci(metric_values(res[m], qs, "recall@5", t))
            cells.append(f"{v:.3f} [{lo:.2f},{hi:.2f}]")
        print(f"  {m:14s}" + "".join(f"{c:>22s}" for c in cells))
    both = {m: np.mean([res[m][q.id]["recall@10"] == 1.0 for q in ans if q.type == "multihop"])
            for m in METHODS}
    print("  multi-hop questions with BOTH evidence slots in the top 10: " +
          ", ".join(f"{m} {v:.0%}" for m, v in both.items()))

    print("\nPaired bootstrap on recall@5 (same questions, so the noise mostly cancels):")
    for a, b in [("hybrid", "bm25"), ("hybrid", "dense"), ("hybrid+rerank", "hybrid"),
                 ("bm25", "tfidf"), ("dense", "bm25")]:
        d, lo, hi, p = paired_bootstrap(metric_values(res[a], qs, "recall@5"),
                                        metric_values(res[b], qs, "recall@5"))
        print(f"  {a:13s} - {b:6s}: {d:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  "
              f"P(no improvement) = {p:.3f}")

    w = ret.reranker
    print("\nreranker weights (trained on all answerable questions; features are 0..1):")
    print("  " + ", ".join(f"{n} {v:+.2f}" for n, v in zip(FEATURES + ["bias"], w)))

    # where each retriever misses: who wins on which question
    print("\nquestions where BM25 and dense disagree at recall@5:")
    for q in ans:
        b, d = res["bm25"][q.id]["recall@5"], res["dense"][q.id]["recall@5"]
        if b != d:
            print(f"  {'bm25 ' if b > d else 'dense'} wins {q.id} ({q.type:8s}) {q.question[:70]}")

    if not args.no_ablation:
        print("\nAblation: chunking strategy x retriever (recall@5 / nDCG@10)")
        names = list(CHUNKERS)
        print(f"  {'chunker':13s} {'chunks':>6s}" + "".join(f"{m:>16s}" for m in METHODS[:5]))
        for name in names:
            qs2, rel2 = load_eval(name)
            r2 = Retriever(name, embedder=ret.enc)
            res2 = run_all(r2, qs2, rel2, METHODS[:5])
            cells = [f"{np.mean(metric_values(res2[m], qs2, 'recall@5')):.3f} / "
                     f"{np.mean(metric_values(res2[m], qs2, 'ndcg@10')):.3f}" for m in METHODS[:5]]
            print(f"  {name:13s} {len(r2.chunks):6d}" + "".join(f"{c:>16s}" for c in cells))
    print(f"\n({time.time() - t0:.1f} s)")


if __name__ == "__main__":
    main()
