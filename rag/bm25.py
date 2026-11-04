"""
Sparse retrieval from scratch: a tokenizer, an inverted index, TF-IDF and BM25.

Sparse = one dimension per vocabulary word, almost all zeros. A chunk matches a query
only through words they SHARE, weighted so that rare words count more than common
ones. Crude, but hard to beat on queries full of exact names ("pack_padded_sequence",
"ALiBi", "124M"), it needs no training, and it explains itself: every score is a sum
of per-term contributions you can print (`BM25.explain`).

BM25 (Robertson & Zaragoza 2009), for query q and chunk d of length |d|:

    score(q, d) = Σ_{t ∈ q}  idf(t) · tf(t,d) · (k1 + 1) / ( tf(t,d) + k1 · (1 − b + b · |d| / avgdl) )
    idf(t)      = ln(1 + (N − df(t) + 0.5) / (df(t) + 0.5))

  - tf saturates: the 10th "attention" in a chunk adds far less than the 1st (k1 sets
    how fast; k1 → 0 is "present or not", k1 → ∞ is raw tf),
  - long chunks are penalised (b = 0 no length normalisation, b = 1 full),
  - rare terms (small df) get a big idf.

TF-IDF (the baseline): weight (1 + ln tf) · idf per term, cosine between the query and
chunk vectors. No saturation knob, and length handled by the vector norm.

Both are scored "term at a time" from an inverted index: for each query term, walk
its posting list (the chunks that contain it) and add its contribution. Chunks with
no query term are never touched, which is why search engines scale to billions of
documents. gradient_check.py verifies both against a dense brute-force formula.

Run (index stats, one query explained term by term, and a k1 x b sensitivity grid):
    python rag/bm25.py
    python rag/bm25.py --query "why divide by sqrt d_k"
"""
from __future__ import annotations

import argparse
import math
import re
from collections import Counter
from typing import Dict, List, Sequence

import numpy as np

STOPWORDS = frozenset("""
a an the and or but if of to in on at by for with from as is are was were be been being
it its this that these those which who whom what when where why how do does did done
not no can could should would will shall may might must than then so such into over
under about after before between through during each per via we you i our your their
they them he she his her my me us also just only very more most less much many some any
all both other same own there here up down out off again once too nor get gets got use
used uses using
""".split())

TOKEN_RE = re.compile(r"[a-z0-9]+")


def stem(w: str) -> str:
    """A deliberately tiny suffix stripper (a few of Porter's step-1 rules). It only has
    to map a word and its variants to the SAME string, not to a real word:
    gates/gate/gated -> gat, computing/compute -> comput, layers -> layer."""
    if len(w) <= 3 or w.isdigit():
        return w
    if w.endswith("ies") and len(w) > 4:
        w = w[:-3] + "y"
    elif w.endswith("sses"):
        w = w[:-2]
    elif w.endswith("s") and not w.endswith(("ss", "us", "is")):
        w = w[:-1]
    for suf in ("ing", "ed"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            w = w[:-len(suf)]
            if len(w) > 3 and w[-1] == w[-2] and w[-1] not in "lsz":
                w = w[:-1]  # running -> runn -> run
            break
    if w.endswith("ly") and len(w) > 5:
        w = w[:-2]
    if w.endswith("e") and len(w) > 4:
        w = w[:-1]
    return w


def tokenize(text: str, stemming: bool = True, stopwords: bool = True) -> List[str]:
    """lowercase -> alphanumeric runs -> drop stopwords -> stem.
    "W_hh" -> ["w", "hh"], "top-k" -> ["top", "k"], "d_k" -> ["d", "k"]."""
    toks = TOKEN_RE.findall(text.lower())
    if stopwords:
        toks = [t for t in toks if t not in STOPWORDS]
    return [stem(t) for t in toks] if stemming else toks


class InvertedIndex:
    """term -> posting list: (ids of the chunks containing it, term frequency in each)."""

    def __init__(self, docs_tokens: Sequence[List[str]]):
        self.N = len(docs_tokens)
        self.doc_len = np.array([len(t) for t in docs_tokens], dtype=np.float64)
        self.avgdl = float(self.doc_len.mean())
        lists: Dict[str, List] = {}
        for i, toks in enumerate(docs_tokens):
            for term, tf in Counter(toks).items():
                lists.setdefault(term, []).append((i, tf))
        self.vocab = {t: k for k, t in enumerate(sorted(lists))}
        self.postings = []
        for t in sorted(lists):
            ids, tfs = zip(*lists[t])
            self.postings.append((np.array(ids, dtype=np.int64), np.array(tfs, dtype=np.float64)))
        self.df = np.array([len(p[0]) for p in self.postings], dtype=np.float64)

    def n_postings(self):
        return int(self.df.sum())


class BM25:
    def __init__(self, docs_tokens: Sequence[List[str]], k1: float = 1.2, b: float = 0.75):
        self.index = InvertedIndex(docs_tokens)
        self.k1, self.b = k1, b
        N, df = self.index.N, self.index.df
        self.idf = np.log(1.0 + (N - df + 0.5) / (df + 0.5))
        self.norm = k1 * (1 - b + b * self.index.doc_len / self.index.avgdl)  # per chunk

    def term_contrib(self, term: str):
        """(chunk ids, contribution of `term` to each of those chunks' scores)."""
        k = self.index.vocab.get(term)
        if k is None:
            return np.zeros(0, dtype=np.int64), np.zeros(0)
        ids, tf = self.index.postings[k]
        return ids, self.idf[k] * tf * (self.k1 + 1) / (tf + self.norm[ids])

    def score(self, q_tokens: List[str]) -> np.ndarray:
        scores = np.zeros(self.index.N)
        for term in dict.fromkeys(q_tokens):  # unique terms, query order kept
            ids, c = self.term_contrib(term)
            scores[ids] += c
        return scores

    def explain(self, q_tokens: List[str], doc_id: int):
        """[(term, tf, df, idf, contribution)] for one chunk: why did it score this?"""
        rows = []
        for term in dict.fromkeys(q_tokens):
            k = self.index.vocab.get(term)
            if k is None:
                rows.append((term, 0, 0, 0.0, 0.0))
                continue
            ids, c = self.term_contrib(term)
            hit = np.nonzero(ids == doc_id)[0]
            tf = int(self.index.postings[k][1][hit[0]]) if len(hit) else 0
            rows.append((term, tf, int(self.index.df[k]), float(self.idf[k]),
                         float(c[hit[0]]) if len(hit) else 0.0))
        return rows


class TFIDF:
    """Cosine similarity of (1 + ln tf) · idf vectors, scored from the same inverted index."""

    def __init__(self, docs_tokens: Sequence[List[str]]):
        self.index = InvertedIndex(docs_tokens)
        N, df = self.index.N, self.index.df
        self.idf = np.log((1 + N) / (1 + df)) + 1.0  # smoothed, as in scikit-learn
        sq = np.zeros(N)
        self.weights = []
        for k, (ids, tf) in enumerate(self.index.postings):
            w = (1 + np.log(tf)) * self.idf[k]
            self.weights.append(w)
            np.add.at(sq, ids, w ** 2)
        self.doc_norm = np.sqrt(np.maximum(sq, 1e-12))

    def score(self, q_tokens: List[str]) -> np.ndarray:
        scores, qn = np.zeros(self.index.N), 0.0
        for term, qtf in Counter(q_tokens).items():
            k = self.index.vocab.get(term)
            if k is None:
                continue
            wq = (1 + math.log(qtf)) * self.idf[k]
            qn += wq ** 2
            ids, _ = self.index.postings[k]
            scores[ids] += wq * self.weights[k]
        return scores / self.doc_norm / max(math.sqrt(qn), 1e-12)


# ----------------------------------------------------------------------------
# Dense brute-force references (for gradient_check.py): same formulas, no index
# ----------------------------------------------------------------------------
def tf_matrix(docs_tokens, vocab):
    M = np.zeros((len(docs_tokens), len(vocab)))
    for i, toks in enumerate(docs_tokens):
        for t in toks:
            M[i, vocab[t]] += 1
    return M


def bm25_bruteforce(docs_tokens, q_tokens, k1=1.2, b=0.75):
    vocab = {t: k for k, t in enumerate(sorted({t for d in docs_tokens for t in d}))}
    TF = tf_matrix(docs_tokens, vocab)
    N, dl = TF.shape[0], TF.sum(1)
    df = (TF > 0).sum(0)
    idf = np.log(1 + (N - df + 0.5) / (df + 0.5))
    q = np.zeros(len(vocab))
    for t in set(q_tokens):
        if t in vocab:
            q[vocab[t]] = 1
    sat = TF * (k1 + 1) / (TF + k1 * (1 - b + b * dl / dl.mean())[:, None])
    return sat @ (q * idf)


def tfidf_bruteforce(docs_tokens, q_tokens):
    vocab = {t: k for k, t in enumerate(sorted({t for d in docs_tokens for t in d}))}
    TF = tf_matrix(docs_tokens, vocab)
    N = TF.shape[0]
    idf = np.log((1 + N) / (1 + (TF > 0).sum(0))) + 1
    W = np.where(TF > 0, 1 + np.log(np.maximum(TF, 1)), 0) * idf
    q_tf = tf_matrix([[t for t in q_tokens if t in vocab]], vocab)[0]
    wq = np.where(q_tf > 0, 1 + np.log(np.maximum(q_tf, 1)), 0) * idf
    return (W @ wq) / np.linalg.norm(W, axis=1) / max(np.linalg.norm(wq), 1e-12)


def main():
    from chunking import get_chunks
    from retrieval_eval import evaluate_ranking, load_eval

    ap = argparse.ArgumentParser()
    ap.add_argument("--query", default="What does the causal mask do?")
    args = ap.parse_args()

    chunks = get_chunks()
    toks = [tokenize(c.text) for c in chunks]
    bm = BM25(toks)
    ix = bm.index
    print(f"Inverted index over {ix.N} chunks (markdown+ctx): {len(ix.vocab):,} terms, "
          f"{ix.n_postings():,} postings, avg chunk length {ix.avgdl:.1f} tokens")
    top_df = sorted(ix.vocab, key=lambda t: -ix.df[ix.vocab[t]])[:8]
    print("most common terms (df): " + ", ".join(f"{t} {int(ix.df[ix.vocab[t]])}" for t in top_df))
    print(f"terms in only one chunk: {(ix.df == 1).sum():,} ({(ix.df == 1).mean():.0%} of the vocabulary)")

    q = tokenize(args.query)
    print(f"\nQuery: {args.query!r}\n  tokens after lowercase / stopwords / stemming: {q}")
    s = bm.score(q)
    print(f"\nBM25 top 5 (k1={bm.k1}, b={bm.b}):")
    for r, i in enumerate(np.argsort(-s)[:5]):
        parts = ", ".join(f"{t}: tf {tf} x idf {idf:.2f} -> {c:.2f}"
                          for t, tf, df, idf, c in bm.explain(q, i) if tf)
        print(f"  {r + 1}. {s[i]:6.2f}  [{chunks[i].doc}] {chunks[i].section[:48]}\n"
              f"             {parts}")
    tf = TFIDF(toks)
    s2 = tf.score(q)
    print("\nTF-IDF cosine top 5:")
    for r, i in enumerate(np.argsort(-s2)[:5]):
        print(f"  {r + 1}. {s2[i]:6.3f}  [{chunks[i].doc}] {chunks[i].section[:48]}")

    qs, rel = load_eval("markdown+ctx")
    ans = [q for q in qs if q.answerable]
    qtoks = [tokenize(q.question) for q in ans]
    print(f"\nSensitivity on qa.jsonl ({len(ans)} answerable questions): recall@5 of BM25")
    k1s, bs = [0.3, 0.6, 0.9, 1.2, 1.5, 2.0, 3.0], [0.0, 0.25, 0.5, 0.75, 1.0]
    print("   k1 \\ b " + "".join(f"{b:7.2f}" for b in bs))
    for k1 in k1s:
        row = []
        for b in bs:
            m = BM25(toks, k1, b)
            row.append(np.mean([evaluate_ranking(np.argsort(-m.score(t)), rel[q.id], q)["recall@5"]
                                for q, t in zip(ans, qtoks)]))
        print(f"  {k1:6.1f}  " + "".join(f"{v:7.3f}" for v in row))
    print("  (a sensitivity check, not tuning: the defaults k1=1.2, b=0.75 were fixed beforehand)")
    for stemming, sw in [(False, True), (True, False)]:
        t2 = [tokenize(c.text, stemming, sw) for c in chunks]
        m = BM25(t2)
        r5 = np.mean([evaluate_ranking(np.argsort(-m.score(tokenize(q.question, stemming, sw))),
                                       rel[q.id], q)["recall@5"] for q in ans])
        print(f"  stemming={'on ' if stemming else 'off'} stopwords={'removed' if sw else 'kept   '}:"
              f" recall@5 = {r5:.3f}")


if __name__ == "__main__":
    main()
