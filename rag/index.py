"""
Vector search from scratch: exact brute force, IVF (k-means buckets) and product
quantization (PQ), with the recall / latency / memory trade-offs measured.

Once chunks are vectors, retrieval is "find the k rows of X (n, D) with the largest
dot product with q". Exact search is one matrix-vector product plus a partial sort:

    scores = X q                 (n,)      n·D multiply-adds, reads all n·D floats
    top-k  = argpartition(-scores, k)      O(n), then sort only those k

At n = 261 chunks that takes microseconds and nothing else is worth discussing. At
100 million vectors × 768 dims (307 GB in float32) it is impossible per query, and you
need APPROXIMATE nearest-neighbour (ANN) search, which trades a little recall for a lot
of speed and/or memory:

  IVF (inverted file): k-means the vectors into nlist clusters. At query time score
      the nlist centroids, open only the nprobe closest clusters, and search exactly
      inside them. Speed-up ≈ nlist / nprobe; misses happen when a true neighbour sits
      in a cluster you didn't open.
  PQ (product quantization, Jégou et al. 2011): split each D-dim vector into m
      sub-vectors, k-means each subspace to 256 centroids, and store only the m centroid
      ids: m BYTES per vector instead of 4·D. Dot products are computed from an (m, 256)
      lookup table per query ("asymmetric distance computation", ADC). Memory ÷ 4D/m;
      recall drops because vectors are approximated. Re-scoring the top R candidates
      with the exact vectors recovers most of it.
  HNSW (Malkov & Yashunin 2018, not built here): a layered proximity graph walked
      greedily; the default in most vector databases. See the README.

Run (~40 s; saves rag/index_results.npz for make_figures.py):
    python rag/index.py
    python rag/index.py --n 200000
"""
from __future__ import annotations

import argparse
import os
import time
import warnings
from typing import Optional

import numpy as np

warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "index_results.npz")


def topk(scores: np.ndarray, k: int):
    """Indices of the k largest scores, best first: argpartition is O(n), and only the
    k winners get sorted."""
    k = min(k, scores.shape[-1])
    part = np.argpartition(-scores, k - 1, axis=-1)[..., :k]
    order = np.argsort(-np.take_along_axis(scores, part, -1), axis=-1, kind="stable")
    return np.take_along_axis(part, order, -1)


class ExactIndex:
    def __init__(self, X: np.ndarray):
        self.X = np.ascontiguousarray(X, dtype=np.float32)

    def scores(self, q: np.ndarray) -> np.ndarray:
        return self.X @ q.astype(np.float32)

    def search(self, Q: np.ndarray, k: int = 10):
        """Q: (nq, D) or (D,). Returns ids (nq, k) [or (k,)]."""
        S = Q.astype(np.float32) @ self.X.T
        return topk(S, k)

    def nbytes(self):
        return self.X.nbytes


def kmeans(X, k, iters=15, seed=0, spherical=True, sample=None, verbose=False):
    """Lloyd's algorithm with k-means++ seeding. spherical=True clusters unit vectors by
    cosine (assign by max dot product, renormalise the centroids), which is what we
    want for IVF on normalised embeddings. Returns centroids (k, D) and the objective
    per iteration (mean squared distance, or mean cosine if spherical)."""
    rng = np.random.default_rng(seed)
    if sample and len(X) > sample:
        X = X[rng.choice(len(X), sample, replace=False)]
    X = X.astype(np.float32)
    n = len(X)
    sq = (X ** 2).sum(1)
    # k-means++: each new centroid is drawn with probability ∝ squared distance
    # to the nearest centroid chosen so far, so the seeds are spread out.
    C = np.empty((k, X.shape[1]), np.float32)
    C[0] = X[rng.integers(n)]
    d2 = np.maximum(sq - 2 * X @ C[0] + (C[0] ** 2).sum(), 0)
    for j in range(1, k):
        p = d2 / d2.sum() if d2.sum() > 0 else np.full(n, 1 / n)
        C[j] = X[rng.choice(n, p=p)]
        d2 = np.minimum(d2, np.maximum(sq - 2 * X @ C[j] + (C[j] ** 2).sum(), 0))
    hist = []
    for _ in range(iters):
        if spherical:
            S = X @ C.T
            a = S.argmax(1)
            hist.append(float(S[np.arange(n), a].mean()))
        else:
            D2 = sq[:, None] - 2 * X @ C.T + (C ** 2).sum(1)[None]
            a = D2.argmin(1)
            hist.append(float(np.maximum(D2[np.arange(n), a], 0).mean()))
        counts = np.bincount(a, minlength=k)
        sums = np.zeros_like(C)
        np.add.at(sums, a, X)
        empty = counts == 0
        C[~empty] = sums[~empty] / counts[~empty, None]
        if empty.any():  # re-seed empty clusters on random points
            C[empty] = X[rng.choice(n, empty.sum(), replace=False)]
        if spherical:
            C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-12
    return C, np.array(hist)


def assign(X, C, spherical=True, batch=20000):
    out = np.empty(len(X), np.int64)
    for i in range(0, len(X), batch):
        xb = X[i:i + batch]
        if spherical:
            out[i:i + batch] = (xb @ C.T).argmax(1)
        else:
            out[i:i + batch] = ((C ** 2).sum(1)[None] - 2 * xb @ C.T).argmin(1)
    return out


class IVFIndex:
    def __init__(self, X, nlist, seed=0, iters=15, train_size=30000):
        X = np.ascontiguousarray(X, dtype=np.float32)
        self.C, self.kmeans_hist = kmeans(X, nlist, iters, seed, True, train_size)
        a = assign(X, self.C)
        order = np.argsort(a, kind="stable")
        self.ids = order                                   # original id of each stored row
        self.Xs = X[order]                                 # vectors stored list by list
        self.offsets = np.concatenate([[0], np.cumsum(np.bincount(a, minlength=nlist))])
        self.nlist = nlist

    def search_one(self, q, k=10, nprobe=8):
        q = q.astype(np.float32)
        probe = topk(self.C @ q, nprobe)
        ranges = [(self.offsets[c], self.offsets[c + 1]) for c in probe]
        ids = np.concatenate([self.ids[a:b] for a, b in ranges])
        scores = np.concatenate([self.Xs[a:b] @ q for a, b in ranges])
        found = ids[topk(scores, k)] if len(ids) else ids
        # fewer than k candidates in the opened lists: pad with -1 ("no result")
        return np.concatenate([found, np.full(k - len(found), -1, dtype=found.dtype)])

    def search(self, Q, k=10, nprobe=8):
        return np.stack([self.search_one(q, k, nprobe) for q in Q])

    def candidates(self, nprobe):
        """Average number of vectors scored per query (the work actually done)."""
        sizes = np.diff(self.offsets)
        return nprobe * sizes.mean()

    def nbytes(self):
        return self.Xs.nbytes + self.C.nbytes + self.ids.nbytes


class PQIndex:
    def __init__(self, X, m=16, ks=256, iters=15, seed=0, train_size=20000):
        X = np.ascontiguousarray(X, dtype=np.float32)
        n, D = X.shape
        assert D % m == 0, "D must be divisible by m"
        self.m, self.ks, self.dsub = m, ks, D // m
        self.codebooks = np.empty((m, ks, self.dsub), np.float32)
        self.codes = np.empty((n, m), np.uint8)
        for j in range(m):
            sub = X[:, j * self.dsub:(j + 1) * self.dsub]
            self.codebooks[j], _ = kmeans(sub, ks, iters, seed + j, False, train_size)
            self.codes[:, j] = assign(sub, self.codebooks[j], spherical=False)
        self.X = X  # kept ONLY for the optional exact re-scoring; real systems keep it on disk

    def reconstruct(self, ids=None):
        codes = self.codes if ids is None else self.codes[ids]
        return np.concatenate([self.codebooks[j][codes[:, j]] for j in range(self.m)], axis=1)

    def adc_scores(self, q):
        """<q, x̂> for every stored x̂ from an (m, 256) table of sub-dot-products:
        table[j, c] = <q_j, codebook_j[c]>, score = Σ_j table[j, code_j]. n·m lookups and
        adds instead of n·D multiply-adds, and it reads n·m bytes instead of 4·n·D."""
        table = np.einsum("jcd,jd->jc", self.codebooks, q.reshape(self.m, self.dsub).astype(np.float32))
        flat = table.ravel()
        return flat[self.codes.astype(np.int64) + np.arange(self.m) * self.ks].sum(1)

    def search_one(self, q, k=10, rerank=0):
        s = self.adc_scores(q)
        if not rerank:
            return topk(s, k)
        cand = topk(s, rerank)  # shortlist by the compressed codes, then exact re-score
        return cand[topk(self.X[cand] @ q.astype(np.float32), k)]

    def search(self, Q, k=10, rerank=0):
        return np.stack([self.search_one(q, k, rerank) for q in Q])

    def code_bytes(self):
        return self.codes.nbytes + self.codebooks.nbytes


def recall_at_k(found, truth):
    """Fraction of the exact top-k that the approximate search also returned."""
    k = truth.shape[1]
    return float(np.mean([len(set(f[:k]) & set(t)) / k for f, t in zip(found, truth)]))


def time_per_query(fn, Q, repeats=1):
    """Median wall time of one query, in milliseconds (one query at a time, like a
    request to a server; batching queries is a different, throughput-oriented regime)."""
    ts = []
    for _ in range(repeats):
        for q in Q:
            t = time.perf_counter()
            fn(q)
            ts.append(time.perf_counter() - t)
    return 1000 * float(np.median(ts))


def synthetic(n, D=128, n_clusters=1000, spread=1.25, seed=0, nq=300):
    """Clustered unit vectors ("topics" + noise), like real embeddings, plus held-out
    queries from the same distribution. spread = length of the noise vector relative to
    the (unit) centre: 1.25 gives cosine ≈ 0.62 to the centre, ≈ 0.39 between two
    points of a cluster and ≈ 0 ± 0.09 between clusters: clusters that overlap a bit,
    so IVF has to work for its recall (0.75 makes it trivial, 2.0 nearly uniform)."""
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n_clusters, D)).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)

    def draw(m):
        c = centers[rng.integers(n_clusters, size=m)]
        x = c + spread / np.sqrt(D) * rng.standard_normal((m, D)).astype(np.float32)
        return x / np.linalg.norm(x, axis=1, keepdims=True)

    return draw(n), draw(nq)


def uniform(n, D=128, seed=0, nq=300):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n + nq, D)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X[:n], X[n:]


def benchmark_ivf(X, Q, truth, nlist, nprobes, k=10):
    t = time.perf_counter()
    ivf = IVFIndex(X, nlist)
    build = time.perf_counter() - t
    rows = []
    for p in nprobes:
        if p > nlist:
            continue
        found = ivf.search(Q, k, p)
        rows.append((p, recall_at_k(found, truth), time_per_query(lambda q: ivf.search_one(q, k, p), Q[:200]),
                     ivf.candidates(p)))
    return ivf, build, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100_000)
    ap.add_argument("--dim", type=int, default=128)
    args = ap.parse_args()
    k = 10
    res = {}

    # ---------------------------------------------------------------- 1. our corpus
    from chunking import get_chunks
    from corpus import load_qa
    from embed_numpy import get_embedder
    enc, _ = get_embedder()
    chunks = get_chunks()
    Xc = enc.encode([c.text for c in chunks])
    Qc = enc.encode([q.question for q in load_qa()])
    exact = ExactIndex(Xc)
    truth = exact.search(Qc, k)
    t_exact = time_per_query(lambda q: topk(exact.scores(q), k), Qc, repeats=20)
    print(f"1. Our corpus: {len(chunks)} chunk vectors x D={Xc.shape[1]} "
          f"({exact.nbytes() / 1024:.0f} KB), {len(Qc)} qa.jsonl questions as queries")
    print(f"   exact search: {t_exact * 1000:.0f} µs per query. Nothing to optimise here.")
    nlist_c = 16
    _, _, rows = benchmark_ivf(Xc, Qc, truth, nlist_c, [1, 2, 4, 8, 16])
    print(f"   IVF nlist={nlist_c}: " + ", ".join(f"nprobe {p}: recall@10 {r:.2f}" for p, r, _, _ in rows))
    res["corpus_ivf"] = np.array([(p, r, t, c) for p, r, t, c in rows])
    res["corpus_exact_ms"] = t_exact

    # ---------------------------------------------------------------- 2. synthetic 100k
    for part, (kind, gen) in enumerate([("clustered", synthetic), ("uniform", uniform)], 2):
        X, Q = gen(args.n, args.dim)
        ex = ExactIndex(X)
        t = time.perf_counter()
        truth = ex.search(Q, k)
        t_batch = (time.perf_counter() - t) / len(Q) * 1000
        t_one = time_per_query(lambda q: topk(ex.scores(q), k), Q[:200])
        nlist = int(round(np.sqrt(args.n)))
        print(f"\n{part}. Synthetic {kind} unit vectors: n={args.n:,} x D={args.dim} "
              f"({ex.nbytes() / 2 ** 20:.1f} MB float32), {len(Q)} held-out queries")
        print(f"   exact: {t_one:.2f} ms per query one at a time "
              f"({t_batch:.3f} ms/query when all {len(Q)} are batched into one matmul)")
        ivf, build, rows = benchmark_ivf(X, Q, truth, nlist, [1, 2, 4, 8, 16, 32, 64, 128])
        print(f"   IVF nlist={nlist} (≈ √n), k-means on 30k samples, built in {build:.1f} s:")
        print(f"     {'nprobe':>6s} {'recall@10':>9s} {'ms/query':>9s} {'speed-up':>8s} {'vectors scored':>15s}")
        for p, r, tq, cand in rows:
            print(f"     {p:6d} {r:9.3f} {tq:9.3f} {t_one / tq:7.1f}x {cand:15,.0f}")
        res[f"{kind}_exact_ms"] = t_one
        res[f"{kind}_ivf"] = np.array(rows)
        if kind != "clustered":
            continue
        print(f"   PQ (256 centroids per subspace; float32 = {4 * args.dim} bytes/vector):")
        print(f"     {'m':>3s} {'bytes/vec':>9s} {'compress':>8s} {'recall@10':>9s} {'ms/query':>9s}"
              f" {'+ exact re-score of top 100':>28s}")
        pq_rows = []
        for m in [8, 16, 32]:
            t = time.perf_counter()
            pq = PQIndex(X, m)
            build = time.perf_counter() - t
            r = recall_at_k(pq.search(Q, k), truth)
            tq = time_per_query(lambda q: pq.search_one(q, k), Q[:100])
            r2 = recall_at_k(pq.search(Q, k, rerank=100), truth)
            tq2 = time_per_query(lambda q: pq.search_one(q, k, rerank=100), Q[:100])
            print(f"     {m:3d} {m:9d} {4 * args.dim / m:7.0f}x {r:9.3f} {tq:9.3f} "
                  f"{'recall ' + format(r2, '.3f') + ', ' + format(tq2, '.2f') + ' ms':>28s}"
                  f"   (built in {build:.1f} s)")
            pq_rows.append((m, r, tq, r2, tq2, 4 * args.dim / m))
        res["pq"] = np.array(pq_rows)
        print("   Honest note: NumPy's fancy-indexed table lookups are slower than its BLAS matmul,")
        print("   so PQ is not faster here; in FAISS (SIMD lookups) it is. The 16-64x memory cut is real.")
    np.savez(RESULTS, n=args.n, dim=args.dim, **res)
    print(f"\nsaved {os.path.relpath(RESULTS)}")


if __name__ == "__main__":
    main()
