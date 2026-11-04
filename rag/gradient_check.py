"""
The check script for this folder: everything hand-written is compared with a
brute-force or numerical reference. Runs in a few seconds.

Part 1  the eval set: every gold evidence string in qa.jsonl occurs EXACTLY ONCE in
        its README, inside the section it names (so labels can't silently rot when the
        READMEs change), and chunk spans are exact slices of the source.
Part 2  gradients (float64, central differences):
          dL/dz of L2-normalisation,
          InfoNCE wrt the query AND passage embeddings (with extra hard negatives and a
          false-negative mask),
          the embedding bag + residual MLP encoder (E, W1, b1, W2) on its own,
          the full pipeline: texts -> encoder -> InfoNCE loss -> every parameter.
Part 3  retrieval: BM25 and TF-IDF from the inverted index == a dense brute-force
        formula; recall / precision / MRR / nDCG / RRF on hand-computed examples; the
        faithfulness metric flags an invented sentence.
Part 4  vector indexes: argpartition top-k == full sort; IVF with nprobe = nlist ==
        exact search; PQ's lookup-table scores == dot products with the reconstructed
        vectors; spherical k-means never lowers its objective.

Try breaking a line (e.g. drop the `- e * ...` term in l2_normalize_backward, or the
`/ tau` in info_nce's dd) and re-run.

Run:
    python rag/gradient_check.py
"""
import warnings

import numpy as np

from bm25 import BM25, TFIDF, bm25_bruteforce, tfidf_bruteforce, tokenize
from chunking import CHUNKERS, get_chunks
from corpus import load_corpus, load_qa, validate_qa
from embed_numpy import Encoder, info_nce, l2_normalize, l2_normalize_backward
from index import ExactIndex, IVFIndex, PQIndex, kmeans, topk
from rag_pipeline import Answer, faithfulness
from retrieval_eval import bootstrap_ci, evaluate_ranking, rrf

warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)

F64 = np.float64
EPS = 1e-5
TOL = 1e-6


def numerical_grad(f, arr, eps=EPS):
    grad = np.zeros_like(arr)
    it = np.nditer(arr, flags=["multi_index"])
    for _ in it:
        idx = it.multi_index
        old = arr[idx]
        arr[idx] = old + eps
        lp = f()
        arr[idx] = old - eps
        lm = f()
        arr[idx] = old
        grad[idx] = (lp - lm) / (2 * eps)
    return grad


def rel_error(a, b):
    """‖a − b‖ / (‖a‖ + ‖b‖) over the whole tensor (see transformer/gradient_check.py)."""
    return float(np.linalg.norm(a - b) / max(1e-20, np.linalg.norm(a) + np.linalg.norm(b)))


def report(name, ok, detail=""):
    print(f"  {name:56s} {detail:34s} {'OK' if ok else 'FAIL'}")
    return ok


def grad_report(name, shape, err):
    return report(f"{name} shape={shape}", err < TOL, f"relative error = {err:.2e}")


# ----------------------------------------------------------------------------
def part1_eval_set():
    print("Part 1: the eval set and the chunks\n")
    docs = load_corpus()
    qs = load_qa(docs=docs)
    problems = validate_qa(qs, docs)
    for p in problems:
        print("    " + p)
    n_ev = sum(len(slot) for q in qs for slot in q.slots)
    ok = report(f"qa.jsonl: {len(qs)} questions, {n_ev} evidence strings", not problems,
                "each found once, in its section")
    by = {d.name: d for d in docs}
    for name in CHUNKERS:
        chunks = get_chunks(name, docs)
        exact = all(by[c.doc].text[c.start:c.end] == c.body for c in chunks)
        prefix = all(c.text.endswith(c.body) for c in chunks)
        detail = f"{len(chunks)} chunks, body == source slice"
        if name.startswith("markdown"):
            same_sec = all(by[c.doc].section_at(c.start).start == by[c.doc].section_at(c.end - 1).start
                           for c in chunks)
            exact &= same_sec
            detail = f"{len(chunks)} chunks, exact slices, 1 section"
        ok &= report(f"chunker {name}", exact and prefix, detail)
    return ok


# ----------------------------------------------------------------------------
def tiny_encoder(rng, B=40, D=6):
    enc = Encoder(n_buckets=B, dim=D, seed=1, dtype=F64, idf=rng.uniform(1, 3, B), tau=0.1)
    for k, p in enc.params.items():  # bigger-than-init weights so nothing is ~0
        p[...] = rng.standard_normal(p.shape) * 0.5
    return enc


def part2_gradients():
    print("\nPart 2: gradients (numerical vs hand-written backward, float64)\n")
    rng = np.random.default_rng(0)
    ok = True

    z = rng.standard_normal((4, 5))
    G = rng.standard_normal((4, 5))
    f = lambda: float(np.sum(l2_normalize(z)[0] * G))
    e, n = l2_normalize(z)
    ok &= grad_report("l2_normalize  dz", z.shape, rel_error(l2_normalize_backward(G, e, n),
                                                              numerical_grad(f, z)))

    N, M, D, tau = 4, 6, 5, 0.1  # 4 queries, 4 positives + 2 hard negatives
    q = l2_normalize(rng.standard_normal((N, D)))[0]
    d = l2_normalize(rng.standard_normal((M, D)))[0]
    mask = np.zeros((N, M), bool)
    mask[0, 2] = True  # pretend passage 2 is a false negative for query 0
    _, dq, dd, _ = info_nce(q, d, tau, mask)
    ok &= grad_report("info_nce  dq (queries)", q.shape,
                      rel_error(dq, numerical_grad(lambda: info_nce(q, d, tau, mask)[0], q)))
    ok &= grad_report("info_nce  dd (positives + hard negatives)", d.shape,
                      rel_error(dd, numerical_grad(lambda: info_nce(q, d, tau, mask)[0], d)))

    texts = ["the causal mask hides the future", "attention weights sum to one",
             "BM25 saturates term frequency", "gates in an LSTM"]
    enc = tiny_encoder(rng)
    bags = enc.bags(texts)
    G = rng.standard_normal((len(texts), enc.config["dim"]))
    enc.forward(bags)
    grads = enc.backward(G, dense_E=True)
    f = lambda: float(np.sum(enc.forward(bags) * G))
    for k, p in enc.params.items():
        ok &= grad_report(f"encoder alone  {k}", p.shape, rel_error(grads[k], numerical_grad(f, p)))
    enc.forward(bags)  # fresh cache (the numerical loop left a perturbed one)
    sparse = enc.backward(G)["E"]
    dense = np.zeros_like(enc.params["E"])
    dense[sparse[0]] = sparse[1]
    ok &= report("encoder  sparse dE (rows, values) == dense dE", np.allclose(dense, grads["E"]),
                 f"{len(sparse[0])} of {len(dense)} rows touched")

    # full pipeline: shared encoder for queries and passages, InfoNCE with a hard negative
    qs_t = ["what hides the future", "do attention weights sum to one", "term frequency saturation"]
    ds_t = ["the causal mask sets future scores to minus infinity",
            "softmax rows of attention sum to one", "BM25 saturates tf with k1",
            "LSTM forget gates scale the cell state"]  # the last one: a hard negative
    enc = tiny_encoder(rng)
    bags = enc.bags(qs_t + ds_t)

    def loss():
        e = enc.forward(bags)
        return info_nce(e[:3], e[3:], enc.tau)[0]

    e = enc.forward(bags)
    _, dq, dd, _ = info_nce(e[:3], e[3:], enc.tau)
    grads = enc.backward(np.concatenate([dq, dd]), dense_E=True)
    for k, p in enc.params.items():
        ok &= grad_report(f"full pipeline (texts -> InfoNCE)  {k}", p.shape,
                          rel_error(grads[k], numerical_grad(loss, p)))
    return ok


# ----------------------------------------------------------------------------
def part3_retrieval():
    print("\nPart 3: retrieval scoring and metrics\n")
    ok = True
    chunks = get_chunks()
    toks = [tokenize(c.text) for c in chunks]
    queries = ["What does the causal mask do?", "why divide by sqrt d_k", "forget gate bias LSTM",
               "im2col matrix multiply", "a query with zzzunknownzzz words"]
    for k1, b in [(1.2, 0.75), (2.0, 0.0), (0.5, 1.0)]:
        bm = BM25(toks, k1, b)
        err = max(np.abs(bm.score(tokenize(q)) - bm25_bruteforce(toks, tokenize(q), k1, b)).max()
                  for q in queries)
        ok &= report(f"BM25 inverted index == brute force (k1={k1}, b={b})", err < 1e-9,
                     f"max abs diff = {err:.1e}")
    tf = TFIDF(toks)
    err = max(np.abs(tf.score(tokenize(q)) - tfidf_bruteforce(toks, tokenize(q))).max() for q in queries)
    ok &= report("TF-IDF inverted index == brute force", err < 1e-9, f"max abs diff = {err:.1e}")
    bm = BM25(toks)
    q = tokenize(queries[0])
    i = int(np.argmax(bm.score(q)))
    ok &= report("BM25.explain contributions sum to the score",
                 abs(sum(r[4] for r in bm.explain(q, i)) - bm.score(q)[i]) < 1e-9)

    # ranking [5, 2, 9, 1]; slot 0 = {2}, slot 1 = {1, 7}
    m = evaluate_ranking([5, 2, 9, 1], [{2}, {1, 7}])
    want = {"recall@1": 0.0, "recall@3": 0.5, "recall@5": 1.0, "precision@3": 1 / 3,
            "precision@5": 2 / 5, "mrr": 0.5,
            "ndcg@10": (1 / np.log2(3) + 1 / np.log2(5)) / (1 + 1 / np.log2(3))}
    ok &= report("recall / precision / MRR / nDCG on a hand example",
                 all(abs(m[k] - v) < 1e-12 for k, v in want.items()),
                 f"nDCG@10 = {m['ndcg@10']:.4f}")
    m2 = evaluate_ranking([7, 3], [{2, 7}, {7}])  # one chunk satisfying both slots at rank 1
    ok &= report("nDCG = 1 when rank 1 satisfies both slots", abs(m2["ndcg@10"] - 1) < 1e-12)
    s = rrf([[0, 1, 2], [2, 0, 1]], 3, k=60)
    want = [1 / 61 + 1 / 62, 1 / 62 + 1 / 63, 1 / 63 + 1 / 61]
    ok &= report("reciprocal rank fusion", np.allclose(s, want) and list(np.argsort(-s)) == [0, 2, 1],
                 "fused order [0, 2, 1]")
    mean, lo, hi = bootstrap_ci([0.5] * 20)
    ok &= report("bootstrap CI of a constant is a point", mean == lo == hi == 0.5)

    c = chunks[i]
    real = " ".join(c.body.split()[:12])
    good = Answer(real, False, [i], [(real, [i])], [i], 1.0)
    bad = Answer("the mask was invented by Turing in 1950", False, [i],
                 [(real, [i]), ("the mask was invented by Turing in 1950", [i])], [i], 1.0)
    ok &= report("faithfulness: copied sentence 1.0, invented one flagged",
                 faithfulness(good, chunks) == 1.0 and faithfulness(bad, chunks) == 0.5,
                 f"{faithfulness(good, chunks):.1f} and {faithfulness(bad, chunks):.1f}")
    return ok


# ----------------------------------------------------------------------------
def part4_index():
    print("\nPart 4: vector indexes\n")
    rng = np.random.default_rng(0)
    ok = True
    X = rng.standard_normal((3000, 32)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    Q = X[:20] + 0.1 * rng.standard_normal((20, 32)).astype(np.float32)
    S = Q @ X.T
    ok &= report("argpartition top-k == full argsort top-k",
                 np.array_equal(topk(S, 10), np.argsort(-S, axis=1, kind="stable")[:, :10]))
    exact = ExactIndex(X).search(Q, 10)
    ivf = IVFIndex(X, 16, iters=5)
    ok &= report("IVF with nprobe = nlist == exact search",
                 np.array_equal(ivf.search(Q, 10, nprobe=16), exact), "16 lists, all probed")
    pq = PQIndex(X, m=8, ks=16, iters=5)
    adc = pq.adc_scores(Q[0])
    ok &= report("PQ lookup-table scores == q · reconstructed vectors",
                 np.allclose(adc, pq.reconstruct() @ Q[0], atol=1e-5),
                 f"max diff {np.abs(adc - pq.reconstruct() @ Q[0]).max():.1e}")
    _, hist = kmeans(X, 20, iters=8, seed=0, spherical=True)
    ok &= report("spherical k-means objective never decreases", bool(np.all(np.diff(hist) >= -1e-6)),
                 f"mean cosine {hist[0]:.3f} -> {hist[-1]:.3f}")
    return ok


def main():
    results = [part1_eval_set(), part2_gradients(), part3_retrieval(), part4_index()]
    ok = all(results)
    print("\nGradient check", "PASSED" if ok else "FAILED")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
