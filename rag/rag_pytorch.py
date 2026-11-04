"""
The same bi-encoder in PyTorch: first prove it is the same function as the NumPy one,
then train it the idiomatic way at a larger scale, with BM25-mined hard negatives.

1. Equivalence. nn.EmbeddingBag(mode="sum", per_sample_weights=...) is exactly our
   weighted mean of hashed-feature rows (the weights already sum to 1), nn.Linear +
   tanh + a residual is our MLP, F.normalize is our L2 step, and F.cross_entropy on
   Q Pᵀ / τ is InfoNCE. We copy the trained NumPy weights in (float64) and assert:
     - identical embeddings for every chunk and every qa.jsonl question,
     - identical gradients (autograd vs our hand-written backward) for every
       parameter, through the full InfoNCE loss on a real batch.
2. Idiomatic training: sparse embedding gradients (nn.EmbeddingBag(sparse=True) +
   torch.optim.SparseAdam, the lazy Adam we wrote by hand), AdamW for the dense layers,
   a cosine LR schedule, a larger batch (256), twice as many pairs per epoch, and
   HARD NEGATIVES mined with BM25: for every query, the training paragraph that BM25
   ranks highest among the ones from OTHER sections. In-batch negatives are mostly
   easy (a random paragraph about CNNs vs a question about LSTMs); a hard negative
   shares the query's words but is the wrong passage, which is exactly the confusion
   a dense retriever must learn to resolve (DPR, Karpukhin et al. 2020).
   We train with and without hard negatives and report the retrieval metrics on
   qa.jsonl, whatever they turn out to be.

Run (~70 s):
    python rag/rag_pytorch.py
    python rag/rag_pytorch.py --steps 600 --skip-baseline
"""
from __future__ import annotations

import argparse
import math
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from bm25 import BM25, tokenize
from chunking import get_chunks
from corpus import load_qa
from embed_numpy import (Encoder, corpus_units, feature_idf, get_embedder, group_mask, info_nce,
                         make_pairs, split_groups, text_tokens)
from retrieval_eval import Retriever, load_eval, run_all, metric_values

warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)


class BiEncoder(nn.Module):
    def __init__(self, n_buckets, dim, sparse=False):
        super().__init__()
        self.bag = nn.EmbeddingBag(n_buckets, dim, mode="sum", sparse=sparse)
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim, bias=False)

    def forward(self, ids, offsets, weights):
        x = self.bag(ids, offsets, per_sample_weights=weights)  # Σ_j w_j E[id_j]
        z = x + self.fc2(torch.tanh(self.fc1(x)))
        return F.normalize(z, dim=-1)


def to_torch(bags, dtype):
    return (torch.from_numpy(bags.ids), torch.from_numpy(bags.offsets[:-1]),
            torch.from_numpy(bags.w).to(dtype))


def load_numpy_weights(model: BiEncoder, enc: Encoder):
    """Ours are row-vector (x @ W); nn.Linear stores the transpose (W x)."""
    p = enc.params
    with torch.no_grad():
        model.bag.weight.copy_(torch.from_numpy(p["E"]))
        model.fc1.weight.copy_(torch.from_numpy(p["W1"].T))
        model.fc1.bias.copy_(torch.from_numpy(p["b1"]))
        model.fc2.weight.copy_(torch.from_numpy(p["W2"].T))


def to_numpy_encoder(model: BiEncoder, like: Encoder) -> Encoder:
    enc = Encoder(like.config["n_buckets"], like.config["dim"], idf=like.idf, tau=like.tau,
                  trigram_weight=like.trigram_weight)
    enc.params["E"] = model.bag.weight.detach().float().numpy().copy()
    enc.params["W1"] = model.fc1.weight.detach().float().numpy().T.copy()
    enc.params["b1"] = model.fc1.bias.detach().float().numpy().copy()
    enc.params["W2"] = model.fc2.weight.detach().float().numpy().T.copy()
    return enc


def infonce_torch(q, d, tau, mask=None):
    S = q @ d.T / tau
    if mask is not None:
        S = S.masked_fill(torch.from_numpy(mask), float("-inf"))
    return F.cross_entropy(S, torch.arange(len(q)))


# ----------------------------------------------------------------------------
def part1_equivalence(enc32):
    print("1. PyTorch == our NumPy bi-encoder (float64, trained weights from embedder.npz)\n")
    enc = Encoder(enc32.config["n_buckets"], enc32.config["dim"], dtype=np.float64, idf=enc32.idf,
                  tau=enc32.tau, trigram_weight=enc32.trigram_weight)
    for k in enc.params:
        enc.params[k] = enc32.params[k].astype(np.float64)
    model = BiEncoder(enc.config["n_buckets"], enc.config["dim"]).double()
    load_numpy_weights(model, enc)

    texts = [c.text for c in get_chunks()] + [q.question for q in load_qa()]
    bags = enc.bags(texts)
    e_np = enc.forward(bags)
    e_pt = model(*to_torch(bags, torch.float64)).detach().numpy()
    diff = np.abs(e_np - e_pt).max()
    print(f"  embeddings of {len(texts)} texts (261 chunks + 63 questions): max |diff| = {diff:.1e}")
    assert diff < 1e-12

    rng = np.random.default_rng(0)
    units = corpus_units()
    pairs = make_pairs(units, rng)[:64]
    bags = enc.bags([p[0] for p in pairs] + [p[1] for p in pairs])
    mask = group_mask([p[2] for p in pairs], [p[2] for p in pairs])
    e = enc.forward(bags)
    loss_np, dq, dd, _ = info_nce(e[:64], e[64:], enc.tau, mask)
    g_np = enc.backward(np.concatenate([dq, dd]), dense_E=True)

    model.zero_grad()
    e_t = model(*to_torch(bags, torch.float64))
    loss_t = infonce_torch(e_t[:64], e_t[64:], enc.tau, mask)
    loss_t.backward()
    print(f"  InfoNCE on 64 real pairs: NumPy {loss_np:.10f}  PyTorch {loss_t.item():.10f}")
    assert abs(loss_np - loss_t.item()) < 1e-10
    pairs_t = [("E", model.bag.weight.grad, g_np["E"]), ("W1", model.fc1.weight.grad.T, g_np["W1"]),
               ("b1", model.fc1.bias.grad, g_np["b1"]), ("W2", model.fc2.weight.grad.T, g_np["W2"])]
    for name, gt, gn in pairs_t:
        d = np.abs(gt.numpy() - gn).max()
        rel = d / (np.abs(gn).max() + 1e-30)
        print(f"  autograd d{name:2s} == hand-written backward: max |diff| = {d:.1e} (relative {rel:.1e})")
        assert rel < 1e-9
    print("  ✔ same function, same gradients\n")


# ----------------------------------------------------------------------------
class HardNegativeMiner:
    """BM25 over the training paragraphs; for a query, the best-scoring paragraph from a
    DIFFERENT section (same-section ones would be false negatives)."""

    def __init__(self, units):
        self.paras = sorted({(u[0], u[2]) for u in units})
        self.groups = np.array([g for g, _ in self.paras])
        self.bm25 = BM25([tokenize(p) for _, p in self.paras])

    def __call__(self, query, group, rng):
        s = self.bm25.score(tokenize(query))
        s[self.groups == group] = -np.inf
        top = np.argsort(-s)[:3]
        top = top[np.isfinite(s[top])]
        if len(top) == 0:
            j = int(rng.integers(len(self.paras)))
        else:
            j = int(top[0])
        return self.paras[j][1], self.paras[j][0]


def train_torch(steps=1000, batch=256, lr=3e-3, tau=0.05, dim=128, hard_negatives=True,
                feat_drop=0.5, trigram_weight=0.5, seed=0, verbose=True):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    units = corpus_units()
    train_units, _ = split_groups(units, 0.1, seed)  # same held-out sections as NumPy
    idf = feature_idf([u[2] for u in units], 2 ** 15)
    enc_np = Encoder(2 ** 15, dim, seed, idf=idf, tau=tau, trigram_weight=trigram_weight)
    model = BiEncoder(2 ** 15, dim, sparse=True)
    load_numpy_weights(model, enc_np)  # same init as the NumPy run
    opt_sparse = torch.optim.SparseAdam([model.bag.weight], lr=lr)
    opt_dense = torch.optim.AdamW([p for n, p in model.named_parameters() if not n.startswith("bag")],
                                  lr=lr, weight_decay=0.01)
    sched = [torch.optim.lr_scheduler.LambdaLR(o, lambda s: max(0.05, 0.5 * (1 + math.cos(math.pi * s / steps))))
             for o in (opt_sparse, opt_dense)]
    miner = HardNegativeMiner(train_units) if hard_negatives else None

    def epoch_pairs():
        # twice the ict / drop samples of the NumPy recipe: more (noisier) pairs per epoch
        ps = make_pairs(train_units, rng) + make_pairs(train_units, rng, ("ict", "drop"))
        out = []
        for q, d, g, kind in ps:
            neg = miner(q, g, rng) if miner else None
            out.append((q, d, g, kind, neg))
        return [out[i] for i in rng.permutation(len(out))]

    t0 = time.time()
    pairs, pos, losses = [], 0, []
    for step in range(1, steps + 1):
        if pos + batch > len(pairs):
            pairs, pos = epoch_pairs(), 0
        bp = pairs[pos:pos + batch]
        pos += batch
        texts = [p[0] for p in bp] + [p[1] for p in bp]
        groups_d = [p[2] for p in bp]
        if hard_negatives:
            texts += [p[4][0] for p in bp]
            groups_d += [p[4][1] for p in bp]
        bags = enc_np.bags(texts, feat_drop, rng)
        e = model(*to_torch(bags, torch.float32))
        n = len(bp)
        mask = group_mask([p[2] for p in bp], groups_d)
        loss = infonce_torch(e[:n], e[n:], tau, mask)
        opt_sparse.zero_grad()
        opt_dense.zero_grad()
        loss.backward()
        opt_sparse.step()
        opt_dense.step()
        for s in sched:
            s.step()
        losses.append(loss.item())
        if verbose and (step % 200 == 0 or step == steps):
            print(f"    step {step:5d} | InfoNCE {np.mean(losses[-50:]):.3f} | {time.time() - t0:5.1f} s")
    return to_numpy_encoder(model, enc_np), time.time() - t0


def retrieval_row(name, enc, qs, rel):
    ret = Retriever(embedder=enc)
    res = run_all(ret, qs, rel, ["dense", "hybrid"])
    r = {m: {k: np.mean(metric_values(res[m], qs, k)) for k in ("recall@5", "mrr", "ndcg@10")}
         for m in res}
    sem = np.mean(metric_values(res["dense"], qs, "recall@5", "semantic"))
    print(f"  {name:44s} {r['dense']['recall@5']:9.3f} {r['dense']['mrr']:7.3f} "
          f"{r['dense']['ndcg@10']:8.3f} {sem:10.3f} {r['hybrid']['recall@5']:11.3f}")
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--skip-baseline", action="store_true",
                    help="don't train the no-hard-negatives PyTorch model")
    args = ap.parse_args()
    torch.set_num_threads(min(8, torch.get_num_threads()))

    enc_np, _ = get_embedder()
    part1_equivalence(enc_np)

    print(f"2. Idiomatic PyTorch training: batch {args.batch}, 2x pairs per epoch, SparseAdam + "
          f"AdamW, cosine LR, {args.steps} steps")
    runs = []
    if not args.skip_baseline:
        print("  (a) in-batch negatives only")
        e_a, t_a = train_torch(args.steps, args.batch, hard_negatives=False)
        runs.append((f"PyTorch, batch {args.batch} ({t_a:.0f} s)", e_a))
    print("  (b) + one BM25-mined hard negative per query")
    e_b, t_b = train_torch(args.steps, args.batch, hard_negatives=True)
    runs.append((f"PyTorch, batch {args.batch} + hard negatives ({t_b:.0f} s)", e_b))

    qs, rel = load_eval()
    print("\nRetrieval on qa.jsonl (57 answerable questions; questions never used in training):")
    print(f"  {'dense model':44s} {'recall@5':>9s} {'MRR':>7s} {'nDCG@10':>8s} "
          f"{'semantic':>10s} {'hybrid R@5':>11s}")
    print(f"  {'':44s} {'':>9s} {'':>7s} {'':>8s} {'R@5':>10s} {'(+BM25)':>11s}")
    retrieval_row("NumPy, batch 128 (embedder.npz)", enc_np, qs, rel)
    for name, e in runs:
        retrieval_row(name, e, qs, rel)
    print("\n  With 57 questions a difference of 0.05 in recall@5 is ~3 questions: within the noise "
          "(see the bootstrap CIs in retrieval_eval.py).")


if __name__ == "__main__":
    main()
