"""
The tool router in PyTorch.

1. Equivalence: builds the same router (Linear -> ReLU -> Linear) with nn.Linear,
   copies the NumPy weights from router_numpy.py into it, and checks in float64 that
       - the logits and the loss are identical, and
       - autograd's gradients equal our hand-written backward() for every parameter.
2. The sparse view: a bag of hashed features times W1 is just a SUM of rows of W1,
   which is exactly nn.EmbeddingBag(mode="sum"). Checks that it gives the same
   hidden layer as the dense matmul, without ever building the (N, D) matrix. This is
   how production text classifiers (fastText, Joulin et al. 2016) handle millions
   of hash buckets.
3. Idiomatic training at a larger scale: EmbeddingBag over 2^16 buckets, dropout,
   AdamW, a validation split carved out of the training set, early stopping on the
   validation loss, then test accuracy and per-request routing latency.

Run (~4 s):
    python agents/router_pytorch.py
"""
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from router_numpy import LABELS, Router, feature_ids, featurize, load_jsonl, train

warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)


class TorchRouter(nn.Module):
    def __init__(self, D, hidden, n_out=len(LABELS)):
        super().__init__()
        self.fc1 = nn.Linear(D, hidden)
        self.fc2 = nn.Linear(hidden, n_out)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


def part1_equivalence():
    print("1. NumPy router == PyTorch router (float64)\n")
    texts, y = load_jsonl("router_train.jsonl")
    texts, y = texts[:64], y[:64]
    D, H = 2048, 64
    npm = Router(D=D, hidden=H, seed=0, dtype=np.float64)
    rng = np.random.default_rng(1)
    for p in npm.params().values():  # non-trivial biases too
        p += rng.standard_normal(p.shape) * 0.05
    X = featurize(texts, D, dtype=np.float64)
    loss_np = npm.loss(X, y)
    grads_np = npm.backward()

    tm = TorchRouter(D, H).double()
    with torch.no_grad():  # nn.Linear stores W as (out, in): transpose ours
        tm.fc1.weight.copy_(torch.from_numpy(npm.fc1.params["W"].T))
        tm.fc1.bias.copy_(torch.from_numpy(npm.fc1.params["b"]))
        tm.fc2.weight.copy_(torch.from_numpy(npm.fc2.params["W"].T))
        tm.fc2.bias.copy_(torch.from_numpy(npm.fc2.params["b"]))
    logits = tm(torch.from_numpy(X))
    loss_t = F.cross_entropy(logits, torch.from_numpy(y))
    loss_t.backward()
    pairs = [("logits", npm.logits, logits.detach().numpy()),
             ("loss", np.array(loss_np), loss_t.detach().numpy()),
             ("fc1.W", grads_np["fc1.W"], tm.fc1.weight.grad.numpy().T),
             ("fc1.b", grads_np["fc1.b"], tm.fc1.bias.grad.numpy()),
             ("fc2.W", grads_np["fc2.W"], tm.fc2.weight.grad.numpy().T),
             ("fc2.b", grads_np["fc2.b"], tm.fc2.bias.grad.numpy())]
    ok = True
    for name, a, b in pairs:
        err = float(np.abs(a - b).max())
        ok &= err < 1e-10
        what = "value" if name in ("logits", "loss") else "grad "
        print(f"   {what} {name:7s} max |numpy - torch| = {err:.1e}")
    print("   (the loss differs by ~1e-12 only because our NumPy loss adds 1e-12 inside the log)")
    print(f"   -> {'identical' if ok else 'MISMATCH'}\n")
    return ok, npm, tm


def bag(texts, D, dtype=torch.float32):
    """Flat ids + offsets + weights for nn.EmbeddingBag (the same 1/sqrt(n) scaling as featurize)."""
    ids, offsets, weights = [], [], []
    for t in texts:
        f = feature_ids(t, D)
        offsets.append(len(ids))
        ids += f
        weights += [1.0 / float(np.sqrt(len(f)))] * len(f)
    return torch.tensor(ids), torch.tensor(offsets), torch.tensor(weights, dtype=dtype)


def part2_embedding_bag(npm, tm):
    print("2. Bag of hashed features @ W1  ==  nn.EmbeddingBag(mode='sum')\n")
    texts, _ = load_jsonl("router_test.jsonl")
    texts = texts[:50]
    D = npm.config["D"]
    dense = torch.from_numpy(featurize(texts, D, dtype=np.float64)) @ tm.fc1.weight.detach().T
    eb = nn.EmbeddingBag(D, tm.fc1.out_features, mode="sum").double()
    with torch.no_grad():
        eb.weight.copy_(tm.fc1.weight.T)  # row i = the hidden-layer contribution of bucket i
    ids, offsets, w = bag(texts, D, dtype=torch.float64)
    with torch.no_grad():
        sparse = eb(ids, offsets, per_sample_weights=w)
    err = float((dense - sparse).abs().max())
    nnz = len(ids) / len(texts)
    print(f"   max |dense - EmbeddingBag| = {err:.1e}; on average a request touches {nnz:.0f} of "
          f"{D} buckets, so the sparse version does {D / nnz:.0f}x less work\n")
    return err < 1e-10


class BagRouter(nn.Module):
    def __init__(self, D, hidden, n_out=len(LABELS), dropout=0.2):
        super().__init__()
        self.emb = nn.EmbeddingBag(D, hidden, mode="sum")
        self.b1 = nn.Parameter(torch.zeros(hidden))
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden, n_out)

    def forward(self, ids, offsets, w):
        return self.out(self.drop(F.relu(self.emb(ids, offsets, per_sample_weights=w) + self.b1)))


def part3_idiomatic(D=2 ** 16, hidden=128, epochs=15, batch=64, patience=5, seed=0):
    print(f"3. Idiomatic training: EmbeddingBag over D = 2^16 = {D:,} buckets, hidden {hidden}, "
          "dropout, AdamW, early stopping\n")
    torch.manual_seed(seed)
    texts, y = load_jsonl("router_train.jsonl")
    te_texts, yte = load_jsonl("router_test.jsonl")
    perm = np.random.default_rng(seed).permutation(len(texts))
    n_val = len(texts) // 10
    va, tr = perm[:n_val], perm[n_val:]
    tr_texts, ytr = [texts[i] for i in tr], torch.tensor(y[tr])
    va_bag, yva = bag([texts[i] for i in va], D), torch.tensor(y[va])
    te_bag = bag(te_texts, D)
    model = BagRouter(D, hidden)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    best, best_state, bad, t0 = float("inf"), None, 0, time.time()
    for ep in range(epochs):
        model.train()
        order = torch.randperm(len(tr_texts))
        for i in range(0, len(order), batch):
            idx = order[i:i + batch]
            ids, off, w = bag([tr_texts[j] for j in idx], D)
            loss = F.cross_entropy(model(ids, off, w), ytr[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = F.cross_entropy(model(*va_bag), yva).item()
            va_acc = (model(*va_bag).argmax(1) == yva).float().mean().item()
        if ep < 3 or ep % 5 == 4:
            print(f"   epoch {ep + 1:2d}  train loss {loss.item():.4f}  val loss {vl:.4f}  val acc {va_acc:.3f}")
        if vl < best - 1e-4:
            best, best_state, bad = vl, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                print(f"   early stop at epoch {ep + 1} (val loss hasn't improved for {patience} epochs)")
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(*te_bag).argmax(1).numpy()
    acc = float((pred == yte).mean())
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n   trained in {time.time() - t0:.1f} s, {n_params:,} parameters (almost all in the bag table)")
    np_model, _ = train(verbose=False)
    np_acc = float((np_model.forward(featurize(te_texts)).argmax(1) == yte).mean())
    print(f"   test accuracy (held-out phrasings): PyTorch EmbeddingBag {acc:.3f}  vs  NumPy router {np_acc:.3f}")
    print(f"   note: validation accuracy {va_acc:.3f} vs test {acc:.3f}. The validation split comes from the SAME "
          "templates\n   as training, the test set from held-out phrasings: a validation set that doesn't "
          "look like\n   production traffic tells you nothing about it.")
    one = bag(["Convert 3 GiB to MB"], D)
    with torch.no_grad():
        for _ in range(20):
            model(*one)
        t0 = time.perf_counter()
        for _ in range(500):
            model(*one)
    us = (time.perf_counter() - t0) / 500 * 1e6
    print(f"   routing one request takes {us:.0f} µs on this CPU, vs hundreds of ms for one LLM call: that's "
          "why\n   production systems put small classifiers in front of big models")
    return acc


def main():
    ok1, npm, tm = part1_equivalence()
    ok2 = part2_embedding_bag(npm, tm)
    part3_idiomatic()
    if not (ok1 and ok2):
        raise SystemExit("equivalence check FAILED")


if __name__ == "__main__":
    main()
