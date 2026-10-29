"""
Attention from scratch in NumPy: forward AND hand-written backward.

1. Scaled dot-product attention (Vaswani et al. 2017, the Transformer's attention):

       S   = Q Kᵀ / √d_k              scores      (n, m)   how well each query matches each key
       S   = S + mask (-1e9 where     masked      (n, m)   e.g. padding: never look at <pad>
             a key is forbidden)
       A   = softmax(S, over keys)    weights     (n, m)   each row sums to 1
       Out = A V                      output      (n, d_v) weighted sum of the values

   Backward (given dOut, the gradient of the loss w.r.t. Out):

       dV = Aᵀ dOut
       dA = dOut Vᵀ
       dS = A ⊙ (dA − sum(dA ⊙ A, axis=-1))       softmax Jacobian, row by row
       dQ = dS K / √d_k
       dK = dSᵀ Q / √d_k

   Everything also works with leading batch dimensions: Q (B, n, d), K (B, m, d), ...

2. Additive attention (Bahdanau et al. 2014, the original seq2seq attention):

       score(q, k) = vᵀ tanh(q W_q + k W_k)

   Same softmax and weighted sum afterwards; only the scoring function differs.

3. A tiny model where attention is the WHOLE point: key-value retrieval.
   The input is a set of (key, value) pairs plus a query key, e.g.

       pairs: c→7  k→2  a→5  f→9      query: a     answer: 5

   Model (d = 16, ~1.2k parameters, trained with hand-written backprop and Adam):

       q    = E_key[query] W_q          (B, 1, d)   what am I looking for?
       k_i  = E_key[key_i] W_k          (B, P, d)   what does pair i contain?
       v_i  = E_val[value_i]            (B, P, d)   what does pair i give me if picked?
       ctx  = attention(q, k, v, mask)  (B, 1, d)   soft dictionary lookup
       p    = softmax(ctx W_o + b_o)    (B, 10)     which value digit?

   Without attention the model would have to average the pairs and could only
   guess (about 1 / #pairs). With attention it learns to put almost all weight
   on the matching key and copy that pair's value.

Run:
    python attention/attention_numpy.py
    python attention/attention_numpy.py --no-attention      (ablation: uniform weights)
    python attention/attention_numpy.py --dim 4 --max-pairs 12
"""
import argparse
import os
import warnings

import numpy as np

# NumPy 2.0 + Apple Accelerate raises spurious "divide by zero / overflow /
# invalid value encountered in matmul" warnings even when results are correct.
# Only silence those; real numerical warnings (exp, log, ...) still show.
warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)

NEG = -1e9  # "minus infinity" for masked scores: exp(NEG - max) is exactly 0


def softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))  # subtract max for numerical stability
    return e / e.sum(axis=axis, keepdims=True)


def softmax_backward(dA, A):
    """Gradient through a row-wise softmax: dS = A ⊙ (dA − Σ_j dA_j A_j)."""
    return A * (dA - np.sum(dA * A, axis=-1, keepdims=True))


def swap(x):
    """Transpose the last two axes (works with or without a batch dimension)."""
    return np.swapaxes(x, -1, -2)


# ---------------------------------------------------------------- scaled dot-product

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q (..., n, d)   K (..., m, d)   V (..., m, d_v)
    mask: bool, broadcastable to (..., n, m). True = may attend, False = blocked.
    Returns Out (..., n, d_v), A (..., n, m) and a cache for the backward pass.
    """
    d = Q.shape[-1]
    S = Q @ swap(K) / np.sqrt(d)  # (..., n, m)
    if mask is not None:
        S = np.where(mask, S, NEG)
    A = softmax(S)  # (..., n, m), rows sum to 1
    Out = A @ V  # (..., n, d_v)
    return Out, A, (Q, K, V, A)


def scaled_dot_product_attention_backward(dOut, cache):
    Q, K, V, A = cache
    d = Q.shape[-1]
    dV = swap(A) @ dOut  # (..., m, d_v)
    dA = dOut @ swap(V)  # (..., n, m)
    dS = softmax_backward(dA, A)  # masked entries have A = 0, so dS = 0 there
    dQ = dS @ K / np.sqrt(d)  # (..., n, d)
    dK = swap(dS) @ Q / np.sqrt(d)  # (..., m, d)
    return dQ, dK, dV


# ---------------------------------------------------------------- additive (Bahdanau)

def additive_attention(q, K, V, W_q, W_k, v, mask=None):
    """
    Bahdanau attention: score(q_i, k_j) = vᵀ tanh(q_i W_q + k_j W_k).
    q (B, n, d_q)   K (B, m, d_k)   V (B, m, d_v)
    W_q (d_q, d_a)  W_k (d_k, d_a)  v (d_a,)
    """
    Qp = q @ W_q  # (B, n, d_a)
    Kp = K @ W_k  # (B, m, d_a)
    H = np.tanh(Qp[:, :, None, :] + Kp[:, None, :, :])  # (B, n, m, d_a): every query-key pair
    S = H @ v  # (B, n, m)
    if mask is not None:
        S = np.where(mask, S, NEG)
    A = softmax(S)
    Out = A @ V  # (B, n, d_v)
    return Out, A, (q, K, V, W_q, W_k, v, H, A)


def additive_attention_backward(dOut, cache):
    q, K, V, W_q, W_k, v, H, A = cache
    dV = swap(A) @ dOut
    dS = softmax_backward(dOut @ swap(V), A)  # (B, n, m)
    dv = np.einsum("bnm,bnma->a", dS, H)
    dZ = dS[..., None] * v * (1.0 - H ** 2)  # through vᵀ(.) and tanh: (B, n, m, d_a)
    dQp = dZ.sum(axis=2)  # each query appears in m pairs: (B, n, d_a)
    dKp = dZ.sum(axis=1)  # each key appears in n pairs:   (B, m, d_a)
    dW_q = q.reshape(-1, q.shape[-1]).T @ dQp.reshape(-1, dQp.shape[-1])
    dW_k = K.reshape(-1, K.shape[-1]).T @ dKp.reshape(-1, dKp.shape[-1])
    return dQp @ W_q.T, dKp @ W_k.T, dV, dW_q, dW_k, dv


# ---------------------------------------------------------------- key-value retrieval task

KEYS = "abcdefghijklmnop"  # 16 possible key symbols
VALUES = "0123456789"  # 10 possible value symbols


def make_kv_batch(B, rng, min_pairs=2, max_pairs=8):
    """
    Returns keys (B, P), vals (B, P), mask (B, P) [True = real pair], query (B,), answer (B,)
    and target (B,) = index of the matching pair. P = max_pairs; short sets are padded.
    """
    P = max_pairs
    keys = np.zeros((B, P), dtype=np.int64)
    vals = np.zeros((B, P), dtype=np.int64)
    mask = np.zeros((B, P), dtype=bool)
    query = np.zeros(B, dtype=np.int64)
    target = np.zeros(B, dtype=np.int64)
    for b in range(B):
        n = int(rng.integers(min_pairs, max_pairs + 1))
        keys[b, :n] = rng.choice(len(KEYS), n, replace=False)  # distinct keys in one example
        vals[b, :n] = rng.integers(0, len(VALUES), n)
        mask[b, :n] = True
        target[b] = rng.integers(n)
        query[b] = keys[b, target[b]]
    answer = vals[np.arange(B), target]
    return keys, vals, mask, query, answer, target


def kv_describe(keys, vals, mask, query, b):
    pairs = "  ".join(f"{KEYS[k]}→{VALUES[v]}" for k, v, m in zip(keys[b], vals[b], mask[b]) if m)
    return f"{pairs}   query: {KEYS[query[b]]}"


class KVModel:
    """embed → attention(query, pairs) → linear → softmax.  See the module docstring."""

    def __init__(self, dim=16, seed=0):
        rng = np.random.default_rng(seed)
        d = dim
        self.params = {
            "E_key": rng.standard_normal((len(KEYS), d)) * 0.5,
            "W_q": rng.standard_normal((d, d)) / np.sqrt(d),
            "W_k": rng.standard_normal((d, d)) / np.sqrt(d),
            "E_val": rng.standard_normal((len(VALUES), d)) * 0.5,
            "W_o": rng.standard_normal((d, len(VALUES))) / np.sqrt(d),
            "b_o": np.zeros(len(VALUES)),
        }

    def forward(self, keys, vals, mask, query):
        P = self.params
        eq = P["E_key"][query][:, None, :]  # (B, 1, d)  embedding of the query symbol
        ek = P["E_key"][keys]  # (B, P, d)  same table: a key is a key
        q = eq @ P["W_q"]  # (B, 1, d)
        k = ek @ P["W_k"]  # (B, P, d)
        v = P["E_val"][vals]  # (B, P, d)
        ctx, A, att_cache = scaled_dot_product_attention(q, k, v, mask[:, None, :])
        logits = ctx[:, 0, :] @ P["W_o"] + P["b_o"]  # (B, 10)
        cache = (keys, vals, query, eq, ek, ctx, att_cache)
        return logits, A[:, 0, :], cache

    def loss(self, keys, vals, mask, query, answer):
        logits, _, _ = self.forward(keys, vals, mask, query)
        p = softmax(logits)
        return -np.mean(np.log(p[np.arange(len(answer)), answer] + 1e-12))

    def loss_and_grads(self, keys, vals, mask, query, answer):
        P = self.params
        logits, A, cache = self.forward(keys, vals, mask, query)
        keys, vals, query, eq, ek, ctx, att_cache = cache
        B = len(answer)
        p = softmax(logits)
        loss = -np.mean(np.log(p[np.arange(B), answer] + 1e-12))

        g = {name: np.zeros_like(w) for name, w in P.items()}
        dlogits = p.copy()
        dlogits[np.arange(B), answer] -= 1.0  # softmax + cross-entropy: p - onehot
        dlogits /= B
        g["W_o"] = ctx[:, 0, :].T @ dlogits
        g["b_o"] = dlogits.sum(axis=0)
        dctx = (dlogits @ P["W_o"].T)[:, None, :]  # (B, 1, d)

        dq, dk, dv = scaled_dot_product_attention_backward(dctx, att_cache)
        g["W_q"] = eq.reshape(-1, eq.shape[-1]).T @ dq.reshape(-1, dq.shape[-1])
        g["W_k"] = ek.reshape(-1, ek.shape[-1]).T @ dk.reshape(-1, dk.shape[-1])
        deq = dq @ P["W_q"].T  # (B, 1, d)
        dek = dk @ P["W_k"].T  # (B, P, d)
        # Embedding lookups backward: add each row's gradient into the table row it came from
        # (np.add.at handles repeated indices correctly).
        np.add.at(g["E_key"], query, deq[:, 0, :])
        np.add.at(g["E_key"], keys, dek)
        np.add.at(g["E_val"], vals, dv)
        return loss, g, A


class Adam:
    def __init__(self, params, lr=0.01, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps, self.t = lr, b1, b2, eps, 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params, grads):
        self.t += 1
        for k in params:
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * grads[k]
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * grads[k] ** 2
            m_hat = self.m[k] / (1 - self.b1 ** self.t)
            v_hat = self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def evaluate(model, n=2000, seed=123, max_pairs=8):
    keys, vals, mask, query, answer, target = make_kv_batch(n, np.random.default_rng(seed),
                                                            max_pairs=max_pairs)
    logits, A, _ = model.forward(keys, vals, mask, query)
    acc = float(np.mean(logits.argmax(1) == answer))
    w_correct = float(np.mean(A[np.arange(n), target]))
    return acc, w_correct


def train(dim=16, steps=600, batch=64, lr=0.003, max_pairs=8, seed=0, print_every=50,
          no_attention=False, verbose=True):
    """Returns the trained model and a history of (step, loss, acc, weight on the right pair)."""
    rng = np.random.default_rng(seed)
    model = KVModel(dim, seed)
    if no_attention:
        # q = 0 makes every score 0, so the weights are uniform over the real pairs:
        # the model can only AVERAGE the values, it can't look anything up.
        model.params["W_q"][:] = 0.0
    opt = Adam(model.params, lr)
    history = []
    for step in range(steps + 1):
        keys, vals, mask, query, answer, target = make_kv_batch(batch, rng, max_pairs=max_pairs)
        loss, grads, A = model.loss_and_grads(keys, vals, mask, query, answer)
        if no_attention:
            grads["W_q"][:] = 0.0
        opt.step(model.params, grads)
        if step % 50 == 0:
            acc, w = evaluate(model, n=500, max_pairs=max_pairs)
            history.append((step, loss, acc, w))
            if verbose and step % print_every == 0:
                print(f"step {step:5d} | loss {loss:.3f} | test acc {acc:6.1%} | "
                      f"avg attention on the matching key {w:.2f}")
    return model, np.array(history)


def bars(weights, width=30):
    return ["█" * int(round(w * width)) + "·" * (width - int(round(w * width))) for w in weights]


def show_examples(model, n=3, seed=7, max_pairs=8):
    keys, vals, mask, query, answer, target = make_kv_batch(n, np.random.default_rng(seed),
                                                            min_pairs=4, max_pairs=max_pairs)
    logits, A, _ = model.forward(keys, vals, mask, query)
    for b in range(n):
        print(f"\n  {kv_describe(keys, vals, mask, query, b)}")
        print(f"  answer {VALUES[answer[b]]}, model says {VALUES[logits[b].argmax()]}  "
              f"(p = {softmax(logits[b])[logits[b].argmax()]:.2f})")
        for j, line in enumerate(bars(A[b])):
            if not mask[b, j]:
                continue
            mark = "  ← matching key" if j == target[b] else ""
            print(f"    {KEYS[keys[b, j]]}→{VALUES[vals[b, j]]}  {line} {A[b, j]:.3f}{mark}")


def save_model(path, model):
    np.savez(path, **model.params)


def load_model(path):
    d = np.load(path)
    model = KVModel(dim=d["W_q"].shape[0])
    for k in model.params:
        model.params[k] = d[k]
    return model


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dim", type=int, default=16, help="embedding / attention size d")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.003)
    ap.add_argument("--max-pairs", type=int, default=8, help="up to this many (key, value) pairs")
    ap.add_argument("--no-attention", action="store_true",
                    help="ablation: freeze q = 0, so attention is a plain average of the values")
    ap.add_argument("--save", default=os.path.join(here, "kv_model.npz"))
    args = ap.parse_args()

    print(f"Key-value retrieval: {args.max_pairs} pairs max, {len(KEYS)} key symbols, "
          f"{len(VALUES)} value symbols, d = {args.dim}")
    model = KVModel(args.dim)
    n_params = sum(p.size for p in model.params.values())
    print(f"{n_params:,} parameters. Random guessing: loss ln(10) = {np.log(10):.3f}, "
          f"acc 10%\n")
    if args.no_attention:
        print("ABLATION: q = 0, so every pair gets the same weight (a plain average).\n")
    model, _ = train(args.dim, args.steps, args.batch, args.lr, args.max_pairs,
                     no_attention=args.no_attention)

    acc, w = evaluate(model, max_pairs=args.max_pairs)
    print(f"\nfinal test accuracy on 2000 new examples: {acc:.1%}  "
          f"(average attention weight on the matching key: {w:.3f})")
    print("\nWhat the model looks at (attention weights over the pairs):")
    show_examples(model, max_pairs=args.max_pairs)

    if args.no_attention:
        print("\n(Ablation model not saved.)")
        return
    save_model(args.save, model)
    print(f"\nSaved model to {os.path.relpath(args.save)}.")


if __name__ == "__main__":
    main()
