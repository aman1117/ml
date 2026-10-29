"""
Gradient check: compare the hand-written attention backward passes in
attention_numpy.py against numerical gradients from central differences:

    dL/dw ≈ (L(w + eps) - L(w - eps)) / (2 * eps)

Checked:
  1. scaled dot-product attention: dQ, dK, dV (batched), without and with a padding
     mask and with a causal mask,
  2. additive (Bahdanau) attention: dq, dK, dV, dW_q, dW_k, dv (with a mask),
  3. every parameter of the key-value retrieval model (end to end, through the
     embedding lookups, the attention and the softmax + cross-entropy).

For 1 and 2 the "loss" is L = Σ Out ⊙ R with a fixed random R, so dOut = R.
Everything is float64. Try breaking a line of a backward function and re-run.

Run:
    python attention/gradient_check.py
"""
import numpy as np

from attention_numpy import (KVModel, additive_attention, additive_attention_backward,
                             make_kv_batch, scaled_dot_product_attention,
                             scaled_dot_product_attention_backward)

TOL = 1e-6


def numerical_grad(f, param, eps=1e-5):
    grad = np.zeros_like(param)
    it = np.nditer(param, flags=["multi_index"])
    for _ in it:
        idx = it.multi_index
        old = param[idx]
        param[idx] = old + eps
        lp = f()
        param[idx] = old - eps
        lm = f()
        param[idx] = old
        grad[idx] = (lp - lm) / (2 * eps)
    return grad


def rel_error(a, b):
    return np.max(np.abs(a - b) / np.maximum(1e-12, np.abs(a) + np.abs(b)))


def report(label, name, param, ana, num):
    err = rel_error(ana, num)
    ok = err < TOL
    print(f"  {label:<22} {name:<6} shape={str(param.shape):12s} "
          f"max relative error = {err:.2e}  {'OK' if ok else 'FAIL'}")
    return ok


def check_sdpa(label, mask, rng, B=2, n=3, m=5, d=4, dv=3):
    Q = rng.standard_normal((B, n, d))
    K = rng.standard_normal((B, m, d))
    V = rng.standard_normal((B, m, dv))
    R = rng.standard_normal((B, n, dv))  # L = Σ Out ⊙ R  →  dOut = R

    def f():
        return np.sum(scaled_dot_product_attention(Q, K, V, mask)[0] * R)

    _, _, cache = scaled_dot_product_attention(Q, K, V, mask)
    dQ, dK, dV = scaled_dot_product_attention_backward(R, cache)
    ok = True
    for name, p, g in [("dQ", Q, dQ), ("dK", K, dK), ("dV", V, dV)]:
        ok &= report(label, name, p, g, numerical_grad(f, p))
    return ok


def check_additive(rng, B=2, n=3, m=4, dq=5, dk=4, da=6, dv=3):
    q = rng.standard_normal((B, n, dq))
    K = rng.standard_normal((B, m, dk))
    V = rng.standard_normal((B, m, dv))
    W_q = rng.standard_normal((dq, da)) * 0.5
    W_k = rng.standard_normal((dk, da)) * 0.5
    v = rng.standard_normal(da)
    mask = np.array([[True, True, True, True], [True, True, False, False]])[:, None, :]
    R = rng.standard_normal((B, n, dv))

    def f():
        return np.sum(additive_attention(q, K, V, W_q, W_k, v, mask)[0] * R)

    _, _, cache = additive_attention(q, K, V, W_q, W_k, v, mask)
    grads = additive_attention_backward(R, cache)
    ok = True
    for name, p, g in zip(["dq", "dK", "dV", "dW_q", "dW_k", "dv"],
                          [q, K, V, W_q, W_k, v], grads):
        ok &= report("additive + mask", name, p, g, numerical_grad(f, p))
    return ok


def check_kv_model(rng):
    model = KVModel(dim=6, seed=1)
    for k in model.params:  # bigger, non-zero weights so every path carries gradient
        model.params[k] = rng.standard_normal(model.params[k].shape) * 0.7
    keys, vals, mask, query, answer, _ = make_kv_batch(6, rng, min_pairs=2, max_pairs=5)

    def f():
        return model.loss(keys, vals, mask, query, answer)

    _, grads, _ = model.loss_and_grads(keys, vals, mask, query, answer)
    ok = True
    for name, p in model.params.items():
        ok &= report("KV retrieval model", name, p, grads[name], numerical_grad(f, p))
    return ok


def main():
    rng = np.random.default_rng(0)
    padding = np.array([[True, True, True, False, False],
                        [True, True, True, True, False]])[:, None, :]  # (B, 1, m): per-example lengths
    causal = np.tril(np.ones((5, 5), dtype=bool))  # (n, m): query i sees keys 0..i

    print("Scaled dot-product attention  softmax(Q Kᵀ / √d) V")
    ok = check_sdpa("no mask", None, rng)
    ok &= check_sdpa("padding mask", padding, rng)
    ok &= check_sdpa("causal mask (self)", causal, rng, n=5)
    print("\nAdditive (Bahdanau) attention  vᵀ tanh(q W_q + k W_k)")
    ok &= check_additive(rng)
    print("\nKey-value retrieval model (embeddings → attention → linear → cross-entropy)")
    ok &= check_kv_model(rng)

    print("\nGradient check", "PASSED" if ok else "FAILED")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
