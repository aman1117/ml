"""
Gradient check: compare every hand-written backward() in llama_numpy.py against
numerical gradients from central finite differences (float64):

    dL/dw ≈ (L(w + eps) - L(w - eps)) / (2 * eps)

Part 1 checks each NEW layer on its own (input AND parameters), using the trick
L = Σ (layer(x) ⊙ G) for a fixed random G, so dL/d(output) = G:
    RMSNorm, RoPE, grouped-query attention with n_kv_head = n_head (MHA), 2 (GQA)
    and 1 (MQA), the SwiGLU feed-forward, and one whole LLaMA block.
Part 2 checks EVERY parameter of a tiny full LLaMA through the real loss
(GQA with separate output matrix, MQA with tied embeddings).
Part 3 checks RoPE's two defining properties numerically:
    - it is a rotation (norms unchanged), and
    - the score q_m · k_n depends only on the offset m - n (relative position).

Relative errors should be tiny (< 1e-6). Try breaking a line (e.g. drop the
.sum(axis=2) in Attention.backward, or the minus sign in RoPE.backward) and re-run.

Run:
    python llama/gradient_check.py
"""
import warnings

import numpy as np

from llama_numpy import (LLaMA, Attention, Block, FeedForward, RMSNorm, RoPE, rope_angles,
                         rope_rotate)

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
    """‖a - b‖ / (‖a‖ + ‖b‖) over the whole tensor (see ../transformer/gradient_check.py
    for why not element by element)."""
    return float(np.linalg.norm(a - b) / max(1e-20, np.linalg.norm(a) + np.linalg.norm(b)))


def report(name, shape, err, tol=TOL):
    ok = err < tol
    print(f"  {name:34s} shape={str(shape):14s} relative error = {err:.2e}  "
          f"{'OK' if ok else 'FAIL'}")
    return ok


def randomize(layer, rng, scale=0.5):
    """Bigger-than-init random weights (in place) so no gradient is accidentally ~0."""
    for k, p in layer.params.items():
        p[...] = (1.0 + rng.standard_normal(p.shape) * 0.3 if k == "weight" and p.ndim == 1
                  else rng.standard_normal(p.shape) * scale)


def all_params(layer, prefix=""):
    """{name: (owner_layer, key)} for a layer and its sublayers, recursively."""
    out = {f"{prefix}{k}": (layer, k) for k in layer.params}
    for name, sub in getattr(layer, "sublayers", lambda: {})().items():
        out.update(all_params(sub, f"{prefix}{name}."))
    return out


def check_layer(name, layer, x, rng):
    """L = Σ layer(x) ⊙ G. Checks dL/dx and every parameter (including sublayers')."""
    out = layer.forward(x)
    G = rng.standard_normal(out.shape)
    f = lambda: float(np.sum(layer.forward(x) * G))
    layer.forward(x)
    dx = layer.backward(G)
    ok = report(f"{name}  input x", x.shape, rel_error(dx, numerical_grad(f, x)))
    params = all_params(layer)
    analytic = {pn: owner.grads[key].copy() for pn, (owner, key) in params.items()}
    for pname, (owner, key) in params.items():
        arr = owner.params[key]
        ok &= report(f"{name}  {pname}", arr.shape, rel_error(analytic[pname],
                                                              numerical_grad(f, arr)))
    return ok


def part1_layers(seed=0):
    B, T, C, nh = 2, 5, 16, 4
    print(f"Part 1: each layer on its own (B={B}, T={T}, C={C}, {nh} query heads of size "
          f"{C // nh})\n")
    rng = np.random.default_rng(seed)
    ok = True

    norm = RMSNorm(C, dtype=F64)
    randomize(norm, rng)
    ok &= check_layer("RMSNorm", norm, rng.standard_normal((B, T, C)) * 2 + 0.5, rng)

    ok &= check_layer("RoPE", RoPE(C // nh), rng.standard_normal((B, nh, T, C // nh)), rng)

    for nkv in (nh, 2, 1):
        kind = {nh: "MHA", 1: "MQA"}.get(nkv, "GQA")
        attn = Attention(C, nh, nkv, rng, dtype=F64)
        for sub in attn.sublayers().values():
            randomize(sub, rng)
        x = rng.standard_normal((B, T, C))
        ok &= check_layer(f"Attn {kind} n_kv={nkv}", attn, x, rng)
        # The causal mask: output at position 0 may only depend on input position 0.
        attn.forward(x)
        G = np.zeros((B, T, C))
        G[:, 0] = 1.0
        leak = float(np.abs(attn.backward(G)[:, 1:]).max())
        print(f"  {f'Attn {kind} causal mask':34s} d(out[0])/d(x[1:]) = {leak:.1e}  "
              f"{'OK' if leak == 0 else 'FAIL'}  (the future can't influence the past)")
        ok &= leak == 0

    ffn = FeedForward(C, 24, rng, dtype=F64)
    for sub in ffn.sublayers().values():
        randomize(sub, rng)
    ok &= check_layer("SwiGLU", ffn, rng.standard_normal((B, T, C)), rng)

    blk = Block(C, nh, 2, 24, rng, proj_std=0.5, dtype=F64)
    for _, (owner, _) in all_params(blk).items():
        randomize(owner, rng)
    ok &= check_layer("Block (GQA)", blk, rng.standard_normal((B, T, C)), rng)
    return ok


def part2_full_model(n_kv_head, tie, seed=0):
    rng = np.random.default_rng(seed)
    V, B, T, d, nh = 7, 2, 5, 16, 4
    model = LLaMA(V, n_layer=2, n_head=nh, n_kv_head=n_kv_head, d_model=d, block_size=T,
                  ffn_hidden=16, tie_weights=tie, seed=seed, dtype=F64)
    for name, p in model.params().items():
        p[...] = (1.0 + rng.standard_normal(p.shape) * 0.3 if p.ndim == 1
                  else rng.standard_normal(p.shape) * 0.3)
    idx = rng.integers(0, V, (B, T))
    targets = rng.integers(0, V, (B, T))
    print(f"\nPart 2: every parameter of a tiny LLaMA through the real loss (V={V}, L=2, "
          f"heads={nh}, kv heads={n_kv_head}, d={d}, T={T}, "
          f"{'TIED' if tie else 'separate'} embeddings, {model.n_params()} params)\n")
    model.forward(idx, targets)
    grads = model.backward()
    f = lambda: model.forward(idx, targets)
    ok = True
    for name, p in model.params().items():
        ok &= report(name, p.shape, rel_error(grads[name], numerical_grad(f, p)))
    return ok


def part3_rope_properties(seed=0):
    print("\nPart 3: RoPE properties (head size 16, base 10000)\n")
    rng = np.random.default_rng(seed)
    hs = 16
    q, k = rng.standard_normal(hs), rng.standard_normal(hs)

    def at(v, m):  # the vector v placed at position m
        cos, sin = rope_angles([m], hs, dtype=F64)
        return rope_rotate(v[None], cos, sin)[0]

    norms = [abs(np.linalg.norm(at(q, m)) / np.linalg.norm(q) - 1) for m in (1, 7, 1000)]
    ok = report("rotation keeps the norm", (hs,), max(norms), tol=1e-12)
    # Same offset m - n at very different absolute positions -> same score.
    worst = 0.0
    for offset in (0, 1, 5, 31):
        base = at(q, offset) @ at(k, 0)
        for n in (3, 100, 4321):
            worst = max(worst, abs(at(q, n + offset) @ at(k, n) - base) / abs(base))
    ok &= report("q_m·k_n depends only on m - n", (hs,), worst, tol=1e-9)
    # ...and it does change with the offset (otherwise the check above is vacuous).
    spread = np.ptp([at(q, m) @ at(k, 0) for m in range(8)])
    print(f"  {'score varies with the offset':34s} range over m-n = 0..7: {spread:.2f}  "
          f"{'OK' if spread > 0.1 else 'FAIL'}")
    return ok and spread > 0.1


def main():
    results = [part1_layers(),
               part2_full_model(n_kv_head=2, tie=False),
               part2_full_model(n_kv_head=1, tie=True),
               part3_rope_properties()]
    ok = all(results)
    print("\nGradient check", "PASSED" if ok else "FAILED")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
