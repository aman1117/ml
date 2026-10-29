"""
Gradient check: compare every hand-written backward() in transformer_numpy.py
against numerical gradients from central finite differences (float64):

    dL/dw ≈ (L(w + eps) - L(w - eps)) / (2 * eps)

Part 1 checks each layer ON ITS OWN (input AND parameters), using the trick
L = Σ (layer(x) ⊙ G) for a fixed random G, so dL/d(output) = G:
    Linear, Embedding, LayerNorm, GELU, causal multi-head attention (with the
    mask), softmax + cross-entropy, and one whole Transformer block.
Part 2 checks EVERY parameter of a tiny full GPT through the real loss, with
separate and with tied input/output embeddings.

If backward() is correct the relative error is tiny (< 1e-6). Try breaking a
line in a backward() (e.g. drop a term in LayerNorm.backward) and re-run.

Run:
    python transformer/gradient_check.py
"""
import numpy as np

from transformer_numpy import (GELU, GPT, Block, CausalSelfAttention, Embedding,
                               LayerNorm, Linear, cross_entropy)

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
    """Relative error of the whole tensor: ‖a - b‖ / (‖a‖ + ‖b‖).

    Why not element by element? Finite differences can't resolve gradients below
    ~1e-10 (float64 round-off divided by eps), and some entries are legitimately
    ~0 or exactly 0 (e.g. the key bias, see below), which would make an
    element-wise ratio meaningless noise. A real bug moves this number to >1e-3."""
    return float(np.linalg.norm(a - b) / max(1e-20, np.linalg.norm(a) + np.linalg.norm(b)))


def report(name, shape, err):
    ok = err < TOL
    print(f"  {name:28s} shape={str(shape):14s} relative error = {err:.2e}  "
          f"{'OK' if ok else 'FAIL'}")
    return ok


def randomize(layer_params, rng, scale=0.5):
    """Bigger-than-init random weights so no gradient is accidentally ~0."""
    for k, p in layer_params.items():
        if k == "gamma":
            p[...] = 1.0 + rng.standard_normal(p.shape) * 0.3
        else:
            p[...] = rng.standard_normal(p.shape) * scale


def check_layer(name, layer, x, rng, check_input=True):
    """L = Σ layer(x) ⊙ G. Checks dL/dx (if float input) and every param."""
    out = layer.forward(x)
    G = rng.standard_normal(out.shape)
    f = lambda: float(np.sum(layer.forward(x) * G))
    layer.forward(x)
    dx = layer.backward(G)
    ok = True
    if check_input:
        ok &= report(f"{name}  input x", x.shape, rel_error(dx, numerical_grad(f, x)))
    params = dict(layer.params)
    for sub_name, sub in getattr(layer, "sublayers", lambda: {})().items():
        if hasattr(sub, "sublayers"):
            for n2, s2 in sub.sublayers().items():
                params.update({f"{sub_name}.{n2}.{k}": (s2, k) for k in s2.params})
        else:
            params.update({f"{sub_name}.{k}": (sub, k) for k in sub.params})
    for pname, entry in params.items():
        if isinstance(entry, tuple):
            owner, key = entry
            arr, ana = owner.params[key], owner.grads[key]
        else:
            arr, ana = entry, layer.grads[pname]
        ok &= report(f"{name}  {pname}", arr.shape, rel_error(ana, numerical_grad(f, arr)))
    return ok


def part1_layers(seed=0):
    print("Part 1: each layer on its own (B=2, T=5, C=8, 2 heads)\n")
    rng = np.random.default_rng(seed)
    B, T, C, V = 2, 5, 8, 7
    ok = True

    lin = Linear(C, 6, rng, dtype=F64)
    randomize(lin.params, rng)
    ok &= check_layer("Linear", lin, rng.standard_normal((B, T, C)), rng)

    emb = Embedding(V, C, rng, dtype=F64)
    randomize(emb.params, rng)
    idx = rng.integers(0, V, (B, T))
    ok &= check_layer("Embedding", emb, idx, rng, check_input=False)

    ln = LayerNorm(C, dtype=F64)
    randomize(ln.params, rng)
    ok &= check_layer("LayerNorm", ln, rng.standard_normal((B, T, C)) * 2 + 0.5, rng)

    ok &= check_layer("GELU", GELU(), rng.standard_normal((B, T, C)) * 2, rng)

    attn = CausalSelfAttention(C, 2, rng, dtype=F64)
    for sub in attn.sublayers().values():
        randomize(sub.params, rng)
    x = rng.standard_normal((B, T, C))
    ok &= check_layer("Attention", attn, x, rng)
    # Fun fact the check reveals: the KEY bias b_k gets exactly zero gradient.
    # It adds q·b_k to every score in a row, and softmax ignores a constant shift.
    gk = attn.c_attn.grads["b"][C:2 * C]
    print(f"  {'Attention  key bias b_k':28s} max |dL/db_k| = {np.abs(gk).max():.1e}  "
          "(softmax ignores q·b_k, which is the same for every key in a row)")
    # The causal mask must stop ALL gradient flowing from the past to the future:
    # output at position 0 may only depend on input position 0.
    attn.forward(x)
    G = np.zeros((B, T, C))
    G[:, 0] = 1.0
    dx = attn.backward(G)
    leak = float(np.abs(dx[:, 1:]).max())
    print(f"  {'Attention  causal mask':28s} d(out[0])/d(x[1:]) = {leak:.1e}  "
          f"{'OK' if leak == 0 else 'FAIL'}  (the future can't influence the past)")
    ok &= leak == 0

    logits = rng.standard_normal((B, T, V)) * 2
    targets = rng.integers(0, V, (B, T))
    _, dlogits = cross_entropy(logits.copy(), targets)
    num = numerical_grad(lambda: cross_entropy(logits.copy(), targets)[0], logits)
    ok &= report("softmax + cross-entropy", logits.shape, rel_error(dlogits, num))

    blk = Block(C, 2, rng, proj_std=0.5, dtype=F64)
    for sub in [blk.ln_1, blk.ln_2, *blk.attn.sublayers().values(),
                *blk.mlp.sublayers().values()]:
        randomize(sub.params, rng)
    ok &= check_layer("Block", blk, rng.standard_normal((B, T, C)), rng)
    return ok


def part2_full_model(tie, seed=0):
    rng = np.random.default_rng(seed)
    V, B, T = 7, 2, 5
    model = GPT(V, n_layer=2, n_head=2, d_model=8, block_size=6, tie_weights=tie,
                seed=seed, dtype=F64)
    for name, p in model.params().items():
        if name.endswith("gamma"):
            p[...] = 1.0 + rng.standard_normal(p.shape) * 0.3
        else:
            p[...] = rng.standard_normal(p.shape) * 0.3
    idx = rng.integers(0, V, (B, T))
    targets = rng.integers(0, V, (B, T))

    print(f"\nPart 2: every parameter of a tiny GPT through the real loss "
          f"(V={V}, L=2, heads=2, d=8, T={T}, {'TIED' if tie else 'separate'} embeddings, "
          f"{model.n_params()} params)\n")
    model.forward(idx, targets)
    grads = model.backward()
    f = lambda: model.forward(idx, targets)
    ok = True
    for name, p in model.params().items():
        ok &= report(name, p.shape, rel_error(grads[name], numerical_grad(f, p)))
    return ok


def main():
    results = [part1_layers(), part2_full_model(tie=False), part2_full_model(tie=True)]
    ok = all(results)
    print("\nGradient check", "PASSED" if ok else "FAILED")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
