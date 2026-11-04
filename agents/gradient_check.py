"""
Gradient check for the tool router: compare every hand-written backward() in
router_numpy.py with numerical gradients from central finite differences (float64):

    dL/dw ≈ (L(w + eps) - L(w - eps)) / (2 * eps)

Part 1 checks each layer ON ITS OWN (input AND parameters) with the trick
L = Σ (layer(x) ⊙ G) for a fixed random G, so dL/d(output) = G:
    Linear, ReLU, softmax + cross-entropy.
Part 2 checks EVERY parameter of a small Router through the real loss, on real
hashed bag-of-words features of real requests from router_train.jsonl.

If backward() is correct the relative error is tiny (< 1e-6). Try breaking a line
in a backward() (e.g. drop the `- 1.0` in cross_entropy, or return `dy` from
ReLU.backward) and re-run: the check fails.

Run (< 1 s):
    python agents/gradient_check.py
"""
import warnings

import numpy as np

from router_numpy import LABELS, Linear, ReLU, Router, cross_entropy, featurize, load_jsonl

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
    """‖a − b‖ / (‖a‖ + ‖b‖) over the whole tensor (robust to entries that are exactly 0)."""
    return float(np.linalg.norm(a - b) / max(1e-20, np.linalg.norm(a) + np.linalg.norm(b)))


def report(name, shape, err):
    ok = err < TOL
    print(f"  {name:34s} shape={str(shape):12s} relative error = {err:.2e}  {'OK' if ok else 'FAIL'}")
    return ok


def check_layer(name, layer, x, rng):
    out = layer.forward(x)
    G = rng.standard_normal(out.shape)
    f = lambda: float(np.sum(layer.forward(x) * G))  # noqa: E731
    layer.forward(x)
    dx = layer.backward(G)
    ok = report(f"{name}  input x", x.shape, rel_error(dx, numerical_grad(f, x)))
    for k, p in layer.params.items():
        layer.forward(x)
        layer.backward(G)
        ok &= report(f"{name}  {k}", p.shape, rel_error(layer.grads[k], numerical_grad(f, p)))
    return ok


def part1(seed=0):
    print("Part 1: each layer on its own (N=5)\n")
    rng = np.random.default_rng(seed)
    ok = True
    lin = Linear(7, 4, rng, dtype=F64)
    lin.params["b"][...] = rng.standard_normal(4)
    ok &= check_layer("Linear", lin, rng.standard_normal((5, 7)), rng)
    # Keep inputs away from 0, where ReLU has a kink and finite differences straddle it.
    x = rng.standard_normal((5, 6))
    x = np.where(np.abs(x) < 0.1, 0.3, x)
    ok &= check_layer("ReLU", ReLU(), x, rng)
    logits = rng.standard_normal((5, len(LABELS))) * 2
    y = rng.integers(0, len(LABELS), 5)
    _, d = cross_entropy(logits.copy(), y)
    ok &= report("softmax + cross-entropy", logits.shape,
                 rel_error(d, numerical_grad(lambda: cross_entropy(logits, y)[0], logits)))
    return ok


def part2(seed=0):
    texts, y = load_jsonl("router_train.jsonl")
    texts, y = texts[:6], y[:6]
    D, H = 64, 8
    model = Router(D=D, hidden=H, seed=seed, dtype=F64)
    rng = np.random.default_rng(seed)
    for p in model.params().values():
        p[...] = rng.standard_normal(p.shape) * 0.5
    X = featurize(texts, D=D, dtype=F64)
    print(f"\nPart 2: every parameter of a small Router through the real loss "
          f"(D={D} hash buckets, H={H}, {len(texts)} real requests)\n")
    model.loss(X, y)
    grads = model.backward()
    f = lambda: model.loss(X, y)  # noqa: E731
    ok = True
    for name, p in model.params().items():
        ok &= report(name, p.shape, rel_error(grads[name], numerical_grad(f, p)))
    dead = float((model.fc1.forward(X) <= 0).mean())
    print(f"  ({dead:.0%} of (example, hidden unit) pairs are inactive: ReLU passes no "
          "gradient through them)")
    return ok


def main():
    ok = all([part1(), part2()])
    print("\nGradient check", "PASSED" if ok else "FAILED")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
