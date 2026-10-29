"""
Gradient check: compare the hand-written backward() of every CNN layer against
numerical gradients from central finite differences:

    dL/dw ≈ (L(w + eps) - L(w - eps)) / (2 * eps)

Checks, in float64:
  - Conv2D with padding 1 and stride 2 (W, b and the input gradient dX)
  - Conv2D with no padding, stride 1
  - MaxPool2D (dX: the gradient must go to the argmax only)
  - GlobalAvgPool and Dense (dX, W, b)
  - the whole network, flatten head and gap head (every parameter AND dX)

For a single layer we use the loss L = Σ out ⊙ R with a fixed random R, so the
upstream gradient dout is exactly R. Try breaking a line in Conv2D.backward()
or col2im() and re-run to see the check fail.

Run:
    python cnn/gradient_check.py
"""
import numpy as np

from cnn_numpy import CNN, Conv2D, Dense, GlobalAvgPool, MaxPool2D, softmax_cross_entropy

TOL = 1e-6


def numerical_grad(f, x, eps=1e-4):
    """Central differences. eps = 1e-4 (not 1e-5): global average pooling makes some
    input gradients tiny (~1e-6), and with a smaller eps the float round-off of
    L(w+eps) - L(w-eps), about 1e-16·L/eps, would swamp them. Inputs are random, so a
    1e-4 step practically never crosses a ReLU kink or changes a max-pool winner."""
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    for _ in it:
        idx = it.multi_index
        old = x[idx]
        x[idx] = old + eps
        lp = f()
        x[idx] = old - eps
        lm = f()
        x[idx] = old
        grad[idx] = (lp - lm) / (2 * eps)
    return grad


def rel_error(a, b):
    return np.max(np.abs(a - b) / np.maximum(1e-12, np.abs(a) + np.abs(b)))


def report(name, arr, ana, num):
    err = rel_error(ana, num)
    ok = err < TOL
    print(f"  {name:22s} shape={str(arr.shape):15s} max relative error = {err:.2e}  "
          f"{'OK' if ok else 'FAIL'}")
    return ok


def check_layer(title, layer, x, rng):
    """L = Σ layer(x) ⊙ R  →  dL/dout = R. Checks dX and every parameter of the layer."""
    print(title)
    R = rng.standard_normal(layer.forward(x).shape)
    f = lambda: np.sum(layer.forward(x) * R)
    layer.forward(x)
    dx = layer.backward(R)
    ok = report("dX", x, dx, numerical_grad(f, x))
    for name, p in layer.params.items():
        layer.forward(x)
        layer.backward(R)
        ok &= report(name, p, layer.grads[name], numerical_grad(f, p))
    return ok


def check_network(head, rng):
    print(f"Full network (head = {head}), softmax cross-entropy loss")
    model = CNN(n_classes=4, size=8, c1=3, c2=4, head=head, seed=1)
    for _, p, _ in model.named_params():  # non-zero biases so every path is exercised
        if p.ndim == 1:
            p += rng.standard_normal(p.shape) * 0.1
    x = rng.standard_normal((3, 1, 8, 8))
    y = rng.integers(0, 4, 3)
    f = lambda: softmax_cross_entropy(model.forward(x), y)[0]

    _, dlogits = softmax_cross_entropy(model.forward(x), y)
    dx = model.backward(dlogits)
    grads = {name: g.copy() for name, _, g in model.named_params()}
    ok = True
    for name, p, _ in model.named_params():
        ok &= report(name, p, grads[name], numerical_grad(f, p))
    ok &= report("dX (the input image)", x, dx, numerical_grad(f, x))
    return ok


def main():
    rng = np.random.default_rng(0)
    ok = True
    # Random real-valued inputs: max-pool ties and ReLU kinks at exactly 0 then have
    # probability ~0, so finite differences are valid everywhere.
    ok &= check_layer("Conv2D 3x3, C_in=2 → C_out=3, pad=1, stride=2",
                      Conv2D(2, 3, 3, stride=2, pad=1, rng=rng), rng.standard_normal((2, 2, 7, 7)),
                      rng)
    layer = Conv2D(2, 3, 3, stride=1, pad=0, rng=rng)
    layer.params["b"] = rng.standard_normal(3)
    ok &= check_layer("Conv2D 3x3, C_in=2 → C_out=3, pad=0, stride=1", layer,
                      rng.standard_normal((2, 2, 6, 6)), rng)
    ok &= check_layer("MaxPool2D 2x2", MaxPool2D(2), rng.standard_normal((2, 3, 6, 6)), rng)
    ok &= check_layer("GlobalAvgPool", GlobalAvgPool(), rng.standard_normal((2, 3, 4, 4)), rng)
    layer = Dense(5, 4, rng)
    layer.params["b"] = rng.standard_normal(4)
    ok &= check_layer("Dense 5 → 4", layer, rng.standard_normal((3, 5)), rng)
    ok &= check_network("flatten", rng)
    ok &= check_network("gap", rng)

    print("\nGradient check", "PASSED" if ok else "FAILED")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
