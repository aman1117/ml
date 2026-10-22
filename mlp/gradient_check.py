"""
Gradient check: compare the backprop gradients from mlp_numpy.MLP against
numerical gradients from central finite differences:

    dL/dw ≈ (L(w + eps) - L(w - eps)) / (2 * eps)

Checks every activation function, with L2 regularization switched on.
Try breaking a line in MLP.backward() and re-run to see the check fail.

Run:
    python mlp/gradient_check.py
"""
import numpy as np

from mlp_numpy import ACTIVATIONS, MLP


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


def grad_check(activation, tol=1e-6, seed=0):
    rng = np.random.default_rng(seed)
    model = MLP([4, 6, 5, 3], activation, seed)
    for i in range(model.n_layers):
        # Non-zero biases so every gradient path is exercised.
        model.b[i] = rng.standard_normal(model.b[i].shape) * 0.1
    X = rng.standard_normal((8, 4))
    y = rng.integers(0, 3, 8)
    reg = 0.1

    _, dW, db = model.loss_and_grads(X, y, reg)
    f = lambda: model.loss(X, y, reg)

    ok = True
    for i in range(model.n_layers):
        for name, param, grad in [(f"W{i}", model.W[i], dW[i]), (f"b{i}", model.b[i], db[i])]:
            err = rel_error(grad, numerical_grad(f, param))
            ok &= err < tol
            print(f"  {activation:7s} {name}  shape={str(param.shape):7s} "
                  f"max relative error = {err:.2e}  {'OK' if err < tol else 'FAIL'}")
    return ok


if __name__ == "__main__":
    # ReLU has a kink at 0. If a unit sits right at it, the numerical gradient
    # can disagree; with random inputs this practically never happens.
    results = [grad_check(a) for a in ACTIVATIONS]
    print("\nGradient check", "PASSED" if all(results) else "FAILED")
    raise SystemExit(0 if all(results) else 1)
