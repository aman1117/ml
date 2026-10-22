"""
Gradient check: compare the analytic BPTT gradients from rnn_numpy.RNN.backward
against numerical gradients from central finite differences:

    dL/dw ≈ (L(w + eps) - L(w - eps)) / (2 * eps)

If backward() is correct, the relative error should be tiny (< 1e-6).
Try breaking a line in backward() and re-run to see the check fail.

Run:
    python rnn/gradient_check.py
"""
import numpy as np

from rnn_numpy import RNN


def grad_check(V=6, H=8, T=7, eps=1e-5, tol=1e-6, seed=0):
    rng = np.random.default_rng(seed)
    model = RNN(V, H, seed)
    # Use larger weights than the training init so gradients aren't ~0.
    for k in model.params:
        model.params[k] = rng.standard_normal(model.params[k].shape) * 0.5

    inputs = rng.integers(0, V, T).tolist()
    targets = rng.integers(0, V, T).tolist()
    h0 = rng.standard_normal((H, 1)) * 0.1

    def loss_fn():
        return model.forward(inputs, targets, h0)[0]

    _, _, cache = model.forward(inputs, targets, h0)
    grads = model.backward(cache)

    ok = True
    for name, param in model.params.items():
        num = np.zeros_like(param)
        it = np.nditer(param, flags=["multi_index"])
        for _ in it:
            idx = it.multi_index
            old = param[idx]
            param[idx] = old + eps
            lp = loss_fn()
            param[idx] = old - eps
            lm = loss_fn()
            param[idx] = old
            num[idx] = (lp - lm) / (2 * eps)

        ana = grads[name]
        rel = np.abs(ana - num) / np.maximum(1e-12, np.abs(ana) + np.abs(num))
        max_rel = rel.max()
        status = "OK" if max_rel < tol else "FAIL"
        ok &= max_rel < tol
        print(f"{name:5s} shape={str(param.shape):8s} max relative error = {max_rel:.2e}  {status}")

    print("\nGradient check", "PASSED" if ok else "FAILED")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if grad_check() else 1)
