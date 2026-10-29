"""
Gradient check: compare the hand-written BPTT gradients of lstm_numpy.LSTM and
lstm_numpy.GRU against numerical gradients from central finite differences:

    dL/dw ≈ (L(w + eps) - L(w - eps)) / (2 * eps)

Checks every parameter of both cells AND the initial state (h0, c0), whose
gradient is what would flow into the previous chunk without truncation.
If backward() is correct, every relative error is tiny (< 1e-6).
Try breaking a line in LSTM.backward() (e.g. drop "+ dc_next") and re-run.

Run:
    python lstm/gradient_check.py
"""
import numpy as np

from lstm_numpy import GRU, LSTM


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


def grad_check(cls, V=6, H=5, T=8, tol=1e-6, seed=0):
    rng = np.random.default_rng(seed)
    model = cls(V, H, seed)
    # Use larger weights than the training init so every gate is far from 0.5
    # and every gradient path is exercised.
    for k in model.params:
        model.params[k] = rng.standard_normal(model.params[k].shape) * 0.5

    inputs = rng.integers(0, V, T).tolist()
    targets = rng.integers(0, V, T).tolist()
    state = tuple(rng.standard_normal((H, 1)) * 0.5 for _ in model.init_state())

    def loss_fn():
        return model.forward(inputs, targets, state)[0]

    _, _, cache = model.forward(inputs, targets, state)
    grads, dstate = model.backward(cache)

    checks = [(name, model.params[name], grads[name]) for name in model.params]
    checks += [(name, s, ds) for name, s, ds in zip(["h0", "c0"], state, dstate)]

    ok = True
    print(f"{cls.__name__} (V={V}, H={H}, T={T} steps)")
    for name, param, ana in checks:
        err = rel_error(ana, numerical_grad(loss_fn, param))
        ok &= err < tol
        print(f"  {name:5s} shape={str(param.shape):8s} max relative error = {err:.2e}  "
              f"{'OK' if err < tol else 'FAIL'}")
    print()
    return ok


if __name__ == "__main__":
    results = [grad_check(LSTM), grad_check(GRU)]
    print("Gradient check", "PASSED" if all(results) else "FAILED")
    raise SystemExit(0 if all(results) else 1)
