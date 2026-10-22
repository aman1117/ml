"""
Why do vanilla RNNs struggle with long-range dependencies?

The gradient from step t back to step k is a product of Jacobians:

    dh_t/dh_k = prod_{i=k+1..t} diag(1 - h_i^2) @ W_hh

This script builds random RNNs whose W_hh has a chosen spectral radius
(largest |eigenvalue|), runs them forward, and prints ||dh_T/dh_k|| as the
distance T-k grows.

- radius < 1  -> norm shrinks exponentially (vanishing)
- radius > 1  -> with a linear activation, norm grows exponentially (exploding);
                 with tanh, saturation (slope ~ 0) often pulls it back down.

Run:
    python rnn/vanishing_gradients.py
"""
import warnings

import numpy as np

# Spurious matmul warnings from NumPy 2.0 + Apple Accelerate (see rnn_numpy.py).
warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)


def jacobian_norms(radius, activation="tanh", H=64, T=60, seed=0):
    rng = np.random.default_rng(seed)
    W_hh = rng.standard_normal((H, H))
    W_hh *= radius / np.max(np.abs(np.linalg.eigvals(W_hh)))
    W_xh = rng.standard_normal((H, H)) * 0.1
    xs = rng.standard_normal((T, H, 1))

    h = np.zeros((H, 1))
    slopes = []  # derivative of the activation at each step
    for t in range(T):
        a = W_xh @ xs[t] + W_hh @ h
        if activation == "tanh":
            h = np.tanh(a)
            slopes.append(1.0 - h ** 2)
        else:  # linear: isolates the effect of W_hh alone
            h = a
            slopes.append(np.ones_like(a))

    # J = dh_{T-1}/dh_k, accumulated backwards from k = T-2 down to 0.
    J = np.eye(H)
    norms = {}
    for k in range(T - 2, -1, -1):
        J = J @ (slopes[k + 1] * W_hh)  # diag(slope) @ W_hh, via broadcasting
        norms[T - 1 - k] = np.linalg.norm(J, 2)
    return norms


def main():
    distances = [1, 5, 10, 20, 40, 59]
    header = "radius  activation | " + " ".join(f"d={d:<8d}" for d in distances)
    print("||dh_T / dh_(T-d)||  (spectral norm of the Jacobian product)\n")
    print(header)
    print("-" * len(header))
    for activation in ["linear", "tanh"]:
        for radius in [0.5, 0.9, 1.0, 1.1, 1.5]:
            norms = jacobian_norms(radius, activation)
            row = " ".join(f"{norms[d]:<10.2e}" for d in distances)
            print(f"{radius:<6} {activation:<10} | {row}")
        print()

    print("Takeaways:")
    print(" * The gradient signal from d steps back scales roughly like radius^d.")
    print(" * radius < 1: it vanishes, so the RNN can't learn long-range dependencies.")
    print(" * radius > 1 (linear): it explodes, so gradient clipping is needed.")
    print(" * tanh saturation shrinks gradients further. LSTMs/GRUs fix this with")
    print("   additive, gated memory paths.")


if __name__ == "__main__":
    main()
