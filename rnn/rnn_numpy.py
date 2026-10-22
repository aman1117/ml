"""
Vanilla RNN from scratch in NumPy, trained as a character-level language model.

Equations (column vectors, see README section 2):
    h_t = tanh(W_xh @ x_t + W_hh @ h_{t-1} + b_h)
    y_t = W_hy @ h_t + b_y
    p_t = softmax(y_t)
    L   = sum_t -log p_t[target_t]

Run:
    python rnn/rnn_numpy.py
    python rnn/rnn_numpy.py --hidden 128 --seq-len 50 --iters 20000
"""
import argparse
import os
import warnings

import numpy as np

# NumPy 2.0 + Apple Accelerate raises spurious "divide by zero / overflow /
# invalid value encountered in matmul" warnings even when results are correct.
# Only silence those; real numerical warnings (exp, log, ...) still show.
warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)


def softmax(y):
    e = np.exp(y - np.max(y))  # subtract max for numerical stability
    return e / np.sum(e)


class RNN:
    def __init__(self, vocab_size, hidden_size, seed=0):
        rng = np.random.default_rng(seed)
        V, H = vocab_size, hidden_size
        self.V, self.H = V, H
        self.rng = rng
        # Small init keeps tanh out of saturation and W_hh's spectral radius < 1.
        self.params = {
            "W_xh": rng.standard_normal((H, V)) * 0.01,
            "W_hh": rng.standard_normal((H, H)) * 0.01,
            "W_hy": rng.standard_normal((V, H)) * 0.01,
            "b_h": np.zeros((H, 1)),
            "b_y": np.zeros((V, 1)),
        }

    def forward(self, inputs, targets, h_prev):
        """
        inputs, targets: lists of integer char indices (same length T).
        h_prev: (H, 1) hidden state carried in from the previous chunk.
        Returns (loss, last hidden state, cache for backward).
        """
        P = self.params
        xs, hs, ps = {}, {-1: np.copy(h_prev)}, {}
        loss = 0.0
        for t, (ix, target) in enumerate(zip(inputs, targets)):
            xs[t] = np.zeros((self.V, 1))
            xs[t][ix] = 1.0  # one-hot
            hs[t] = np.tanh(P["W_xh"] @ xs[t] + P["W_hh"] @ hs[t - 1] + P["b_h"])
            y = P["W_hy"] @ hs[t] + P["b_y"]
            ps[t] = softmax(y)
            loss += -np.log(ps[t][target, 0])
        cache = (xs, hs, ps, targets)
        return loss, hs[len(inputs) - 1], cache

    def backward(self, cache):
        """Backpropagation through time. Returns a dict of gradients."""
        P = self.params
        xs, hs, ps, targets = cache
        grads = {k: np.zeros_like(v) for k, v in P.items()}
        dh_next = np.zeros((self.H, 1))  # gradient flowing in from step t+1

        for t in reversed(range(len(targets))):
            dy = np.copy(ps[t])
            dy[targets[t]] -= 1.0  # d(cross-entropy)/d(logits) = p - onehot
            grads["W_hy"] += dy @ hs[t].T
            grads["b_y"] += dy

            dh = P["W_hy"].T @ dy + dh_next  # from output now + from the future
            da = (1.0 - hs[t] ** 2) * dh  # backprop through tanh

            grads["b_h"] += da
            grads["W_xh"] += da @ xs[t].T
            grads["W_hh"] += da @ hs[t - 1].T
            dh_next = P["W_hh"].T @ da  # send gradient to step t-1

        return grads

    def sample(self, h, seed_ix, n, temperature=1.0):
        """Generate n character indices, feeding each prediction back as input."""
        P = self.params
        x = np.zeros((self.V, 1))
        x[seed_ix] = 1.0
        out = []
        for _ in range(n):
            h = np.tanh(P["W_xh"] @ x + P["W_hh"] @ h + P["b_h"])
            y = P["W_hy"] @ h + P["b_y"]
            p = softmax(y / temperature)
            ix = self.rng.choice(self.V, p=p.ravel())
            x = np.zeros((self.V, 1))
            x[ix] = 1.0
            out.append(ix)
        return out


def train(text, hidden_size=100, seq_len=25, lr=0.1, iters=10000, clip=5.0,
          sample_every=1000, seed=0):
    chars = sorted(set(text))
    V = len(chars)
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for i, c in enumerate(chars)}
    data = [stoi[c] for c in text]
    print(f"data: {len(text)} chars, vocab size: {V}")

    model = RNN(V, hidden_size, seed)
    # Adagrad memory: per-parameter sum of squared gradients.
    mem = {k: np.zeros_like(v) for k, v in model.params.items()}
    smooth_loss = -np.log(1.0 / V) * seq_len  # loss of uniform guessing
    p = 0
    h = np.zeros((hidden_size, 1))

    for it in range(iters + 1):
        # Wrap around at end of data and reset memory.
        if p + seq_len + 1 > len(data):
            p = 0
            h = np.zeros((hidden_size, 1))
        inputs = data[p:p + seq_len]
        targets = data[p + 1:p + seq_len + 1]  # next char is the target

        if it % sample_every == 0:
            ixs = model.sample(h, inputs[0], 200)
            print(f"\n----- iter {it}, loss {smooth_loss:.3f} -----")
            print("".join(itos[i] for i in ixs))

        # h is carried forward (context), but gradients stop at the chunk
        # boundary: this is truncated BPTT.
        loss, h, cache = model.forward(inputs, targets, h)
        grads = model.backward(cache)

        for k, g in grads.items():
            np.clip(g, -clip, clip, out=g)  # tame exploding gradients
            mem[k] += g * g
            model.params[k] -= lr * g / np.sqrt(mem[k] + 1e-8)

        smooth_loss = 0.999 * smooth_loss + 0.001 * loss
        p += seq_len

    return model, stoi, itos


def save_model(path, model, chars):
    """Save the weights and the vocabulary (needed to map chars <-> indices)."""
    np.savez(path, chars=np.array(chars), **model.params)


def load_model(path):
    d = np.load(path)
    chars = [str(c) for c in d["chars"]]
    model = RNN(len(chars), d["W_hh"].shape[0])
    for k in model.params:
        model.params[k] = d[k]
    return model, chars


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", default=os.path.join(here, "input.txt"))
    ap.add_argument("--hidden", type=int, default=100)
    ap.add_argument("--seq-len", type=int, default=25)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--iters", type=int, default=10000)
    ap.add_argument("--sample-every", type=int, default=1000)
    ap.add_argument("--temperature", type=float, default=0.8,
                    help="temperature for the final sample")
    ap.add_argument("--save", default=os.path.join(here, "model.npz"),
                    help="where to save the trained model (used by generate.py)")
    args = ap.parse_args()

    with open(args.file, encoding="utf-8") as f:
        text = f.read()

    model, stoi, itos = train(text, args.hidden, args.seq_len, args.lr,
                              args.iters, sample_every=args.sample_every)

    h = np.zeros((args.hidden, 1))
    ixs = model.sample(h, stoi[text[0]], 400, temperature=args.temperature)
    print(f"\n===== final sample (temperature {args.temperature}) =====")
    print(text[0] + "".join(itos[i] for i in ixs))

    save_model(args.save, model, [itos[i] for i in range(len(itos))])
    print(f"\nSaved model to {args.save}. Try: python rnn/generate.py")


if __name__ == "__main__":
    main()
