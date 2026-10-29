"""
LSTM (and GRU) from scratch in NumPy, trained as a character-level language
model on the SAME text as ../rnn/rnn_numpy.py, so the losses are comparable.

LSTM equations (column vectors, gate order i, f, g, o like PyTorch; README section 3):
    a_t = W_xh @ x_t + W_hh @ h_{t-1} + b_h            # (4H, 1): all four gates at once
    i_t = sigmoid(a_t[0:H])      input gate   ("how much to write")
    f_t = sigmoid(a_t[H:2H])     forget gate  ("how much old memory to keep")
    g_t = tanh(a_t[2H:3H])       candidate    ("what to write")
    o_t = sigmoid(a_t[3H:4H])    output gate  ("how much memory to show")
    c_t = f_t * c_{t-1} + i_t * g_t                    # ADDITIVE cell update
    h_t = o_t * tanh(c_t)
    y_t = W_hy @ h_t + b_y,   p_t = softmax(y_t),   L = sum_t -log p_t[target_t]

GRU equations (PyTorch's convention, gate order r, z, n; README section 8):
    r_t = sigmoid(W_xr x_t + b_xr + W_hr h_{t-1} + b_hr)      reset gate
    z_t = sigmoid(W_xz x_t + b_xz + W_hz h_{t-1} + b_hz)      update gate
    n_t = tanh(W_xn x_t + b_xn + r_t * (W_hn h_{t-1} + b_hn)) candidate
    h_t = (1 - z_t) * n_t + z_t * h_{t-1}

Same training recipe as the vanilla RNN: truncated BPTT over 25-char chunks
(the state (h, c) is carried across chunks, gradients are not), element-wise
gradient clipping, Adagrad.

Run:
    python lstm/lstm_numpy.py                    # LSTM, saves lstm/model.npz
    python lstm/lstm_numpy.py --cell gru         # GRU,  saves lstm/model_gru.npz
    python lstm/lstm_numpy.py --hidden 128 --seq-len 50 --iters 20000
"""
import argparse
import os
import time
import warnings

import numpy as np

# NumPy 2.0 + Apple Accelerate raises spurious "divide by zero / overflow /
# invalid value encountered in matmul" warnings even when results are correct.
# Only silence those; real numerical warnings (exp, log, ...) still show.
warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TEXT = os.path.join(HERE, "..", "rnn", "input.txt")  # same text as the vanilla RNN


def softmax(y):
    e = np.exp(y - np.max(y))  # subtract max for numerical stability
    return e / np.sum(e)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def onehot(ix, V):
    x = np.zeros((V, 1))
    x[ix] = 1.0
    return x


class LSTM:
    """Char-level LSTM language model. The state is a tuple (h, c)."""
    cell = "lstm"

    def __init__(self, vocab_size, hidden_size, seed=0):
        rng = np.random.default_rng(seed)
        V, H = vocab_size, hidden_size
        self.V, self.H = V, H
        self.rng = rng
        b_h = np.zeros((4 * H, 1))
        # Forget-gate bias = 1: f starts at sigmoid(1) ≈ 0.73 instead of 0.5,
        # so the cell remembers by default until it learns what to forget.
        b_h[H:2 * H] = 1.0
        # Same names as rnn_numpy.py: W_xh, W_hh and b_h are just 4x taller,
        # one (H, .) block per gate, stacked in the order i, f, g, o.
        self.params = {
            "W_xh": rng.standard_normal((4 * H, V)) * 0.01,
            "W_hh": rng.standard_normal((4 * H, H)) * 0.01,
            "b_h": b_h,
            "W_hy": rng.standard_normal((V, H)) * 0.01,
            "b_y": np.zeros((V, 1)),
        }

    def init_state(self):
        return (np.zeros((self.H, 1)), np.zeros((self.H, 1)))

    def cell_step(self, x, h_prev, c_prev):
        """One LSTM step. Returns h, c and the gates (for the cache and the visuals)."""
        P, H = self.params, self.H
        a = P["W_xh"] @ x + P["W_hh"] @ h_prev + P["b_h"]  # (4H, 1)
        i = sigmoid(a[0:H])
        f = sigmoid(a[H:2 * H])
        g = np.tanh(a[2 * H:3 * H])
        o = sigmoid(a[3 * H:4 * H])
        c = f * c_prev + i * g
        tc = np.tanh(c)
        h = o * tc
        return h, c, (i, f, g, o, tc)

    def step(self, ix, state, temperature=1.0):
        """Read one char index. Returns (new state, next-char probabilities (V, 1))."""
        h, c, _ = self.cell_step(onehot(ix, self.V), *state)
        y = self.params["W_hy"] @ h + self.params["b_y"]
        return (h, c), softmax(y / temperature)

    def forward(self, inputs, targets, state):
        """
        inputs, targets: lists of integer char indices (same length T).
        state: (h, c), each (H, 1), carried in from the previous chunk.
        Returns (loss, last state, cache for backward).
        """
        P = self.params
        h_prev, c_prev = state
        xs, hs, cs, gates, ps = {}, {-1: np.copy(h_prev)}, {-1: np.copy(c_prev)}, {}, {}
        loss = 0.0
        for t, (ix, target) in enumerate(zip(inputs, targets)):
            xs[t] = onehot(ix, self.V)
            hs[t], cs[t], gates[t] = self.cell_step(xs[t], hs[t - 1], cs[t - 1])
            ps[t] = softmax(P["W_hy"] @ hs[t] + P["b_y"])
            loss += -np.log(ps[t][target, 0])
        cache = (xs, hs, cs, gates, ps, targets)
        T = len(inputs)
        return loss, (hs[T - 1], cs[T - 1]), cache

    def backward(self, cache):
        """BPTT. Returns (grads dict, (dL/dh_{-1}, dL/dc_{-1}))."""
        P, H = self.params, self.H
        xs, hs, cs, gates, ps, targets = cache
        grads = {k: np.zeros_like(v) for k, v in P.items()}
        dh_next = np.zeros((H, 1))  # gradient flowing in from step t+1 via h
        dc_next = np.zeros((H, 1))  # ... and via the cell state c

        for t in reversed(range(len(targets))):
            i, f, g, o, tc = gates[t]
            dy = np.copy(ps[t])
            dy[targets[t]] -= 1.0  # d(cross-entropy)/d(logits) = p - onehot
            grads["W_hy"] += dy @ hs[t].T
            grads["b_y"] += dy

            dh = P["W_hy"].T @ dy + dh_next              # from output now + from the future
            do = dh * tc                                 # h = o * tanh(c)
            dc = dh * o * (1.0 - tc ** 2) + dc_next      # through tanh(c), plus c's own future
            di = dc * g                                  # c = f * c_prev + i * g
            dg = dc * i
            df = dc * cs[t - 1]
            dc_next = dc * f                             # the "constant error carousel"

            # Back through the gate non-linearities: sigmoid' = s(1-s), tanh' = 1-tanh².
            da = np.vstack([di * i * (1.0 - i),
                            df * f * (1.0 - f),
                            dg * (1.0 - g ** 2),
                            do * o * (1.0 - o)])         # (4H, 1)
            grads["W_xh"] += da @ xs[t].T
            grads["W_hh"] += da @ hs[t - 1].T
            grads["b_h"] += da
            dh_next = P["W_hh"].T @ da                   # send gradient to h_{t-1}

        return grads, (dh_next, dc_next)


class GRU:
    """Char-level GRU language model (PyTorch's equations). The state is a tuple (h,)."""
    cell = "gru"

    def __init__(self, vocab_size, hidden_size, seed=0):
        rng = np.random.default_rng(seed)
        V, H = vocab_size, hidden_size
        self.V, self.H = V, H
        self.rng = rng
        # Blocks stacked in PyTorch's order r, z, n. The GRU needs TWO biases
        # because b_hn sits inside the reset gate: r * (W_hn h + b_hn).
        self.params = {
            "W_xh": rng.standard_normal((3 * H, V)) * 0.01,
            "W_hh": rng.standard_normal((3 * H, H)) * 0.01,
            "b_xh": np.zeros((3 * H, 1)),
            "b_hh": np.zeros((3 * H, 1)),
            "W_hy": rng.standard_normal((V, H)) * 0.01,
            "b_y": np.zeros((V, 1)),
        }

    def init_state(self):
        return (np.zeros((self.H, 1)),)

    def cell_step(self, x, h_prev):
        """One GRU step. Returns h and the gates (r, z, n, W_hn h + b_hn)."""
        P, H = self.params, self.H
        ax = P["W_xh"] @ x + P["b_xh"]        # (3H, 1) the input's vote for r, z, n
        ah = P["W_hh"] @ h_prev + P["b_hh"]   # (3H, 1) the memory's vote for r, z, n
        r = sigmoid(ax[0:H] + ah[0:H])
        z = sigmoid(ax[H:2 * H] + ah[H:2 * H])
        n = np.tanh(ax[2 * H:] + r * ah[2 * H:])
        h = (1.0 - z) * n + z * h_prev        # blend: new content vs. old memory
        return h, (r, z, n, ah[2 * H:])

    def step(self, ix, state, temperature=1.0):
        h, _ = self.cell_step(onehot(ix, self.V), state[0])
        y = self.params["W_hy"] @ h + self.params["b_y"]
        return (h,), softmax(y / temperature)

    def forward(self, inputs, targets, state):
        P = self.params
        xs, hs, gates, ps = {}, {-1: np.copy(state[0])}, {}, {}
        loss = 0.0
        for t, (ix, target) in enumerate(zip(inputs, targets)):
            xs[t] = onehot(ix, self.V)
            hs[t], gates[t] = self.cell_step(xs[t], hs[t - 1])
            ps[t] = softmax(P["W_hy"] @ hs[t] + P["b_y"])
            loss += -np.log(ps[t][target, 0])
        cache = (xs, hs, gates, ps, targets)
        return loss, (hs[len(inputs) - 1],), cache

    def backward(self, cache):
        """BPTT. Returns (grads dict, (dL/dh_{-1},))."""
        P, H = self.params, self.H
        xs, hs, gates, ps, targets = cache
        grads = {k: np.zeros_like(v) for k, v in P.items()}
        dh_next = np.zeros((H, 1))

        for t in reversed(range(len(targets))):
            r, z, n, ahn = gates[t]
            dy = np.copy(ps[t])
            dy[targets[t]] -= 1.0
            grads["W_hy"] += dy @ hs[t].T
            grads["b_y"] += dy

            dh = P["W_hy"].T @ dy + dh_next
            dn = dh * (1.0 - z)                   # h = (1 - z) * n + z * h_prev
            dz = dh * (hs[t - 1] - n)
            dh_direct = dh * z                    # the GRU's carousel: scaled by z, not W_hh
            dan = dn * (1.0 - n ** 2)             # through tanh
            dr = dan * ahn                        # n = tanh(... + r * (W_hn h + b_hn))
            dar = dr * r * (1.0 - r)
            daz = dz * z * (1.0 - z)

            dax = np.vstack([dar, daz, dan])      # (3H, 1) grad of the input's vote
            dah = np.vstack([dar, daz, dan * r])  # (3H, 1) the memory's n-vote was scaled by r
            grads["W_xh"] += dax @ xs[t].T
            grads["b_xh"] += dax
            grads["W_hh"] += dah @ hs[t - 1].T
            grads["b_hh"] += dah
            dh_next = dh_direct + P["W_hh"].T @ dah

        return grads, (dh_next,)


CELLS = {"lstm": LSTM, "gru": GRU}


def sample(model, state, seed_ix, n, temperature=1.0):
    """Generate n character indices, feeding each prediction back as input."""
    ix, out = seed_ix, []
    for _ in range(n):
        state, p = model.step(ix, state, temperature)
        ix = model.rng.choice(model.V, p=p.ravel())
        out.append(ix)
    return out


def train(text, cell="lstm", hidden_size=100, seq_len=25, lr=0.1, iters=10000, clip=5.0,
          sample_every=1000, seed=0, verbose=True):
    """Same loop as rnn_numpy.train. Returns (model, stoi, itos, per-char loss of every chunk)."""
    chars = sorted(set(text))
    V = len(chars)
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for i, c in enumerate(chars)}
    data = [stoi[c] for c in text]
    log = print if verbose else (lambda *a, **k: None)
    model = CELLS[cell](V, hidden_size, seed)
    n_params = sum(v.size for v in model.params.values())
    log(f"data: {len(text)} chars, vocab size: {V}, cell: {cell.upper()}, params: {n_params:,}")

    # Adagrad memory: per-parameter sum of squared gradients.
    mem = {k: np.zeros_like(v) for k, v in model.params.items()}
    smooth_loss = -np.log(1.0 / V) * seq_len  # loss of uniform guessing
    p = 0
    state = model.init_state()
    losses = []

    for it in range(iters + 1):
        # Wrap around at end of data and reset memory.
        if p + seq_len + 1 > len(data):
            p = 0
            state = model.init_state()
        inputs = data[p:p + seq_len]
        targets = data[p + 1:p + seq_len + 1]

        if sample_every and it % sample_every == 0:
            ixs = sample(model, state, inputs[0], 200)
            log(f"\n----- iter {it}, loss {smooth_loss:.3f} -----")
            log("".join(itos[i] for i in ixs))

        # The state (h, c) is carried forward, but gradients stop at the
        # chunk boundary: truncated BPTT, exactly like the vanilla RNN.
        loss, state, cache = model.forward(inputs, targets, state)
        grads, _ = model.backward(cache)

        for k, g in grads.items():
            np.clip(g, -clip, clip, out=g)  # still useful: LSTMs can explode too
            mem[k] += g * g
            model.params[k] -= lr * g / np.sqrt(mem[k] + 1e-8)

        smooth_loss = 0.999 * smooth_loss + 0.001 * loss
        losses.append(loss / seq_len)
        p += seq_len

    return model, stoi, itos, np.array(losses)


def run_config(cell, hidden, seq_len, lr, iters, text):
    """A string that identifies a training run (make_figures.py reuses matching models)."""
    return f"{cell} H={hidden} seq_len={seq_len} lr={lr} iters={iters} text_chars={len(text)}"


def save_model(path, model, chars, **extra):
    """Save the weights, the cell type, the vocabulary (and extras like the loss history)."""
    np.savez(path, cell=np.array(model.cell), chars=np.array(chars), **model.params, **extra)


def load_model(path):
    d = np.load(path)
    chars = [str(c) for c in d["chars"]]
    model = CELLS[str(d["cell"])](len(chars), d["W_hh"].shape[1])
    for k in model.params:
        model.params[k] = d[k]
    return model, chars


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", default=DEFAULT_TEXT,
                    help="training text (default: ../rnn/input.txt, same as the vanilla RNN)")
    ap.add_argument("--cell", choices=sorted(CELLS), default="lstm")
    ap.add_argument("--hidden", type=int, default=100)
    ap.add_argument("--seq-len", type=int, default=25)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--iters", type=int, default=10000)
    ap.add_argument("--sample-every", type=int, default=1000)
    ap.add_argument("--temperature", type=float, default=0.8,
                    help="temperature for the final sample")
    ap.add_argument("--save", default=None,
                    help="where to save the model (default lstm/model.npz, or "
                         "lstm/model_gru.npz with --cell gru)")
    args = ap.parse_args()
    save = args.save or os.path.join(HERE, "model.npz" if args.cell == "lstm" else "model_gru.npz")

    with open(args.file, encoding="utf-8") as f:
        text = f.read()

    t0 = time.time()
    model, stoi, itos, losses = train(text, args.cell, args.hidden, args.seq_len, args.lr,
                                      args.iters, sample_every=args.sample_every)
    print(f"\ntrained in {time.time() - t0:.1f} s. Average loss per char over the last "
          f"200 chunks: {losses[-200:].mean():.3f}")
    print(f"(the 'loss' printed above is summed over a {args.seq_len}-char chunk, "
          "like rnn_numpy.py)")

    ixs = sample(model, model.init_state(), stoi[text[0]], 400, temperature=args.temperature)
    print(f"\n===== final sample (temperature {args.temperature}) =====")
    print(text[0] + "".join(itos[i] for i in ixs))

    config = run_config(args.cell, args.hidden, args.seq_len, args.lr, args.iters, text)
    save_model(save, model, [itos[i] for i in range(len(itos))], losses=losses,
               config=np.array(config))
    flag = "" if args.cell == "lstm" else " --cell gru"
    print(f"\nSaved model to {os.path.relpath(save)}. Try: python lstm/generate.py{flag}")


if __name__ == "__main__":
    main()
