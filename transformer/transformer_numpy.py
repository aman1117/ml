"""
A GPT-style Transformer (decoder only) from scratch in NumPy, with a
HAND-WRITTEN backward pass for every layer. It is trained as a character-level
language model on the same text as ../rnn/ so the numbers compare directly.

Forward pass (one row per token; x has shape (B, T, C), see README sections 3-6):
    x = wte[idx] + wpe[0..T-1]                    token + position embeddings
    repeat n_layer times (one "block", pre-LayerNorm):
        x = x + Attn(LN_1(x))                     tokens look at earlier tokens
        x = x + MLP(LN_2(x))                      each token "thinks" on its own
    logits = LN_f(x) @ W_head                     (B, T, V)
    L = mean over all (b, t) of  -log softmax(logits[b, t])[target[b, t]]

    Attn(x): q, k, v = x W_q, x W_k, x W_v        split into n_head heads of size hs
             A = softmax(q kᵀ / √hs + causal mask)   y = A v   out = concat(heads) W_o
    MLP(x):  GELU(x W_1 + b_1) W_2 + b_2          C → 4C → C
    LN(x):   γ ⊙ (x - mean) / √(var + ε) + β      over the C features of each token

Every layer is a small class with forward() (which caches what it needs) and
backward() (which turns d(loss)/d(output) into d(loss)/d(input) and fills
self.grads). GPT chains them. gradient_check.py verifies every backward().

Run:
    python transformer/transformer_numpy.py
    python transformer/transformer_numpy.py --n-layer 4 --d-model 128 --steps 3000
    python transformer/transformer_numpy.py --file some_other_text.txt
"""
import argparse
import json
import math
import os
import time
import warnings

import numpy as np

# NumPy 2.0 + Apple Accelerate raises spurious "divide by zero / overflow /
# invalid value encountered in matmul" warnings even when results are correct.
# Only silence those; real numerical warnings (exp, log, ...) still show.
warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TEXT = os.path.join(HERE, "..", "rnn", "input.txt")


# ----------------------------------------------------------------------------
# Layers. Each one: params / grads dicts, forward(x) -> y, backward(dy) -> dx.
# ----------------------------------------------------------------------------
class Layer:
    def __init__(self):
        self.params, self.grads = {}, {}


class Linear(Layer):
    """y = x @ W + b.   x: (..., n_in)   W: (n_in, n_out)   b: (n_out,)"""

    def __init__(self, n_in, n_out, rng, std=0.02, bias=True, dtype=np.float32):
        super().__init__()
        self.params["W"] = (rng.standard_normal((n_in, n_out)) * std).astype(dtype)
        if bias:
            self.params["b"] = np.zeros(n_out, dtype=dtype)

    def forward(self, x):
        self.x = x
        y = x @ self.params["W"]
        return y + self.params["b"] if "b" in self.params else y

    def backward(self, dy):
        W = self.params["W"]
        x2 = self.x.reshape(-1, W.shape[0])  # all tokens of all sequences as rows
        dy2 = dy.reshape(-1, W.shape[1])
        self.grads["W"] = x2.T @ dy2  # (n_in, N) @ (N, n_out)
        if "b" in self.params:
            self.grads["b"] = dy2.sum(axis=0)
        return dy @ W.T


class Embedding(Layer):
    """A lookup table: row i of the (n, C) weight is the vector for id i."""

    def __init__(self, n, C, rng, std=0.02, dtype=np.float32):
        super().__init__()
        self.params["weight"] = (rng.standard_normal((n, C)) * std).astype(dtype)

    def forward(self, idx):
        self.idx = idx
        return self.params["weight"][idx]

    def backward(self, dy):
        # Same as a one-hot matmul: each row's gradient is the sum of the
        # gradients of every place that id was used (np.add.at handles repeats).
        dW = np.zeros_like(self.params["weight"])
        np.add.at(dW, self.idx, dy)
        self.grads["weight"] = dW


class LayerNorm(Layer):
    """Normalize each token's C features to mean 0 / var 1, then scale and shift."""

    def __init__(self, C, eps=1e-5, dtype=np.float32):
        super().__init__()
        self.eps = eps
        self.params["gamma"] = np.ones(C, dtype=dtype)
        self.params["beta"] = np.zeros(C, dtype=dtype)

    def forward(self, x):
        mu = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        self.rstd = 1.0 / np.sqrt(var + self.eps)  # (B, T, 1)
        self.xhat = (x - mu) * self.rstd
        return self.params["gamma"] * self.xhat + self.params["beta"]

    def backward(self, dy):
        xhat, rstd = self.xhat, self.rstd
        C = xhat.shape[-1]
        self.grads["gamma"] = (dy * xhat).reshape(-1, C).sum(axis=0)
        self.grads["beta"] = dy.reshape(-1, C).sum(axis=0)
        dxhat = dy * self.params["gamma"]
        # xhat depends on x directly AND through mean and var, which gives the
        # two subtracted terms (README section 7).
        return rstd * (dxhat - dxhat.mean(axis=-1, keepdims=True)
                       - xhat * (dxhat * xhat).mean(axis=-1, keepdims=True))


GELU_C = math.sqrt(2.0 / math.pi)  # a Python float, so float32 arrays stay float32


class GELU(Layer):
    """GELU, tanh approximation (as in GPT-2):
    y = 0.5 x (1 + tanh(√(2/π) (x + 0.044715 x³)))"""

    def forward(self, x):
        self.x = x
        self.t = np.tanh(GELU_C * (x + 0.044715 * x ** 3))
        return 0.5 * x * (1.0 + self.t)

    def backward(self, dy):
        x, t = self.x, self.t
        du_dx = GELU_C * (1.0 + 3 * 0.044715 * x ** 2)
        return dy * (0.5 * (1.0 + t) + 0.5 * x * (1.0 - t ** 2) * du_dx)


def softmax(z, axis=-1):
    e = np.exp(z - z.max(axis=axis, keepdims=True))  # subtract max for stability
    return e / e.sum(axis=axis, keepdims=True)


class CausalSelfAttention(Layer):
    """Multi-head causal self-attention.

    q, k, v come from the SAME sequence x (one big matmul with W_qkv = [W_q W_k W_v]).
    Shapes: x (B, T, C) -> q, k, v (B, nh, T, hs) with hs = C / nh
            scores, A (B, nh, T, T) -> y (B, nh, T, hs) -> concat (B, T, C) -> W_o
    """

    def __init__(self, C, n_head, rng, proj_std=0.02, dtype=np.float32):
        super().__init__()
        assert C % n_head == 0, "d_model must be divisible by n_head"
        self.n_head, self.hs = n_head, C // n_head
        self.c_attn = Linear(C, 3 * C, rng, dtype=dtype)  # W_qkv: (C, 3C)
        self.c_proj = Linear(C, C, rng, std=proj_std, dtype=dtype)  # W_o: (C, C)

    def sublayers(self):
        return {"c_attn": self.c_attn, "c_proj": self.c_proj}

    def split_heads(self, z):  # (B, T, C) -> (B, nh, T, hs)
        B, T, C = z.shape
        return z.reshape(B, T, self.n_head, self.hs).transpose(0, 2, 1, 3)

    def merge_heads(self, z):  # (B, nh, T, hs) -> (B, T, C)
        B, nh, T, hs = z.shape
        return z.transpose(0, 2, 1, 3).reshape(B, T, nh * hs)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = np.split(self.c_attn.forward(x), 3, axis=-1)
        q, k, v = self.split_heads(q), self.split_heads(k), self.split_heads(v)
        scale = 1.0 / math.sqrt(self.hs)
        scores = (q @ k.transpose(0, 1, 3, 2)) * scale  # (B, nh, T, T)
        mask = np.tril(np.ones((T, T), dtype=bool))  # token t may see 0..t only
        scores = np.where(mask, scores, -np.inf)
        att = softmax(scores)  # exp(-inf) = 0: future tokens get weight exactly 0
        y = att @ v  # (B, nh, T, hs): weighted mix of the value vectors
        self.q, self.k, self.v, self.att, self.scale = q, k, v, att, scale
        return self.c_proj.forward(self.merge_heads(y))

    def backward(self, dout):
        q, k, v, att = self.q, self.k, self.v, self.att
        dy = self.split_heads(self.c_proj.backward(dout))  # (B, nh, T, hs)
        datt = dy @ v.transpose(0, 1, 3, 2)  # y = A v  ->  dA = dy vᵀ
        dv = att.transpose(0, 1, 3, 2) @ dy  # dv = Aᵀ dy
        # softmax backward, row by row: dS = A ⊙ (dA - Σ_j dA_j A_j).
        # Masked entries have A = 0, so they get zero gradient automatically.
        dscores = att * (datt - (datt * att).sum(axis=-1, keepdims=True))
        dscores *= self.scale
        dq = dscores @ k  # scores = q kᵀ  ->  dq = dS k,  dk = dSᵀ q
        dk = dscores.transpose(0, 1, 3, 2) @ q
        dqkv = np.concatenate([self.merge_heads(dq), self.merge_heads(dk),
                               self.merge_heads(dv)], axis=-1)
        return self.c_attn.backward(dqkv)


class MLP(Layer):
    """Position-wise feed-forward: Linear(C -> 4C) -> GELU -> Linear(4C -> C)."""

    def __init__(self, C, rng, proj_std=0.02, dtype=np.float32):
        super().__init__()
        self.c_fc = Linear(C, 4 * C, rng, dtype=dtype)
        self.gelu = GELU()
        self.c_proj = Linear(4 * C, C, rng, std=proj_std, dtype=dtype)

    def sublayers(self):
        return {"c_fc": self.c_fc, "c_proj": self.c_proj}

    def forward(self, x):
        return self.c_proj.forward(self.gelu.forward(self.c_fc.forward(x)))

    def backward(self, dy):
        return self.c_fc.backward(self.gelu.backward(self.c_proj.backward(dy)))


class Block(Layer):
    """One Transformer block, pre-LayerNorm (GPT-2 style):
        x = x + Attn(LN_1(x))
        x = x + MLP(LN_2(x))"""

    def __init__(self, C, n_head, rng, proj_std, dtype=np.float32):
        super().__init__()
        self.ln_1 = LayerNorm(C, dtype=dtype)
        self.attn = CausalSelfAttention(C, n_head, rng, proj_std, dtype)
        self.ln_2 = LayerNorm(C, dtype=dtype)
        self.mlp = MLP(C, rng, proj_std, dtype)

    def sublayers(self):
        return {"ln_1": self.ln_1, "attn": self.attn, "ln_2": self.ln_2, "mlp": self.mlp}

    def forward(self, x):
        x = x + self.attn.forward(self.ln_1.forward(x))
        x = x + self.mlp.forward(self.ln_2.forward(x))
        return x

    def backward(self, dx):
        # The residual "+" copies the gradient: one copy skips the sublayer
        # untouched (the gradient highway), the other goes through it.
        dx = dx + self.ln_2.backward(self.mlp.backward(dx))
        dx = dx + self.ln_1.backward(self.attn.backward(dx))
        return dx


def cross_entropy(logits, targets):
    """Mean cross-entropy over all positions, and its gradient w.r.t. the logits."""
    V = logits.shape[-1]
    z = logits.reshape(-1, V)
    t = targets.reshape(-1)
    N = len(t)
    probs = softmax(z)
    loss = -np.log(probs[np.arange(N), t]).mean()
    dz = probs
    dz[np.arange(N), t] -= 1.0  # p - onehot(target)
    return float(loss), (dz / N).reshape(logits.shape)


# ----------------------------------------------------------------------------
# The model
# ----------------------------------------------------------------------------
DEFAULT_CONFIG = dict(n_layer=2, n_head=4, d_model=64, block_size=32, tie_weights=False)


class GPT:
    def __init__(self, vocab_size, n_layer=2, n_head=4, d_model=64, block_size=32,
                 tie_weights=False, seed=0, dtype=np.float32):
        rng = np.random.default_rng(seed)
        self.config = dict(vocab_size=vocab_size, n_layer=n_layer, n_head=n_head,
                           d_model=d_model, block_size=block_size, tie_weights=tie_weights)
        C, V = d_model, vocab_size
        # GPT-2 init: N(0, 0.02), and the projections that write into the
        # residual stream scaled by 1/√(2·n_layer) so the stream doesn't grow with depth.
        proj_std = 0.02 / math.sqrt(2 * n_layer)
        self.wte = Embedding(V, C, rng, dtype=dtype)  # token embedding (V, C)
        self.wpe = Embedding(block_size, C, rng, dtype=dtype)  # position embedding (T, C)
        self.blocks = [Block(C, n_head, rng, proj_std, dtype) for _ in range(n_layer)]
        self.ln_f = LayerNorm(C, dtype=dtype)
        # Weight tying: reuse wte as the output matrix (logits = x @ wteᵀ).
        self.lm_head = None if tie_weights else Linear(C, V, rng, bias=False, dtype=dtype)

    # -- bookkeeping: a flat {name: layer} view so every param has a unique name
    def layers(self):
        out = {"wte": self.wte, "wpe": self.wpe}
        for i, blk in enumerate(self.blocks):
            for name, sub in blk.sublayers().items():
                if hasattr(sub, "sublayers"):
                    for name2, sub2 in sub.sublayers().items():
                        out[f"h.{i}.{name}.{name2}"] = sub2
                else:
                    out[f"h.{i}.{name}"] = sub
        out["ln_f"] = self.ln_f
        if self.lm_head is not None:
            out["lm_head"] = self.lm_head
        return out

    def params(self):
        return {f"{ln}.{pn}": p for ln, layer in self.layers().items()
                for pn, p in layer.params.items()}

    def grads(self):
        return {f"{ln}.{pn}": layer.grads[pn] for ln, layer in self.layers().items()
                for pn in layer.params}

    def n_params(self):
        return sum(p.size for p in self.params().values())

    # -- forward / backward
    def forward(self, idx, targets=None):
        """idx: (B, T) int ids. Returns logits (B, T, V), or the loss if targets given."""
        B, T = idx.shape
        assert T <= self.config["block_size"], "sequence longer than block_size"
        x = self.wte.forward(idx) + self.wpe.forward(np.arange(T))  # (B,T,C) + (T,C)
        for blk in self.blocks:
            x = blk.forward(x)
        x = self.ln_f.forward(x)
        self.x_final = x
        if self.lm_head is None:
            logits = x @ self.wte.params["weight"].T  # (B, T, C) @ (C, V)
        else:
            logits = self.lm_head.forward(x)
        if targets is None:
            return logits
        loss, self.dlogits = cross_entropy(logits, targets)
        return loss

    def backward(self):
        """Backprop the loss from the last forward(idx, targets) into every grad."""
        dlogits = self.dlogits
        C = self.config["d_model"]
        if self.lm_head is None:
            W = self.wte.params["weight"]
            dx = dlogits @ W
            d_tied = dlogits.reshape(-1, W.shape[0]).T @ self.x_final.reshape(-1, C)
        else:
            dx = self.lm_head.backward(dlogits)
        dx = self.ln_f.backward(dx)
        for blk in reversed(self.blocks):
            dx = blk.backward(dx)
        self.wte.backward(dx)  # x = tok + pos: both receive the same gradient
        self.wpe.backward(dx.sum(axis=0))  # pos is shared by every sequence in the batch
        if self.lm_head is None:
            self.wte.grads["weight"] += d_tied  # used twice -> gradients add
        return self.grads()

    def attention_maps(self):
        """Softmax weights of every layer from the last forward: list of (B, nh, T, T)."""
        return [blk.attn.att for blk in self.blocks]

    # -- generation (simple version: recompute everything each step;
    #    generate.py has the fast KV-cache version)
    def generate(self, ids, n, temperature=1.0, top_k=None, rng=None):
        rng = rng or np.random.default_rng()
        ids = list(ids)
        for _ in range(n):
            ctx = np.array(ids[-self.config["block_size"]:])[None]  # crop to block_size
            logits = self.forward(ctx)[0, -1]
            ids.append(sample_next(logits, temperature, top_k, rng))
        return ids


def sample_next(logits, temperature=1.0, top_k=None, rng=None):
    """Pick the next id from the logits of the last position."""
    rng = rng or np.random.default_rng()
    z = logits.astype(np.float64) / max(temperature, 1e-8)
    if top_k is not None and top_k < len(z):
        kth = np.sort(z)[-top_k]
        z = np.where(z < kth, -np.inf, z)  # keep only the k most likely
    return int(rng.choice(len(z), p=softmax(z)))


def param_count_formula(V, T, d, L, tied):
    """Exact parameter count of this architecture (README section 6)."""
    per_layer = 12 * d * d + 13 * d  # attn 4d²+4d, MLP 8d²+5d, two LNs 4d
    return L * per_layer + (V + T) * d + 2 * d + (0 if tied else V * d)


# ----------------------------------------------------------------------------
# Optimizer and schedule
# ----------------------------------------------------------------------------
class AdamW:
    """Adam with decoupled weight decay (Loshchilov & Hutter). Decay only
    matrices (dim >= 2), not biases or LayerNorm gains, as in GPT-2 / nanoGPT."""

    def __init__(self, params, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1):
        self.params = params
        self.b1, self.b2 = betas
        self.eps, self.wd = eps, weight_decay
        self.m = {k: np.zeros_like(p) for k, p in params.items()}
        self.v = {k: np.zeros_like(p) for k, p in params.items()}
        self.t = 0

    def step(self, grads, lr):
        self.t += 1
        for k, p in self.params.items():
            g = grads[k]
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g
            m_hat = self.m[k] / (1 - self.b1 ** self.t)  # bias correction
            v_hat = self.v[k] / (1 - self.b2 ** self.t)
            if p.ndim >= 2:
                p -= lr * self.wd * p  # decoupled: not mixed into the Adam statistics
            p -= lr * m_hat / (np.sqrt(v_hat) + self.eps)  # in place: layers see it


def lr_at(step, max_lr, warmup, total, min_ratio=0.1):
    """Linear warmup, then cosine decay to min_ratio·max_lr."""
    if step < warmup:
        return max_lr * (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return max_lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))


def clip_grad_norm(grads, max_norm):
    """Scale all gradients together so their global L2 norm is <= max_norm."""
    norm = float(np.sqrt(sum(float((g * g).sum()) for g in grads.values())))
    if norm > max_norm:
        for g in grads.values():
            g *= max_norm / (norm + 1e-6)
    return norm


# ----------------------------------------------------------------------------
# Data, training, evaluation
# ----------------------------------------------------------------------------
def load_text(path=DEFAULT_TEXT):
    with open(path, encoding="utf-8") as f:
        text = f.read()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    data = np.array([stoi[c] for c in text], dtype=np.int64)
    return text, chars, data


def get_batch(data, batch_size, block_size, rng):
    """batch_size random windows. Target = the input shifted left by one."""
    ix = rng.integers(0, len(data) - block_size - 1, batch_size)
    x = np.stack([data[i:i + block_size] for i in ix])
    y = np.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x, y


def evaluate(model, data, block_size=None):
    """Mean loss per character over the whole text, in non-overlapping windows."""
    T = block_size or model.config["block_size"]
    starts = np.arange(0, len(data) - T - 1, T)
    x = np.stack([data[i:i + T] for i in starts])
    y = np.stack([data[i + 1:i + T + 1] for i in starts])
    return model.forward(x, y)


def train(data, vocab_size, n_layer=2, n_head=4, d_model=64, block_size=32,
          tie_weights=False, batch_size=16, steps=2000, lr=3e-3, warmup=100,
          weight_decay=0.1, clip=1.0, seed=0, print_every=200, itos=None,
          verbose=True):
    """Train a GPT. Returns (model, history) with the loss of every step."""
    rng = np.random.default_rng(seed)
    model = GPT(vocab_size, n_layer, n_head, d_model, block_size, tie_weights, seed)
    opt = AdamW(model.params(), weight_decay=weight_decay)
    history = {"loss": [], "lr": [], "grad_norm": [], "time": []}
    t0 = time.time()
    for step in range(steps):
        x, y = get_batch(data, batch_size, block_size, rng)
        loss = model.forward(x, y)
        grads = model.backward()
        norm = clip_grad_norm(grads, clip)
        lr_now = lr_at(step, lr, warmup, steps)
        opt.step(grads, lr_now)
        history["loss"].append(loss)
        history["lr"].append(lr_now)
        history["grad_norm"].append(norm)
        history["time"].append(time.time() - t0)
        if verbose and (step % print_every == 0 or step == steps - 1):
            avg = np.mean(history["loss"][-print_every:])
            print(f"step {step:5d} | loss/char {loss:.3f} (avg {avg:.3f}) | "
                  f"ppl {np.exp(avg):6.2f} | lr {lr_now:.2e} | {time.time() - t0:5.1f} s")
            if itos is not None and step % (print_every * 5) == 0:
                s = model.generate([0], 80, temperature=0.8, rng=np.random.default_rng(step))
                print("   sample: " + repr("".join(itos[i] for i in s[1:])))
    return model, history


def save_model(path, model, chars, history=None):
    """Weights + vocabulary + config (+ the training curve, for make_figures.py)."""
    extra = {}
    if history is not None:
        extra = {f"history.{k}": np.array(v) for k, v in history.items()}
    np.savez(path, chars=np.array(chars), config=np.array(json.dumps(model.config)),
             **model.params(), **extra)


def load_model(path, dtype=np.float32):
    """Returns (model, chars, history-or-None)."""
    d = np.load(path)
    config = json.loads(str(d["config"]))
    model = GPT(dtype=dtype, **config)
    for name, p in model.params().items():
        p[...] = d[name]  # in place, so the layers' own arrays are updated
    history = {k.split(".", 1)[1]: d[k] for k in d.files if k.startswith("history.")}
    return model, [str(c) for c in d["chars"]], (history or None)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", default=DEFAULT_TEXT, help="training text (default ../rnn/input.txt)")
    ap.add_argument("--n-layer", type=int, default=DEFAULT_CONFIG["n_layer"])
    ap.add_argument("--n-head", type=int, default=DEFAULT_CONFIG["n_head"])
    ap.add_argument("--d-model", type=int, default=DEFAULT_CONFIG["d_model"])
    ap.add_argument("--block-size", type=int, default=DEFAULT_CONFIG["block_size"],
                    help="context length T")
    ap.add_argument("--tie-weights", action="store_true", help="reuse wte as the output layer")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--save", default=os.path.join(HERE, "model.npz"))
    args = ap.parse_args()

    text, chars, data = load_text(args.file)
    V = len(chars)
    itos = dict(enumerate(chars))
    print(f"data: {len(text)} chars, vocab size V = {V}, "
          f"uniform guessing = ln({V}) = {np.log(V):.3f} per char (perplexity {V})")
    probe = GPT(V, args.n_layer, args.n_head, args.d_model, args.block_size, args.tie_weights)
    n = probe.n_params()
    assert n == param_count_formula(V, args.block_size, args.d_model, args.n_layer,
                                    args.tie_weights)
    print(f"model: n_layer={args.n_layer} n_head={args.n_head} d_model={args.d_model} "
          f"block_size={args.block_size} -> {n:,} parameters "
          f"(12·L·d² = {12 * args.n_layer * args.d_model ** 2:,})\n")

    t0 = time.time()
    model, history = train(data, V, args.n_layer, args.n_head, args.d_model, args.block_size,
                           args.tie_weights, args.batch_size, args.steps, args.lr,
                           args.warmup, itos=itos)
    print(f"\ntrained in {time.time() - t0:.1f} s")

    final = evaluate(model, data)
    print(f"loss over the whole text: {final:.3f} per char = perplexity {np.exp(final):.2f} "
          f"(uniform guessing: {np.log(V):.3f}, perplexity {V})")

    rng = np.random.default_rng(0)
    ids = model.generate([chars.index(text[0])], 400, args.temperature, args.top_k, rng)
    print(f"\n===== sample (temperature {args.temperature}, top-k {args.top_k}) =====")
    print("".join(itos[i] for i in ids))

    save_model(args.save, model, chars, history)
    print(f"\nSaved model to {os.path.relpath(args.save)}. Try: python transformer/generate.py")


if __name__ == "__main__":
    main()
