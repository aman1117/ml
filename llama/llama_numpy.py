"""
A LLaMA-style Transformer (decoder only) from scratch in NumPy, with a
HAND-WRITTEN backward pass for every layer. It is the direct upgrade of the
mini-GPT in ../transformer/transformer_numpy.py and trains on the same text
(../rnn/input.txt) with the same settings, so the losses compare directly.

What changed from GPT-2 (README section 1):
    GPT-2                                   LLaMA (this file)
    x = wte[idx] + wpe[0..T-1]              x = tok_embeddings[idx]       (no position table)
    LayerNorm (γ, β, subtract the mean)     RMSNorm (γ only, no mean)
    q, k, v = x W_qkv + b                   q = x wq, k = x wk, v = x wv  (no biases; n_kv_head
                                            ≤ n_head key/value heads: grouped-query attention)
    learned absolute positions              RoPE: rotate q and k by position·θ_i
    MLP: GELU(x W_1 + b_1) W_2 + b_2        SwiGLU: (silu(x w1) ⊙ x w3) w2,  hidden ≈ 8/3·d

Forward pass (one row per token; x has shape (B, T, C)):
    x = tok_embeddings[idx]
    repeat n_layer times (pre-norm block):
        x = x + Attention(RMSNorm(x))       q, k rotated by RoPE; causal mask; GQA
        x = x + FeedForward(RMSNorm(x))     SwiGLU
    logits = RMSNorm(x) @ W_out             (B, T, V)
    L = mean over (b, t) of -log softmax(logits[b, t])[target[b, t]]

The parts that did NOT change (Linear, Embedding, softmax, cross-entropy, AdamW,
warmup + cosine schedule, gradient clipping, data loading) are imported unchanged
from ../transformer/transformer_numpy.py, so this file contains only what is new.
Parameter names follow Meta's LLaMA code (tok_embeddings, attention.wq/wk/wv/wo,
feed_forward.w1/w2/w3, attention_norm, ffn_norm, norm, output), so you can map
them onto a real checkpoint. gradient_check.py verifies every backward().

Run:
    python llama/llama_numpy.py
    python llama/llama_numpy.py --n-kv-head 4      # plain multi-head attention (MHA)
    python llama/llama_numpy.py --n-kv-head 1      # multi-query attention (MQA)
    python llama/llama_numpy.py --n-layer 4 --d-model 128 --steps 3000
"""
import argparse
import json
import math
import os
import sys
import time
import warnings

import numpy as np

# NumPy 2.0 + Apple Accelerate raises spurious "divide by zero / overflow /
# invalid value encountered in matmul" warnings even when results are correct.
# Only silence those; real numerical warnings (exp, log, ...) still show.
warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)

HERE = os.path.dirname(os.path.abspath(__file__))
# append (not insert at 0): ../transformer also has a generate.py, gradient_check.py and
# make_figures.py, and this folder's files of the same name must win.
sys.path.append(os.path.join(HERE, "..", "transformer"))
from transformer_numpy import (DEFAULT_TEXT, AdamW, Embedding, Layer, Linear,  # noqa: E402,F401
                               clip_grad_norm, cross_entropy, get_batch, load_text,
                               lr_at, sample_next, softmax)


# ----------------------------------------------------------------------------
# RMSNorm
# ----------------------------------------------------------------------------
class RMSNorm(Layer):
    """y = γ ⊙ x / rms(x),   rms(x) = √(mean(x²) + ε)   over the C features of each token.

    LayerNorm without the mean subtraction and without β (Zhang & Sennrich 2019)."""

    def __init__(self, C, eps=1e-5, dtype=np.float32):
        super().__init__()
        self.eps = eps
        self.params["weight"] = np.ones(C, dtype=dtype)

    def forward(self, x):
        self.rstd = 1.0 / np.sqrt((x * x).mean(axis=-1, keepdims=True) + self.eps)  # (..., 1)
        self.xhat = x * self.rstd
        return self.params["weight"] * self.xhat

    def backward(self, dy):
        xhat, rstd = self.xhat, self.rstd
        C = xhat.shape[-1]
        self.grads["weight"] = (dy * xhat).reshape(-1, C).sum(axis=0)
        dxhat = dy * self.params["weight"]
        # xhat_i = x_i·r with r = (mean(x²)+ε)^(-1/2), so ∂r/∂x_j = -r³ x_j / C and
        #   dx_j = r·dxhat_j - r³ x_j/C · Σ_i dxhat_i x_i = r·(dxhat_j - xhat_j·mean(dxhat ⊙ xhat)).
        # LayerNorm's backward had one more term (- mean(dxhat)) because it also subtracts μ.
        return rstd * (dxhat - xhat * (dxhat * xhat).mean(axis=-1, keepdims=True))


# ----------------------------------------------------------------------------
# RoPE: rotary position embedding
# ----------------------------------------------------------------------------
def rope_freqs(hs, base=10000.0):
    """θ_i = base^(-2i/hs) for the hs/2 pairs of dimensions (2i, 2i+1). Pair 0 turns
    1 radian per position; the last pair needs ~2π·base positions for one turn."""
    return base ** (-np.arange(0, hs, 2, dtype=np.float64) / hs)


def rope_angles(positions, hs, base=10000.0, dtype=np.float32):
    """cos and sin of position·θ_i: two (len(positions), hs/2) tables."""
    ang = np.outer(np.asarray(positions, dtype=np.float64), rope_freqs(hs, base))
    return np.cos(ang).astype(dtype), np.sin(ang).astype(dtype)


def rope_rotate(x, cos, sin):
    """Rotate every pair (x[2i], x[2i+1]) of the last axis by the angle whose cos/sin are given.

    x: (..., T, hs)   cos, sin: (T, hs/2). In complex numbers z_i = x[2i] + j·x[2i+1]
    this is z_i · e^{j·m·θ_i}. Calling it with -sin rotates backwards (the inverse)."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = np.empty_like(x)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out


class RoPE(Layer):
    """No parameters. Forward rotates by +m·θ_i, backward by -m·θ_i: a rotation is
    orthogonal (R⁻¹ = Rᵀ), and the gradient of y = R x is dx = Rᵀ dy."""

    def __init__(self, hs, base=10000.0):
        super().__init__()
        assert hs % 2 == 0, "RoPE needs an even head size (it rotates pairs of dims)"
        self.hs, self.base = hs, base

    def forward(self, x, positions=None):
        positions = np.arange(x.shape[-2]) if positions is None else positions
        self.cos, self.sin = rope_angles(positions, self.hs, self.base, x.dtype)
        return rope_rotate(x, self.cos, self.sin)

    def backward(self, dy):
        return rope_rotate(dy, self.cos, -self.sin)


# ----------------------------------------------------------------------------
# Grouped-query causal self-attention (with RoPE, no biases)
# ----------------------------------------------------------------------------
class Attention(Layer):
    """n_head query heads, n_kv_head key/value heads; each group of
    g = n_head / n_kv_head consecutive query heads shares one K/V head
    (query head h uses KV head h // g, like torch.repeat_interleave).
    n_kv_head = n_head: ordinary multi-head attention (MHA); n_kv_head = 1: MQA.

    Shapes: x (B, T, C) -> q (B, nh, T, hs), k, v (B, nkv, T, hs)
            q is viewed as (B, nkv, g, T, hs) and k, v as (B, nkv, 1, T, hs), so one
            broadcast matmul gives scores (B, nkv, g, T, T) without copying K or V."""

    def __init__(self, C, n_head, n_kv_head, rng, proj_std=0.02, rope_base=10000.0,
                 dtype=np.float32):
        super().__init__()
        assert C % n_head == 0, "d_model must be divisible by n_head"
        assert n_head % n_kv_head == 0, "n_kv_head must divide n_head"
        self.n_head, self.n_kv_head = n_head, n_kv_head
        self.hs = C // n_head
        self.groups = n_head // n_kv_head
        self.wq = Linear(C, n_head * self.hs, rng, bias=False, dtype=dtype)
        self.wk = Linear(C, n_kv_head * self.hs, rng, bias=False, dtype=dtype)
        self.wv = Linear(C, n_kv_head * self.hs, rng, bias=False, dtype=dtype)
        self.wo = Linear(n_head * self.hs, C, rng, std=proj_std, bias=False, dtype=dtype)
        self.rope_q, self.rope_k = RoPE(self.hs, rope_base), RoPE(self.hs, rope_base)
        self.scale = 1.0 / math.sqrt(self.hs)

    def sublayers(self):
        return {"wq": self.wq, "wk": self.wk, "wv": self.wv, "wo": self.wo}

    def split_heads(self, z, n):  # (B, T, n·hs) -> (B, n, T, hs)
        B, T, _ = z.shape
        return z.reshape(B, T, n, self.hs).transpose(0, 2, 1, 3)

    @staticmethod
    def merge_heads(z):  # (B, n, T, hs) -> (B, T, n·hs)
        B, n, T, hs = z.shape
        return z.transpose(0, 2, 1, 3).reshape(B, T, n * hs)

    def forward(self, x, window=None):
        """window: if given, token i only sees tokens i-window+1 .. i (sliding-window
        attention, as in Mistral; generate.py uses it to check its rolling cache)."""
        B, T, C = x.shape
        nh, nkv, g, hs = self.n_head, self.n_kv_head, self.groups, self.hs
        q = self.rope_q.forward(self.split_heads(self.wq.forward(x), nh))  # (B, nh, T, hs)
        k = self.rope_k.forward(self.split_heads(self.wk.forward(x), nkv))  # (B, nkv, T, hs)
        v = self.split_heads(self.wv.forward(x), nkv)
        qg = q.reshape(B, nkv, g, T, hs)
        scores = (qg @ k[:, :, None].transpose(0, 1, 2, 4, 3)) * self.scale  # (B,nkv,g,T,T)
        i, j = np.arange(T)[:, None], np.arange(T)[None, :]
        mask = j <= i  # causal: token i sees 0..i
        if window is not None:
            mask &= i - j < window
        att = softmax(np.where(mask, scores, -np.inf))
        y = att @ v[:, :, None]  # (B, nkv, g, T, hs)
        self.qg, self.k, self.v, self.att = qg, k, v, att
        return self.wo.forward(self.merge_heads(y.reshape(B, nh, T, hs)))

    def backward(self, dout):
        qg, k, v, att = self.qg, self.k, self.v, self.att
        B, nkv, g, T, hs = qg.shape
        dy = self.split_heads(self.wo.backward(dout), self.n_head).reshape(B, nkv, g, T, hs)
        datt = dy @ v[:, :, None].transpose(0, 1, 2, 4, 3)  # y = A v -> dA = dy vᵀ
        # One K/V head was used by all g query heads of its group, so its gradient
        # is the SUM over that group (axis 2). This is the only new thing in GQA's backward.
        dv = (att.transpose(0, 1, 2, 4, 3) @ dy).sum(axis=2)  # (B, nkv, T, hs)
        dscores = att * (datt - (datt * att).sum(axis=-1, keepdims=True)) * self.scale
        dq = (dscores @ k[:, :, None]).reshape(B, nkv * g, T, hs)
        dk = (dscores.transpose(0, 1, 2, 4, 3) @ qg).sum(axis=2)
        dq = self.rope_q.backward(dq)  # un-rotate: rotate by -m·θ
        dk = self.rope_k.backward(dk)
        return (self.wq.backward(self.merge_heads(dq)) + self.wk.backward(self.merge_heads(dk))
                + self.wv.backward(self.merge_heads(dv)))


# ----------------------------------------------------------------------------
# SwiGLU feed-forward
# ----------------------------------------------------------------------------
def ffn_hidden_size(d, multiple_of=16, multiplier=None):
    """LLaMA's rule: 2/3 of 4d (so 3 matrices cost what GPT's 2 did), optionally scaled
    (Llama 2 70B / Llama 3 use 1.3), rounded UP to a multiple of `multiple_of`.
    d = 64 -> int(170.67) = 170 -> 176.   d = 4096, multiple_of = 256 -> 11008 (LLaMA-7B)."""
    hidden = int(2 * 4 * d / 3)
    if multiplier is not None:
        hidden = int(multiplier * hidden)
    return multiple_of * ((hidden + multiple_of - 1) // multiple_of)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class FeedForward(Layer):
    """SwiGLU (Shazeer 2020): y = (silu(x w1) ⊙ (x w3)) w2,  silu(a) = a·σ(a).
    w1, w3: (C, H)   w2: (H, C). The 'gate' x w1 decides, per hidden unit, how much of
    the 'value' x w3 gets through."""

    def __init__(self, C, hidden, rng, proj_std=0.02, dtype=np.float32):
        super().__init__()
        self.w1 = Linear(C, hidden, rng, bias=False, dtype=dtype)
        self.w3 = Linear(C, hidden, rng, bias=False, dtype=dtype)
        self.w2 = Linear(hidden, C, rng, std=proj_std, bias=False, dtype=dtype)

    def sublayers(self):
        return {"w1": self.w1, "w2": self.w2, "w3": self.w3}

    def forward(self, x):
        a, b = self.w1.forward(x), self.w3.forward(x)
        s = sigmoid(a)
        self.a, self.b, self.s, self.silu = a, b, s, a * s
        return self.w2.forward(self.silu * b)

    def backward(self, dy):
        dh = self.w2.backward(dy)
        db = dh * self.silu
        # silu'(a) = σ(a) + a·σ(a)(1 - σ(a)) = σ(a)(1 + a(1 - σ(a)))
        da = dh * self.b * self.s * (1.0 + self.a * (1.0 - self.s))
        return self.w1.backward(da) + self.w3.backward(db)


# ----------------------------------------------------------------------------
# Block and model
# ----------------------------------------------------------------------------
class Block(Layer):
    """Pre-norm block:  x = x + Attention(RMSNorm(x));  x = x + FeedForward(RMSNorm(x))."""

    def __init__(self, C, n_head, n_kv_head, ffn_hidden, rng, proj_std, rope_base=10000.0,
                 eps=1e-5, dtype=np.float32):
        super().__init__()
        self.attention_norm = RMSNorm(C, eps, dtype)
        self.attention = Attention(C, n_head, n_kv_head, rng, proj_std, rope_base, dtype)
        self.ffn_norm = RMSNorm(C, eps, dtype)
        self.feed_forward = FeedForward(C, ffn_hidden, rng, proj_std, dtype)

    def sublayers(self):
        return {"attention_norm": self.attention_norm, "attention": self.attention,
                "ffn_norm": self.ffn_norm, "feed_forward": self.feed_forward}

    def forward(self, x, window=None):
        x = x + self.attention.forward(self.attention_norm.forward(x), window)
        x = x + self.feed_forward.forward(self.ffn_norm.forward(x))
        return x

    def backward(self, dx):
        dx = dx + self.ffn_norm.backward(self.feed_forward.backward(dx))
        dx = dx + self.attention_norm.backward(self.attention.backward(dx))
        return dx


DEFAULT_CONFIG = dict(n_layer=2, n_head=4, n_kv_head=2, d_model=64, block_size=32,
                      ffn_hidden=176, rope_base=10000.0, tie_weights=False, norm_eps=1e-5)


class LLaMA:
    def __init__(self, vocab_size, n_layer=2, n_head=4, n_kv_head=2, d_model=64, block_size=32,
                 ffn_hidden=None, rope_base=10000.0, tie_weights=False, norm_eps=1e-5, seed=0,
                 dtype=np.float32):
        rng = np.random.default_rng(seed)
        ffn_hidden = ffn_hidden or ffn_hidden_size(d_model)
        self.config = dict(vocab_size=vocab_size, n_layer=n_layer, n_head=n_head,
                           n_kv_head=n_kv_head, d_model=d_model, block_size=block_size,
                           ffn_hidden=ffn_hidden, rope_base=rope_base, tie_weights=tie_weights,
                           norm_eps=norm_eps)
        C, V = d_model, vocab_size
        # Same init as our GPT: N(0, 0.02), residual-writing projections (wo, w2) / √(2L).
        proj_std = 0.02 / math.sqrt(2 * n_layer)
        self.tok_embeddings = Embedding(V, C, rng, dtype=dtype)  # no position table!
        self.blocks = [Block(C, n_head, n_kv_head, ffn_hidden, rng, proj_std, rope_base,
                             norm_eps, dtype) for _ in range(n_layer)]
        self.norm = RMSNorm(C, norm_eps, dtype)
        self.output = None if tie_weights else Linear(C, V, rng, bias=False, dtype=dtype)

    # -- bookkeeping: a flat {name: layer} view so every param has a unique name
    def layers(self):
        out = {"tok_embeddings": self.tok_embeddings}
        for i, blk in enumerate(self.blocks):
            for name, sub in blk.sublayers().items():
                if hasattr(sub, "sublayers"):
                    for name2, sub2 in sub.sublayers().items():
                        out[f"layers.{i}.{name}.{name2}"] = sub2
                else:
                    out[f"layers.{i}.{name}"] = sub
        out["norm"] = self.norm
        if self.output is not None:
            out["output"] = self.output
        return out

    def params(self):
        return {f"{ln}.{pn}": p for ln, layer in self.layers().items()
                for pn, p in layer.params.items()}

    def grads(self):
        return {f"{ln}.{pn}": layer.grads[pn] for ln, layer in self.layers().items()
                for pn in layer.params}

    def n_params(self):
        return sum(p.size for p in self.params().values())

    def head_weight(self):
        """(C, V) output matrix: W_out, or tok_embeddingsᵀ when tied."""
        if self.output is None:
            return self.tok_embeddings.params["weight"].T
        return self.output.params["W"]

    # -- forward / backward
    def forward(self, idx, targets=None, window=None):
        """idx: (B, T) int ids. Returns logits (B, T, V), or the loss if targets given.
        Unlike GPT there is no hard limit T <= block_size (there is no position table),
        but the model only learned offsets < block_size; `window` caps them."""
        x = self.tok_embeddings.forward(idx)
        for blk in self.blocks:
            x = blk.forward(x, window)
        x = self.norm.forward(x)
        self.x_final = x
        if self.output is None:
            logits = x @ self.tok_embeddings.params["weight"].T  # tied: (B,T,C) @ (C,V)
        else:
            logits = self.output.forward(x)
        if targets is None:
            return logits
        loss, self.dlogits = cross_entropy(logits, targets)
        return loss

    def backward(self):
        """Backprop the loss from the last forward(idx, targets) into every grad."""
        dlogits = self.dlogits
        C = self.config["d_model"]
        if self.output is None:
            W = self.tok_embeddings.params["weight"]
            dx = dlogits @ W
            d_tied = dlogits.reshape(-1, W.shape[0]).T @ self.x_final.reshape(-1, C)
        else:
            dx = self.output.backward(dlogits)
        dx = self.norm.backward(dx)
        for blk in reversed(self.blocks):
            dx = blk.backward(dx)
        self.tok_embeddings.backward(dx)  # the only input: no position table to update
        if self.output is None:
            self.tok_embeddings.grads["weight"] += d_tied  # used twice -> gradients add
        return self.grads()

    def attention_maps(self):
        """Softmax weights of every layer from the last forward: list of (B, nh, T, T)."""
        out = []
        for blk in self.blocks:
            a = blk.attention.att
            out.append(a.reshape(a.shape[0], -1, *a.shape[-2:]))
        return out

    # -- generation (simple version: recompute the last block_size tokens each step;
    #    generate.py has the fast sliding KV-cache version)
    def generate(self, ids, n, temperature=1.0, top_k=None, rng=None):
        rng = rng or np.random.default_rng()
        ids = list(ids)
        for _ in range(n):
            ctx = np.array(ids[-self.config["block_size"]:])[None]
            logits = self.forward(ctx)[0, -1]
            ids.append(sample_next(logits, temperature, top_k, rng))
        return ids


def param_count_formula(V, d, L, n_head, n_kv_head, ffn_hidden, tied):
    """Exact parameter count (README section 11)."""
    hs = d // n_head
    attn = d * d + 2 * d * n_kv_head * hs + d * d  # wq, wk, wv, wo
    ffn = 3 * d * ffn_hidden  # w1, w3: (d, H), w2: (H, d)
    per_layer = attn + ffn + 2 * d  # + two RMSNorm gains
    return L * per_layer + V * d + d + (0 if tied else V * d)


# ----------------------------------------------------------------------------
# Training and evaluation (the loop is the same as ../transformer/)
# ----------------------------------------------------------------------------
def evaluate(model, data, block_size=None):
    """Mean loss per character over the whole text, in non-overlapping windows."""
    T = block_size or model.config["block_size"]
    starts = np.arange(0, len(data) - T - 1, T)
    x = np.stack([data[i:i + T] for i in starts])
    y = np.stack([data[i + 1:i + T + 1] for i in starts])
    return model.forward(x, y)


def train(data, vocab_size, n_layer=2, n_head=4, n_kv_head=2, d_model=64, block_size=32,
          ffn_hidden=None, rope_base=10000.0, tie_weights=False, batch_size=16, steps=2000,
          lr=3e-3, warmup=100, weight_decay=0.1, clip=1.0, seed=0, print_every=200, itos=None,
          verbose=True):
    """Train a LLaMA. Same loop, batches, optimizer and schedule as ../transformer/'s
    train(), so the two models see identical data in identical order."""
    rng = np.random.default_rng(seed)
    model = LLaMA(vocab_size, n_layer, n_head, n_kv_head, d_model, block_size, ffn_hidden,
                  rope_base, tie_weights, seed=seed)
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
    model = LLaMA(dtype=dtype, **config)
    for name, p in model.params().items():
        p[...] = d[name]  # in place, so the layers' own arrays are updated
    history = {k.split(".", 1)[1]: d[k] for k in d.files if k.startswith("history.")}
    return model, [str(c) for c in d["chars"]], (history or None)


def to_float64(model32):
    """A float64 copy with the same weights (for exactness checks)."""
    model = LLaMA(dtype=np.float64, **model32.config)
    for (_, p), (_, p32) in zip(model.params().items(), model32.params().items()):
        p[...] = p32
    return model


def get_trained_model(dtype=np.float64, verbose=True):
    """The model trained with the default settings: loaded from llama/model.npz if it is
    there, otherwise trained now (~6 s, same seed, so identical weights) and saved.
    Returns (model, chars, history)."""
    path = os.path.join(HERE, "model.npz")
    _, chars, data = load_text(DEFAULT_TEXT)
    want = dict(vocab_size=len(chars), **DEFAULT_CONFIG)
    if os.path.exists(path):
        model, chars, hist = load_model(path, dtype)
        if model.config == want and hist is not None and len(hist["loss"]) == 2000:
            return model, chars, hist
    if verbose:
        print("(no llama/model.npz with the default settings yet: training one, ~6 s)")
    model32, hist = train(data, len(chars), verbose=False)
    save_model(path, model32, chars, hist)
    model = to_float64(model32) if dtype == np.float64 else model32
    return model, chars, {k: np.array(v) for k, v in hist.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", default=DEFAULT_TEXT, help="training text (default ../rnn/input.txt)")
    ap.add_argument("--n-layer", type=int, default=DEFAULT_CONFIG["n_layer"])
    ap.add_argument("--n-head", type=int, default=DEFAULT_CONFIG["n_head"])
    ap.add_argument("--n-kv-head", type=int, default=DEFAULT_CONFIG["n_kv_head"],
                    help="key/value heads: = n-head for MHA, 1 for MQA, in between for GQA")
    ap.add_argument("--d-model", type=int, default=DEFAULT_CONFIG["d_model"])
    ap.add_argument("--block-size", type=int, default=DEFAULT_CONFIG["block_size"],
                    help="training context length T")
    ap.add_argument("--ffn-hidden", type=int, default=None,
                    help="SwiGLU hidden size (default: 8/3·d rounded up to a multiple of 16)")
    ap.add_argument("--rope-base", type=float, default=DEFAULT_CONFIG["rope_base"])
    ap.add_argument("--tie-weights", action="store_true", help="reuse tok_embeddings as output")
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
    probe = LLaMA(V, args.n_layer, args.n_head, args.n_kv_head, args.d_model, args.block_size,
                  args.ffn_hidden, args.rope_base, args.tie_weights)
    n = probe.n_params()
    H = probe.config["ffn_hidden"]
    assert n == param_count_formula(V, args.d_model, args.n_layer, args.n_head, args.n_kv_head,
                                    H, args.tie_weights)
    kind = ("MHA" if args.n_kv_head == args.n_head else
            "MQA" if args.n_kv_head == 1 else "GQA")
    print(f"model: n_layer={args.n_layer} n_head={args.n_head} n_kv_head={args.n_kv_head} "
          f"({kind}, {args.n_head // args.n_kv_head} query heads per KV head) "
          f"d_model={args.d_model} ffn_hidden={H} block_size={args.block_size}\n"
          f"       -> {n:,} parameters (no biases, no position table)\n")

    t0 = time.time()
    model, history = train(data, V, args.n_layer, args.n_head, args.n_kv_head, args.d_model,
                           args.block_size, args.ffn_hidden, args.rope_base, args.tie_weights,
                           args.batch_size, args.steps, args.lr, args.warmup, itos=itos)
    print(f"\ntrained in {time.time() - t0:.1f} s")

    final = evaluate(model, data)
    print(f"loss over the whole text: {final:.3f} per char = perplexity {np.exp(final):.2f} "
          f"(uniform guessing: {np.log(V):.3f}, perplexity {V})")

    rng = np.random.default_rng(0)
    ids = model.generate([chars.index(text[0])], 400, args.temperature, args.top_k, rng)
    print(f"\n===== sample (temperature {args.temperature}, top-k {args.top_k}) =====")
    print("".join(itos[i] for i in ids))

    save_model(args.save, model, chars, history)
    print(f"\nSaved model to {os.path.relpath(args.save)}. Try: python llama/generate.py")


if __name__ == "__main__":
    main()
