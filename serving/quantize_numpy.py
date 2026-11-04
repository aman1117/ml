"""
Weight quantization from scratch: store the GPT's weights in 8 or 4 bits instead
of 32, and measure what it costs in quality.

Why: decoding is memory-bound (batching.py section 5), so the time per token is
roughly (bytes of weights + KV cache) / memory bandwidth. Fewer bytes per weight =
faster tokens, a smaller GPU, or room for a bigger batch.

The schemes (W is our Linear's (n_in, n_out) matrix; an output channel is a column):
  int8 absmax PER-TENSOR   s = max|W| / 127          q = round(W / s) ∈ [-127, 127]
  int8 absmax PER-CHANNEL  s_j = max_i |W_ij| / 127  one scale per output column
  int4 GROUP-WISE          every `group` consecutive inputs of a column share
                           s = max|W_g| / 7, q ∈ [-8, 7], two 4-bit values per byte
  dequantize:              Ŵ = q · s, and |W - Ŵ| <= s / 2 elementwise (rounding)

QLinear is a drop-in replacement for transformer_numpy.Linear that keeps only q
and the scales and dequantizes on the fly inside the matmul (the scale is
factored out: y_j = s_j · Σ_i x_i q_ij). Then:
  1. size, max/mean weight error and loss / bits-per-char on held-out text for
     fp32, fp16, int8 (per-tensor, per-channel) and int4 (per-channel, group 64, 32);
  2. the OUTLIER problem: one column scaled up (without changing what the model
     computes) ruins per-tensor int8 but not per-channel or group-wise;
  3. KV-cache quantization (int8 / int4 per token and head).

Run (~10 s):
    python serving/quantize_numpy.py
"""
from __future__ import annotations

import argparse
import copy
import math
import time

import numpy as np

from common import CharTokenizer, evaluate, get_model, heldout_ids, lm_logits, softmax, train_ids

LINEAR_NAMES = ("attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj")


# ----------------------------------------------------------------------------
# Quantizers
# ----------------------------------------------------------------------------
def _safe(s):
    return np.where(s == 0, 1.0, s).astype(np.float32)  # an all-zero column: any scale works


def quant_int8_tensor(W):
    s = _safe(np.abs(W).max() / 127.0)
    return np.clip(np.round(W / s), -127, 127).astype(np.int8), s


def quant_int8_channel(W):
    s = _safe(np.abs(W).max(axis=0) / 127.0)  # (n_out,): one scale per output column
    return np.clip(np.round(W / s), -127, 127).astype(np.int8), s


def quant_int4_group(W, group):
    """Groups of `group` consecutive input rows within each column. q: (n_in, n_out)
    int8 holding values in [-8, 7]; s: (n_in/group, n_out)."""
    n_in, n_out = W.shape
    assert n_in % group == 0, (n_in, group)
    Wg = W.reshape(n_in // group, group, n_out)
    s = _safe(np.abs(Wg).max(axis=1) / 7.0)
    q = np.clip(np.round(Wg / s[:, None]), -8, 7).astype(np.int8)
    return q.reshape(n_in, n_out), s


def pack_int4(q):
    """Two signed 4-bit values per byte along axis 0: byte = (q[2i] & 0xF) | (q[2i+1] << 4).
    Negative numbers are stored in two's complement: -1 -> 0b1111, -8 -> 0b1000."""
    u = (q.astype(np.int16) & 0x0F).astype(np.uint8)
    return (u[0::2] | (u[1::2] << 4)).astype(np.uint8)


def unpack_int4(p):
    lo, hi = (p & 0x0F).astype(np.int8), (p >> 4).astype(np.int8)
    q = np.empty((2 * p.shape[0],) + p.shape[1:], dtype=np.int8)
    q[0::2], q[1::2] = lo, hi
    return (q ^ 8) - 8  # sign-extend a nibble: 0..7 stay, 8..15 -> -8..-1


class QLinear:
    """y = x Ŵ + b with Ŵ stored as int8 / packed int4 / fp16 plus scales."""

    def __init__(self, lin, scheme, group=32):
        W = lin.params["W"]
        self.scheme, self.group, self.dtype = scheme, group, W.dtype
        self.b = lin.params.get("b")
        self.shape = W.shape
        if scheme == "fp16":
            self.q, self.s = W.astype(np.float16), None
        elif scheme == "int8-tensor":
            self.q, self.s = quant_int8_tensor(W)
        elif scheme == "int8":
            self.q, self.s = quant_int8_channel(W)
        elif scheme == "int4":
            g = W.shape[0] if group is None else min(group, W.shape[0])
            self.group = g
            q, self.s = quant_int4_group(W, g)
            self.q = pack_int4(q)  # (n_in/2, n_out) bytes: half a byte per weight
        else:
            raise ValueError(scheme)

    def dequant(self):
        if self.scheme == "fp16":
            return self.q.astype(self.dtype)
        if self.scheme == "int4":
            q = unpack_int4(self.q).astype(self.dtype)
            n_in, n_out = self.shape
            return (q.reshape(n_in // self.group, self.group, n_out)
                    * self.s[:, None]).reshape(n_in, n_out)
        return self.q.astype(self.dtype) * self.s.astype(self.dtype)

    def forward(self, x):
        if self.scheme == "int4":
            # dequantize-on-the-fly, one group at a time: y = Σ_g s_g ⊙ (x_g @ q_g)
            n_in, n_out = self.shape
            G = n_in // self.group
            q = unpack_int4(self.q).astype(x.dtype).reshape(G, self.group, n_out)
            xg = x.reshape(-1, G, self.group).transpose(1, 0, 2)  # (G, N, group)
            y = ((xg @ q) * self.s[:, None].astype(x.dtype)).sum(axis=0)
            y = y.reshape(x.shape[:-1] + (n_out,))
        elif self.scheme == "fp16":
            y = x @ self.q.astype(x.dtype)
        else:
            # the scale is constant along the sum over inputs, so it comes out of
            # the matmul: an int8 kernel multiplies int8 values and rescales once.
            y = (x @ self.q.astype(x.dtype)) * self.s.astype(x.dtype)
        return y + self.b if self.b is not None else y

    @property
    def params(self):
        """Materialized (dequantized) weights, for code that reads W directly."""
        out = {"W": self.dequant()}
        if self.b is not None:
            out["b"] = self.b
        return out

    def nbytes(self):
        n = self.q.nbytes + (0 if self.s is None else self.s.nbytes)
        return n + (0 if self.b is None else self.b.nbytes)


def linears(model):
    """{name: (owner, attribute)} of every Linear we quantize (all but embeddings/LN)."""
    out = {}
    for i, blk in enumerate(model.blocks):
        for name in LINEAR_NAMES:
            mod, attr = name.split(".")
            out[f"h.{i}.{name}"] = (getattr(blk, mod), attr)
    if model.lm_head is not None:
        out["lm_head"] = (model, "lm_head")
    return out


def quantize_model(model, scheme, group=32):
    """A copy of the model with every Linear replaced by a QLinear ('fp32' = plain copy)."""
    m = copy.deepcopy(model)
    if scheme != "fp32":
        for owner, attr in linears(m).values():
            setattr(owner, attr, QLinear(getattr(owner, attr), scheme, group))
    return m


def linear_bytes(model):
    total = 0
    for owner, attr in linears(model).values():
        lin = getattr(owner, attr)
        total += (lin.nbytes() if isinstance(lin, QLinear)
                  else sum(p.size * 4 for p in lin.params.values()))
    return total


def model_bytes(model):
    """Bytes to store the model: Linears at their (quantized) size, the rest fp32."""
    rest = sum(p.size * 4 for name, layer in model.layers().items()
               if not (name == "lm_head" or name.endswith(LINEAR_NAMES))
               for p in layer.params.values())
    return linear_bytes(model) + rest


# ----------------------------------------------------------------------------
# Measurements
# ----------------------------------------------------------------------------
def weight_errors(ref, qm):
    errs = []
    for name, (owner, attr) in linears(qm).items():
        lin = getattr(owner, attr)
        W = getattr(linears(ref)[name][0], linears(ref)[name][1]).params["W"]
        errs.append(np.abs(W - lin.params["W"]).ravel())
    e = np.concatenate(errs)
    return float(e.max()), float(e.mean())


def next_token_stats(ref, qm, data):
    """Top-1 agreement and mean KL(p_fp32 || p_quant) of next-char distributions."""
    T = ref.config["block_size"]
    starts = np.arange(0, len(data) - T - 1, T)
    x = np.stack([data[i:i + T] for i in starts])
    p = softmax(ref.forward(x).astype(np.float64))
    q = softmax(qm.forward(x).astype(np.float64))
    kl = (p * (np.log(p + 1e-12) - np.log(q + 1e-12))).sum(-1).mean()
    return float((p.argmax(-1) == q.argmax(-1)).mean()), float(kl)


def evaluate_variant(ref, scheme, group, held, train):
    qm = quantize_model(ref, scheme, group)
    max_e, mean_e = weight_errors(ref, qm) if scheme != "fp32" else (0.0, 0.0)
    agree, kl = next_token_stats(ref, qm, held)
    h = evaluate(qm, held)
    return dict(scheme=scheme, group=group, bytes=model_bytes(qm), linear_bytes=linear_bytes(qm),
                max_err=max_e, mean_err=mean_e, held_loss=h, held_bpc=h / math.log(2),
                train_loss=evaluate(qm, train), agree=agree, kl=kl, model=qm)


VARIANTS = [("fp32", None, "fp32"), ("fp16", None, "fp16"),
            ("int8-tensor", None, "int8 per-tensor"), ("int8", None, "int8 per-channel"),
            ("int4", None, "int4 per-channel"), ("int4", 64, "int4 group 64"),
            ("int4", 32, "int4 group 32")]


OUTLIER_CHANNEL = 5


def make_outlier_model(model, layer=0, channel=OUTLIER_CHANNEL, factor=50.0):
    """Scale ONE output column of W_v (inside c_attn) by `factor` and divide the
    matching input row of W_o (attn.c_proj) by it. Attention is linear in v, so the
    model computes EXACTLY the same function, but c_attn now has an outlier column
    50x larger than the rest, like the outlier features of real LLMs."""
    m = copy.deepcopy(model)
    a = m.blocks[layer].attn
    C = m.config["d_model"]
    a.c_attn.params["W"][:, 2 * C + channel] *= factor
    a.c_attn.params["b"][2 * C + channel] *= factor
    a.c_proj.params["W"][channel, :] /= factor
    return m


def forward_kv_quant(model, idx, bits):
    """Full forward where every K and V vector goes through the (quantized) cache:
    absmax per (token, head) vector, int8 or int4. Returns logits."""
    B, T = idx.shape
    x = model.wte.params["weight"][idx] + model.wpe.params["weight"][np.arange(T)]
    mask = np.tril(np.ones((T, T), dtype=bool))
    qmax = 2 ** (bits - 1) - 1

    def kvq(z):  # (B, nh, T, hs): one scale per head vector of each token
        s = _safe(np.abs(z).max(axis=-1, keepdims=True) / qmax)
        return np.clip(np.round(z / s), -qmax - 1, qmax) * s

    for blk in model.blocks:
        a = blk.attn
        q, k, v = np.split(a.c_attn.forward(blk.ln_1.forward(x)), 3, axis=-1)
        q, k, v = a.split_heads(q), a.split_heads(k), a.split_heads(v)
        if bits < 32:
            k, v = kvq(k), kvq(v)
        att = softmax(np.where(mask, (q @ k.transpose(0, 1, 3, 2)) / math.sqrt(a.hs), -np.inf))
        x = x + a.c_proj.forward(a.merge_heads(att @ v))
        x = x + blk.mlp.forward(blk.ln_2.forward(x))
    return lm_logits(model, model.ln_f.forward(x))


def kv_quant_eval(model, data, bits):
    T = model.config["block_size"]
    starts = np.arange(0, len(data) - T - 1, T)
    x = np.stack([data[i:i + T] for i in starts])
    y = np.stack([data[i + 1:i + T + 1] for i in starts])
    p = softmax(forward_kv_quant(model, x, bits).astype(np.float64))
    return float(-np.log(np.take_along_axis(p, y[..., None], -1) + 1e-12).mean())


def kv_bytes_per_token(model, bits):
    L, C, nh = len(model.blocks), model.config["d_model"], model.config["n_head"]
    if bits == 32:
        return 2 * L * C * 4
    return 2 * L * C * bits // 8 + 2 * L * nh * 4  # values + one fp32 scale per head vector


def time_decode(model, n=200):
    from common import decode_step, prefill
    _, cache = prefill(model, [1, 2, 3])
    ts = []
    for i in range(n):
        if len(cache) == model.config["block_size"]:
            _, cache = prefill(model, [1, 2, 3])
        t0 = time.perf_counter()
        decode_step(model, i % 40, cache)
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def run_all(model, tok, verbose=True):
    held, train = heldout_ids(tok), train_ids(tok)
    rows = [dict(evaluate_variant(model, s, g, held, train), label=label)
            for s, g, label in VARIANTS]
    out_model = make_outlier_model(model)
    outlier = [dict(evaluate_variant(out_model, s, g, held, train), label=label)
               for s, g, label in VARIANTS if s in ("fp32", "int8-tensor", "int8")
               or (s == "int4" and g == 32)]
    C = model.config["d_model"]
    W = out_model.blocks[0].attn.c_attn.params["W"]
    others = np.ones(W.shape[1], dtype=bool)
    others[2 * C + OUTLIER_CHANNEL] = False
    for r in outlier:  # what happened to the 191 NORMAL columns of the outlier matrix
        Wq = r["model"].blocks[0].attn.c_attn.params["W"]
        r["zero_frac"] = float((Wq[:, others] == 0).mean())
        r["rel_err"] = float(np.abs(Wq - W)[:, others].mean() / np.abs(W[:, others]).mean())
    kv = [dict(bits=b, held_loss=kv_quant_eval(model, held, b),
               bytes_per_token=kv_bytes_per_token(model, b)) for b in (32, 8, 4)]
    return dict(rows=rows, outlier=outlier, kv=kv)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args()
    model, chars, _ = get_model()
    tok = CharTokenizer(chars)

    print("\n== Pack/unpack: 4-bit values two per byte ==")
    q = np.array([[-8], [7], [-1], [0], [3], [-5]], dtype=np.int8)
    p = pack_int4(q)
    print(f"  q = {q.ravel().tolist()} -> bytes {[f'0x{b:02x}' for b in p.ravel()]} -> "
          f"unpacked {unpack_int4(p).ravel().tolist()}")

    print("\n== One matrix: h.0.mlp.c_fc.W (64 x 256) ==")
    W = model.blocks[0].mlp.c_fc.params["W"]
    for scheme, group, label in VARIANTS[2:]:
        ql = QLinear(model.blocks[0].mlp.c_fc, scheme, group)
        err = np.abs(W - ql.dequant())
        bound = (ql.s / 2).max()
        print(f"  {label:17s} {ql.q.nbytes + ql.s.nbytes:6,d} bytes (fp32: {W.nbytes:,}) | "
              f"max err {err.max():.5f} <= max scale/2 {bound:.5f} | mean err {err.mean():.6f}")

    res = run_all(model, tok)
    fp = res["rows"][0]
    print(f"\n== Whole model: every Linear quantized (embeddings, LayerNorms, biases stay "
          f"fp32) ==\n  held-out text: serving/heldout.txt ({len(heldout_ids(tok))} chars "
          f"never trained on); train text: ../rnn/input.txt (memorized)")
    print(f"  {'variant':17s} {'size':>9s} {'x smaller':>9s} {'max|dW|':>8s} {'mean|dW|':>9s}"
          f" {'held bpc':>9s} {'Δ':>7s} {'train loss':>10s} {'top-1 =':>8s} {'KL':>8s}")
    for r in res["rows"]:
        print(f"  {r['label']:17s} {r['bytes']:9,d} {fp['bytes'] / r['bytes']:9.2f} "
              f"{r['max_err']:8.5f} {r['mean_err']:9.6f} {r['held_bpc']:9.4f} "
              f"{r['held_bpc'] - fp['held_bpc']:+7.4f} {r['train_loss']:10.4f} "
              f"{r['agree']:8.1%} {r['kl']:8.5f}")

    print("\n== The outlier problem: W_v column 5 of layer 0 x50, W_o row 5 / 50 "
          "(same function) ==")
    for r in res["outlier"]:
        print(f"  {r['label']:17s} other columns: {r['zero_frac']:6.1%} rounded to 0, "
              f"mean rel. error {r['rel_err']:6.1%} | train loss {r['train_loss']:.4f} | "
              f"held bpc {r['held_bpc']:.4f} | top-1 agreement {r['agree']:6.1%}")
    print("  fp32 is unchanged (the transform is exact). Per-tensor int8 now has a scale 50x\n"
          "  too big for every other weight; per-channel and group scales are local, so only\n"
          "  the outlier column pays. This is why LLM.int8() keeps outlier features in fp16,\n"
          "  SmoothQuant moves activation outliers into the weights, and GPTQ/AWQ quantize\n"
          "  per group with error compensation / activation-aware scaling.")

    print("\n== KV-cache quantization (per token and head, absmax) ==")
    for r in res["kv"]:
        name = "fp32" if r["bits"] == 32 else f"int{r['bits']}"
        print(f"  {name:5s} {r['bytes_per_token']:5d} bytes/token | held-out loss "
              f"{r['held_loss']:.4f} ({r['held_loss'] - res['kv'][0]['held_loss']:+.4f})")

    print("\n== Speed of one decode step in NumPy (median of 200) ==")
    for scheme, group in (("fp32", None), ("int8", None), ("int4", 32)):
        ms = time_decode(quantize_model(model, scheme, group)) * 1e3
        print(f"  {scheme:5s} {ms:.3f} ms")
    print("  NumPy has no int8/int4 matmul kernels, so dequantizing on the fly is SLOWER\n"
          "  here. The speed-up of real kernels (and bitsandbytes / GPTQ / AWQ / llama.cpp)\n"
          "  comes from moving 4-8x fewer bytes in the memory-bound decode step.")


if __name__ == "__main__":
    main()
