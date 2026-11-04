"""
The serving tricks in PyTorch, on the same mini-GPT (../transformer/transformer_pytorch.py).

  1. A KV-cache decode loop for the PyTorch module (prefill with
     F.scaled_dot_product_attention, then one-token steps that append to the
     cache), checked against a full recompute (float64, every position), and the
     trained model's greedy generation checked token-for-token against NumPy's
     generate.py.
  2. torch.ao.quantization.quantize_dynamic: int8 nn.Linear weights (and int8
     activations, quantized on the fly per batch) vs our NumPy int8 from
     quantize_numpy.py: error on held-out text, and speed on a bigger model.
  3. bf16 weights and activations.
  4. torch.compile on the decode step (falls back gracefully if this machine has no
     working compiler toolchain).
  5. Batched decode throughput vs batch size on CPU: the step time barely moves
     while the batch is small (memory-bound: the weights are read once per step
     whatever the batch), then grows with it (compute-bound).

Run (~60 s):
    python serving/serving_pytorch.py
    python serving/serving_pytorch.py --no-compile
"""
from __future__ import annotations

import argparse
import contextlib
import os
import sys
import time
import warnings

import numpy as np
import torch
import torch.nn.functional as F

from common import (TRANSFORMER_DIR, CharTokenizer, generate_ids, get_model, heldout_ids,
                    softmax)
from quantize_numpy import quantize_model as np_quantize
from quantize_numpy import time_decode

sys.path.insert(0, TRANSFORMER_DIR)
from transformer_pytorch import GPT, copy_numpy_weights  # noqa: E402

torch.set_num_threads(min(8, os.cpu_count() or 1))


@contextlib.contextmanager
def quiet_fds():
    """Silence C++-level warnings and subprocess chatter (qnnpack, inductor probing for
    a compiler/OpenMP) by pointing file descriptors 1 and 2 at /dev/null for a while."""
    sys.stdout.flush()
    sys.stderr.flush()
    saved = [os.dup(1), os.dup(2)]
    null = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(null, 1)
        os.dup2(null, 2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved[0], 1)
        os.dup2(saved[1], 2)
        for fd in saved + [null]:
            os.close(fd)


# ----------------------------------------------------------------------------
# 1. KV cache for the PyTorch module
# ----------------------------------------------------------------------------
def _qkv(blk, x):
    B, T, C = x.shape
    nh = blk.attn.n_head
    q, k, v = blk.attn.c_attn(blk.ln_1(x)).split(C, dim=2)
    return [z.view(B, T, nh, C // nh).transpose(1, 2) for z in (q, k, v)]  # (B, nh, T, hs)


def _rest(blk, x, y):
    B, nh, T, hs = y.shape
    x = x + blk.attn.c_proj(y.transpose(1, 2).reshape(B, T, nh * hs))
    return x + blk.mlp(blk.ln_2(x))


@torch.no_grad()
def prefill(model, idx):
    """idx (B, T) -> last logits (B, V) and the cache [(K, V) per layer], (B, nh, T, hs)."""
    T = idx.shape[1]
    x = model.wte(idx) + model.wpe(torch.arange(T))
    cache = []
    for blk in model.blocks:
        q, k, v = _qkv(blk, x)
        x = _rest(blk, x, F.scaled_dot_product_attention(q, k, v, is_causal=True))
        cache.append([k, v])
    return model.lm_head(model.ln_f(x[:, -1])), cache


@torch.no_grad()
def decode(model, tok, pos, cache):
    """One new token per sequence (B,) at position pos; appends to the cache in place."""
    x = (model.wte(tok) + model.wpe(torch.as_tensor(pos)))[:, None]  # (B, 1, C)
    for blk, kv in zip(model.blocks, cache):
        q, k, v = _qkv(blk, x)
        kv[0] = torch.cat([kv[0], k], dim=2)
        kv[1] = torch.cat([kv[1], v], dim=2)
        x = _rest(blk, x, F.scaled_dot_product_attention(q, kv[0], kv[1]))  # no mask: all past
    return model.lm_head(model.ln_f(x[:, 0]))


@torch.no_grad()
def generate_greedy(model, ids, n, T):
    """generate.py's semantics (window full -> keep the last T/2 tokens), greedy."""
    ids = list(ids)
    logits, cache = prefill(model, torch.tensor([ids[-T:]]))
    out = []
    for _ in range(n):
        nxt = int(logits[0].argmax())
        out.append(nxt)
        ids.append(nxt)
        if cache[0][0].shape[2] == T:
            logits, cache = prefill(model, torch.tensor([ids[-(T // 2):]]))
        else:
            logits = decode(model, torch.tensor([nxt]), cache[0][0].shape[2], cache)
    return out


def check_cache(np_model):
    """Prefill 1 token + decode the rest == one full forward (float64)."""
    cfg = np_model.config
    pt = GPT(cfg["vocab_size"], cfg["n_layer"], cfg["n_head"], cfg["d_model"],
             cfg["block_size"]).double().eval()
    copy_numpy_weights(np_model, pt)
    idx = torch.randint(0, cfg["vocab_size"], (3, cfg["block_size"]),
                        generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        full = pt(idx)  # (B, T, V)
    logits, cache = prefill(pt, idx[:, :1])
    steps = [logits]
    for t in range(1, idx.shape[1]):
        steps.append(decode(pt, idx[:, t], t, cache))
    return float((torch.stack(steps, 1) - full).abs().max())


# ----------------------------------------------------------------------------
# 2-4. Quantization, bf16, compile
# ----------------------------------------------------------------------------
def to_torch(np_model, dtype=torch.float32):
    cfg = np_model.config
    pt = GPT(cfg["vocab_size"], cfg["n_layer"], cfg["n_head"], cfg["d_model"],
             cfg["block_size"]).eval()
    copy_numpy_weights(np_model, pt)
    return pt.to(dtype)


def torch_int8(pt):
    """Dynamic quantization: Linear weights stored int8 (per tensor); activations
    quantized to int8 on the fly per batch; int8 GEMM with int32 accumulation."""
    if "qnnpack" in torch.backends.quantized.supported_engines and \
            torch.backends.quantized.engine == "none":
        torch.backends.quantized.engine = "qnnpack"  # the ARM backend
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return torch.ao.quantization.quantize_dynamic(pt, {torch.nn.Linear}, dtype=torch.qint8)


@torch.no_grad()
def torch_quality(pt, ref_probs, x, y):
    logits = pt(torch.from_numpy(x)).double()
    p = torch.softmax(logits, -1).numpy()
    loss = float(-np.log(np.take_along_axis(p, y[..., None], -1)).mean())
    kl = float((ref_probs * (np.log(ref_probs + 1e-12) - np.log(p + 1e-12))).sum(-1).mean())
    return loss, float((p.argmax(-1) == ref_probs.argmax(-1)).mean()), kl


def numpy_quality(m, ref_probs, x, y):
    p = softmax(m.forward(x).astype(np.float64))
    loss = float(-np.log(np.take_along_axis(p, y[..., None], -1)).mean())
    kl = float((ref_probs * (np.log(ref_probs + 1e-12) - np.log(p + 1e-12))).sum(-1).mean())
    return loss, float((p.argmax(-1) == ref_probs.argmax(-1)).mean()), kl


def decode_time(pt, B=1, ctx=64, steps=30):
    """Median seconds per decode step at batch B with ctx cached tokens."""
    V = pt.lm_head.out_features
    idx = torch.randint(0, V, (B, ctx), generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        _, cache = prefill(pt, idx)
        tok = torch.zeros(B, dtype=torch.long)
        ts = []
        for i in range(steps):
            c = [[k[:, :, :ctx], v[:, :, :ctx]] for k, v in cache]  # same length every step
            t0 = time.perf_counter()
            decode(pt, tok, ctx, c)
            ts.append(time.perf_counter() - t0)
    return float(np.median(ts[3:]))


def big_model(d=1024, n_layer=4, V=1024, T=256):
    torch.manual_seed(0)
    return GPT(V, n_layer, 8, d, T).eval()


def try_compile(pt, B=1, ctx=64):
    """torch.compile the decode step. Returns (compile s, eager ms, compiled ms) or the error."""
    try:
        import torch._dynamo
        torch._dynamo.config.suppress_errors = False
        cdec = torch.compile(decode, dynamic=True)
        V = pt.lm_head.out_features
        idx = torch.randint(0, V, (B, ctx), generator=torch.Generator().manual_seed(0))
        with torch.no_grad():
            _, cache = prefill(pt, idx)
            tok = torch.zeros(B, dtype=torch.long)
            fresh = lambda: [[k[:, :, :ctx].clone(), v[:, :, :ctx].clone()] for k, v in cache]
            t0 = time.perf_counter()
            out_c = cdec(pt, tok, ctx, fresh())
            t_compile = time.perf_counter() - t0
            out_e = decode(pt, tok, ctx, fresh())
            diff = float((out_c - out_e).abs().max())
            ts_c, ts_e = [], []
            for _ in range(20):
                c = fresh()
                t0 = time.perf_counter()
                cdec(pt, tok, ctx, c)
                ts_c.append(time.perf_counter() - t0)
                c = fresh()
                t0 = time.perf_counter()
                decode(pt, tok, ctx, c)
                ts_e.append(time.perf_counter() - t0)
        return dict(ok=True, compile_s=t_compile, eager_ms=np.median(ts_e) * 1e3,
                    compiled_ms=np.median(ts_c) * 1e3, diff=diff)
    except Exception as e:  # no C++ toolchain, unsupported platform, ...
        return dict(ok=False, error=f"{type(e).__name__}: {str(e).splitlines()[0][:150]}")


# ----------------------------------------------------------------------------
# 5. Throughput vs batch size
# ----------------------------------------------------------------------------
def throughput_curve(batches=(1, 2, 4, 8, 16, 32, 64, 128, 256), d=1024, n_layer=4, ctx=64):
    pt = big_model(d, n_layer)
    n_params = sum(p.numel() for p in pt.parameters())
    rows = []
    for B in batches:
        t = decode_time(pt, B, ctx, steps=15)
        rows.append(dict(B=B, step_ms=t * 1e3, tok_s=B / t))
    return dict(rows=rows, n_params=n_params, d=d, n_layer=n_layer, ctx=ctx)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-compile", action="store_true")
    args = ap.parse_args()

    np64, chars, _ = get_model(np.float64)
    tok = CharTokenizer(chars)
    T = np64.config["block_size"]

    print("\n== 1. KV-cache decode loop in PyTorch ==")
    diff = check_cache(np64)
    print(f"  prefill(1 token) + {T - 1} decode steps vs one full forward, 3 sequences, "
          f"float64: max |logit diff| = {diff:.1e}")
    assert diff < 1e-10
    pt64 = to_torch(np64, torch.float64)
    for p in ("The keeper", "One winter night"):
        ids = tok.encode(p)[0]
        a = generate_greedy(pt64, ids, 120, T)
        b = generate_ids(np64, ids, 120, temperature=0.0, rng=np.random.default_rng(0))
        print(f"  greedy, 120 tokens (3 window slides) from {p!r}: PyTorch == NumPy "
              f"generate.py: {a == b}")

    print("\n== 2-3. int8 (torch dynamic quantization vs ours) and bf16, on the trained model ==")
    np32 = get_model(np.float32, verbose=False)[0]
    held = heldout_ids(tok)
    starts = np.arange(0, len(held) - T - 1, T)
    x = np.stack([held[i:i + T] for i in starts])
    y = np.stack([held[i + 1:i + T + 1] for i in starts])
    ref = softmax(np32.forward(x).astype(np.float64))
    pt32 = to_torch(np32)
    with quiet_fds():
        q8 = torch_quality(torch_int8(to_torch(np32)), ref, x, y)
    rows = [("torch fp32", torch_quality(pt32, ref, x, y)),
            ("torch int8 dynamic", q8),
            ("torch bf16", torch_quality(to_torch(np32, torch.bfloat16), ref, x, y)),
            ("numpy int8 per-tensor", numpy_quality(np_quantize(np32, "int8-tensor"), ref, x, y)),
            ("numpy int8 per-channel", numpy_quality(np_quantize(np32, "int8"), ref, x, y))]
    print(f"  {'variant':24s} {'held-out loss':>13s} {'top-1 = fp32':>13s} {'KL vs fp32':>11s}")
    for name, (loss, agree, kl) in rows:
        print(f"  {name:24s} {loss:13.4f} {agree:13.1%} {kl:11.5f}")
    print("  torch's dynamic int8 also quantizes the ACTIVATIONS (per batch), so its error is\n"
          "  larger than our weight-only int8; per-channel weights beat per-tensor.")

    big = big_model()
    n = sum(p.numel() for p in big.parameters())
    print(f"\n  Speed: one decode step, batch 1, 64 cached tokens, random {n / 1e6:.0f}M-param "
          f"GPT (d=1024, 4 layers):")
    t_fp32 = decode_time(big)
    with quiet_fds():
        t_int8 = decode_time(torch_int8(big_model()))
    t_bf16 = decode_time(big_model().to(torch.bfloat16))
    from common import GPT as NumpyGPT
    npbig = NumpyGPT(1024, 4, 8, 1024, 256, seed=0)
    t_np = time_decode(npbig, 30)
    t_np8 = time_decode(np_quantize(npbig, "int8"), 30)
    for name, t in (("torch fp32", t_fp32), ("torch int8 dynamic", t_int8),
                    ("torch bf16", t_bf16), ("numpy fp32", t_np),
                    ("numpy int8 (dequant on the fly)", t_np8)):
        print(f"  {name:32s} {t * 1e3:7.2f} ms/token  ({t_fp32 / t:4.2f}x torch fp32)")
    print(f"  weights: fp32 {4 * n / 1e6:.0f} MB, int8 {n / 1e6:.0f} MB, bf16 {2 * n / 1e6:.0f} MB."
          "\n  Smaller weights only pay off with fast low-bit kernels on memory-bound hardware\n"
          "  (GPUs, x86 VNNI). Here qnnpack also re-quantizes the activations every step, and\n"
          "  NumPy's dequantize-then-matmul materializes fp32 weights: both are SLOWER.")

    print("\n== 4. torch.compile on the decode step ==")
    if args.no_compile:
        print("  skipped (--no-compile)")
    else:
        with quiet_fds():
            r = try_compile(big)
        if r["ok"]:
            print(f"  compiled in {r['compile_s']:.1f} s; eager {r['eager_ms']:.2f} ms vs "
                  f"compiled {r['compiled_ms']:.2f} ms per step "
                  f"({r['eager_ms'] / r['compiled_ms']:.2f}x), max |diff| {r['diff']:.1e}")
        else:
            print(f"  torch.compile is not usable here ({r['error']}).\n"
                  f"  That is fine: eager mode is the reference; compile is an optimization.")

    tc = throughput_curve()
    print(f"\n== 5. Batched decode throughput vs batch size (the same {tc['n_params'] / 1e6:.0f}M-"
          f"param model, fp32) ==")
    for r in tc["rows"]:
        print(f"  batch {r['B']:4d}: {r['step_ms']:7.2f} ms/step -> {r['tok_s']:9,.0f} tokens/s")
    r1, rb = tc["rows"][0], tc["rows"][-1]
    print(f"  batch 1 -> {rb['B']}: {rb['B']}x the tokens per step for "
          f"{rb['step_ms'] / r1['step_ms']:.1f}x the time. At small batch every step re-reads "
          f"{4 * tc['n_params'] / 1e6:.0f} MB\n  of weights for very little arithmetic "
          f"(memory-bound); at large batch the matmuls dominate (compute-bound).")


if __name__ == "__main__":
    main()
