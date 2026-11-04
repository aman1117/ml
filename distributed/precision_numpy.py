"""
Number formats and mixed-precision training, from scratch in NumPy.

What this teaches (README sections 2-3):
  1. The bit layouts of fp32, fp16, bf16 and the two FP8 formats (e4m3, e5m2):
     value = (-1)^sign · 1.mantissa · 2^(exponent - bias), and what that means
     for RANGE (exponent bits) and PRECISION (mantissa bits).
  2. A generic "round to any format" function (round-to-nearest-even, with
     subnormals), checked against NumPy's own float16 and against the classic
     bf16 trick on the raw float32 bits.
  3. Why fp16 needs LOSS SCALING: real gradients of the mini-GPT from
     ../transformer/ fall below fp16's smallest number (2^-24 ≈ 6e-8) and
     flush to zero; multiplying the loss by S shifts them back into range.
     Static scaling, and dynamic scaling (skip the step and halve S on overflow).
  4. Why bf16 does NOT need it (same 8 exponent bits as fp32), and why every
     recipe keeps an fp32 MASTER copy of the weights: in bf16, 1 + 1e-4 == 1,
     so small updates vanish. Stochastic rounding is the alternative.
  5. Trains the mini-GPT of ../transformer/transformer_numpy.py with every
     tensor that crosses a layer boundary rounded to the chosen format
     (the way fp16/bf16 kernels store their outputs; matmuls accumulate in
     fp32 as tensor cores do) and compares the losses:
        fp32 | fp16, no scaling | fp16 + static scaling | fp16 + dynamic scaling
        | bf16 (fp32 master) | bf16 weights, no master | bf16, no master + stochastic rounding

Run (~80 s):
    python distributed/precision_numpy.py
    python distributed/precision_numpy.py --steps 2000
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")  # regenerated outputs (gitignored)
sys.path.insert(0, os.path.join(HERE, "..", "transformer"))
from transformer_numpy import (GPT, AdamW, clip_grad_norm, evaluate, get_batch,  # noqa: E402
                               load_text, lr_at)

# ----------------------------------------------------------------------------
# 1. The formats: (exponent bits, mantissa bits, bias, kind)
#    kind "ieee": the all-ones exponent means inf/NaN.
#    kind "fn"  : FP8 e4m3 ("finite"): no inf; only S.1111.111 is NaN, so the
#                 all-ones exponent still holds numbers and the max is 448.
# ----------------------------------------------------------------------------
FORMATS = {
    "fp32": (8, 23, 127, "ieee"),
    "fp16": (5, 10, 15, "ieee"),
    "bf16": (8, 7, 127, "ieee"),
    "fp8_e4m3": (4, 3, 7, "fn"),
    "fp8_e5m2": (5, 2, 15, "ieee"),
}


def format_info(name):
    """Range and precision of a format, straight from its bit counts."""
    e_bits, m_bits, bias, kind = FORMATS[name]
    if kind == "ieee":  # top exponent reserved for inf / NaN
        max_val = (2.0 - 2.0 ** -m_bits) * 2.0 ** ((2 ** e_bits - 2) - bias)
    else:  # e4m3fn: top exponent usable, except mantissa all-ones (= NaN)
        max_val = (2.0 - 2.0 ** (1 - m_bits)) * 2.0 ** ((2 ** e_bits - 1) - bias)
    return dict(
        bits=1 + e_bits + m_bits, e_bits=e_bits, m_bits=m_bits, bias=bias,
        max=max_val,
        min_normal=2.0 ** (1 - bias),
        min_subnormal=2.0 ** (1 - bias - m_bits),
        eps=2.0 ** -m_bits,  # gap between 1 and the next number
        digits=(m_bits + 1) * math.log10(2),  # ≈ significant decimal digits
    )


def round_to_format(x, name):
    """Round x to the nearest number representable in `name` (ties to even),
    including subnormals and overflow (inf for IEEE formats, saturate to ±448
    for e4m3, as FP8 casts with scaling do in practice). Returns float64."""
    e_bits, m_bits, bias, kind = FORMATS[name]
    x = np.asarray(x, dtype=np.float64)
    a = np.abs(x)
    _, e = np.frexp(a)  # a = f · 2^e with f in [0.5, 1)  ->  leading bit is 2^(e-1)
    e = np.maximum(e - 1, 1 - bias)  # below the min normal, the spacing stays fixed (subnormals)
    step = np.ldexp(1.0, e - m_bits)  # distance between neighbouring numbers at this exponent
    with np.errstate(invalid="ignore"):
        q = np.round(a / step) * step  # np.round rounds halves to even: exactly IEEE RNE
    mx = format_info(name)["max"]
    q = np.minimum(q, mx) if kind == "fn" else np.where(q > mx, np.inf, q)
    q = np.where(np.isnan(x), np.nan, q)
    return np.copysign(q, x)


def encode_bits(x, name):
    """(sign, exponent, mantissa) bit strings of x after rounding to `name`."""
    e_bits, m_bits, bias, kind = FORMATS[name]
    q = float(round_to_format(x, name))
    sign = "1" if math.copysign(1.0, q) < 0 else "0"
    a = abs(q)
    if math.isnan(q):
        exp, man = 2 ** e_bits - 1, 2 ** m_bits - 1
    elif math.isinf(a):
        exp, man = 2 ** e_bits - 1, 0
    elif a == 0:
        exp, man = 0, 0
    else:
        e = math.frexp(a)[1] - 1
        if e < 1 - bias:  # subnormal: exponent field 0, no implicit leading 1
            exp, man = 0, int(round(a / 2.0 ** (1 - bias - m_bits)))
        else:
            exp, man = e + bias, int(round((a / 2.0 ** e - 1.0) * 2 ** m_bits))
    return sign, format(exp, f"0{e_bits}b"), format(man, f"0{m_bits}b")


# ----------------------------------------------------------------------------
# 2. Fast quantizers used during training (float32 in, float32 out)
# ----------------------------------------------------------------------------
def to_fp16(x):
    """Round float32 to fp16 on the raw bits (4x faster than NumPy's float16 cast,
    identical results; equivalence_check.py verifies it). Normal range: round to
    nearest even at 10 mantissa bits (add 0xFFF + the last kept bit, clear 13 bits).
    Subnormal range (|x| < 2^-14): fixed spacing 2^-24. Too big: ±inf."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    u = x.view(np.uint32)
    lsb = (u >> 13) & 1
    r = ((u + np.uint32(0x0FFF) + lsb) & np.uint32(0xFFFFE000)).view(np.float32)
    r = np.where(np.abs(x) < np.float32(2.0 ** -14),
                 np.round(x * np.float32(2.0 ** 24)) * np.float32(2.0 ** -24), r)
    r = np.where(np.abs(r) > np.float32(65504), np.copysign(np.float32(np.inf), x), r)
    return np.where(np.isnan(x), x, r)


def to_bf16(x):
    """bf16 = the top 16 bits of a float32. Round to nearest even on the bits:
    add 0x7FFF (+1 if the last kept bit is 1, so exact halves go to even),
    then clear the low 16 bits. Carries ripple into the exponent correctly."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    u = x.view(np.uint32)
    lsb = (u >> 16) & 1
    r = ((u + np.uint32(0x7FFF) + lsb) & np.uint32(0xFFFF0000)).view(np.float32)
    return np.where(np.isnan(x), np.float32(np.nan), r)


def to_bf16_stochastic(x, rng):
    """Stochastic rounding: add uniform random noise below the kept bits, then
    truncate. Rounds up with probability = the distance to the lower neighbour,
    so E[result] = x exactly and tiny updates survive on average."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    noise = rng.integers(0, 1 << 16, size=x.shape, dtype=np.uint32)
    r = ((x.view(np.uint32) + noise) & np.uint32(0xFFFF0000)).view(np.float32)
    return np.where(np.isfinite(x), r, x)


QUANTIZERS = {"fp32": lambda x: np.asarray(x, np.float32), "fp16": to_fp16, "bf16": to_bf16}


# ----------------------------------------------------------------------------
# 3. Low-precision emulation of the transformer_numpy GPT
# ----------------------------------------------------------------------------
def emulate_precision(model, q, recorder=None):
    """Round every tensor that crosses a layer boundary (activations going
    forward, gradients going backward, parameter gradients) with q. Inside a
    layer, NumPy computes in fp32, like a tensor core accumulating in fp32.
    The softmax/cross-entropy stays in fp32, as autocast does.
    `recorder`, if given, collects every incoming activation gradient."""
    layers = list(model.layers().values())
    for blk in model.blocks:
        layers += [blk.mlp.gelu, blk]
    for layer in layers:
        fwd, bwd = layer.forward, layer.backward

        def forward(x, fwd=fwd):
            if np.issubdtype(np.asarray(x).dtype, np.floating):
                x = q(x)
            return q(fwd(x))

        def backward(dy, bwd=bwd, layer=layer):
            dy = q(dy)
            if recorder is not None and layer is not model.blocks[0]:
                recorder.append(dy)
            dx = bwd(dy)
            for k in layer.grads:
                layer.grads[k] = q(layer.grads[k])
            return None if dx is None else q(dx)

        layer.forward, layer.backward = forward, backward
    return model


class DynamicLossScaler:
    """torch.cuda.amp.GradScaler's algorithm: start high; if any gradient is
    inf/NaN, SKIP the step and halve the scale; after `growth_interval`
    clean steps in a row, double it. The scale hovers just below overflow.
    We start deliberately high (2^32; PyTorch's default is 2^16) so you can
    watch it back off from overflow in the first steps."""

    def __init__(self, init_scale=2.0 ** 32, growth_interval=200, factor=2.0):
        self.scale, self.growth_interval, self.factor = init_scale, growth_interval, factor
        self.good_steps, self.skipped = 0, 0

    def update(self, found_inf):
        if found_inf:
            self.scale /= self.factor
            self.good_steps = 0
            self.skipped += 1
        else:
            self.good_steps += 1
            if self.good_steps % self.growth_interval == 0:
                self.scale *= self.factor


MODES = {
    # key: (label, compute format, loss scaling, fp32 master weights?, weight rounding)
    "fp32": ("fp32", "fp32", None, True, None),
    "fp16": ("fp16, no scaling", "fp16", None, True, None),
    "fp16-static": ("fp16 + static scale", "fp16", "static", True, None),
    "fp16-dynamic": ("fp16 + dynamic scale", "fp16", "dynamic", True, None),
    "bf16": ("bf16 (fp32 master)", "bf16", None, True, None),
    "bf16-nomaster": ("bf16, no master", "bf16", None, False, "nearest"),
    "bf16-nomaster-sr": ("bf16, no master + SR", "bf16", None, False, "stochastic"),
}
STATIC_SCALE = 2.0 ** 10
# Real runs average the loss over far more tokens than our 512 per step: with
# gradient accumulation each micro-batch's loss is divided by the number of
# micro-batches k before its (fp16) backward, and the fp32 gradient buffer adds
# them up. We emulate one micro-batch of a k = 1024 accumulation, i.e. a
# 524,288-token (0.5M) batch, GPT-3 Small's batch size: divide by 1024 before
# the backward, multiply by 1024 in fp32 after. In fp32/bf16 this changes
# nothing; in fp16 it decides which gradients survive. (--accum 1 = plain 512 tokens.)
ACCUM = 1024


def train_mode(mode, data, vocab_size, steps=1000, batch_size=16, block_size=32,
               lr=3e-3, warmup=100, seed=0, accum=ACCUM):
    """Train the default mini-GPT under one precision recipe. The batches,
    init and schedule are identical across modes; only the rounding differs.

    Loss is mean over the B·T tokens as usual, so dlogits = (p - onehot)/(B·T)."""
    _, fmt, scaling, master_fp32, weight_rounding = MODES[mode]
    q = QUANTIZERS[fmt]
    model = emulate_precision(GPT(vocab_size, block_size=block_size, seed=seed), q)
    params = model.params()  # the arrays the layers compute with (low-precision values)
    master = {k: p.copy() for k, p in params.items()}  # what the optimizer updates
    if not master_fp32:  # the weights themselves live in bf16
        for p in master.values():
            p[...] = q(p)
    opt = AdamW(master)
    scaler = DynamicLossScaler() if scaling == "dynamic" else None
    rng, sr_rng = np.random.default_rng(seed), np.random.default_rng(seed + 1)
    hist = {"loss": [], "scale": [], "skipped": [], "zero_frac": [], "param_zero": []}
    for step in range(steps):
        for k, p in params.items():  # compute copy = master cast to the low format
            p[...] = q(master[k])
        x, y = get_batch(data, batch_size, block_size, rng)
        loss = model.forward(x, y)
        S = STATIC_SCALE if scaling == "static" else (scaler.scale if scaler else 1.0)
        # Loss · S/accum: S is the loss scale, 1/accum the gradient-accumulation
        # weight of one micro-batch. Both multiply every gradient of the backward.
        model.dlogits *= S / accum
        grads = {k: g.astype(np.float32) * np.float32(accum / S)  # unscale in fp32
                 for k, g in model.backward().items()}
        hist["zero_frac"].append(float(np.mean(q(model.dlogits) == 0)))
        hist["param_zero"].append(float(np.mean(np.concatenate([(g == 0).ravel() for g in grads.values()]))))
        found_inf = not all(np.isfinite(g).all() for g in grads.values())
        if scaler is not None:
            scaler.update(found_inf)
        hist["loss"].append(loss)
        hist["scale"].append(S)
        hist["skipped"].append(found_inf)
        if found_inf:  # skip: never let inf/NaN into the weights or Adam's state
            continue
        clip_grad_norm(grads, 1.0)
        opt.step(grads, lr_at(step, lr, warmup, steps))
        if weight_rounding == "nearest":
            for p in master.values():
                p[...] = to_bf16(p)
        elif weight_rounding == "stochastic":
            for p in master.values():
                p[...] = to_bf16_stochastic(p, sr_rng)
    clean = GPT(vocab_size, block_size=block_size, seed=seed)  # un-emulated, for evaluation
    for k, p in clean.params().items():
        p[...] = master[k]
    return clean, hist


def gradient_census(model, data, batch_size=16, block_size=32, seed=1):
    """Real fp32 gradients of a model: the logit gradient, every activation
    gradient that crosses a layer boundary, and every parameter gradient."""
    rec = []
    emulate_precision(model, QUANTIZERS["fp32"], recorder=rec)
    x, y = get_batch(data, batch_size, block_size, np.random.default_rng(seed))
    model.forward(x, y)
    dlogits = model.dlogits.copy()
    pgrads = model.backward()
    acts = np.concatenate([r.ravel() for r in rec])
    return dict(dlogits=dlogits.ravel(), activations=acts,
                params=np.concatenate([g.ravel() for g in pgrads.values()]))


def underflow_stats(g, scale=1.0):
    """Fractions of non-zero fp32 values that become 0 / subnormal in fp16."""
    a = np.abs(g[g != 0]).astype(np.float64) * scale
    return dict(zero=float(np.mean(a < 2.0 ** -25)),  # rounds to 0 (half the min subnormal)
                subnormal=float(np.mean((a >= 2.0 ** -25) & (a < 2.0 ** -14))),
                overflow=float(np.mean(a > 65504)))


# ----------------------------------------------------------------------------
# Demos
# ----------------------------------------------------------------------------
def show_formats():
    print("1. The formats: value = (-1)^s · 1.m · 2^(e - bias)\n")
    print(f"  {'format':9s} {'bits':>4s} {'s|exp|man':>9s} {'max':>10s} {'min normal':>11s} "
          f"{'min subnorm':>11s} {'eps (1→next)':>13s} {'digits':>6s}")
    for name in FORMATS:
        f = format_info(name)
        print(f"  {name:9s} {f['bits']:4d} {'1|' + str(f['e_bits']) + '|' + str(f['m_bits']):>9s} "
              f"{f['max']:10.3g} {f['min_normal']:11.3g} {f['min_subnormal']:11.3g} "
              f"{f['eps']:13.3g} {f['digits']:6.1f}")
    print("\n  Exponent bits buy RANGE (bf16 = fp32's 8), mantissa bits buy PRECISION "
          "(fp16's 10 > bf16's 7).")
    print("  fp8_e4m3 has no inf (max 448); fp8_e5m2 keeps IEEE inf/NaN.\n")
    for v in (math.pi, -1e-4, 70000.0):
        print(f"  x = {v!r}")
        for name in FORMATS:
            s, e, m = encode_bits(v, name)
            r = float(round_to_format(v, name))
            err = abs(r - v) / abs(v)
            print(f"    {name:9s} {s} {e:>8s} {m:<23s} -> {r:<14.8g} rel. error {err:.1e}")
        print()


def show_tiny_updates():
    print("2. Why the weights need an fp32 MASTER copy\n")
    for name in ("fp32", "fp16", "bf16"):
        r = float(round_to_format(round_to_format(1.0, name) + 1e-4, name))
        print(f"  {name}: 1 + 1e-4 = {r!r}")
    n = 1000  # 1000 independent weights, to show the average and spread
    w32, wbf, wsr = (np.ones(n, np.float32) for _ in range(3))
    rng = np.random.default_rng(0)
    for _ in range(10000):
        w32 = w32 + np.float32(1e-4)
        wbf = to_bf16(wbf + np.float32(1e-4))
        wsr = to_bf16_stochastic(wsr + np.float32(1e-4), rng)
    print(f"\n  Add 1e-4 to a weight of 1.0, 10,000 times (the exact answer is 2.0):")
    print(f"    fp32 accumulator               : {w32.mean():.4f}")
    print(f"    bf16, round to nearest         : {wbf.mean():.4f}   every update is "
          f"below half a bf16 step (2^-8 = {2 ** -8:.4f}): all lost")
    print(f"    bf16, stochastic rounding      : {wsr.mean():.4f} ± {wsr.std():.3f} "
          f"(mean ± std over {n} weights): right on average\n")


def show_gradient_census(model, data):
    print("5. Where the gradients live: real gradients of the fp32-trained mini-GPT\n")
    census = gradient_census(model, data)
    print(f"  fp16: smallest subnormal 2^-24 = {2 ** -24:.1e}, smallest normal 2^-14 = "
          f"{2 ** -14:.1e}, largest 65504. bf16/fp32 go down to ~1e-38.\n")
    print(f"  {'':18s} {'':10s} {'flushed to 0 in fp16':^47s}")
    print(f"  {'gradient':18s} {'median |g|':>10s} {'512-tok batch':>14s} "
          f"{'0.5M-tok batch':>15s} {'0.5M + S=2^16':>15s}")
    for k, label in (("dlogits", "dL/dlogits"), ("activations", "activation grads"),
                     ("params", "parameter grads")):
        g = census[k]
        z = [underflow_stats(g, s)["zero"] for s in (1.0, 1.0 / ACCUM, 2.0 ** 16 / ACCUM)]
        med = float(np.median(np.abs(g[g != 0])))
        print(f"  {label:18s} {med:10.1e} {z[0]:14.1%} {z[1]:15.1%} {z[2]:15.1%}")
    print(f"\n  dlogits = (softmax - onehot) / (tokens in the batch): once the model is "
          f"confident, the\n  wrong-class probabilities are tiny, and divided by the batch "
          f"size they fall below 2^-24.\n  Scaling by S shifts the whole histogram right "
          f"(log2 S = 16 bins). Too big an S overflows\n  instead (inf): that is what the "
          f"dynamic scaler detects, skipping the step and halving S.\n")
    return census


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--modes", default=",".join(MODES),
                    help="comma-separated subset of: " + ",".join(MODES))
    ap.add_argument("--accum", type=int, default=ACCUM,
                    help="virtual gradient-accumulation factor (1 = a plain 512-token batch)")
    args = ap.parse_args()

    show_formats()
    show_tiny_updates()

    _, chars, data = load_text()
    V = len(chars)
    print(f"3-4. Training the mini-GPT ({args.steps} steps of B=16, T=32) under each recipe.")
    print(f"     Each step is one micro-batch of a {ACCUM}-way gradient accumulation "
          f"({ACCUM * 512:,} tokens per step),\n     so the fp16 backward sees gradients "
          f"{ACCUM}x smaller than with a 512-token batch.\n")
    results = {}
    for mode in args.modes.split(","):
        t0 = time.time()
        model, hist = train_mode(mode, data, V, steps=args.steps, accum=args.accum)
        final = float(np.mean(hist["loss"][-100:]))
        whole = evaluate(model, data)
        results[mode] = (model, hist, whole)
        extra = ""
        if mode.startswith("fp16"):
            extra = (f"| skipped {sum(hist['skipped']):2d} steps, final S = 2^"
                     f"{int(np.log2(hist['scale'][-1])):<2d} | = 0 in fp16: dlogits "
                     f"{np.mean(hist['zero_frac'][-100:]):3.0%}, param grads "
                     f"{np.mean(hist['param_zero'][-100:]):3.0%}")
        print(f"  {MODES[mode][0]:22s} loss (last 100 steps) {final:.3f} | whole text "
              f"{whole:.3f} | {time.time() - t0:4.1f} s {extra}")
    p = np.bincount(data) / len(data)
    unigram = float(-(p[p > 0] * np.log(p[p > 0])).sum())
    print(f"\n  fp16 without scaling: the gradients reaching the lower layers underflow to 0, the "
          f"body stops\n  learning and the model falls back to character frequencies (unigram "
          f"entropy of the text:\n  {unigram:.3f}). A static or dynamic loss scale gets fp32's "
          f"result back. bf16 needs no scaling.\n  But bf16 WEIGHTS without an fp32 master copy "
          f"learn less: updates below half a bf16 step\n  are rounded away (stochastic rounding "
          f"recovers it).\n")

    if "fp32" in results:
        census = show_gradient_census(results["fp32"][0], data)
        os.makedirs(RUNS, exist_ok=True)
        np.savez(os.path.join(RUNS, "precision.npz"), modes=np.array(list(results)),
                 steps=args.steps, accum=args.accum,
                 **{f"loss.{m}": np.array(r[1]["loss"]) for m, r in results.items()},
                 **{f"scale.{m}": np.array(r[1]["scale"]) for m, r in results.items()},
                 **{f"skipped.{m}": np.array(r[1]["skipped"]) for m, r in results.items()},
                 **{f"whole.{m}": r[2] for m, r in results.items()},
                 **{f"census.{k}": v for k, v in census.items()})
        print(f"  saved the curves to {os.path.relpath(os.path.join(RUNS, 'precision.npz'))} "
              "(make_figures.py uses them)")

if __name__ == "__main__":
    main()
