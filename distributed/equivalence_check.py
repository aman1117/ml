"""
Equivalence check: every distributed trick in this folder must compute EXACTLY
what one device computes on the full batch (float64, so "equal" means equal up
to summation order, ~1e-15). The analog of ../transformer/gradient_check.py.

    collectives        ring all-reduce == sum; reduce-scatter / all-gather;
                       bytes sent per rank == 2(W-1)/W · n
    data parallel      W workers' averaged gradients == 1 worker on the full batch;
                       after 3 AdamW steps all replicas identical and == 1 worker
    grad accumulation  k micro-batches (loss / k) == one big batch
    ZeRO-1/2/3 (FSDP)  sharded AdamW updates == plain AdamW (stage 3: freed
                       parameters are NaN, so a missing all-gather would fail)
    tensor parallel    Megatron MLP and attention (t = 2, 4): outputs, input
                       gradients and weight gradients == unsharded
    pipeline parallel  GPipe and 1F1B micro-batch gradients == full batch;
                       simulated bubble == (p-1)/(m+p-1)
    mixed precision    our fp16/bf16 rounding == NumPy's float16 / the bf16 bit
                       trick; power-of-2 loss scaling recovers the gradients
                       exactly; dynamic scaling skips overflowing steps

Try breaking a line (e.g. drop the `/ W` in data_parallel_grads, or the
`c.dlogits = dlogits / m` in run_pipeline) and re-run.

Run (~2 s):
    python distributed/equivalence_check.py
"""
from __future__ import annotations

import sys
import time

import numpy as np

import parallel_numpy as P
import precision_numpy as Q

TOL = 1e-12  # float64, different summation orders
TOL_ADAM = 1e-10  # after AdamW steps: see max_diff_split in parallel_numpy.py (key bias b_k)
results = []


def report(name, err, tol=TOL, extra=""):
    ok = bool(err <= tol)
    results.append(ok)
    print(f"  {name:62s} max |diff| = {err:.1e}  {'OK' if ok else 'FAIL'}{extra}")
    return ok


def check_flag(name, ok, extra=""):
    results.append(bool(ok))
    print(f"  {name:62s} {extra}  {'OK' if ok else 'FAIL'}")


def collectives(rng):
    print("Collectives (simulated ranks, float64)")
    for W, n in ((2, 10), (3, 11), (4, 1000), (7, 99)):
        arrays = [rng.standard_normal(n) for _ in range(W)]
        log = P.CommLog(W)
        out = P.ring_all_reduce(arrays, log)
        total = np.sum(arrays, axis=0)
        err = max(float(np.abs(o - total).max()) for o in out)
        expect = 2 * (W - 1) / W * n * 8
        report(f"ring all-reduce W={W} n={n} == sum (every rank)", err,
               extra=f"  [sent/rank {log.sent.max()} B, 2(W-1)/W·n·8 = {expect:.0f}]"
               if n % W == 0 else "")
        if n % W == 0:
            check_flag(f"ring all-reduce W={W}: every rank sends 2(W-1)/W·n", np.all(log.sent == expect))
    W = 4
    arrays = [rng.standard_normal(12) for _ in range(W)]
    rs = P.reduce_scatter(arrays)
    total = np.sum(arrays, axis=0)
    report("reduce-scatter: rank r gets chunk r of the sum",
           max(float(np.abs(rs[r] - np.array_split(total, W)[r]).max()) for r in range(W)))
    ag = P.all_gather(rs)
    report("all-gather(reduce-scatter(x)) == all-reduce(x)",
           max(float(np.abs(a - total).max()) for a in ag))


def data_parallel(model, data, rng):
    print("\nData parallelism and gradient accumulation")
    x, y = P.get_batch(data, 8, 8, rng)
    ref = P.clone(model)
    ref.forward(x, y)
    g_ref = ref.backward()
    keys = list(g_ref)
    for W in (2, 4, 8):
        grads, _ = P.data_parallel_grads([P.clone(model) for _ in range(W)], x, y)
        report(f"DP W={W}: all-reduced grads / W == full-batch grads",
               max(float(np.abs(g - P.flatten(g_ref, keys)).max()) for g in grads))
    replicas, single = P.data_parallel_train(model, data, W=4, steps=3)
    spread = max(P.max_diff(replicas[0].params(), r.params()) for r in replicas[1:])
    report("DP W=4, 3 AdamW steps: replicas identical", spread, tol=0.0)
    report("DP W=4, 3 AdamW steps: params == 1 worker, full batch",
           P.max_diff(replicas[0].params(), single.params()), tol=TOL_ADAM)
    for k in (2, 4, 8):
        report(f"grad accumulation k={k} (loss / k) == one batch of 8",
               P.max_diff(P.accumulated_grads(P.clone(model), x, y, k), g_ref))


def zero(model, data):
    print("\nZeRO / FSDP (W = 4, 3 steps) vs plain AdamW on the full batch")
    batches = [P.get_batch(data, 8, 8, np.random.default_rng(10 + i)) for i in range(3)]
    ref = P.clone(model)
    opt = P.AdamW(ref.params())
    for xb, yb in batches:
        ref.forward(xb, yb)
        opt.step(ref.backward(), 1e-2)
    for stage in (1, 2, 3):
        z = P.ZeroSimulator(model, 4, stage)
        for xb, yb in batches:
            z.step(xb, yb, 1e-2)
        full = z.full_params()
        report(f"ZeRO-{stage}: updated params == AdamW", P.max_diff(full, ref.params()), tol=TOL_ADAM)
        if stage < 3:
            report(f"ZeRO-{stage}: every rank's full params == the shards (all-gather)",
                   max(P.max_diff(r.params(), full) for r in z.replicas))
        else:
            freed = all(np.isnan(p).all() for r in z.replicas for p in r.params().values())
            check_flag("ZeRO-3: between steps no rank holds full params", freed, "(all freed)")
        b = z.bytes_per_rank()
        n = sum(s * 4 for _, s in z.sizes.values())
        expect = {1: 4 * n + 12 * n // 4, 2: 2 * n + 14 * n // 4, 3: 16 * n // 4}[stage]
        check_flag(f"ZeRO-{stage}: bytes/rank == ZeRO paper formula", b["total"] == expect,
                   f"({b['total']} B)")


def tensor_parallel(rng):
    print("\nMegatron tensor parallelism (one MLP, one attention layer)")
    C, nh, B, T = 16, 4, 2, 8
    mlp = P.MLP(C, rng, dtype=P.F64)
    attn = P.CausalSelfAttention(C, nh, rng, dtype=P.F64)
    for layer in (mlp.c_fc, mlp.c_proj, attn.c_attn, attn.c_proj):
        for p in layer.params.values():
            p[...] = 0.3 * rng.standard_normal(p.shape)
    x, dy = rng.standard_normal((B, T, C)), rng.standard_normal((B, T, C))
    for name, layer, fn in (("MLP", mlp, P.tp_mlp), ("attention", attn, P.tp_attention)):
        y_ref = layer.forward(x)
        dx_ref = layer.backward(dy)
        for t in (2, 4):
            y, dx, g = fn(layer, x, dy, t)
            gerr = max(float(np.abs(g[k] - getattr(layer, k.split(".")[0]).grads[k.split(".")[1]]).max())
                       for k in g)
            report(f"TP {name} t={t}: output y", float(np.abs(y - y_ref).max()))
            report(f"TP {name} t={t}: input gradient dx", float(np.abs(dx - dx_ref).max()))
            report(f"TP {name} t={t}: all weight gradients", gerr)


def pipeline(data, V, rng):
    print("\nPipeline parallelism (4-block GPT, p = 4 stages)")
    model = P.small_gpt(V, n_layer=4)
    x, y = P.get_batch(data, 8, 8, rng)
    full = P.clone(model)
    loss_ref = full.forward(x, y)
    g_ref = full.backward()
    for name, fn in (("GPipe", P.schedule_gpipe), ("1F1B", P.schedule_1f1b)):
        for p, m in ((4, 4), (2, 8)):
            g, loss = P.run_pipeline(model, x, y, p, m, fn)
            report(f"{name} p={p} m={m}: grads == full batch (loss {loss:.4f})",
                   max(P.max_diff(g, g_ref), abs(loss - loss_ref)))
    for p, m in ((4, 4), (4, 8), (8, 16)):
        for name, fn in (("GPipe", P.schedule_gpipe), ("1F1B", P.schedule_1f1b)):
            _, _, bubble, peak = P.simulate_schedule(fn(p, m))
            report(f"{name} p={p} m={m}: bubble == (p-1)/(m+p-1)", abs(bubble - (p - 1) / (m + p - 1)))
        _, _, _, peak = P.simulate_schedule(P.schedule_1f1b(p, m))
        check_flag(f"1F1B p={p} m={m}: stage s stashes at most p-s micro-batches",
                   peak == [min(p - s, m) for s in range(p)], str(peak))


def mixed_precision(data, V, rng):
    print("\nNumber formats and loss scaling")
    x = (rng.standard_normal(200_000) * np.exp(rng.uniform(-22, 13, 200_000))).astype(np.float32)
    x[:6] = [65504, 65519.99, 65520, 2.0 ** -25, 3 * 2.0 ** -26, -0.0]
    with np.errstate(over="ignore"):
        ref16 = x.astype(np.float16).astype(np.float64)
    same = lambda a, b: bool(np.all((a == b) | (np.isnan(a) & np.isnan(b))))
    check_flag("round_to_format(fp16) == NumPy float16 (incl. subnormals, inf)",
               same(Q.round_to_format(x, "fp16"), ref16))
    check_flag("to_fp16 (bit trick) == NumPy float16", same(Q.to_fp16(x).astype(np.float64), ref16))
    check_flag("to_bf16 (bit trick) == round_to_format(bf16)",
               same(Q.to_bf16(x).astype(np.float64), Q.round_to_format(x, "bf16")))
    vals = x[:2000]
    with np.errstate(over="ignore"):
        bits_ok = all("".join(Q.encode_bits(v, "fp16")) == format(int(np.float16(v).view(np.uint16)), "016b")
                      for v in vals if np.isfinite(np.float16(v)))
    check_flag("encode_bits(fp16) == NumPy float16's raw bits", bits_ok)
    sr = Q.to_bf16_stochastic(np.full(1_000_000, 1.0 + 2 ** -10, np.float32), np.random.default_rng(0))
    report("stochastic rounding is unbiased: E[SR(1 + 2^-10)]",  # std of the mean: 2.6e-6
           abs(float(sr.mean(dtype=np.float64)) - (1 + 2 ** -10)), tol=1.5e-5)

    model = P.small_gpt(V)
    xb, yb = P.get_batch(data, 8, 8, rng)
    model.forward(xb, yb)
    g1 = model.backward()
    for S in (2.0 ** 10, 2.0 ** 20):
        model.forward(xb, yb)
        model.dlogits *= S
        gS = {k: v / S for k, v in model.backward().items()}
        report(f"float64: loss x 2^{int(np.log2(S))} then grads / S == unscaled grads", P.max_diff(gS, g1), tol=0.0)

    # fp16 emulation at a 0.5M-token batch: without scaling small grads vanish; with S they come back.
    def fp16_grads(S, accum=1024):
        m = Q.emulate_precision(Q.GPT(V, n_layer=2, n_head=4, d_model=16, block_size=8, seed=0), Q.to_fp16)
        m.forward(xb, yb)
        m.dlogits *= S / accum
        return {k: v.astype(np.float64) * accum / S for k, v in m.backward().items()}

    m32 = Q.GPT(V, n_layer=2, n_head=4, d_model=16, block_size=8, seed=0)
    m32.forward(xb, yb)
    g32 = m32.backward()
    # worst relative error ‖g16 - g32‖ / ‖g32‖ over the parameter tensors
    rel = lambda g: max(float(np.linalg.norm(g[k] - g32[k]) / np.linalg.norm(g32[k]))
                        for k in g if np.linalg.norm(g32[k]) > 0)
    e_none, e_scaled = rel(fp16_grads(1.0)), rel(fp16_grads(2.0 ** 16))
    report("fp16 + loss scale 2^16: grads == fp32 grads (worst rel.)", e_scaled, tol=5e-3,
           extra=f"  [no scaling: {e_none:.2f}]")
    check_flag("fp16, no scaling, 0.5M-token batch: a tensor's grads are lost", e_none > 0.3,
               f"(worst rel. error {e_none:.2f})")
    sc = Q.DynamicLossScaler(init_scale=2.0 ** 16, growth_interval=2)
    sc.update(True)
    a = (sc.scale, sc.skipped)
    sc.update(False)
    sc.update(False)
    check_flag("dynamic scaler: overflow -> skip + halve; 2 clean steps -> double",
               a == (2.0 ** 15, 1) and sc.scale == 2.0 ** 16)


def main():
    t0 = time.time()
    _, chars, data = P.load_text()
    V = len(chars)
    rng = np.random.default_rng(0)
    model = P.small_gpt(V)
    collectives(rng)
    data_parallel(model, data, rng)
    zero(model, data)
    tensor_parallel(rng)
    pipeline(data, V, rng)
    mixed_precision(data, V, rng)
    ok = all(results)
    print(f"\n{sum(results)}/{len(results)} checks passed in {time.time() - t0:.1f} s")
    print("Equivalence check", "PASSED" if ok else "FAILED")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
