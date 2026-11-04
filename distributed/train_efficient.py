"""
Single-process efficiency tools on the PyTorch mini-GPT (../transformer/
transformer_pytorch.py), measured on this CPU:

  1. Activation checkpointing (torch.utils.checkpoint): peak memory of a
     training step (from torch.profiler with profile_memory=True) and the extra
     time of recomputing the forward. None / every 3rd / every 2nd / every block.
     Plus what autograd actually stores per layer, vs Korthikanti et al.'s 34·s·b·h.
  2. Gradient accumulation: 4 micro-batches of 2 == one batch of 8 (gradients),
     at a quarter of the activation memory.
  3. bf16 autocast on CPU: speed and the loss difference vs fp32.
  4. torch.compile: compile time and steady-state step time (skipped gracefully
     if the compiler toolchain is missing).
  5. torch.profiler: which ops the time goes to.
  6. MFU on this CPU: model FLOPs per second / the best matmul FLOP/s we can measure.

Run (~20 s):
    python distributed/train_efficient.py
    python distributed/train_efficient.py --no-compile
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
import warnings

import torch
import torch.nn as nn
from torch.profiler import ProfilerActivity, profile
from torch.utils.checkpoint import checkpoint

warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")
sys.path.insert(0, os.path.join(HERE, "..", "transformer"))
from transformer_numpy import load_text  # noqa: E402
from transformer_pytorch import GPT  # noqa: E402

CFG = dict(n_layer=6, n_head=8, d_model=256, block_size=256)  # bigger than transformer/'s, still CPU-sized
SMALL = dict(n_layer=2, n_head=4, d_model=64, block_size=32)  # transformer/'s default
BATCH = 8
BF16_STEPS = 60


class Checkpointed(nn.Module):
    """Run a block under torch.utils.checkpoint: keep only its INPUT for the
    backward; recompute everything inside the block when the gradient arrives."""

    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, x):
        return checkpoint(self.block, x, use_reentrant=False)


def make_model(V, every=0, seed=0, dtype=torch.float32, cfg=CFG):
    """every = k: checkpoint every k-th block (0 = none, 1 = all)."""
    torch.manual_seed(seed)
    model = GPT(V, **cfg, dropout=0.0).to(dtype)
    if every:
        model.blocks = nn.ModuleList([Checkpointed(b) if i % every == 0 else b
                                      for i, b in enumerate(model.blocks)])
    return model


def batch(X, Y, B, seed):
    g = torch.Generator().manual_seed(seed)
    ix = torch.randint(len(X), (B,), generator=g)
    return X[ix], Y[ix]


def windows(T):
    _, chars, data = load_text()
    data = torch.from_numpy(data)
    n = len(data) - T - 1
    X = torch.stack([data[i:i + T] for i in range(n)])
    Y = torch.stack([data[i + 1:i + T + 1] for i in range(n)])
    return len(chars), X, Y


def train_step(model, x, y, accum=1):
    for xm, ym in zip(x.chunk(accum), y.chunk(accum)):
        _, loss = model(xm, ym)
        (loss / accum).backward()
    return loss


def peak_step_memory(model, x, y, accum=1):
    """Peak bytes allocated during one forward+backward, from torch.profiler.

    With profile_memory=True every op records the bytes it allocated net of
    what it freed (self_cpu_memory_usage); frees outside an op appear as
    '[memory]' events. Their running sum in time order is the allocation
    timeline; its maximum is the peak above the starting point."""
    for p in model.parameters():
        p.grad = None
    with profile(activities=[ProfilerActivity.CPU], profile_memory=True) as prof:
        train_step(model, x, y, accum)
    cur = peak = 0
    for e in sorted(prof.events(), key=lambda e: e.time_range.start):
        cur += e.self_cpu_memory_usage
        peak = max(peak, cur)
    return peak


def saved_activation_bytes(model, x, y):
    """Bytes of ACTIVATIONS autograd keeps for the backward (each storage once).
    Autograd also saves the weights themselves (dX = dY·Wᵀ needs W); they are
    model state, not activations, so they are left out."""
    params = {p.untyped_storage().data_ptr() for p in model.parameters()}
    seen = {}

    def pack(t):
        s = t.untyped_storage()
        if s.data_ptr() not in params:
            seen[s.data_ptr()] = s.nbytes()
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        model(x, y)
    return sum(seen.values())


def time_steps(model, x, y, n=5, accum=1, warmup=1, autocast=None):
    for i in range(warmup + n):
        if i == warmup:
            t0 = time.perf_counter()
        for p in model.parameters():
            p.grad = None
        if autocast is None:
            train_step(model, x, y, accum)
        else:
            with torch.autocast("cpu", dtype=autocast):
                train_step(model, x, y, accum)
    return (time.perf_counter() - t0) / n


def best_times(models, x, y, rounds=3, n=3, accums=None):
    """Laptop timings drift (turbo, other processes, thermals): time the variants
    in interleaved rounds and keep each one's best."""
    accums = accums or [1] * len(models)
    best = [float("inf")] * len(models)
    for _ in range(rounds):
        for i, (m, a) in enumerate(zip(models, accums)):
            best[i] = min(best[i], time_steps(m, x, y, n=n, accum=a))
    return best


def model_flops_per_token(model, T):
    """6N (matmuls, forward + backward) + 12·L·T·d (attention scores, PaLM's formula)."""
    N = sum(p.numel() for n, p in model.named_parameters() if "wpe" not in n)
    return 6 * N + 12 * CFG["n_layer"] * T * CFG["d_model"]


def matmul_peak(dtype=torch.float32, n=2048, reps=10):
    a, b = torch.randn(n, n, dtype=dtype), torch.randn(n, n, dtype=dtype)
    a @ b
    t0 = time.perf_counter()
    for _ in range(reps):
        a @ b
    return 2 * n ** 3 * reps / (time.perf_counter() - t0)


@contextlib.contextmanager
def quiet_stderr():
    """Silence subprocesses' stderr (file descriptor 2); Python exceptions still propagate."""
    sys.stderr.flush()
    saved = os.dup(2)
    with open(os.devnull, "w") as devnull:
        os.dup2(devnull.fileno(), 2)
        try:
            yield
        finally:
            os.dup2(saved, 2)
            os.close(saved)


def mb(x):
    return f"{x / 1e6:6.1f} MB"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-compile", action="store_true", help="skip the torch.compile part")
    ap.add_argument("--threads", type=int, default=torch.get_num_threads())
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    V, X, Y = windows(CFG["block_size"])
    x, y = batch(X, Y, BATCH, 0)
    T, C, L = CFG["block_size"], CFG["d_model"], CFG["n_layer"]
    base = make_model(V)
    N = sum(p.numel() for p in base.parameters())
    print(f"GPT: {L} layers, d_model {C}, {CFG['n_head']} heads, context {T}, {N:,} params "
          f"({N * 4 / 1e6:.1f} MB fp32); batch {BATCH} x {T} = {BATCH * T:,} tokens; "
          f"{args.threads} threads\n")
    results = {}

    # ------------------------------------------------------------------ 1
    print("1. Activation checkpointing: peak memory of one training step vs time\n")
    saved = saved_activation_bytes(base, x, y)
    sbh = T * BATCH * C
    head = 4 * BATCH * T * (V + 2 * C)  # logits, and the final LayerNorm's input and output
    per_layer = (saved - head) / L
    print(f"   autograd stores {mb(saved)} of activations (weights excluded) for the backward:\n   "
          f"{mb(per_layer).strip()} per layer = "
          f"{per_layer / (2 * sbh):.1f}·s·b·h in 16-bit units")
    print(f"   Korthikanti et al. predict 34·s·b·h + 5·a·s²·b, where the 34 includes two 1-byte "
          f"dropout\n   masks: 32·s·b·h with dropout 0. F.scaled_dot_product_attention's fused CPU "
          f"kernel keeps\n   no T×T matrix, so the 5·a·s²·b term ({5 * CFG['n_head'] * T / C:.0f}·s·b·h "
          f"here) is gone.\n")
    print(f"   {'checkpoint':16s} {'peak memory':>12s} {'vs none':>8s} {'step time':>10s} {'vs none':>8s}")
    configs = ((0, "none"), (3, "every 3rd block"), (2, "every 2nd block"), (1, "every block"))
    models = [make_model(V, every) for every, _ in configs]
    peaks = [peak_step_memory(m, x, y) for m in models]
    times = best_times(models, x, y)
    rows = []
    for (every, label), peak, secs in zip(configs, peaks, times):
        rows.append(dict(label=label, peak=peak, time=secs))
        print(f"   {label:16s} {mb(peak):>12s} {peak / peaks[0]:7.0%} {secs * 1e3:8.0f} ms "
              f"{secs / times[0]:7.0%}")
    grads = N * 4
    print(f"\n   (peak includes the {mb(grads).strip()} of gradients created by the backward)."
          f"\n   Checkpointing every block keeps only each block's input and pays one extra forward"
          f"\n   (forward ≈ 1/3 of a step, so ≈ +33% in theory). Big runs often checkpoint only the"
          f"\n   cheap-to-recompute, memory-hungry parts ('selective', Korthikanti et al. 2022).\n")
    results["checkpointing"] = rows

    # ------------------------------------------------------------------ 2
    print("2. Gradient accumulation: same gradients, a fraction of the activations\n")
    m64 = make_model(V, dtype=torch.float64)
    for p in m64.parameters():
        p.grad = None
    train_step(m64, x, y)
    big = [p.grad.clone() for p in m64.parameters()]
    print(f"   {'micro-batch x steps':20s} {'peak memory':>12s} {'step time':>10s} "
          f"{'max |grad - batch-8 grad|':>26s}")
    acc_rows = []
    accums = (1, 2, 4, 8)
    models = [make_model(V) for _ in accums]
    times = best_times(models, x, y, accums=accums)
    for accum, model, secs in zip(accums, models, times):
        for p in m64.parameters():
            p.grad = None
        train_step(m64, x, y, accum)
        diff = max(float((p.grad - g).abs().max()) for p, g in zip(m64.parameters(), big))
        peak = peak_step_memory(model, x, y, accum)
        acc_rows.append(dict(accum=accum, peak=peak, time=secs, diff=diff))
        print(f"   {f'{BATCH // accum} x {accum}':20s} {mb(peak):>12s} {secs * 1e3:8.0f} ms "
              f"{diff:26.1e}")
    print("\n   (float64 for the gradient comparison.) Each micro-loss is divided by the number of"
          "\n   micro-batches. Memory falls with the micro-batch; time per sample rises a little"
          "\n   because small matmuls use the cores less well.\n")
    results["accumulation"] = acc_rows

    # ------------------------------------------------------------------ 3
    print("3. bf16 autocast on this CPU\n")
    mm32, mm16 = matmul_peak(torch.float32, 512, 5), matmul_peak(torch.bfloat16, 512, 2)
    print(f"   a 512 x 512 matmul: fp32 {mm32 / 1e9:.0f} GFLOP/s, bf16 {mm16 / 1e9:.1f} GFLOP/s "
          f"({mm32 / mm16:.0f}x slower)")
    print("   This PyTorch build has fast fp32 matmuls on this CPU (Apple Accelerate) but no bf16"
          "\n   kernels (no oneDNN/AMX), so bf16 falls back to a slow reference loop. Measured on"
          "\n   transformer/'s small default GPT (2 layers, d 64, T 32, batch 16) to keep it short:")
    V2, X2, Y2 = windows(SMALL["block_size"])
    small = lambda: make_model(V2, cfg=SMALL)  # noqa: E731
    x2, y2 = batch(X2, Y2, 16, 0)
    fp32_t = time_steps(small(), x2, y2, n=3)
    bf16_t = time_steps(small(), x2, y2, n=3, autocast=torch.bfloat16)
    probe = small()
    with torch.no_grad():
        l32 = probe(x2, y2)[1].item()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            logits, l16 = probe(x2, y2)
    print(f"   step time: fp32 {fp32_t * 1e3:.1f} ms, bf16 autocast {bf16_t * 1e3:.0f} ms")
    print(f"   loss at the same weights: fp32 {l32:.5f}, bf16 autocast {l16.item():.5f} "
          f"(|Δ| = {abs(l32 - l16.item()):.1e}); logits come out as {logits.dtype}")
    curves = {}
    for name, ac in (("fp32", None), ("bf16", torch.bfloat16)):
        model = small()
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3, betas=(0.9, 0.95), weight_decay=0.1)
        losses = []
        for step in range(BF16_STEPS):
            xb, yb = batch(X2, Y2, 16, 100 + step)
            with torch.autocast("cpu", dtype=torch.bfloat16, enabled=ac is not None):
                _, loss = model(xb, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        curves[name] = losses
    print(f"   {BF16_STEPS} training steps, same batches: mean loss of the last 10 steps fp32 "
          f"{sum(curves['fp32'][-10:]) / 10:.4f}, bf16 {sum(curves['bf16'][-10:]) / 10:.4f}")
    print("   Autocast runs matmuls in bf16 and keeps reductions (softmax, LayerNorm, loss) and the"
          "\n   weights in fp32: the master-weights recipe of precision_numpy.py, done for you. No"
          "\n   GradScaler needed for bf16. Whether it is FASTER depends on the hardware: on GPUs"
          "\n   (tensor cores) bf16 matmuls are 2x+ faster than tf32/fp32 and use half the memory"
          "\n   traffic; here they are much slower.\n")
    results["bf16"] = dict(mm32=mm32, mm16=mm16, fp32_ms=fp32_t * 1e3, bf16_ms=bf16_t * 1e3,
                           loss32=l32, loss16=l16.item(), curves=curves)

    # ------------------------------------------------------------------ 4
    print("4. torch.compile")
    if args.no_compile:
        print("   skipped (--no-compile)\n")
    else:
        try:
            model = make_model(V)
            compiled = torch.compile(model)
            t0 = time.perf_counter()
            with quiet_stderr():  # inductor runs `brew --prefix libomp` on macOS, which can be noisy
                train_step(compiled, x, y)
            first = time.perf_counter() - t0
            eager_t, comp_t = best_times([make_model(V), compiled], x, y)
            print(f"   first step (tracing + codegen): {first:.1f} s; then {comp_t * 1e3:.0f} ms per "
                  f"step vs eager {eager_t * 1e3:.0f} ms ({eager_t / comp_t:.2f}x)")
            print("   Inductor fuses the element-wise ops (GELU, LayerNorm, residual adds) into a few"
                  "\n   kernels, saving memory traffic; the matmuls are unchanged. Gains are larger on"
                  "\n   GPUs. Compile once, run millions of steps.\n")
            results["compile"] = dict(first_s=first, compiled_ms=comp_t * 1e3, eager_ms=eager_t * 1e3)
        except Exception as e:  # noqa: BLE001
            msg = str(e).strip().splitlines()[0][:200] if str(e).strip() else ""
            print(f"   torch.compile failed here ({type(e).__name__}: {msg}).")
            print("   Usually a missing C++ toolchain / OpenMP on macOS; the rest of the script does "
                  "not need it.\n")
            results["compile"] = dict(error=f"{type(e).__name__}: {msg}")

    # ------------------------------------------------------------------ 5
    print("5. torch.profiler: where one training step's time goes (eager, fp32)\n")
    model = make_model(V)
    train_step(model, x, y)
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        train_step(model, x, y)
    table = prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10,
                                      max_name_column_width=38)
    print("\n".join("   " + line for line in table.splitlines()))
    print("\n   Matmuls (aten::mm / addmm / bmm, forward and backward) dominate; everything else is"
          "\n   memory-bound element-wise work, which is what fusion (torch.compile) targets.\n")

    # ------------------------------------------------------------------ 6
    print("6. MFU on this CPU\n")
    peak32 = matmul_peak(torch.float32)
    step_t = best_times([base], x, y)[0]
    tok_s = BATCH * T / step_t
    fpt = model_flops_per_token(base, T)
    achieved = tok_s * fpt
    print(f"   best fp32 matmul (2048 x 2048): {peak32 / 1e9:.0f} GFLOP/s  <- our 'peak'")
    print(f"   training: {tok_s:,.0f} tokens/s x {fpt / 1e6:.1f} MFLOP/token (6N + 12·L·T·d) = "
          f"{achieved / 1e9:.0f} GFLOP/s")
    print(f"   MFU = {achieved / peak32:.0%} of the measured fp32 matmul peak")
    print("   (On GPUs MFU uses the datasheet peak, e.g. 989 TFLOP/s bf16 for an H100; 40-55% is"
          "\n   good for LLM training. Small matrices, element-wise ops, the optimizer step and,"
          "\n   at scale, communication and pipeline bubbles are what keep it below 100%.)")
    results["mfu"] = dict(peak32=peak32, tokens_per_s=tok_s, flops_per_token=fpt,
                          mfu=achieved / peak32)

    os.makedirs(RUNS, exist_ok=True)
    with open(os.path.join(RUNS, "efficient.json"), "w") as f:
        json.dump(results, f, indent=1)
    print(f"\nsaved {os.path.relpath(os.path.join(RUNS, 'efficient.json'))}")


if __name__ == "__main__":
    main()
