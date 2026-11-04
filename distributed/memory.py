"""
Memory and compute calculator for training a GPT: where the bytes and the
FLOPs go, and how each parallelism strategy divides them.

What this teaches (README sections 1, 4, 5, 9-12):
  1. Parameter counts of four real shapes: GPT-3 Small (125M), GPT-3 XL (1.3B),
     LLaMA-2-7B and LLaMA-2-70B, from the layer sizes (checked against the
     published totals).
  2. "16 bytes per parameter": mixed-precision Adam keeps bf16 weights (2) and
     grads (2), plus fp32 master weights (4), Adam m (4) and v (4).
  3. Activation memory per layer, from Korthikanti et al. 2022, eq. (1)-(2):
        s·b·h·(34 + 5·a·s/h) bytes     (16-bit activations, no recompute)
     and what selective / full activation checkpointing and tensor+sequence
     parallelism do to it.
  4. ZeRO-1/2/3 per-GPU model-state memory for N data-parallel GPUs (checked
     against the numbers in the ZeRO paper), and TP x PP x DP layouts.
  5. FLOPs = 6·N·D (+ the attention term), training time, MFU, and the
     bytes each strategy has to send per step.

Run:
    python distributed/memory.py
    python distributed/memory.py --model 70B --gpus 1024 --tp 8 --pp 4 --micro-batch 1
"""
from __future__ import annotations

import argparse

GB = 1e9  # decimal GB, as GPU spec sheets use (an "80 GB" H100 has 80e9 bytes)

# Shapes. GPT-3 rows: Brown et al. 2020 Table 2.1 (we use 16 heads of 128 for
# 1.3B; the table's 24 heads x 128 does not multiply to 2048). LLaMA-2 rows:
# Touvron et al. 2023 (SwiGLU MLP, no biases, RMSNorm, untied embeddings, GQA for 70B).
MODELS = {
    "125M": dict(L=12, d=768, heads=12, kv_heads=12, ffn=3072, glu=False, V=50257,
                 seq=2048, tied=True, published=125e6),
    "1.3B": dict(L=24, d=2048, heads=16, kv_heads=16, ffn=8192, glu=False, V=50257,
                 seq=2048, tied=True, published=1.3e9),
    "7B": dict(L=32, d=4096, heads=32, kv_heads=32, ffn=11008, glu=True, V=32000,
               seq=4096, tied=False, published=6.74e9),
    "70B": dict(L=80, d=8192, heads=64, kv_heads=8, ffn=28672, glu=True, V=32000,
                seq=4096, tied=False, published=68.98e9),
}

GPUS = {  # dense bf16 tensor-core peak FLOP/s, memory
    "A100-80GB": dict(peak=312e12, mem=80 * GB),
    "H100-SXM": dict(peak=989e12, mem=80 * GB),
}


# ----------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------
def n_params(m):
    """Exact parameter count from the shapes (GPT-style: biases + LayerNorm +
    learned positions; LLaMA-style: no biases, RMSNorm, RoPE has no params)."""
    L, d, f, V = m["L"], m["d"], m["ffn"], m["V"]
    hd = d // m["heads"]
    kv = m["kv_heads"] * hd
    attn = d * d + 2 * d * kv + d * d  # W_q, W_k, W_v (GQA: smaller), W_o
    mlp = (3 if m["glu"] else 2) * d * f  # SwiGLU has a third (gate) matrix
    if m["glu"]:  # LLaMA: two RMSNorm gains per layer, final norm
        per_layer, extra = attn + mlp + 2 * d, d
    else:  # GPT: biases on every Linear, two LayerNorms (γ, β), final LN, wpe
        per_layer = attn + (d + 2 * kv + d) + mlp + (f + d) + 4 * d
        extra = 2 * d + m["seq"] * d
    emb = V * d * (1 if m["tied"] else 2)
    return L * per_layer + emb + extra


def model_state_bytes(N):
    """Mixed-precision AdamW (Micikevicius et al. 2018; ZeRO paper's K = 12)."""
    return dict(weights_bf16=2 * N, grads_bf16=2 * N, master_fp32=4 * N,
                adam_m=4 * N, adam_v=4 * N)


# ----------------------------------------------------------------------------
# Activations (Korthikanti et al. 2022, section 4)
# ----------------------------------------------------------------------------
def activation_bytes_per_layer(m, b=1, s=None, tp=1, sp=True, recompute="none"):
    """Bytes stored for backward by ONE transformer layer, 16-bit activations.

        no recompute         s·b·h·(34 + 5·a·s/h)           (/t with TP + sequence parallel)
        selective recompute  s·b·h·34                        (attention scores/softmax recomputed)
        full recompute       s·b·h·2                         (keep only the layer's input)

    34·s·b·h: attention 11sbh (LN input, QKV input, Q/K for scores, V, dropout
    mask, W_o input), MLP 19sbh (input, 4h GELU input/output, mask), 2 LNs 4sbh.
    5·a·s²·b: the a×s×s attention scores and softmax (2+2 bytes) and its dropout
    mask (1 byte). The paper derives it for GPT (4h GELU MLP); for SwiGLU models
    it is an approximation. FlashAttention also removes the 5as/h term."""
    s = s or m["seq"]
    h, a = m["d"], m["heads"]
    if recompute == "full":
        return 2 * s * b * h / (tp if sp else 1)
    linear_part = 34 * s * b * h
    attn_part = 5 * a * s * s * b if recompute == "none" else 0
    if tp == 1:
        return linear_part + attn_part
    # With TP the attention/MLP internals are split t ways; without sequence
    # parallelism the 10sbh of LayerNorm/dropout outputs stay replicated.
    return (linear_part / tp if sp else (24 * s * b * h / tp + 10 * s * b * h)) + attn_part / tp


def activation_bytes(m, b=1, s=None, tp=1, recompute="none"):
    """All L layers, plus the fp32 logits of the output layer (4·s·b·V)."""
    s = s or m["seq"]
    return m["L"] * activation_bytes_per_layer(m, b, s, tp, True, recompute) + 4 * s * b * m["V"] / tp


# ----------------------------------------------------------------------------
# Sharding
# ----------------------------------------------------------------------------
def zero_bytes_per_gpu(N, n_gpus, stage):
    """Model-state bytes per GPU (Rajbhandari et al. 2020, Figure 1):
        stage 0 (plain DP)     2N + 2N + 12N
        ZeRO-1 (optimizer)     2N + 2N + 12N/n
        ZeRO-2 (+ gradients)   2N + (2N + 12N)/n
        ZeRO-3 (+ parameters)  (2N + 2N + 12N)/n           (= FSDP FULL_SHARD)"""
    w, g, o = 2 * N, 2 * N, 12 * N
    if stage >= 1:
        o /= n_gpus
    if stage >= 2:
        g /= n_gpus
    if stage >= 3:
        w /= n_gpus
    return dict(weights=w, grads=g, optimizer=o, total=w + g + o)


def layout_bytes_per_gpu(m, tp, pp, dp, micro_batch=1, zero=1, recompute="selective"):
    """Per-GPU memory of a TP x PP x DP layout: model states are split t·p ways
    by TP/PP, then the ZeRO stage shards them over DP. With 1F1B, the first
    stage holds activations of p micro-batches of L/p layers = L layers' worth."""
    N = n_params(m)
    states = zero_bytes_per_gpu(N / (tp * pp), dp, zero)["total"]
    acts = activation_bytes(m, micro_batch, None, tp, recompute)
    return states, acts


# ----------------------------------------------------------------------------
# Compute and communication
# ----------------------------------------------------------------------------
def train_flops(m, tokens, with_attention=True):
    """6·N·D: forward 2N per token (a multiply-add per weight), backward 4N.
    The attention scores add 12·L·s·d per token (PaLM, Chowdhery et al. 2022, appendix B)."""
    N = n_params(m)
    per_token = 6 * N + (12 * m["L"] * m["seq"] * m["d"] if with_attention else 0)
    return per_token * tokens


def training_days(m, tokens, n_gpus, gpu="H100-SXM", mfu=0.4):
    return train_flops(m, tokens) / (n_gpus * GPUS[gpu]["peak"] * mfu) / 86400


def mfu(m, tokens_per_sec, n_gpus, gpu="H100-SXM"):
    """Model FLOPs utilization: the FLOPs the MODEL needs (no recompute counted)
    per second, over the hardware peak. Recompute makes hardware FLOPs (HFU) > MFU."""
    return tokens_per_sec * train_flops(m, 1) / (n_gpus * GPUS[gpu]["peak"])


def comm_bytes_per_gpu(N, n, strategy):
    """Bytes each GPU sends per optimizer step (bf16 = 2 bytes, ring algorithms):
        DP / ZeRO-1/2: one all-reduce (or reduce-scatter + all-gather) of 2N bytes: 2(n-1)/n · 2N
        ZeRO-3 / FSDP: all-gather params in forward AND backward + reduce-scatter grads: 3(n-1)/n · 2N"""
    factor = {"dp": 2, "zero1": 2, "zero2": 2, "zero3": 3}[strategy]
    return factor * (n - 1) / n * 2 * N


# ----------------------------------------------------------------------------
# Printing
# ----------------------------------------------------------------------------
def fmt_bytes(x):
    if x >= 1e12:
        return f"{x / 1e12:.1f} TB"
    if x >= GB:
        return f"{x / GB:.1f} GB"
    return f"{x / 1e6:.0f} MB"


def fmt_n(x):
    return f"{x / 1e9:.2f}B" if x >= 1e9 else f"{x / 1e6:.0f}M"


def print_params():
    print("1. Parameter counts from the shapes\n")
    print(f"  {'model':6s} {'L':>3s} {'d':>6s} {'heads':>5s} {'kv':>3s} {'ffn':>6s} {'MLP':>7s} "
          f"{'V':>6s} {'seq':>5s} {'params':>8s} {'published':>9s} {'12·L·d²':>8s}")
    for name, m in MODELS.items():
        N = n_params(m)
        print(f"  {name:6s} {m['L']:3d} {m['d']:6d} {m['heads']:5d} {m['kv_heads']:3d} "
              f"{m['ffn']:6d} {'SwiGLU' if m['glu'] else 'GELU':>7s} {m['V']:6d} {m['seq']:5d} "
              f"{fmt_n(N):>8s} {fmt_n(m['published']):>9s} {fmt_n(12 * m['L'] * m['d'] ** 2):>8s}")
    print()


def print_model_states():
    print("2. Model states: the 16 bytes/param rule (mixed-precision AdamW, no sharding)\n")
    print(f"  {'model':6s} {'bf16 W (2)':>11s} {'bf16 grad (2)':>14s} {'fp32 master (4)':>16s} "
          f"{'Adam m (4)':>11s} {'Adam v (4)':>11s} {'total (16)':>11s} {'80 GB GPUs':>10s}")
    for name, m in MODELS.items():
        N = n_params(m)
        b = model_state_bytes(N)
        tot = sum(b.values())
        print(f"  {name:6s} {fmt_bytes(b['weights_bf16']):>11s} {fmt_bytes(b['grads_bf16']):>14s} "
              f"{fmt_bytes(b['master_fp32']):>16s} {fmt_bytes(b['adam_m']):>11s} "
              f"{fmt_bytes(b['adam_v']):>11s} {fmt_bytes(tot):>11s} {tot / (80 * GB):10.1f}")
    print("\n  Inference needs only the 2 bytes/param of weights (+ KV cache); training needs 8x that"
          "\n  before a single activation is stored.\n")


def print_activations():
    print("3. Activations for ONE sequence (b = 1, s = model's context), all layers + logits\n")
    print(f"  {'model':6s} {'seq':>5s} {'per layer':>10s} {'no recompute':>13s} "
          f"{'selective':>10s} {'full ckpt':>10s} {'attn share':>10s}")
    for name, m in MODELS.items():
        per = activation_bytes_per_layer(m)
        none_, sel, full = (activation_bytes(m, recompute=r) for r in ("none", "selective", "full"))
        share = 5 * m["heads"] * m["seq"] / m["d"] / (34 + 5 * m["heads"] * m["seq"] / m["d"])
        print(f"  {name:6s} {m['seq']:5d} {fmt_bytes(per):>10s} {fmt_bytes(none_):>13s} "
              f"{fmt_bytes(sel):>10s} {fmt_bytes(full):>10s} {share:10.0%}")
    print("\n  'attn share' = the 5·a·s/h part (scores + softmax) of a layer's activations. It"
          "\n  grows with s: that is why FlashAttention / selective recompute matter so much."
          "\n  Activations scale with batch x sequence; model states do not.\n")


def print_zero():
    print("4. ZeRO: model-state bytes per GPU\n")
    N = 7.5e9
    print(f"  Sanity check vs the ZeRO paper (Figure 1: Ψ = 7.5B, N_d = 64): "
          + ", ".join(f"stage {s}: {zero_bytes_per_gpu(N, 64, s)['total'] / GB:.1f} GB"
                      for s in range(4)))
    print("  (paper: 120 GB, 31.4 GB, 16.6 GB, 1.9 GB)\n")
    for n in (8, 64, 1024):
        print(f"  N = {n} GPUs   {'plain DP':>10s} {'ZeRO-1':>10s} {'ZeRO-2':>10s} {'ZeRO-3':>10s}")
        for name, m in MODELS.items():
            P = n_params(m)
            row = [zero_bytes_per_gpu(P, n, s)["total"] for s in range(4)]
            print(f"  {name:12s} " + " ".join(f"{fmt_bytes(x):>10s}" for x in row))
        print()


def print_layouts():
    print("5. 3D layouts: per-GPU memory = model states / (TP·PP), sharded over DP, + activations")
    print("   (micro-batch 1, selective recompute, TP with sequence parallelism, 1F1B)\n")
    cases = [("7B", 1, 1, 64, 0), ("7B", 1, 1, 64, 3), ("7B", 2, 1, 32, 1),
             ("70B", 1, 1, 64, 3), ("70B", 8, 1, 8, 1), ("70B", 8, 4, 32, 1)]
    print(f"  {'model':5s} {'TP':>3s} {'PP':>3s} {'DP':>4s} {'GPUs':>5s} {'ZeRO':>4s} "
          f"{'states':>9s} {'activ.':>8s} {'total':>9s}  fits 80 GB?")
    for name, tp, pp, dp, z in cases:
        st, ac = layout_bytes_per_gpu(MODELS[name], tp, pp, dp, 1, z)
        tot = st + ac
        print(f"  {name:5s} {tp:3d} {pp:3d} {dp:4d} {tp * pp * dp:5d} {z:4d} {fmt_bytes(st):>9s} "
              f"{fmt_bytes(ac):>8s} {fmt_bytes(tot):>9s}  {'yes' if tot < 80 * GB else 'NO'}")
    print()


def print_compute():
    print("6. Compute: FLOPs = 6·N·D (+ attention), time, MFU\n")
    m = MODELS["7B"]
    D, n, util = 1e12, 1024, 0.4
    F = train_flops(m, D)
    days = training_days(m, D, n, "H100-SXM", util)
    print(f"  7B on 1T tokens: 6·N·D = {6 * n_params(m) * D:.2e} FLOPs, with attention "
          f"{F:.2e} FLOPs")
    print(f"  on {n} H100s (989 TFLOP/s bf16 dense each) at {util:.0%} MFU: "
          f"{F / (n * 989e12 * util) / 3600:.1f} hours = {days:.2f} days\n")
    print(f"  {'model':6s} {'tokens (20/param)':>17s} {'FLOPs':>9s} {'H100-hours @40%':>16s} "
          f"{'wall-clock on 1024 H100s':>25s}")
    for name, mm in MODELS.items():
        Dc = 20 * n_params(mm)  # Chinchilla-optimal data
        hours = training_days(mm, Dc, 1024, "H100-SXM", 0.4) * 24
        wall = f"{hours * 60:.1f} min" if hours < 1 else f"{hours:.1f} h"
        print(f"  {name:6s} {Dc:17.2e} {train_flops(mm, Dc):9.2e} {hours * 1024:16,.0f} {wall:>25s}")
    print("\n  Reality check, LLaMA-2 (Touvron et al. 2023, Table 2): 2T tokens on A100-80GB")
    for name, hours in (("7B", 184320), ("70B", 1720320)):
        mm = MODELS[name]
        implied = train_flops(mm, 2e12, with_attention=False) / (hours * 3600 * 312e12)
        print(f"    {name:4s} {hours:>9,} GPU-hours  ->  implied MFU (6ND / peak·time) = {implied:.0%}")
    print("  PaLM 540B reached 46.2% MFU on 6144 TPU v4 (Chowdhery et al. 2022); LLaMA-3 405B"
          "\n  38-43% on 8-16k H100s (Llama team 2024). 40% is a good, not a pessimistic, guess.\n")


def print_comm():
    print("7. Bytes each GPU sends per step (bf16 gradients/params, ring collectives)\n")
    print(f"  {'model':6s} {'N GPUs':>6s} {'DP all-reduce':>14s} {'ZeRO-3/FSDP':>12s} "
          f"{'at 50 GB/s (IB)':>16s} {'at 450 GB/s (NVLink)':>21s}")
    for name in ("1.3B", "7B", "70B"):
        P = n_params(MODELS[name])
        for n in (8, 64):
            dp, z3 = comm_bytes_per_gpu(P, n, "dp"), comm_bytes_per_gpu(P, n, "zero3")
            print(f"  {name:6s} {n:6d} {fmt_bytes(dp):>14s} {fmt_bytes(z3):>12s} "
                  f"{dp / 50e9:15.2f}s {dp / 450e9:20.2f}s")
    print("\n  Per-GPU traffic is ~constant in N (ring algorithms): that is why data parallelism"
          "\n  scales. It must hide behind the backward pass (DDP buckets) or it stalls the GPUs.\n")


def custom(args):
    m = MODELS[args.model]
    st, ac = layout_bytes_per_gpu(m, args.tp, args.pp, args.gpus // (args.tp * args.pp),
                                  args.micro_batch, args.zero, args.recompute)
    print(f"{args.model}: {fmt_n(n_params(m))} params on {args.gpus} GPUs = TP {args.tp} x PP "
          f"{args.pp} x DP {args.gpus // (args.tp * args.pp)}, ZeRO-{args.zero}, micro-batch "
          f"{args.micro_batch}, {args.recompute} recompute")
    print(f"  model states {fmt_bytes(st)} + activations {fmt_bytes(ac)} = {fmt_bytes(st + ac)} per GPU")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=list(MODELS), help="custom layout: model")
    ap.add_argument("--gpus", type=int, default=64)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--pp", type=int, default=1)
    ap.add_argument("--zero", type=int, default=1, choices=[0, 1, 2, 3])
    ap.add_argument("--micro-batch", type=int, default=1)
    ap.add_argument("--recompute", default="selective", choices=["none", "selective", "full"])
    args = ap.parse_args()
    if args.model:
        custom(args)
        return
    print_params()
    print_model_states()
    print_activations()
    print_zero()
    print_layouts()
    print_compute()
    print_comm()


if __name__ == "__main__":
    main()
