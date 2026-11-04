"""
Real multi-process data parallelism with torch.distributed, on this CPU
(gloo backend), training the mini-GPT of ../transformer/transformer_pytorch.py.

  (a) Manual DDP: every rank starts from DIFFERENT random weights, rank 0's are
      broadcast to all; after backward, dist.all_reduce(grad) / W for every
      parameter; AdamW. Checks that all ranks hold identical weights and that
      they equal a single process trained on the concatenated batch (float64).
  (b) The same with torch.nn.parallel.DistributedDataParallel: DistributedSampler
      to give each rank different data, no_sync() to skip the all-reduce on the
      first micro-batches of a gradient accumulation, gradient buckets.
      Checked against a single process on the union of all ranks' micro-batches.
  (c) FSDP2 (torch.distributed.fsdp.fully_shard) on CPU/gloo: per-rank parameter
      and optimizer memory, and the same equivalence check.
  (d) Throughput vs number of processes W (weak scaling: fixed batch per rank),
      and how long one gradient all-reduce takes.

Two ways to launch:
    python distributed/ddp_pytorch.py                      (spawns W = 4 processes itself,
                                                            then W = 1, 2, 4, 8 for throughput)
    torchrun --nproc_per_node=2 distributed/ddp_pytorch.py --torchrun
(On Python 3.9, torch 2.8's torchrun launcher needs `pip install importlib_metadata`;
the spawn mode needs nothing extra.)

On GPUs (one process per GPU; the code paths are the same, untested here):
    torchrun --nproc_per_node=8 distributed/ddp_pytorch.py --torchrun --backend nccl
    torchrun --nnodes=2 --nproc_per_node=8 --rdzv_backend=c10d \\
             --rdzv_endpoint=<node0-host>:29500 distributed/ddp_pytorch.py --torchrun --backend nccl
What changes with nccl: device = cuda:LOCAL_RANK (torch.cuda.set_device), the
model and every batch move to that GPU, the FSDP device mesh is "cuda", and the
collectives run over NVLink / InfiniBand instead of localhost TCP.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import socket
import sys
import time
import warnings

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")
sys.path.insert(0, os.path.join(HERE, "..", "transformer"))
from transformer_numpy import load_text  # noqa: E402
from transformer_pytorch import GPT  # noqa: E402

CHECK_CFG = dict(n_layer=2, n_head=4, d_model=64, block_size=32)  # the transformer/ default
SPEED_CFG = dict(n_layer=4, n_head=4, d_model=128, block_size=64)


# ----------------------------------------------------------------------------
# Setup
# ----------------------------------------------------------------------------
def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def init(rank, world, backend, port=None):
    """mp.spawn mode: we set the rendezvous address ourselves. torchrun mode:
    torchrun already set MASTER_ADDR/PORT, RANK, WORLD_SIZE and LOCAL_RANK."""
    if port is not None:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend, rank=rank, world_size=world)
    if backend == "nccl":
        local = int(os.environ.get("LOCAL_RANK", rank))
        torch.cuda.set_device(local)
        return torch.device("cuda", local)
    return torch.device("cpu")


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def say(*a, **k):
    if dist.get_rank() == 0:
        print(*a, **k, flush=True)


def windows(T):
    """Every T-character window of the text and its shifted target."""
    _, chars, data = load_text()
    data = torch.from_numpy(data)
    n = len(data) - T - 1
    X = torch.stack([data[i:i + T] for i in range(n)])
    Y = torch.stack([data[i + 1:i + T + 1] for i in range(n)])
    return len(chars), X, Y


def make_model(V, seed, cfg=CHECK_CFG, dtype=torch.float64, device="cpu"):
    torch.manual_seed(seed)
    return GPT(V, **cfg, dropout=0.0).to(device=device, dtype=dtype)


def make_opt(params):
    return torch.optim.AdamW(params, lr=3e-3, betas=(0.9, 0.95), weight_decay=0.1)


def flat_params(model):
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


def spread_across_ranks(vec):
    """max |rank r's vector - rank 0's| over all ranks (an all_gather)."""
    out = [torch.empty_like(vec) for _ in range(dist.get_world_size())]
    dist.all_gather(out, vec)
    return max(float((o - out[0]).abs().max()) for o in out)


def global_batches(X, Y, steps, B, seed=0):
    """The same sequence of global batches on every rank (same generator)."""
    g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        ix = torch.randint(len(X), (B,), generator=g)
        yield X[ix], Y[ix]


def train_reference(V, seed, batches, device):
    """One process, the full batch every step: what DP must reproduce."""
    model = make_model(V, seed, device=device)
    opt = make_opt(model.parameters())
    for x, y in batches:
        _, loss = model(x.to(device), y.to(device))
        opt.zero_grad()
        loss.backward()
        opt.step()
    return flat_params(model)


# ----------------------------------------------------------------------------
# (a) Manual data parallelism
# ----------------------------------------------------------------------------
def part_manual(rank, W, device, steps=5, local_B=4):
    V, X, Y = windows(CHECK_CFG["block_size"])
    model = make_model(V, seed=1000 + rank, device=device)  # deliberately different on every rank
    before = spread_across_ranks(flat_params(model))
    for p in model.parameters():
        dist.broadcast(p.data, src=0)  # everyone starts from rank 0's weights
    after = spread_across_ranks(flat_params(model))
    say(f"(a) manual DDP, W = {W}, float64, {steps} AdamW steps, {local_B} sequences per rank")
    say(f"    weights differ across ranks before broadcast: max |Δ| = {before:.2f}; after: {after:.1e}")
    opt = make_opt(model.parameters())
    batches = list(global_batches(X, Y, steps, W * local_B))
    comm = 0.0
    for x, y in batches:
        xs, ys = x.chunk(W)[rank].to(device), y.chunk(W)[rank].to(device)  # my slice
        _, loss = model(xs, ys)
        opt.zero_grad()
        loss.backward()
        sync(device)
        t0 = time.perf_counter()
        for p in model.parameters():  # one all-reduce per tensor (DDP buckets them)
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
            p.grad /= W  # sum of per-rank means / W = mean over the global batch
        sync(device)  # NCCL collectives are asynchronous: wait before reading the clock
        comm += time.perf_counter() - t0
        opt.step()
    mine = flat_params(model)
    spread = spread_across_ranks(mine)
    if rank == 0:
        ref = train_reference(V, 1000, batches, device)
        diff = float((mine - ref).abs().max())
        say(f"    after training: ranks differ by {spread:.1e}; rank 0 vs ONE process on the "
            f"concatenated batch: {diff:.1e}")
        say(f"    {sum(1 for _ in model.parameters())} all-reduce calls per step, "
            f"{comm / steps * 1e3:.1f} ms of communication per step on rank 0\n")
        return dict(spread=spread, diff=diff)
    return None


# ----------------------------------------------------------------------------
# (b) DistributedDataParallel + DistributedSampler + no_sync
# ----------------------------------------------------------------------------
def part_ddp(rank, W, device, steps=4, micro_B=2, accum=2):
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

    V, X, Y = windows(CHECK_CFG["block_size"])
    ds = TensorDataset(X, Y)
    model = make_model(V, seed=1000 + rank, device=device)
    ddp = DDP(model, device_ids=[device.index] if device.type == "cuda" else None,
              bucket_cap_mb=0.1)  # small buckets so this tiny model gets several
    same = spread_across_ranks(flat_params(model))  # DDP broadcast rank 0's weights in __init__
    sampler = DistributedSampler(ds, num_replicas=W, rank=rank, shuffle=True, seed=0, drop_last=True)
    sampler.set_epoch(0)  # reshuffles differently every epoch; call it at each epoch start
    loader = iter(DataLoader(ds, batch_size=micro_B, sampler=sampler, drop_last=True))
    opt = make_opt(ddp.parameters())
    for step in range(steps):
        opt.zero_grad()
        for k in range(accum):
            x, y = next(loader)
            # no_sync(): skip the all-reduce, just accumulate into .grad locally.
            # The LAST micro-batch's backward all-reduces the accumulated sum.
            ctx = ddp.no_sync() if k < accum - 1 else contextlib.nullcontext()
            with ctx:
                _, loss = ddp(x.to(device), y.to(device))
                (loss / accum).backward()
        opt.step()
    mine = flat_params(model)
    spread = spread_across_ranks(mine)
    buckets = None
    try:  # private API, only for showing the buckets
        info = ddp._get_ddp_logging_data()  # buckets are rebuilt after the first step,
        buckets = info.get("rebuilt_bucket_sizes") or info.get("bucket_sizes")  # in grad order
    except Exception:  # noqa: BLE001
        pass
    if rank == 0:
        # Rebuild what every rank saw: rank r's sampler indices, in micro-batches.
        per_rank = []
        for r in range(W):
            s = DistributedSampler(ds, num_replicas=W, rank=r, shuffle=True, seed=0, drop_last=True)
            s.set_epoch(0)
            per_rank.append(list(s))
        n = micro_B * accum
        batches = []
        for step in range(steps):
            idx = torch.tensor([i for r in range(W) for i in per_rank[r][step * n:(step + 1) * n]])
            batches.append((X[idx], Y[idx]))
        ref = train_reference(V, 1000, batches, device)
        diff = float((mine - ref).abs().max())
        say(f"(b) DistributedDataParallel, W = {W}: DistributedSampler gives each rank "
            f"{len(per_rank[0])} of the {len(ds)} windows;")
        say(f"    {accum} micro-batches of {micro_B} per step (no_sync on the first), "
            f"global batch = {W}·{accum}·{micro_B} = {W * accum * micro_B} sequences")
        say(f"    weights identical after DDP(model) (it broadcasts rank 0's): {same:.1e}; "
            f"after {steps} steps: {spread:.1e}")
        say(f"    rank 0 vs ONE process on the union of all ranks' micro-batches: {diff:.1e}")
        if buckets:
            sizes = [int(b) for b in str(buckets).split(",")]
            say(f"    gradient buckets (bucket_cap_mb=0.1): {len(sizes)} buckets of "
                f"{', '.join(f'{b / 1e3:.0f}' for b in sizes)} kB;\n    each is all-reduced as "
                "soon as all its gradients exist, overlapping the rest of the backward")
        say()
        return dict(spread=spread, diff=diff)
    return None


# ----------------------------------------------------------------------------
# (c) FSDP2
# ----------------------------------------------------------------------------
def part_fsdp(rank, W, device, steps=5, local_B=4):
    V, X, Y = windows(CHECK_CFG["block_size"])
    try:
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.fsdp import fully_shard
    except ImportError as e:
        say(f"(c) FSDP2 not available in torch {torch.__version__} ({e}); skipped\n")
        return None
    # fully_shard keeps shard r of whatever rank r holds, so all ranks must
    # start from the same weights: same seed here (or meta-device init + load).
    model = make_model(V, seed=1000, device=device)
    full_numel = sum(p.numel() for p in model.parameters())
    try:
        mesh = init_device_mesh(device.type, (W,))
        for blk in model.blocks:
            fully_shard(blk, mesh=mesh)  # one FSDP unit per block: gathered one at a time
        fully_shard(model, mesh=mesh)  # root unit: embeddings, final LN, head
        opt = make_opt(model.parameters())
        batches = list(global_batches(X, Y, steps, W * local_B))
        for x, y in batches:
            xs, ys = x.chunk(W)[rank].to(device), y.chunk(W)[rank].to(device)
            _, loss = model(xs, ys)
            opt.zero_grad()
            loss.backward()  # gradients are reduce-scattered (averaged) into the shards
            opt.step()  # each rank updates its own shard only
    except Exception as e:  # noqa: BLE001
        say(f"(c) FSDP2 failed on {device.type}/{dist.get_backend()}: {type(e).__name__}: {e}; "
            "skipped\n")
        return None
    local = sum(p.to_local().numel() for p in model.parameters())
    opt_local = sum(v.to_local().numel() for st in opt.state.values() for v in st.values()
                    if hasattr(v, "to_local"))
    full = torch.cat([p.full_tensor().detach().reshape(-1) for p in model.parameters()])  # all-gathers
    locals_ = [torch.zeros(1, device=device) for _ in range(W)]
    dist.all_gather(locals_, torch.tensor([float(local)], device=device))
    if rank == 0:
        ref = train_reference(V, 1000, batches, device)
        diff = float((full - ref).abs().max())
        say(f"(c) FSDP2 (fully_shard per block + root), W = {W}, on {device.type}/"
            f"{dist.get_backend()}")
        say(f"    parameters per rank: {', '.join(f'{int(v):,}' for v in locals_)} of "
            f"{full_numel:,} ({local / full_numel:.0%}); Adam m+v per rank: {opt_local:,} "
            f"numbers (vs {2 * full_numel:,} unsharded)")
        say(f"    after {steps} steps, gathered params vs ONE process on the full batch: {diff:.1e}\n")
        return dict(diff=diff, local=local, full=full_numel)
    return None


# ----------------------------------------------------------------------------
# (d) Throughput
# ----------------------------------------------------------------------------
def timed_steps(model, opt, X, Y, device, steps, warmup, local_B, seed):
    g = torch.Generator().manual_seed(seed)
    for i in range(warmup + steps):
        if i == warmup:
            dist.barrier()
            t0 = time.perf_counter()
        ix = torch.randint(len(X), (local_B,), generator=g)
        _, loss = model(X[ix].to(device), Y[ix].to(device))
        opt.zero_grad()
        loss.backward()
        opt.step()
    sync(device)
    dist.barrier()
    return time.perf_counter() - t0


def part_speed(rank, W, device, steps=20, warmup=3, local_B=16):
    """Weak scaling. Two runs in the same W processes: first each process trains
    ALONE (no communication: measures how W processes slow each other down by
    sharing the CPU's cores, caches and memory bandwidth), then with DDP. The
    gap between the two is the price of the gradient all-reduce."""
    from torch.nn.parallel import DistributedDataParallel as DDP

    V, X, Y = windows(SPEED_CFG["block_size"])
    T = SPEED_CFG["block_size"]
    alone = make_model(V, 0, SPEED_CFG, torch.float32, device)
    t_alone = timed_steps(alone, make_opt(alone.parameters()), X, Y, device, steps, warmup,
                          local_B, rank)
    model = make_model(V, 0, SPEED_CFG, torch.float32, device)
    ddp = DDP(model, device_ids=[device.index] if device.type == "cuda" else None)
    t_ddp = timed_steps(ddp, make_opt(ddp.parameters()), X, Y, device, steps, warmup, local_B, rank)
    # One all-reduce of all the gradients (what DDP does per step, in buckets).
    flat = torch.cat([p.grad.reshape(-1) for p in model.parameters()])
    sync(device)
    dist.barrier()
    t1 = time.perf_counter()
    for _ in range(10):
        dist.all_reduce(flat)
    sync(device)
    ar = (time.perf_counter() - t1) / 10
    tokens = W * local_B * T * steps
    if rank == 0:
        return dict(W=W, tokens_per_s=tokens / t_ddp, alone_tokens_per_s=tokens / t_alone,
                    step_ms=t_ddp / steps * 1e3, alone_step_ms=t_alone / steps * 1e3,
                    allreduce_ms=ar * 1e3, params=sum(p.numel() for p in model.parameters()),
                    grad_mb=flat.numel() * 4 / 1e6, local_B=local_B, T=T)
    return None


# ----------------------------------------------------------------------------
# Entry points
# ----------------------------------------------------------------------------
def worker(rank, W, backend, port, threads, parts, queue):
    torch.set_num_threads(threads)
    device = init(rank, W, backend, port)
    out = {}
    try:
        if "a" in parts:
            out["a"] = part_manual(rank, W, device)
        if "b" in parts:
            out["b"] = part_ddp(rank, W, device)
        if "c" in parts:
            out["c"] = part_fsdp(rank, W, device)
        if "d" in parts:
            out["d"] = part_speed(rank, W, device)
    finally:
        dist.barrier()
        dist.destroy_process_group()
    if rank == 0 and queue is not None:
        queue.put(out)


def spawn(W, parts, threads, backend="gloo"):
    ctx = mp.get_context("spawn")
    queue = ctx.SimpleQueue()
    mp.spawn(worker, args=(W, backend, free_port(), threads, parts, queue), nprocs=W, join=True)
    return queue.get()


def print_speed(rows):
    base = rows[0]["alone_tokens_per_s"]
    print(f"    {'W':>3s} {'no comm: tok/s':>15s} {'DDP: tok/s':>11s} {'DDP speed-up':>13s} "
          f"{'step alone/DDP (ms)':>20s} {'1 all-reduce (ms)':>18s}")
    for r in rows:
        print(f"    {r['W']:3d} {r['alone_tokens_per_s']:15,.0f} {r['tokens_per_s']:11,.0f} "
              f"{r['tokens_per_s'] / base:12.2f}x {r['alone_step_ms']:11.1f} / {r['step_ms']:5.1f} "
              f"{r['allreduce_ms']:18.2f}")


def base_rate(rows):
    return rows[0]["alone_tokens_per_s"] / rows[0]["W"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--torchrun", action="store_true",
                    help="launched by torchrun: read RANK/WORLD_SIZE from the environment")
    ap.add_argument("--backend", default="gloo", choices=["gloo", "nccl"])
    ap.add_argument("--world", type=int, default=4, help="processes for parts a-c (spawn mode)")
    ap.add_argument("--speed-worlds", default="1,2,4,8", help="W values for the throughput sweep")
    ap.add_argument("--threads", type=int, default=2, help="torch threads per process")
    ap.add_argument("--parts", default="abcd", help="which parts to run (a, b, c, d)")
    args = ap.parse_args()

    if args.torchrun:
        rank, W = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
        torch.set_num_threads(args.threads)
        device = init(rank, W, args.backend)
        say(f"torchrun mode: rank {rank} of {W}, backend {args.backend}, device {device}\n")
        try:
            for p, fn in (("a", part_manual), ("b", part_ddp), ("c", part_fsdp)):
                if p in args.parts:
                    fn(rank, W, device)
            if "d" in args.parts:
                r = part_speed(rank, W, device)
                if rank == 0:
                    say(f"(d) W = {W}: DDP {r['tokens_per_s']:,.0f} tokens/s ({r['step_ms']:.1f} ms "
                        f"per step; {r['alone_tokens_per_s']:,.0f} tokens/s without communication); "
                        f"one all-reduce of {r['grad_mb']:.1f} MB: {r['allreduce_ms']:.2f} ms")
        finally:
            dist.barrier()
            dist.destroy_process_group()
        return

    t0 = time.time()
    print(f"Spawning {args.world} CPU processes (gloo backend, {args.threads} threads each)\n")
    spawn(args.world, args.parts.replace("d", ""), args.threads)
    if "d" in args.parts:
        rows = []
        for W in [int(w) for w in args.speed_worlds.split(",")]:
            rows.append(spawn(W, "d", args.threads)["d"])
        r0 = rows[0]
        print(f"(d) Throughput, weak scaling: {r0['local_B']} sequences x {r0['T']} tokens per rank "
              f"per step,\n    GPT with {r0['params']:,} params ({r0['grad_mb']:.1f} MB of fp32 "
              f"gradients), {args.threads} threads per process, DDP")
        print_speed(rows)
        last = rows[-1]
        print(f"\n    At W = {last['W']}: W processes that never talk already run "
              f"{last['alone_tokens_per_s'] / (last['W'] * base_rate(rows)):.0%} as fast per "
              f"process as one alone\n    (they share cores, caches and memory bandwidth); DDP "
              f"then costs another "
              f"{1 - last['tokens_per_s'] / last['alone_tokens_per_s']:.0%}\n    (one all-reduce = "
              f"{last['allreduce_ms']:.1f} ms vs a {last['alone_step_ms']:.1f} ms step, partly "
              f"overlapped with the backward).")
        os.makedirs(RUNS, exist_ok=True)
        with open(os.path.join(RUNS, "ddp_scaling.json"), "w") as f:
            json.dump(dict(threads=args.threads, rows=rows), f, indent=1)
        print(f"    saved {os.path.relpath(os.path.join(RUNS, 'ddp_scaling.json'))}")
    print(f"\nTotal {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
