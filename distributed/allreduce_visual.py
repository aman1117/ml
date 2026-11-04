"""
Watch a ring all-reduce happen, message by message, in your terminal. Then
watch one ZeRO-3 (FSDP) training step: who holds which parameters when.

Part 1: 4 workers, each with its own gradient vector of 8 numbers, split into
4 chunks. Goal: every worker ends with the SUM of the 4 vectors.
  - reduce-scatter (3 steps): each worker passes one chunk to its right
    neighbour, which adds it to its own copy. After 3 steps worker r holds the
    complete sum of chunk r (green).
  - all-gather (3 steps): the complete chunks travel once around the ring.
  - each worker sent 6 messages of 2 numbers = 12 numbers = 2(W-1)/W · 8,
    instead of rank 0 receiving and sending 3 x 8 = 24 each way.
Part 2: one ZeRO-3 step of the mini-GPT on 4 ranks (parallel_numpy.ZeroSimulator):
    gather a unit's parameters, use them, free them; reduce-scatter the
    gradients; each rank updates only its own shard.

Run:
    python distributed/allreduce_visual.py              (press Enter for the next step)
    python distributed/allreduce_visual.py --no-pause   (or with stdin not a terminal)
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

import parallel_numpy as P

TTY = sys.stdout.isatty()
PAUSE = True


def color(code, s):
    return f"\033[{code}m{s}\033[0m" if TTY else s


def cyan(s): return color("36;1", s)      # the workers' own data
def yellow(s): return color("33;1", s)    # what just arrived
def green(s): return color("32;1", s)     # complete sums
def magenta(s): return color("35;1", s)   # messages on the wire
def dim(s): return color("2", s)


def pause():
    if PAUSE and sys.stdin.isatty():
        try:
            input(dim("\n  [Enter] next step "))
        except EOFError:
            print()
    print("\n" + "=" * 78 + "\n")


def cell(vals, contrib, W, fresh):
    """One chunk on one worker: its numbers and whose gradients are summed in."""
    nums = "[" + " ".join(f"{v:2.0f}" for v in vals) + "]"
    who = "+".join(str(w) for w in sorted(contrib))
    text = f"{nums} {who:<8s}  "
    if len(contrib) == W:
        return green(text)
    if fresh:
        return yellow(text)
    return text


def show_table(chunks, contrib, W, fresh=()):
    print("            " + "".join(f"{'chunk ' + str(c):<18s}" for c in range(W)))
    for r in range(W):
        print(f"  worker {r}  " + "".join(cell(chunks[r][c], contrib[r][c], W, (r, c) in fresh)
                                       for c in range(W)))
    print(dim("\n  [numbers] then which workers' gradients are summed into this copy "
              "(yellow = just arrived, green = complete)"))


def part1(W=4, n=8):
    rng = np.random.default_rng(3)
    grads = [rng.integers(1, 10, n).astype(float) for _ in range(W)]
    chunks = [P.split_chunks(g, W) for g in grads]
    contrib = [[{r} for _ in range(W)] for r in range(W)]
    total = np.sum(grads, axis=0)

    print("PART 1: RING ALL-REDUCE\n")
    print(f"  {W} workers did forward + backward on their own slice of the batch. Each has a "
          f"gradient\n  vector of {n} numbers. Data parallelism needs every worker to end up with "
          "the SUM.\n")
    for r in range(W):
        print(f"   worker {r}: {cyan(' '.join(f'{v:2.0f}' for v in grads[r]))}")
    print(f"   {'sum':8s}: {green(' '.join(f'{v:2.0f}' for v in total))}   <- the goal, on every worker")
    print(f"\n  Workers sit on a ring 0 → 1 → 2 → 3 → 0 and only talk to their right neighbour."
          f"\n  Each vector is cut into {W} chunks of {n // W} numbers:\n")
    show_table(chunks, contrib, W)
    pause()

    log = P.CommLog(W)
    for phase, s, sends in P.ring_reduce_scatter_steps(chunks, log):
        fresh = set()
        for src, dst, c in sends:
            contrib[dst][c] = contrib[dst][c] | contrib[src][c]
            fresh.add((dst, c))
        print(f"REDUCE-SCATTER step {s + 1}/{W - 1}: worker r sends chunk (r - {s + 1}) mod {W} "
              f"to worker r+1, which ADDS it\n")
        print("  " + "   ".join(magenta(f"W{src} ──chunk {c}──▶ W{dst}") for src, dst, c in sends))
        print(dim("  (all 4 sends happen at the same time, each on its own link)\n"))
        show_table(chunks, contrib, W, fresh)
        if s == W - 2:
            print(f"\n  After {W - 1} steps worker r holds the COMPLETE sum of chunk r (diagonal, "
                  "green).\n  That alone is a reduce-scatter: ZeRO and FSDP stop here for gradients.")
        pause()

    for phase, s, sends in P.ring_all_gather_steps(chunks, log):
        fresh = set()
        for src, dst, c in sends:
            contrib[dst][c] = set(contrib[src][c])
            fresh.add((dst, c))
        print(f"ALL-GATHER step {s + 1}/{W - 1}: worker r forwards complete chunk (r - {s}) mod "
              f"{W} to worker r+1, which OVERWRITES its copy\n")
        print("  " + "   ".join(magenta(f"W{src} ──chunk {c}──▶ W{dst}") for src, dst, c in sends))
        print()
        show_table(chunks, contrib, W, fresh)
        pause()

    ok = all(np.array_equal(np.concatenate(chunks[r]), total) for r in range(W))
    per = log.sent.max() // 8
    print("RESULT\n")
    for r in range(W):
        print(f"   worker {r}: {green(' '.join(f'{v:2.0f}' for v in np.concatenate(chunks[r])))}")
    print(f"\n  Every worker has the sum: {green('yes') if ok else 'NO'}.  "
          f"{2 * (W - 1)} steps, and each worker sent {per} numbers = 2(W-1)/W · n = "
          f"2·{W - 1}/{W}·{n}.")
    print(f"  Naive (everyone sends to worker 0, it sends the sum back): worker 0 receives "
          f"{(W - 1) * n} and sends {(W - 1) * n}.")
    print("\n  Per-worker traffic → 2n as W grows, so a ring all-reduce over 1,000 GPUs costs "
          "each GPU\n  about the same bandwidth as over 8. Its latency grows with the 2(W-1) "
          "steps, which is why\n  NCCL also uses trees, and why gradients are sent in big buckets.")
    pause()


def part2(W=4):
    _, chars, data = P.load_text()
    model = P.small_gpt(len(chars))
    z = P.ZeroSimulator(model, W, stage=3)
    x, y = P.get_batch(data, 8, 8, np.random.default_rng(0))
    z.step(x, y, 1e-2)
    total = model.n_params()
    sizes = {u: n for u, (n, _) in z.sizes.items()}

    print("PART 2: ONE ZeRO-3 / FSDP STEP\n")
    print(f"  The mini-GPT ({total:,} parameters) on {W} ranks. Its parameters are grouped into "
          "FSDP units,\n  each flattened and cut into 4 equal shards. Rank r PERMANENTLY owns "
          "shard r of each unit:")
    print(f"  its bf16 parameters, its fp32 master copy, Adam's m and v, and its slice of the "
          "gradient.\n")
    print(f"  {'unit':8s} {'params':>7s}  " + "".join(f"{'rank ' + str(r):>9s}" for r in range(W)))
    for u, n in sizes.items():
        s = z.sizes[u][1]
        print(f"  {u:8s} {n:7,d}  " + "".join(cyan(f"{s:9,d}") for _ in range(W)))
    own = sum(s for _, s in z.sizes.values())
    print(f"  {'total':8s} {total:7,d}  " + "".join(f"{own:9,d}" for _ in range(W))
          + dim("   (shards are padded to a multiple of 4)"))
    print("\n  Memory per rank at rest: 1/4 of everything. (16 bytes/param / 4 = 4 bytes/param.)")
    pause()

    print("WHAT RANK 0 DOES DURING THE STEP (every rank does the same, in lockstep)\n")
    width = 40
    peak = max(h for r, *_, h in z.trace if r == 0)
    phase = {0: "forward", 1: "backward"}
    seen = {}
    for r, event, u, held in z.trace:
        if r != 0:
            continue
        note = ""
        if event == "all-gather":
            k = seen.get(u, 0)
            seen[u] = k + 1
            note = "for the whole step" if u == "root" else f"for the {phase[k]}"
            ev = yellow(f"all-gather {u:6s}") + f" {note:18s}"
        elif event == "free":
            ev = dim(f"free       {u:6s} {'':18s}")
        else:
            ev = green(f"reduce-scatter grads + Adam on my shard of {u}")
        bar = "█" * int(round(held / peak * width))
        print(f"  {ev:<60s} holds {held:5,d} {bar}")
    print(f"\n  Peak: {peak:,} of {total:,} parameters (my shards + the full root unit + ONE "
          f"block),\n  not the whole model."
          f"\n  Real FSDP prefetches the next unit while computing this one (2 blocks in memory)"
          f"\n  and reduce-scatters each block's gradients as soon as its backward is done.")
    pause()

    sent = z.log.sent[0] // 8
    print("WHY IT IS STILL EXACT\n")
    print("  - The all-gather rebuilds exactly the parameters plain training would use.")
    print("  - The reduce-scatter gives rank r the sum of everyone's gradient for ITS shard, "
          "which is\n    all it needs: AdamW is element-wise, so updating 4 shards separately "
          "= updating the whole.")
    print(f"  - Freed parameters are overwritten with NaN in this simulation: if anything used "
          "them,\n    the loss would be NaN. equivalence_check.py compares with plain AdamW: "
          "equal to 1e-12.")
    print(f"\n  Cost: rank 0 sent {sent:,} numbers this step (gather twice + reduce-scatter) vs "
          f"{2 * 3 * total // 4:,} for\n  plain data parallelism's all-reduce: ~1.5x the "
          f"traffic for 1/{W} of the memory.")


def main():
    global PAUSE
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-pause", action="store_true", help="print everything without waiting")
    ap.add_argument("--part", type=int, choices=[1, 2], help="show only part 1 or 2")
    args = ap.parse_args()
    PAUSE = not args.no_pause
    if args.part in (None, 1):
        part1()
    if args.part in (None, 2):
        part2()


if __name__ == "__main__":
    main()
