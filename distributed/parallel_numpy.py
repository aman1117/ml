"""
Distributed training, simulated: W "workers" in one process, every message
explicit, in NumPy. The model is the mini-GPT of ../transformer/transformer_numpy.py
with its hand-written backward, in float64 so equal means equal.

What this teaches (README sections 6-12):
  1. Collectives: a hand-written RING all-reduce = reduce-scatter (W-1 steps)
     + all-gather (W-1 steps). Each worker sends 2(W-1)/W of the data, whatever W.
  2. Data parallelism: W replicas, each gets B/W of the batch, all-reduce the
     gradients, divide by W, identical AdamW steps -> identical replicas, equal
     to one worker on the full batch.
  3. Gradient accumulation: k micro-batches, each loss / k, summed = one big batch.
  4. ZeRO-1/2/3 (= FSDP): every rank keeps 1/W of the optimizer state (1),
     of the gradients (2), of the parameters (3). Parameters are all-gathered
     unit by unit right before use and freed (filled with NaN here, so a missing
     gather cannot go unnoticed); gradients are reduce-scattered; each rank
     updates only its shard.
  5. Megatron tensor parallelism: the MLP as column-parallel W1 + row-parallel
     W2 (one all-reduce forward, one backward), attention split by heads.
  6. Pipeline parallelism: GPipe and 1F1B schedules, simulated in time (bubble
     = (p-1)/(m+p-1)), then executed on the real GPT split into stages.

equivalence_check.py proves each of these equals the single-device computation.

Run:
    python distributed/parallel_numpy.py
"""
from __future__ import annotations

import copy
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "transformer"))
from transformer_numpy import (GPT, MLP, AdamW, CausalSelfAttention, Linear,  # noqa: E402
                               cross_entropy, get_batch, load_text)

F64 = np.float64


# ============================================================================
# 1. Collectives. A "rank" is an index into a Python list; a message is a copy.
# ============================================================================
class CommLog:
    """Counts what every rank sends (bytes) and, optionally, every message."""

    def __init__(self, world, keep_messages=False):
        self.sent = np.zeros(world, dtype=np.int64)
        self.messages = [] if keep_messages else None

    def send(self, src, dst, arr, what=""):
        self.sent[src] += arr.nbytes
        if self.messages is not None:
            self.messages.append((what, src, dst, arr.size))
        return arr.copy()


def split_chunks(x, W):
    return [c.copy() for c in np.array_split(np.ravel(x), W)]


def ring_reduce_scatter_steps(chunks, log=None):
    """chunks[r][c] = rank r's copy of chunk c. Yields after each of the W-1 steps.

    Step s: rank r sends its partial sum of chunk (r - s - 1) mod W to rank r+1,
    which adds it to its own copy. A chunk travels once around the ring picking
    up every rank's contribution; after W-1 steps rank r holds the FULL sum of
    chunk r (and stale partial sums of the others)."""
    W = len(chunks)
    for s in range(W - 1):
        msgs = []
        for r in range(W):  # all sends of a step happen at the same time
            c = (r - s - 1) % W
            dst = (r + 1) % W
            payload = chunks[r][c] if log is None else log.send(r, dst, chunks[r][c], "rs")
            msgs.append((dst, c, payload.copy(), r))
        for dst, c, payload, src in msgs:
            chunks[dst][c] = chunks[dst][c] + payload
        yield "reduce-scatter", s, [(src, dst, c) for dst, c, _, src in msgs]


def ring_all_gather_steps(chunks, log=None):
    """Step s: rank r forwards chunk (r - s) mod W, which it owns (s = 0) or
    received in the previous step, to rank r+1, which overwrites its copy."""
    W = len(chunks)
    for s in range(W - 1):
        msgs = []
        for r in range(W):
            c = (r - s) % W
            dst = (r + 1) % W
            payload = chunks[r][c] if log is None else log.send(r, dst, chunks[r][c], "ag")
            msgs.append((dst, c, payload.copy(), r))
        for dst, c, payload, src in msgs:
            chunks[dst][c] = payload
        yield "all-gather", s, [(src, dst, c) for dst, c, _, src in msgs]


def ring_all_reduce(arrays, log=None):
    """Sum of the W arrays, delivered to every rank. 2(W-1) steps."""
    W = len(arrays)
    shape = np.shape(arrays[0])
    chunks = [split_chunks(a, W) for a in arrays]
    for _ in ring_reduce_scatter_steps(chunks, log):
        pass
    for _ in ring_all_gather_steps(chunks, log):
        pass
    return [np.concatenate(c).reshape(shape) for c in chunks]


def reduce_scatter(arrays, log=None):
    """Rank r gets the sum over ranks of chunk r (np.array_split chunks)."""
    W = len(arrays)
    chunks = [split_chunks(a, W) for a in arrays]
    for _ in ring_reduce_scatter_steps(chunks, log):
        pass
    return [chunks[r][r] for r in range(W)]


def all_gather(shards, log=None):
    """Every rank gets the concatenation of all ranks' shards."""
    W = len(shards)
    chunks = [[shards[r].copy() if c == r else np.zeros_like(shards[c]) for c in range(W)]
              for r in range(W)]
    for _ in ring_all_gather_steps(chunks, log):
        pass
    return [np.concatenate(c) for c in chunks]


def naive_all_reduce(arrays, log=None):
    """Everyone sends everything to rank 0, which sums and sends the result back.
    Rank 0's link carries 2(W-1)·n: it does not scale."""
    W = len(arrays)
    total = arrays[0].copy()
    for r in range(1, W):
        total = total + (arrays[r] if log is None else log.send(r, 0, arrays[r]))
    return [total.copy() if r == 0 else (total.copy() if log is None else log.send(0, r, total))
            for r in range(W)]


# ============================================================================
# Helpers: a small GPT, flattening gradients
# ============================================================================
def small_gpt(V, n_layer=2, n_head=4, d_model=16, block_size=8, seed=0):
    model = GPT(V, n_layer, n_head, d_model, block_size, seed=seed, dtype=F64)
    rng = np.random.default_rng(seed + 100)
    for name, p in model.params().items():  # non-trivial values everywhere
        p[...] = (1.0 + 0.2 * rng.standard_normal(p.shape) if name.endswith("gamma")
                  else 0.2 * rng.standard_normal(p.shape))
    return model


def clone(model):
    return copy.deepcopy(model)


def flatten(d, keys):
    return np.concatenate([np.ravel(d[k]) for k in keys])


def unflatten_into(vec, d, keys):
    i = 0
    for k in keys:
        n = d[k].size
        d[k][...] = vec[i:i + n].reshape(d[k].shape)
        i += n


def max_diff(a, b):
    return max(float(np.abs(a[k] - b[k]).max()) for k in a)


def max_diff_split(a, b, C):
    """(max diff outside the key biases, max diff in the key biases b_k).
    b_k's true gradient is exactly 0 (softmax ignores a constant shift, see
    ../transformer section 7), so its computed gradient is pure round-off
    (~1e-16), and Adam divides it by its own RMS: round-off differences between
    two summation orders become 1e-12-sized parameter differences there."""
    rest, kb = 0.0, 0.0
    for k in a:
        d = np.abs(a[k] - b[k])
        if k.endswith("c_attn.b"):
            kb = max(kb, float(d[C:2 * C].max()))
            d = np.concatenate([d[:C], d[2 * C:]])
        rest = max(rest, float(d.max()))
    return rest, kb


# ============================================================================
# 2. Data parallelism and 3. gradient accumulation
# ============================================================================
def data_parallel_grads(replicas, x, y, log=None):
    """Each replica takes B/W rows, computes the mean loss over ITS tokens;
    the all-reduced sum / W is the gradient of the mean over all tokens
    (the shards are equal-sized)."""
    W = len(replicas)
    xs, ys = np.array_split(x, W), np.array_split(y, W)
    keys = list(replicas[0].params())
    local = []
    for r, model in enumerate(replicas):
        model.forward(xs[r], ys[r])
        local.append(flatten(model.backward(), keys))
    summed = ring_all_reduce(local, log)
    return [s / W for s in summed], keys


def data_parallel_train(model, data, W, steps=3, B=8, lr=1e-2, seed=0):
    """W replicas (rank 0's weights broadcast to all), `steps` AdamW steps.
    Returns the replicas and the single-worker reference trained on full batches."""
    replicas = [clone(model) for _ in range(W)]
    opts = [AdamW(m.params()) for m in replicas]
    ref = clone(model)
    ref_opt = AdamW(ref.params())
    rng = np.random.default_rng(seed)
    T = model.config["block_size"]
    for _ in range(steps):
        x, y = get_batch(data, B, T, rng)
        grads, keys = data_parallel_grads(replicas, x, y)
        for r, (m, opt) in enumerate(zip(replicas, opts)):
            g = {k: np.empty_like(p) for k, p in m.params().items()}
            unflatten_into(grads[r], g, keys)
            opt.step(g, lr)
        ref.forward(x, y)
        ref_opt.step(ref.backward(), lr)
    return replicas, ref


def accumulated_grads(model, x, y, k):
    """k micro-batches of B/k rows; each micro-loss / k; gradients summed."""
    total = None
    for xm, ym in zip(np.array_split(x, k), np.array_split(y, k)):
        model.forward(xm, ym)
        g = {n: v / k for n, v in model.backward().items()}
        total = g if total is None else {n: total[n] + g[n] for n in g}
    return total


# ============================================================================
# 4. ZeRO / FSDP
# ============================================================================
def fsdp_units(model):
    """Parameter groups that are gathered together (FSDP wraps each block).
    The 'root' unit (embeddings, final LN, head) is used at both ends of the
    model, so it is gathered for the whole step, as FSDP's root module is."""
    names = list(model.params())
    units = {"root": [n for n in names if not n.startswith("h.")]}
    for i in range(model.config["n_layer"]):
        units[f"block{i}"] = [n for n in names if n.startswith(f"h.{i}.")]
    return units


class FlatAdamW:
    """The same AdamW as transformer_numpy.AdamW, on a flat 1-D shard.
    Every operation is element-wise, so updating shards separately gives the
    same numbers as updating the whole vector."""

    def __init__(self, n, decay_mask, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1):
        self.m, self.v = np.zeros(n), np.zeros(n)
        self.decay = decay_mask
        self.b1, self.b2 = betas
        self.eps, self.wd, self.t = eps, weight_decay, 0

    def step(self, p, g, lr):
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * g
        self.v = self.b2 * self.v + (1 - self.b2) * g * g
        m_hat = self.m / (1 - self.b1 ** self.t)
        v_hat = self.v / (1 - self.b2 ** self.t)
        p[self.decay] -= lr * self.wd * p[self.decay]  # matrices only, as in AdamW
        p -= lr * m_hat / (np.sqrt(v_hat) + self.eps)


class ZeroSimulator:
    """W ranks training one GPT with ZeRO stage 1, 2 or 3.

    Every FSDP unit's parameters are flattened into one vector (FSDP's
    FlatParameter), padded to a multiple of W, and split into W equal shards.
    Rank r owns shard r of every unit: its fp32 master weights, Adam m and v
    (all stages), its reduced gradient (stage >= 2), its parameters (stage 3)."""

    BYTES = dict(param=2, grad=2, master=4, m=4, v=4)  # what a bf16 mixed-precision run stores

    def __init__(self, model, W, stage):
        self.W, self.stage = W, stage
        self.replicas = [clone(model) for _ in range(W)]  # each rank's compute buffers
        self.units = fsdp_units(model)
        self.log = CommLog(W)
        params0 = model.params()
        self.shards, self.opts, self.sizes = {}, {}, {}
        for u, keys in self.units.items():
            flat = flatten(params0, keys)
            n = len(flat)
            pad = (-n) % W
            flat = np.concatenate([flat, np.zeros(pad)])
            mask = flatten({k: np.full(params0[k].shape, params0[k].ndim >= 2) for k in keys}, keys)
            mask = np.concatenate([mask, np.zeros(pad, dtype=bool)])
            s = len(flat) // W
            self.sizes[u] = (n, s)
            self.shards[u] = [flat[r * s:(r + 1) * s].copy() for r in range(W)]
            self.opts[u] = [FlatAdamW(s, mask[r * s:(r + 1) * s]) for r in range(W)]
        self.peak_gathered = 0
        self.live = {}  # rank -> set of currently gathered units
        self.trace = []  # (rank, event, unit, numbers held) for allreduce_visual.py
        if stage == 3:
            for r in range(W):
                for u in self.units:
                    self._free(r, u)
            self._wrap_blocks()

    # -- stage 3: gather right before use, free right after ------------------
    def _gather(self, r, u):
        """All-gather unit u's shards into rank r's compute buffers. (Real
        FSDP does this as ONE collective for all ranks; here each rank pulls
        the W-1 shards it is missing, which is the same traffic.)"""
        for src in range(self.W):
            if src != r:
                self.log.send(src, r, self.shards[u][src])
        n, _ = self.sizes[u]
        full = np.concatenate(self.shards[u])[:n]
        unflatten_into(full, self.replicas[r].params(), self.units[u])
        self.live.setdefault(r, set()).add(u)
        gathered = sum(self.sizes[x][1] * self.W for x in self.live[r])
        self.peak_gathered = max(self.peak_gathered, gathered)
        self.trace.append((r, "all-gather", u, self.held(r)))

    def _free(self, r, u):
        params = self.replicas[r].params()
        for k in self.units[u]:
            params[k][...] = np.nan  # any use of a freed parameter poisons the result
        if u in self.live.get(r, set()):
            self.live[r].discard(u)
            self.trace.append((r, "free", u, self.held(r)))

    def held(self, r):
        """Parameter numbers rank r holds right now: its shards + gathered units."""
        own = sum(s for _, s in self.sizes.values())
        return own + sum(self.sizes[u][1] * (self.W - 1) for u in self.live.get(r, ()))

    def _wrap_blocks(self):
        for r, model in enumerate(self.replicas):
            for i, blk in enumerate(model.blocks):
                fwd, bwd, u = blk.forward, blk.backward, f"block{i}"

                def forward(x, fwd=fwd, r=r, u=u):
                    self._gather(r, u)
                    y = fwd(x)
                    self._free(r, u)
                    return y

                def backward(dy, bwd=bwd, r=r, u=u):
                    self._gather(r, u)  # backward needs the weights again: gather twice
                    dx = bwd(dy)
                    self._free(r, u)
                    return dx

                blk.forward, blk.backward = forward, backward

    # -- one optimizer step ---------------------------------------------------
    def step(self, x, y, lr):
        W = self.W
        xs, ys = np.array_split(x, W), np.array_split(y, W)
        local = []
        for r, model in enumerate(self.replicas):
            if self.stage == 3:
                self._gather(r, "root")
            model.forward(xs[r], ys[r])
            local.append(model.backward())
            if self.stage == 3:
                self._free(r, "root")
        for u, keys in self.units.items():
            n, s = self.sizes[u]
            flat_grads = [np.concatenate([flatten(g, keys), np.zeros(s * W - n)]) for g in local]
            # Reduce-scatter: rank r receives the SUM of everyone's shard-r gradient.
            # (In real FSDP this runs per unit during the backward, overlapped with it.)
            grad_shards = reduce_scatter(flat_grads, self.log)
            for r in range(W):
                self.opts[u][r].step(self.shards[u][r], grad_shards[r] / W, lr)
                self.trace.append((r, "reduce-scatter + update", u, self.held(r)))
            if self.stage < 3:  # stages 1-2 keep full parameters: all-gather the updated shards
                fulls = all_gather(self.shards[u], self.log)
                for r in range(W):
                    unflatten_into(fulls[r][:n], self.replicas[r].params(), keys)

    def full_params(self):
        """The current parameters, assembled from the shards (for checking)."""
        out = {}
        tmpl = self.replicas[0].params()
        for u, keys in self.units.items():
            n, _ = self.sizes[u]
            flat = np.concatenate(self.shards[u])[:n]
            d = {k: np.empty(tmpl[k].shape) for k in keys}
            unflatten_into(flat, d, keys)
            out.update(d)
        return out

    def bytes_per_rank(self):
        """Persistent per-rank memory if this were a bf16 mixed-precision run."""
        P = sum(s * self.W for _, s in self.sizes.values())  # padded total params
        shard = P // self.W
        b = self.BYTES
        params = (shard if self.stage == 3 else P) * b["param"]
        grads = (shard if self.stage >= 2 else P) * b["grad"]
        optim = shard * (b["master"] + b["m"] + b["v"])
        return dict(params=params, grads=grads, optimizer=optim, total=params + grads + optim)


# ============================================================================
# 5. Megatron tensor parallelism
# ============================================================================
def shard_mlp(mlp, t):
    """Column-parallel c_fc (each rank: C -> 4C/t, its slice of the hidden units),
    row-parallel c_proj (each rank: 4C/t -> C, no bias: added once after the sum)."""
    W1, b1, W2 = mlp.c_fc.params["W"], mlp.c_fc.params["b"], mlp.c_proj.params["W"]
    C, H = W1.shape
    rng = np.random.default_rng(0)
    shards = []
    for i in range(t):
        cols = slice(i * H // t, (i + 1) * H // t)
        sh = MLP(C, rng, dtype=F64)
        sh.c_fc = Linear(C, H // t, rng, dtype=F64)
        sh.c_fc.params["W"][...], sh.c_fc.params["b"][...] = W1[:, cols], b1[cols]
        sh.c_proj = Linear(H // t, C, rng, bias=False, dtype=F64)
        sh.c_proj.params["W"][...] = W2[cols, :]
        shards.append(sh)
    return shards


def tp_mlp(mlp, x, dy, t, log=None):
    """Forward and backward of the MLP on t ranks. Returns y, dx and the
    gradients reassembled in the unsharded layout.

    forward : x is replicated (Megatron's 'f' = identity), each rank computes
              y_i = GELU(x W1_i + b1_i) W2_i   (B, T, C), a PARTIAL sum
              y = all-reduce(y_i) + b2          ('g' = all-reduce)   <- 1 collective
    backward: dy is replicated ('g' backward = identity), each rank computes
              its dW2_i, dW1_i, db1_i locally and a partial dx_i
              dx = all-reduce(dx_i)            ('f' backward = all-reduce) <- 1 collective"""
    shards = shard_mlp(mlp, t)
    partial = [sh.forward(x) for sh in shards]  # no communication: x is already everywhere
    y = ring_all_reduce(partial, log)[0] + mlp.c_proj.params["b"]
    dxs = [sh.backward(dy) for sh in shards]
    dx = ring_all_reduce(dxs, log)[0]
    grads = {"c_fc.W": np.concatenate([sh.c_fc.grads["W"] for sh in shards], axis=1),
             "c_fc.b": np.concatenate([sh.c_fc.grads["b"] for sh in shards]),
             "c_proj.W": np.concatenate([sh.c_proj.grads["W"] for sh in shards], axis=0),
             "c_proj.b": dy.reshape(-1, dy.shape[-1]).sum(axis=0)}
    return y, dx, grads


def shard_attention(attn, t):
    """Split the heads: rank i gets heads [i·nh/t, (i+1)·nh/t). Its slice of
    W_qkv is the matching columns of the q, k AND v blocks (column-parallel);
    its slice of W_o is the matching rows (row-parallel, bias added once)."""
    Wqkv, bqkv, Wo = attn.c_attn.params["W"], attn.c_attn.params["b"], attn.c_proj.params["W"]
    C = Wo.shape[0]
    nh = attn.n_head
    w = C // t  # columns per rank inside each of the q, k, v blocks
    rng = np.random.default_rng(0)
    shards = []
    for i in range(t):
        cols = np.concatenate([np.arange(blk * C + i * w, blk * C + (i + 1) * w) for blk in range(3)])
        sh = CausalSelfAttention(C, nh, rng, dtype=F64)
        sh.n_head = nh // t  # hs stays C / nh: the heads are whole, there are fewer of them
        sh.c_attn = Linear(C, 3 * w, rng, dtype=F64)
        sh.c_attn.params["W"][...], sh.c_attn.params["b"][...] = Wqkv[:, cols], bqkv[cols]
        sh.c_proj = Linear(w, C, rng, bias=False, dtype=F64)
        sh.c_proj.params["W"][...] = Wo[i * w:(i + 1) * w, :]
        shards.append((sh, cols))
    return shards


def tp_attention(attn, x, dy, t, log=None):
    shards = shard_attention(attn, t)
    y = ring_all_reduce([sh.forward(x) for sh, _ in shards], log)[0] + attn.c_proj.params["b"]
    dx = ring_all_reduce([sh.backward(dy) for sh, _ in shards], log)[0]
    C = x.shape[-1]
    gW = np.zeros_like(attn.c_attn.params["W"])
    gb = np.zeros_like(attn.c_attn.params["b"])
    for sh, cols in shards:
        gW[:, cols], gb[cols] = sh.c_attn.grads["W"], sh.c_attn.grads["b"]
    w = C // t
    grads = {"c_attn.W": gW, "c_attn.b": gb,
             "c_proj.W": np.concatenate([sh.c_proj.grads["W"] for sh, _ in shards], axis=0),
             "c_proj.b": dy.reshape(-1, C).sum(axis=0)}
    assert all(sh.c_proj.grads["W"].shape[0] == w for sh, _ in shards)
    return y, dx, grads


# ============================================================================
# 6. Pipeline parallelism
# ============================================================================
def schedule_gpipe(p, m):
    """GPipe: every stage runs ALL m forwards, then all m backwards."""
    return [[("F", i) for i in range(m)] + [("B", i) for i in range(m)] for _ in range(p)]


def schedule_1f1b(p, m):
    """1F1B (PipeDream-Flush, Megatron's default): stage s runs p-s-1 warm-up
    forwards, then alternates one forward / one backward, then drains the
    remaining backwards. Same bubble as GPipe, but at most p-s micro-batches
    are in flight on stage s instead of m."""
    out = []
    for s in range(p):
        warm = min(p - s - 1, m)
        ops = [("F", i) for i in range(warm)]
        for i in range(m - warm):
            ops += [("F", warm + i), ("B", i)]
        ops += [("B", i) for i in range(m - warm, m)]
        out.append(ops)
    return out


def simulate_schedule(schedule, t_f=1.0, t_b=2.0):
    """Run the per-stage op lists in time. F(s, i) needs F(s-1, i) (the
    activations); B(s, i) needs B(s+1, i) (the gradient) and F(s, i).
    Returns the timeline [(stage, op, mb, start, end)], the makespan, the
    bubble (idle) fraction and the peak number of stashed micro-batches per stage."""
    p = len(schedule)
    end = {}
    free_at = [0.0] * p
    pos = [0] * p
    timeline = []
    while any(pos[s] < len(schedule[s]) for s in range(p)):
        progressed = False
        for s in range(p):
            if pos[s] == len(schedule[s]):
                continue
            op, i = schedule[s][pos[s]]
            deps = [("F", s - 1, i)] if op == "F" and s > 0 else []
            if op == "B":
                deps = [("F", s, i)] + ([("B", s + 1, i)] if s < p - 1 else [])
            if all(d in end for d in deps):
                start = max([free_at[s]] + [end[d] for d in deps])
                finish = start + (t_f if op == "F" else t_b)
                end[(op, s, i)] = finish
                free_at[s] = finish
                timeline.append((s, op, i, start, finish))
                pos[s] += 1
                progressed = True
        if not progressed:
            raise RuntimeError("deadlock: the schedule waits on itself")
    makespan = max(e for *_, e in timeline)
    busy = sum(e - b for *_, b, e in timeline)
    peak = []
    for s in range(p):
        live = best = 0
        for op, _ in schedule[s]:
            live += 1 if op == "F" else -1
            best = max(best, live)
        peak.append(best)
    return timeline, makespan, 1 - busy / (p * makespan), peak


def stage_ranges(n_layer, p):
    """Which blocks each stage owns (as even as possible)."""
    edges = np.linspace(0, n_layer, p + 1).round().astype(int)
    return [range(edges[s], edges[s + 1]) for s in range(p)]


def run_pipeline(model, x, y, p, m, schedule_fn=schedule_1f1b, log=None):
    """Execute a pipeline schedule on the real GPT. Stage 0 also owns the
    embeddings, stage p-1 the final LN, the head and the loss.

    Each micro-batch gets its own copy of the model as an activation stash
    (the layers cache their activations on self; the copies share the same
    parameter VALUES). Ops run in the order of the simulated timeline.
    Returns the accumulated gradients and the mean loss."""
    timeline, *_ = simulate_schedule(schedule_fn(p, m))
    ranges = stage_ranges(model.config["n_layer"], p)
    xs, ys = np.array_split(x, m), np.array_split(y, m)
    stash = {i: clone(model) for i in range(m)}
    acts, grads_in, losses = {}, {}, []
    T = x.shape[1]
    for s, op, i, *_ in sorted(timeline, key=lambda e: (e[3], e[0])):
        c = stash[i]
        if op == "F":
            h = c.wte.forward(xs[i]) + c.wpe.forward(np.arange(T)) if s == 0 else acts[(s - 1, i)]
            for b in ranges[s]:
                h = c.blocks[b].forward(h)
            if s == p - 1:
                logits = c.lm_head.forward(c.ln_f.forward(h))
                loss, dlogits = cross_entropy(logits, ys[i])
                losses.append(loss)
                c.dlogits = dlogits / m  # the full-batch loss is the mean of the m micro-losses
            else:  # send the activations to the next stage
                acts[(s, i)] = h if log is None else log.send(s, s + 1, h)
        else:
            if s == p - 1:
                dh = c.ln_f.backward(c.lm_head.backward(c.dlogits))
            else:
                dh = grads_in[(s + 1, i)]
            for b in reversed(ranges[s]):
                dh = c.blocks[b].backward(dh)
            if s == 0:
                c.wte.backward(dh)
                c.wpe.backward(dh.sum(axis=0))
            else:  # send the gradient back to the previous stage
                grads_in[(s, i)] = dh if log is None else log.send(s, s - 1, dh)
    total = None
    for i in range(m):
        g = stash[i].grads()
        total = {k: v.copy() for k, v in g.items()} if total is None else \
            {k: total[k] + g[k] for k in total}
    return total, float(np.mean(losses))


def gantt_text(timeline, p, makespan, unit=1.0):
    """The schedule as text: one row per stage, F/B + micro-batch id, '.' = bubble."""
    width = int(round(makespan / unit))
    rows = [["  ."] * width for _ in range(p)]
    for s, op, i, start, end in timeline:
        for t in range(int(round(start / unit)), int(round(end / unit))):
            rows[s][t] = f" {op}{i}"
    return [f"    stage {s} |" + "".join(r) for s, r in enumerate(rows)]


# ============================================================================
# Demo
# ============================================================================
def main():
    _, chars, data = load_text()
    V = len(chars)
    rng = np.random.default_rng(0)

    print("1. Ring all-reduce: reduce-scatter (W-1 steps) + all-gather (W-1 steps)\n")
    n = 1_000_000  # one million fp32 gradients = 4 MB per worker
    print(f"   n = {n:,} fp32 numbers ({n * 4 / 1e6:.0f} MB) per worker\n")
    print(f"   {'W':>4s} {'steps':>6s} {'ring: MB sent/worker':>21s} {'= 2(W-1)/W·n·4':>15s} "
          f"{'naive: MB via rank 0':>21s} {'max |err|':>10s}")
    for W in (2, 4, 8, 16):
        arrays = [rng.standard_normal(n).astype(np.float32) for _ in range(W)]
        log, nlog = CommLog(W), CommLog(W)
        out = ring_all_reduce(arrays, log)
        naive_all_reduce(arrays, nlog)
        err = max(float(np.abs(o - np.sum(arrays, axis=0)).max()) for o in out)
        print(f"   {W:4d} {2 * (W - 1):6d} {log.sent.max() / 1e6:21.2f} "
              f"{2 * (W - 1) / W * n * 4 / 1e6:15.2f} {nlog.sent[0] / 1e6:21.1f} {err:10.1e}")
    print("\n   Per-worker traffic approaches 2n however many workers there are; the naive"
          "\n   version funnels (W-1)·n through rank 0. (float32 sums in a different order"
          "\n   differ in the last bits; in float64 below the results are exact to 1e-15.)\n")

    print("2. Data parallelism: 4 replicas vs 1 worker on the full batch (3 AdamW steps)\n")
    model = small_gpt(V)
    replicas, ref = data_parallel_train(model, data, W=4, steps=3)
    spread = max(max_diff(replicas[0].params(), r.params()) for r in replicas[1:])
    rest, kb = max_diff_split(replicas[0].params(), ref.params(), model.config["d_model"])
    print(f"   max |param difference| between replicas      : {spread:.1e}")
    print(f"   max |param difference| replica 0 vs 1 worker : {rest:.1e}, "
          f"and {kb:.1e} in the key biases b_k")
    print("   (b_k's true gradient is 0, so its computed gradient is round-off, which Adam"
          "\n   divides by its own RMS: a different summation order shows up there.)\n")

    print("3. Gradient accumulation: 4 micro-batches of 2 == one batch of 8\n")
    x, y = get_batch(data, 8, 8, rng)
    m1 = clone(model)
    m1.forward(x, y)
    big = m1.backward()
    acc = accumulated_grads(clone(model), x, y, 4)
    print(f"   max |grad difference| = {max_diff(big, acc):.1e}"
          "   (each micro-loss divided by 4, the number of micro-batches)\n")

    print("4. ZeRO / FSDP on W = 4 ranks (3 steps), vs plain AdamW on the full batch\n")
    print(f"   {'':8s} {'max |Δ| vs AdamW':>17s} {'(b_k)':>8s} {'params':>8s} {'grads':>8s} "
          f"{'optim':>8s} {'total B/rank':>13s} {'numbers sent/rank/step':>23s}")
    ref = clone(model)
    opt = AdamW(ref.params())
    batches = [get_batch(data, 8, 8, np.random.default_rng(10 + i)) for i in range(3)]
    for xb, yb in batches:
        ref.forward(xb, yb)
        opt.step(ref.backward(), 1e-2)
    P = model.n_params()
    plain = 16 * P
    print(f"   {'DP':8s} {'':>17s} {'':>8s} {2 * P:8d} {2 * P:8d} {12 * P:8d} {plain:13d} "
          f"{2 * 3 * P // 4:23d}")
    for stage in (1, 2, 3):
        z = ZeroSimulator(model, 4, stage)
        for xb, yb in batches:
            z.step(xb, yb, 1e-2)
        b = z.bytes_per_rank()
        rest, kb = max_diff_split(z.full_params(), ref.params(), model.config["d_model"])
        print(f"   ZeRO-{stage:<3d} {rest:17.1e} {kb:8.0e} {b['params']:8d} {b['grads']:8d} "
              f"{b['optimizer']:8d} {b['total']:13d} {z.log.sent.max() // 8 // 3:23d}")
    print(f"\n   {P:,} params; bytes as a bf16 mixed-precision run would store them "
          f"(2 + 2 + 12 per param).")
    print(f"   DP all-reduces 2(W-1)/W·P numbers; ZeRO-1/2 send the same (reduce-scatter grads + "
          f"all-gather\n   params). ZeRO-3 sends ~1.5x: each block's parameters are gathered "
          f"twice (forward and\n   backward; the root unit once) + the reduce-scatter. Peak gathered "
          f"parameters on a rank: {z.peak_gathered:,}\n   of {P:,} (the root unit + one block), "
          f"not the whole model.\n")

    print("5. Megatron tensor parallelism (t = 2 and 4), one MLP and one attention layer\n")
    C, nh, B, T = 16, 4, 2, 8
    mlp = MLP(C, rng, dtype=F64)
    attn = CausalSelfAttention(C, nh, rng, dtype=F64)
    for layer in (mlp.c_fc, mlp.c_proj, attn.c_attn, attn.c_proj):
        for p in layer.params.values():
            p[...] = 0.3 * rng.standard_normal(p.shape)
    xx, dy = rng.standard_normal((B, T, C)), rng.standard_normal((B, T, C))
    for name, layer, fn in (("MLP", mlp, tp_mlp), ("attention", attn, tp_attention)):
        y_ref = layer.forward(xx)
        dx_ref = layer.backward(dy)
        for t in (2, 4):
            log = CommLog(t)
            y_tp, dx_tp, g = fn(layer, xx, dy, t, log)
            gd = max(float(np.abs(g[k] - getattr(layer, k.split(".")[0]).grads[k.split(".")[1]]).max())
                     for k in g)
            print(f"   {name:9s} t={t}: |Δy| {np.abs(y_tp - y_ref).max():.0e}  |Δdx| "
                  f"{np.abs(dx_tp - dx_ref).max():.0e}  |Δgrads| {gd:.0e}  "
                  f"| 2 all-reduces of (B,T,C) = {log.sent.max():,} B sent per rank")
    print("\n   Each rank stores 1/t of the layer's weights and does 1/t of its FLOPs. The price:"
          "\n   an all-reduce of the activations in forward and backward, in EVERY layer, on"
          "\n   the critical path. That is why TP stays inside a node (NVLink), typically t <= 8.\n")

    print("6. Pipeline parallelism: p = 4 stages, m micro-batches, backward = 2x forward\n")
    for m in (4, 8):
        for name, fn in (("GPipe", schedule_gpipe), ("1F1B", schedule_1f1b)):
            tl, span, bubble, peak = simulate_schedule(fn(4, m))
            print(f"   {name} m={m}: makespan {span:.0f} time units, bubble {bubble:.1%} "
                  f"(formula (p-1)/(m+p-1) = {3 / (m + 3):.1%}), stashed micro-batches "
                  f"per stage {peak}")
            if m == 4:
                print("\n".join(gantt_text(tl, 4, span)) + "\n")
    deep = small_gpt(V, n_layer=4)
    xp, yp = get_batch(data, 8, 8, rng)
    full = clone(deep)
    loss_full = full.forward(xp, yp)
    gfull = full.backward()
    for name, fn in (("GPipe", schedule_gpipe), ("1F1B", schedule_1f1b)):
        g, loss = run_pipeline(deep, xp, yp, 4, 4, fn)
        print(f"   executed {name} on a 4-block GPT: loss {loss:.6f} vs {loss_full:.6f}, "
              f"max |grad difference| {max_diff(g, gfull):.1e}")
    print()
    print(f"   {'p':>3s} {'m':>4s} {'bubble':>7s}")
    for p_, m_ in ((4, 4), (4, 16), (8, 8), (8, 32), (16, 64)):
        print(f"   {p_:3d} {m_:4d} {(p_ - 1) / (m_ + p_ - 1):7.1%}")
    print("\n   More micro-batches shrink the bubble, but GPipe must stash all m of them; 1F1B "
          "caps it at p.")


if __name__ == "__main__":
    main()
