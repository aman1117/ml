"""
Batched generation for the NumPy mini-GPT: why a server batches, and how.

One request at a time, the GPU (here: the CPU) spends a decode step reading
ALL the weights to produce ONE token. Batching B requests reads the weights once
for B tokens. This file builds the three ways to do it, from scratch:

  1. STATIC (padded) batching: left-pad B prompts to the same length, run ONE
     masked forward (attention mask hides the pads, every row has its own
     positions 0..L_i-1), then decode all rows together until the LONGEST is done.
     Proven equal to one-at-a-time generation (generate.py) in serving_check.py.
  2. A PAGED KV cache (PagedAttention, Kwon et al. 2023): one pool of fixed-size
     blocks, a free list, and a block table per sequence that maps logical token
     positions to physical blocks, so memory is allocated as tokens arrive and
     freed the moment a request ends. Attention gathers K, V through the table.
  3. CONTINUOUS (in-flight, iteration-level) batching (Orca, Yu et al. 2022): the
     scheduler runs one decode step for every running request, then immediately
     admits waiting requests into the free slots/blocks, instead of waiting for
     the whole batch to finish. Linear layers run on all tokens at once; only
     attention is per-sequence (Orca's "selective batching").

Then it replays a Poisson request trace through static vs continuous batching
on the real model and measures throughput, TTFT (time to first token), TPOT
(time per output token), p50/p95/p99 latency and KV-cache memory, and it
measures the roofline of this CPU: why decode is memory-bound and prefill is
compute-bound.

Our model only learned positions 0..31, so (exactly like generate.py) when a
sequence's window is full we keep its last 16 tokens, re-prefill them and go on.

Run (~40 s):
    python serving/batching.py
    python serving/batching.py --requests 400 --rate 400 --max-batch 16
"""
from __future__ import annotations

import argparse
import itertools
import math
import time
from collections import deque

import numpy as np

from common import (CharTokenizer, get_model, lm_logits, percentile, prefill,
                    sample_next, softmax, train_ids)


def attend(q, K, V, mask, scale):
    """softmax(q Kᵀ·scale, masked) V.   q (B,nh,tq,hs)  K,V (B,nh,tk,hs)  mask -> (B,1,tq,tk)"""
    s = (q @ K.transpose(0, 1, 3, 2)) * scale
    return softmax(np.where(mask, s, -np.inf)) @ V


def embed(model, tokens, positions):
    return model.wte.params["weight"][tokens] + model.wpe.params["weight"][positions]


# ----------------------------------------------------------------------------
# 1. Static batching: left padding + attention mask + per-sequence positions
# ----------------------------------------------------------------------------
class PaddedCache:
    """K, V per layer as rectangles (B, nh, W, hs). valid[b, j] says whether column j
    of row b is a real token (True) or padding. length[b] = real tokens in row b,
    which is also the position the next token of row b gets."""

    def __init__(self, k, v, valid):
        self.k, self.v, self.valid = k, v, valid
        self.length = valid.sum(axis=1)

    @property
    def width(self):
        return self.valid.shape[1]

    def nbytes(self):
        return sum(a.nbytes for a in self.k + self.v)


def padded_prefill(model, prompts):
    """ONE forward pass over B prompts of different lengths, left-padded:

        row 0:  ·  ·  T  h  e        valid: F F T T T   positions: 0 0 0 1 2
        row 1:  T  h  e     k        valid: T T T T T   positions: 0 1 2 3 4

    Left padding puts every prompt's LAST token in the last column, so the
    next-token logits of all rows are logits[:, -1]. Returns (logits (B, V), cache)."""
    B, W = len(prompts), max(len(p) for p in prompts)
    idx = np.zeros((B, W), dtype=np.int64)
    pos = np.zeros((B, W), dtype=np.int64)
    valid = np.zeros((B, W), dtype=bool)
    for i, p in enumerate(prompts):
        L = len(p)
        idx[i, W - L:], pos[i, W - L:], valid[i, W - L:] = p, np.arange(L), True
    # query t may see key j iff j <= t (causal) AND j is a real token. A pad
    # query sees only itself, so its softmax row is never all -inf (no NaN);
    # its output is garbage that no real token ever reads.
    causal = np.tril(np.ones((W, W), dtype=bool))
    mask = (causal[None] & valid[:, None, :]) | np.eye(W, dtype=bool)[None]  # (B, W, W)
    x = embed(model, idx, pos)  # (B, W, C)
    ks, vs = [], []
    for blk in model.blocks:
        a = blk.attn
        q, k, v = np.split(a.c_attn.forward(blk.ln_1.forward(x)), 3, axis=-1)
        q, k, v = a.split_heads(q), a.split_heads(k), a.split_heads(v)  # (B, nh, W, hs)
        y = attend(q, k, v, mask[:, None], 1.0 / math.sqrt(a.hs))
        x = x + a.c_proj.forward(a.merge_heads(y))
        x = x + blk.mlp.forward(blk.ln_2.forward(x))
        ks.append(k)
        vs.append(v)
    logits = lm_logits(model, model.ln_f.forward(x[:, -1:]))[:, 0]
    return logits, PaddedCache(ks, vs, valid)


def padded_decode(model, tokens, cache):
    """Feed one new token per row (row b at position cache.length[b]); append to the cache."""
    B = len(tokens)
    x = embed(model, np.asarray(tokens), cache.length)[:, None]  # (B, 1, C)
    cache.valid = np.concatenate([cache.valid, np.ones((B, 1), dtype=bool)], axis=1)
    mask = cache.valid[:, None, None, :]  # (B, 1, 1, W+1): pads stay invisible
    for l, blk in enumerate(model.blocks):
        a = blk.attn
        q, k, v = np.split(a.c_attn.forward(blk.ln_1.forward(x)), 3, axis=-1)
        q, k, v = a.split_heads(q), a.split_heads(k), a.split_heads(v)  # (B, nh, 1, hs)
        cache.k[l] = np.concatenate([cache.k[l], k], axis=2)
        cache.v[l] = np.concatenate([cache.v[l], v], axis=2)
        y = attend(q, cache.k[l], cache.v[l], mask, 1.0 / math.sqrt(a.hs))
        x = x + a.c_proj.forward(a.merge_heads(y))
        x = x + blk.mlp.forward(blk.ln_2.forward(x))
    cache.length = cache.length + 1
    return lm_logits(model, model.ln_f.forward(x))[:, 0]


def refill_rows(model, cache, rows, windows):
    """Window full for these rows: replace each row's cache by a fresh prefill of
    `windows[i]` (its most recent tokens), right-aligned so the batch stays a
    rectangle. Then drop leading columns that are padding in every row."""
    _, sub = padded_prefill(model, windows)
    W, n = cache.width, sub.width
    for l in range(len(cache.k)):
        for c, s in ((cache.k, sub.k), (cache.v, sub.v)):
            c[l][rows] = 0.0
            c[l][rows, :, W - n:] = s[l]
    cache.valid[rows] = False
    cache.valid[rows, W - n:] = sub.valid
    cache.length[rows] = sub.length
    first = int(cache.valid.any(axis=0).argmax())
    if first:
        cache.k = [k[:, :, first:] for k in cache.k]
        cache.v = [v[:, :, first:] for v in cache.v]
        cache.valid = cache.valid[:, first:]


def static_generate(model, reqs, on_step=None):
    """Generate for a batch of Requests together, until EVERY row is finished.

    Same semantics as generate.py's generate_ids for each row: prefill the last
    T prompt tokens; after sampling a token, if the row's window is full (T),
    re-prefill its last T/2 tokens. (We prefill T/2 - 1 of them and let the
    batched decode step feed the newest one, which gives the same logits and keeps
    every row on one token per step.) Rows that are already finished keep
    decoding greedily until the longest row is done: that wasted compute is
    exactly what static batching costs. on_step(kind, cache) is called after the
    prefill and after every decode step (kind = "prefill" / "decode")."""
    T = model.config["block_size"]
    hist = [list(r.prompt) for r in reqs]
    logits, cache = padded_prefill(model, [h[-T:] for h in hist])
    kind = "prefill"
    while True:
        toks = []
        for i, r in enumerate(reqs):
            if r.finish_reason is None:
                tok = sample_next(logits[i], r.temperature, r.top_k, r.rng)
                r.emit(tok)
            else:
                tok = int(np.argmax(logits[i]))  # wasted slot: keep the rectangle going
            hist[i].append(tok)
            toks.append(tok)
        if on_step:
            on_step(kind, cache)
        if all(r.finish_reason is not None for r in reqs):
            break
        full = [i for i in range(len(reqs)) if cache.length[i] == T]
        if full:
            refill_rows(model, cache, full, [hist[i][-(T // 2):-1] for i in full])
        logits = padded_decode(model, toks, cache)
        kind = "decode"
    return [r.out for r in reqs]


# ----------------------------------------------------------------------------
# 2. Paged KV cache
# ----------------------------------------------------------------------------
class PagedKVCache:
    """vLLM-style paged KV memory for all sequences.

    k, v pools: (n_layer, num_blocks, block_size, nh, hs). A sequence owns a list of
    block ids (its BLOCK TABLE); token position p lives in block table[p // bs],
    slot p % bs. Blocks come from a FREE LIST and go back to it when the sequence
    ends, so no memory is reserved for tokens that were never generated, and the
    only waste is the unfilled tail of each sequence's last block."""

    def __init__(self, n_layer, n_head, head_size, num_blocks, block_size=8,
                 dtype=np.float32):
        shape = (n_layer, num_blocks, block_size, n_head, head_size)
        self.k, self.v = np.zeros(shape, dtype), np.zeros(shape, dtype)
        self.num_blocks, self.block_size = num_blocks, block_size
        self.free = deque(range(num_blocks))
        self.tables, self.lengths = {}, {}

    @classmethod
    def for_model(cls, model, num_blocks, block_size=8):
        a = model.blocks[0].attn
        return cls(len(model.blocks), a.n_head, a.hs, num_blocks, block_size,
                   model.wte.params["weight"].dtype)

    def bytes_per_token(self):
        return 2 * self.k.shape[0] * self.k.shape[3] * self.k.shape[4] * self.k.itemsize

    def ensure(self, seq, n_tokens):
        """Make sure seq has blocks for n_tokens positions. All-or-nothing; False if
        the free list is too short (the caller must wait or preempt)."""
        table = self.tables.setdefault(seq, [])
        need = -(-n_tokens // self.block_size) - len(table)
        if need > len(self.free):
            if not table:
                del self.tables[seq]
            return False
        table.extend(self.free.popleft() for _ in range(max(0, need)))
        self.lengths.setdefault(seq, 0)
        return True

    def release(self, seq):
        self.free.extend(self.tables.pop(seq, []))
        self.lengths.pop(seq, None)

    def _slots(self, seq, positions):
        table = self.tables[seq]
        positions = np.asarray(positions)
        return np.array([table[p // self.block_size] for p in positions]), positions % self.block_size

    def write_seq(self, seq, layer, k, v, start=0):
        """Write a prefill's k, v (nh, t, hs) at positions start..start+t-1."""
        blk, off = self._slots(seq, np.arange(start, start + k.shape[1]))
        self.k[layer, blk, off] = k.transpose(1, 0, 2)
        self.v[layer, blk, off] = v.transpose(1, 0, 2)
        self.lengths[seq] = max(self.lengths[seq], start + k.shape[1])

    def write_tokens(self, layer, seqs, positions, k, v):
        """One new token per sequence: k, v (B, nh, hs) at positions[b]."""
        blk = np.array([self.tables[s][p // self.block_size] for s, p in zip(seqs, positions)])
        off = np.asarray(positions) % self.block_size
        self.k[layer, blk, off] = k
        self.v[layer, blk, off] = v
        for s, p in zip(seqs, positions):
            self.lengths[s] = max(self.lengths[s], p + 1)

    def gather(self, layer, seqs, n_tokens):
        """What the PagedAttention kernel reads: each sequence's K, V through its block
        table, as (B, nh, nb·bs, hs) plus a (B, nb·bs) mask of real positions."""
        nb = max(-(-n // self.block_size) for n in n_tokens)
        table = np.zeros((len(seqs), nb), dtype=np.int64)  # short tables padded with block 0
        for i, s in enumerate(seqs):
            t = self.tables[s][:nb]
            table[i, :len(t)] = t
        B, bs = len(seqs), self.block_size
        K = self.k[layer][table].reshape(B, nb * bs, *self.k.shape[3:]).transpose(0, 2, 1, 3)
        V = self.v[layer][table].reshape(B, nb * bs, *self.v.shape[3:]).transpose(0, 2, 1, 3)
        mask = np.arange(nb * bs)[None] < np.asarray(n_tokens)[:, None]
        return K, V, mask

    def stats(self):
        used = self.num_blocks - len(self.free)
        tokens = sum(self.lengths.values())
        return dict(used_blocks=used, used_tokens=tokens, reserved_tokens=used * self.block_size,
                    fragmentation=(1 - tokens / (used * self.block_size)) if used else 0.0)


# ----------------------------------------------------------------------------
# 3. Continuous batching
# ----------------------------------------------------------------------------
class Request:
    """One generation request. Timestamps are filled in by whoever runs the engine."""
    _next_id = itertools.count()

    def __init__(self, prompt, max_tokens, temperature=0.8, top_k=None, seed=None,
                 stop=(), arrival=0.0, rid=None):
        assert len(prompt) >= 1 and max_tokens >= 1
        self.rid = next(Request._next_id) if rid is None else rid
        self.prompt, self.max_tokens = list(prompt), int(max_tokens)
        self.temperature, self.top_k = temperature, top_k
        self.stop = [tuple(s) for s in stop if len(s)]
        self.rng = np.random.default_rng(seed)
        self.ids = list(prompt)  # prompt + generated, the whole history
        self.out = []
        self.ws = 0  # window start: the KV cache holds ids[ws:-1] (all but the newest)
        self.cached = 0  # = len(ids[ws:-1]) while running
        self.finish_reason = None
        self.arrival, self.t_admit, self.t_first, self.t_finish = arrival, None, None, None
        self.token_times = []
        self.n_preempted, self.order, self.fresh = 0, 0, False

    def emit(self, tok):
        self.out.append(tok)
        self.ids.append(tok)
        for s in self.stop:
            if tuple(self.out[-len(s):]) == s:
                self.finish_reason = "stop"
        if self.finish_reason is None and len(self.out) >= self.max_tokens:
            self.finish_reason = "length"


class StepResult:
    def __init__(self):
        self.tokens = []  # (req, token)
        self.prefill = []  # (req, n_tokens) prefilled this step (admission or window slide)
        self.decoded = []  # requests that got one decode step
        self.finished = []
        self.preempted = []


class ContinuousBatcher:
    """Iteration-level scheduler over a PagedKVCache.

    step():
      1. ADMIT waiting requests (first come, first served) while there is a free
         slot (max_batch) and enough free KV blocks for the prompt: prefill it,
         write its K, V into its blocks, sample its first token.
      2. For every other running request, make room for one more token: slide its
         window if full, allocate a new block if it crosses a block boundary. If the
         pool is empty, PREEMPT the most recently admitted request (free its
         blocks, put it back at the front of the queue; it is re-prefilled later,
         vLLM's "recompute" preemption).
      3. ONE batched decode step for all of them; sample; finished requests free
         their blocks at once, so the next step can admit someone new."""

    def __init__(self, model, kv, max_batch=8):
        self.model, self.kv, self.max_batch = model, kv, max_batch
        self.T = model.config["block_size"]
        self.waiting, self.running = deque(), []
        self._order = itertools.count()

    def add(self, req):
        self.waiting.append(req)

    def has_work(self):
        return bool(self.waiting or self.running)

    def abort(self, req, reason):
        """Cancel (timeout, client gone): free its memory wherever it is."""
        if req in self.waiting:
            self.waiting.remove(req)
        if req in self.running:
            self.running.remove(req)
        self.kv.release(req.rid)
        req.finish_reason = reason

    def _load(self, r, window, reserve=0):
        """Prefill `window` and store its K, V in r's blocks. None if no memory (or if
        fewer than `reserve` blocks would be left free afterwards)."""
        need = -(-len(window) // self.kv.block_size)
        if len(self.kv.free) - need < reserve or not self.kv.ensure(r.rid, len(window)):
            return None
        logits, cache = prefill(self.model, window)
        for l in range(len(self.model.blocks)):
            self.kv.write_seq(r.rid, l, cache.k[l], cache.v[l])
        r.cached = len(window)
        return logits

    def _preempt(self, r, res):
        self.kv.release(r.rid)
        self.running.remove(r)
        self.waiting.appendleft(r)
        r.n_preempted += 1
        res.preempted.append(r)

    def step(self):
        res, T = StepResult(), self.T
        # 1. admission
        while self.waiting and len(self.running) < self.max_batch:
            r = self.waiting[0]
            if not r.out:  # new request: its prompt (last T tokens)
                window = r.prompt[-T:]
                r.ws = len(r.prompt) - len(window)
            else:  # preempted earlier: recompute exactly the cache it had
                window = r.ids[r.ws:-1]
            # Watermark: leave one free block per running request so they can grow
            # (otherwise a new admission is preempted again at once: thrashing).
            logits = self._load(r, window, reserve=len(self.running))
            if logits is None:
                break  # FCFS: don't let later (smaller) requests jump the queue
            self.waiting.popleft()
            r.order, r.fresh = next(self._order), True
            res.prefill.append((r, len(window)))
            if not r.out:
                r.emit(sample_next(logits, r.temperature, r.top_k, r.rng))
                res.tokens.append((r, r.out[-1]))
                if r.finish_reason:
                    self.kv.release(r.rid)
                    res.finished.append(r)
                    continue
            self.running.append(r)
        # 2. room for one more token per running request
        batch = [r for r in self.running if not r.fresh]
        for r in self.running:
            r.fresh = False
        for r in batch:
            if r not in self.running:
                continue  # preempted to make room for an earlier request
            if r.cached == T:  # window full: keep the last T/2 tokens (as generate.py)
                self.kv.release(r.rid)
                r.ws = len(r.ids) - T // 2
                self._load(r, r.ids[r.ws:-1])  # always fits: we just freed more
                res.prefill.append((r, r.cached))
            while not self.kv.ensure(r.rid, r.cached + 1):
                victim = max(self.running, key=lambda s: s.order)
                self._preempt(victim, res)
                if victim is r:
                    break
        batch = [r for r in batch if r in self.running]
        # 3. one batched decode step
        if batch:
            logits = self._decode(batch)
            for r, lg in zip(batch, logits):
                r.emit(sample_next(lg, r.temperature, r.top_k, r.rng))
                res.tokens.append((r, r.out[-1]))
                res.decoded.append(r)
                if r.finish_reason:
                    self.running.remove(r)
                    self.kv.release(r.rid)
                    res.finished.append(r)
        return res

    def _decode(self, batch):
        """Selective batching: every Linear/LN/MLP sees all B new tokens as one (B, C)
        matrix (weights read once); attention reads each sequence's own blocks."""
        model, kv = self.model, self.kv
        seqs = [r.rid for r in batch]
        pos = np.array([r.cached for r in batch])  # newest token's position in its window
        x = embed(model, np.array([r.ids[-1] for r in batch]), pos)[:, None]  # (B, 1, C)
        for l, blk in enumerate(model.blocks):
            a = blk.attn
            q, k, v = np.split(a.c_attn.forward(blk.ln_1.forward(x)), 3, axis=-1)
            q, k, v = a.split_heads(q), a.split_heads(k), a.split_heads(v)  # (B, nh, 1, hs)
            kv.write_tokens(l, seqs, pos, k[:, :, 0], v[:, :, 0])
            K, V, mask = kv.gather(l, seqs, pos + 1)
            y = attend(q, K, V, mask[:, None, None, :], 1.0 / math.sqrt(a.hs))
            x = x + a.c_proj.forward(a.merge_heads(y))
            x = x + blk.mlp.forward(blk.ln_2.forward(x))
        for r in batch:
            r.cached += 1
        return lm_logits(model, model.ln_f.forward(x))[:, 0]


def reference_generate(model, req):
    """The unbatched answer: generate.py's KV-cache sampler, then cut at the stop."""
    from common import generate_ids
    out = generate_ids(model, req.prompt, req.max_tokens, req.temperature, req.top_k,
                       req.rng)
    for i in range(len(out)):
        if any(tuple(out[max(0, i + 1 - len(s)):i + 1]) == s for s in req.stop):
            return out[:i + 1]
    return out


# ----------------------------------------------------------------------------
# 4. Replaying a request trace
# ----------------------------------------------------------------------------
def make_trace(data, n=300, rate=300.0, seed=0, min_prompt=4, max_prompt=32,
               mean_out=40, max_out=160):
    """Poisson arrivals at `rate` req/s; prompts are random slices of the text
    (4..32 chars); output lengths are log-normal (many short, a long tail), as in
    real chat traffic. Returned as plain dicts so every policy gets fresh Requests."""
    rng = np.random.default_rng(seed)
    t = np.cumsum(rng.exponential(1.0 / rate, n))
    trace = []
    for i in range(n):
        L = int(rng.integers(min_prompt, max_prompt + 1))
        s = int(rng.integers(0, len(data) - L))
        out = int(np.clip(rng.lognormal(math.log(mean_out), 0.7), 4, max_out))
        trace.append(dict(rid=i, prompt=[int(c) for c in data[s:s + L]], max_tokens=out,
                          temperature=0.8, seed=i, arrival=float(t[i])))
    return trace


def to_requests(trace):
    return [Request(d["prompt"], d["max_tokens"], d["temperature"], seed=d["seed"],
                    arrival=d["arrival"], rid=d["rid"]) for d in trace]


def simulate(model, trace, policy="continuous", max_batch=8, block_size=8, num_blocks=None):
    """Replay the trace. Compute is REAL (each engine step is timed with
    perf_counter); waiting for the next arrival is simulated (the clock jumps), so a
    300-request trace takes about a second. Returns (requests, timeline)."""
    reqs = to_requests(trace)
    pending = deque(sorted(reqs, key=lambda r: r.arrival))
    T = model.config["block_size"]
    a = model.blocks[0].attn
    bpt = 2 * len(model.blocks) * a.n_head * a.hs * model.wte.params["weight"].itemsize
    timeline = []
    now = 0.0

    def arrivals(queue):
        while pending and pending[0].arrival <= now:
            queue.append(pending.popleft())

    if policy == "continuous":
        num_blocks = num_blocks or max_batch * -(-T // block_size)
        kv = PagedKVCache.for_model(model, num_blocks, block_size)
        eng = ContinuousBatcher(model, kv, max_batch)
        while pending or eng.has_work():
            arrivals(eng.waiting)
            if not eng.has_work():
                now = pending[0].arrival
                continue
            t0 = time.perf_counter()
            res = eng.step()
            start, now = now, now + time.perf_counter() - t0
            for r, _ in res.tokens:
                r.token_times.append(now)
                r.t_first = r.t_first or now
            for r, _ in res.prefill:
                r.t_admit = r.t_admit or start
            for r in res.finished:
                r.t_finish = now
            st = kv.stats()
            timeline.append(dict(t0=start, t1=now, prefill=[r.rid for r, _ in res.prefill],
                                 decode=[r.rid for r in res.decoded],
                                 used_tokens=st["used_tokens"],
                                 reserved_tokens=st["reserved_tokens"],
                                 # what a contiguous allocator would reserve: each running
                                 # request's maximum window, min(T, prompt + max_tokens)
                                 contiguous_tokens=sum(min(T, len(r.prompt) + r.max_tokens)
                                                       for r in eng.running),
                                 tables={r.rid: list(kv.tables.get(r.rid, []))
                                         for r in eng.running},
                                 preempted=[r.rid for r in res.preempted]))
        return reqs, dict(timeline=timeline, bytes_per_token=bpt)

    queue = deque()
    while pending or queue:
        arrivals(queue)
        if not queue:
            now = pending[0].arrival
            continue
        batch = [queue.popleft() for _ in range(min(max_batch, len(queue)))]
        clock = {"t": time.perf_counter(), "start": now}
        n_new = max(r.max_tokens for r in batch)
        W0 = max(len(r.prompt[-T:]) for r in batch)

        def on_step(kind, cache, batch=batch, clock=clock, n_new=n_new, W0=W0):
            nonlocal now
            now += time.perf_counter() - clock["t"]
            for r in batch:
                if len(r.out) > len(r.token_times):
                    r.token_times.append(now)
                    r.t_first = r.t_first or now
                r.t_admit = r.t_admit or clock["start"]
            live = [i for i, r in enumerate(batch) if r.finish_reason is None]
            timeline.append(dict(t0=clock["start"], t1=now, prefill=[r.rid for r in batch]
                                 if kind == "prefill" else [],
                                 decode=[r.rid for r in batch] if kind == "decode" else [],
                                 used_tokens=int(sum(cache.length[i] for i in live)),
                                 # static caches are preallocated for the longest row:
                                 # B x (longest prompt + longest output), capped at T
                                 reserved_tokens=len(batch) * min(T, W0 + n_new),
                                 contiguous_tokens=len(batch) * min(T, W0 + n_new),
                                 tables={}, preempted=[]))
            clock["start"] = now
            clock["t"] = time.perf_counter()

        static_generate(model, batch, on_step)
        # A static batch returns one padded output tensor when its LONGEST row is
        # done, so every request in it finishes then, and its slot was busy until then.
        for r in batch:
            r.t_finish = now
    return reqs, dict(timeline=timeline, bytes_per_token=bpt)


def summarize(reqs, timeline):
    """Throughput and latency percentiles for one replayed trace."""
    t_end = max(r.t_finish for r in reqs)
    t_start = min(r.arrival for r in reqs)
    n_out = sum(len(r.out) for r in reqs)
    ttft = [r.t_first - r.arrival for r in reqs]
    tpot = [(r.token_times[-1] - r.token_times[0]) / (len(r.out) - 1)
            for r in reqs if len(r.out) > 1]
    lat = [r.t_finish - r.arrival for r in reqs]
    busy = sum(s["t1"] - s["t0"] for s in timeline)
    w = np.array([s["t1"] - s["t0"] for s in timeline])
    used = np.array([s["used_tokens"] for s in timeline])
    reserved = np.array([s["reserved_tokens"] for s in timeline])
    out = dict(throughput=n_out / (t_end - t_start), makespan=t_end - t_start, busy=busy,
               tokens=n_out, kv_used=float((used * w).sum() / w.sum()),
               kv_reserved=float((reserved * w).sum() / w.sum()),
               kv_peak=int(reserved.max()))
    for name, xs in (("ttft", ttft), ("tpot", tpot), ("latency", lat)):
        for q in (50, 95, 99):
            out[f"{name}_p{q}"] = percentile(xs, q)
    return out


# ----------------------------------------------------------------------------
# 5. Why decode is memory-bound: the roofline of this machine
# ----------------------------------------------------------------------------
def best_time(f, repeats=5):
    f()
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        f()
        ts.append(time.perf_counter() - t0)
    return min(ts)


def measure_roofline(d=4096, batches=(1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048)):
    """y = x @ W with a (d, d) float32 weight that does not fit in cache, for b input
    rows. FLOPs = 2·b·d², bytes moved ≈ 4·(d² + 2·b·d), so the arithmetic intensity
    is ≈ b/2 FLOPs per byte: one row (a decode step at batch 1) does 0.5 FLOP per
    byte it reads; a 2048-token prefill does ~800."""
    rng = np.random.default_rng(0)
    W = rng.standard_normal((d, d)).astype(np.float32)
    big = np.ones(64 * 2 ** 20 // 4 * 4, dtype=np.float32)  # 256 MB
    dst = np.empty_like(big)
    bw = 2 * big.nbytes / best_time(lambda: np.copyto(dst, big), 3)  # read + write
    pts = []
    for b in batches:
        x = rng.standard_normal((b, d)).astype(np.float32)
        t = best_time(lambda: x @ W, 5 if b < 512 else 3)
        flops, nbytes = 2 * b * d * d, 4 * (d * d + 2 * b * d)
        pts.append(dict(b=b, ai=flops / nbytes, gflops=flops / t / 1e9, ms=t * 1e3))
    return dict(bandwidth=bw, points=pts, peak=max(p["gflops"] for p in pts), d=d)


def measure_prefill_decode(model, ids, batches=(1, 2, 4, 8, 16, 32, 64)):
    """Tokens/s of the real model: prefill of T tokens in one pass vs decode steps."""
    T = model.config["block_size"]
    t_pre = best_time(lambda: prefill(model, ids[:T]), 20)
    rows = []
    for B in batches:
        kv = PagedKVCache.for_model(model, B * (-(-T // 8)), 8)
        eng = ContinuousBatcher(model, kv, max_batch=B)
        reqs = [Request(ids[i:i + T // 2], 10 ** 9, 0.8, seed=i) for i in range(B)]
        for r in reqs:
            eng.add(r)
        eng.step()  # admit + prefill everyone
        ts = []
        for _ in range(T // 2 - 4):  # pure decode steps (no window slide yet)
            t0 = time.perf_counter()
            eng.step()
            ts.append(time.perf_counter() - t0)
        rows.append(dict(B=B, step_ms=float(np.median(ts)) * 1e3,
                         tok_s=B / float(np.median(ts))))
    return dict(prefill_tok_s=T / t_pre, prefill_ms=t_pre * 1e3, T=T, decode=rows)


# ----------------------------------------------------------------------------
def print_summary(name, s):
    print(f"  {name:11s} {s['throughput']:7.0f} tok/s | TTFT p50 {s['ttft_p50'] * 1e3:6.1f} "
          f"p95 {s['ttft_p95'] * 1e3:6.1f} p99 {s['ttft_p99'] * 1e3:6.1f} ms | TPOT p50 "
          f"{s['tpot_p50'] * 1e3:5.2f} ms | latency p50 {s['latency_p50'] * 1e3:6.1f} "
          f"p95 {s['latency_p95'] * 1e3:6.1f} p99 {s['latency_p99'] * 1e3:6.1f} ms")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--requests", type=int, default=300)
    ap.add_argument("--rate", type=float, default=300.0, help="mean arrivals per second")
    ap.add_argument("--max-batch", type=int, default=16)
    ap.add_argument("--block-size", type=int, default=8, help="KV block size in tokens")
    ap.add_argument("--no-roofline", action="store_true")
    args = ap.parse_args()

    model, chars, version = get_model()
    tok = CharTokenizer(chars)
    data = train_ids(tok)
    T = model.config["block_size"]

    print("\n== 1. Static batching = one-at-a-time generation (float64, greedy) ==")
    model64 = get_model(np.float64, verbose=False)[0]
    prompts = ["The keeper", "She", "One winter night a great storm", "Good morning,\" said"]
    reqs = [Request(tok.encode(p)[0], 50, temperature=0.0) for p in prompts]
    static_generate(model64, reqs)
    for p, r in zip(prompts, reqs):
        ref = reference_generate(model64, Request(r.prompt, 50, 0.0))
        print(f"  {p!r:36s} -> {tok.decode(r.out)[:34]!r}...  same as unbatched: "
              f"{r.out == ref}")

    print("\n== 2. Continuous batching with a paged KV cache (float64, seeded sampling) ==")
    kv = PagedKVCache.for_model(model64, num_blocks=10, block_size=args.block_size)
    eng = ContinuousBatcher(model64, kv, max_batch=4)
    reqs = [Request(tok.encode(p)[0], n, 0.8, seed=i)
            for i, (p, n) in enumerate(zip(prompts * 2, [5, 60, 20, 45, 30, 8, 70, 12]))]
    steps, admitted_late = 0, []
    for r in reqs[:4]:
        eng.add(r)
    while eng.has_work():
        if steps in (3, 6, 9, 12):
            eng.add(reqs[4 + steps // 3 - 1])
            admitted_late.append(steps)
        eng.step()
        steps += 1
    same = all(r.out == reference_generate(model64, Request(r.prompt, r.max_tokens, 0.8,
                                                            seed=i))
               for i, r in enumerate(reqs))
    print(f"  8 requests (4 joining mid-flight at steps {admitted_late}), pool of 10 blocks x "
          f"{args.block_size} tokens, {steps} engine steps, "
          f"{sum(r.n_preempted for r in reqs)} preemptions")
    print(f"  every output identical to unbatched generation with the same seed: {same}")

    print(f"\n== 3. Replaying a trace: {args.requests} requests, Poisson {args.rate:.0f} req/s, "
          f"max batch {args.max_batch} ==")
    trace = make_trace(data, args.requests, args.rate)
    outs = [d["max_tokens"] for d in trace]
    print(f"  prompts 4-32 chars, outputs {min(outs)}-{max(outs)} tokens "
          f"(mean {np.mean(outs):.0f}, total {sum(outs):,})")
    results = {}
    for policy in ("static", "continuous"):
        reqs_p, info = simulate(model, trace, policy, args.max_batch, args.block_size)
        s = summarize(reqs_p, info["timeline"])
        results[policy] = s
        print_summary(policy, s)
    for policy in ("static", "continuous"):
        s = results[policy]
        print(f"  {policy:11s} KV cache: {s['kv_used']:6.1f} tokens in use on average, "
              f"{s['kv_reserved']:6.1f} reserved ({s['kv_used'] / s['kv_reserved']:.0%} "
              f"utilized), peak {s['kv_peak']} tokens = "
              f"{s['kv_peak'] * info['bytes_per_token'] / 1024:.0f} KB")
    st, co = results["static"], results["continuous"]
    print(f"  continuous vs static: {co['throughput'] / st['throughput']:.2f}x throughput, "
          f"{st['latency_p99'] / co['latency_p99']:.1f}x lower p99 latency, "
          f"{st['ttft_p50'] / co['ttft_p50']:.1f}x lower median TTFT")

    print(f"\n== 3b. Saturated: the same {args.requests} requests at 3000 req/s (capacity), "
          f"max batch swept ==")
    sat = make_trace(data, args.requests, 3000.0)
    print(f"  {'batch':>5s} {'policy':>11s} {'tokens/s':>9s} {'TPOT p50':>9s} {'TTFT p50':>9s} "
          f"{'latency p50':>11s}")
    for mb in (1, 4, 16, 64):
        for policy in ("static", "continuous"):
            reqs_s, info_s = simulate(model, sat, policy, mb, args.block_size)
            s = summarize(reqs_s, info_s["timeline"])
            print(f"  {mb:5d} {policy:>11s} {s['throughput']:9.0f} {s['tpot_p50'] * 1e3:7.2f} ms "
                  f"{s['ttft_p50'] * 1e3:6.0f} ms {s['latency_p50'] * 1e3:8.0f} ms")
    print("  Bigger batches: more tokens/s for everyone, but each step is longer (TPOT up):\n"
          "  the throughput-latency trade-off. Continuous wins at every batch size > 1.")

    print("\n== 4. Prefill vs decode on the real model (float32) ==")
    pd = measure_prefill_decode(model, list(data[:200]))
    print(f"  prefill: {pd['T']} tokens in one pass = {pd['prefill_ms']:.2f} ms "
          f"-> {pd['prefill_tok_s']:,.0f} tokens/s")
    for row in pd["decode"]:
        print(f"  decode batch {row['B']:3d}: {row['step_ms']:.3f} ms/step -> "
              f"{row['tok_s']:8,.0f} tokens/s")
    print("  A decode step costs about the same for 1 or 16 sequences: the cost is reading\n"
          "  the weights (and here, Python overhead), not the arithmetic. Batching is free\n"
          "  throughput until the step becomes compute-bound.")

    if not args.no_roofline:
        print("\n== 5. Roofline of this CPU: y = x @ W, W = 4096 x 4096 float32 (64 MB) ==")
        rf = measure_roofline()
        print(f"  memory bandwidth (copy): {rf['bandwidth'] / 1e9:.0f} GB/s; "
              f"best matmul: {rf['peak']:.0f} GFLOP/s; ridge point ≈ "
              f"{rf['peak'] * 1e9 / rf['bandwidth']:.0f} FLOPs/byte")
        for p in rf["points"]:
            print(f"  b = {p['b']:5d} rows: intensity {p['ai']:7.1f} FLOP/B -> "
                  f"{p['gflops']:7.0f} GFLOP/s ({p['ms']:.2f} ms)")
        print("  b = 1 is a decode step (memory-bound: GFLOP/s grows ~linearly with b);\n"
              "  b in the hundreds/thousands is a prefill or a big batch (compute-bound).")
    print("\n  For a 7B model in fp16 on an A100 (2 TB/s, 312 TFLOP/s): every decode step\n"
          "  reads 14 GB of weights = 7 ms, so batch 1 tops out at ~140 tokens/s while using\n"
          "  ~0.6% of the FLOPs; the ridge is at ~156 FLOPs/byte, i.e. batch ~156.")


if __name__ == "__main__":
    main()
