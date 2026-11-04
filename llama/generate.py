"""
Give the trained mini-LLaMA a prompt and let it continue, using a KV CACHE with
grouped-query attention and a SLIDING (rolling-buffer) window.

What is new compared with ../transformer/generate.py:
  1. The cache stores only n_kv_head heads per layer: K and V are arrays of shape
     (n_layer, n_kv_head, window, head_size). With GQA (2 KV heads for 4 query heads)
     it is half the size of the GPT's cache; with MQA (1 KV head) a quarter.
  2. The window can SLIDE without re-prefilling. Keys are stored already rotated by
     RoPE at their absolute position p, and a score q_m·k_p depends only on m - p.
     So when the window is full, the newest token simply overwrites the oldest slot
     (slot = p mod window) and every cached key stays valid forever. The GPT could not
     do this: its learned position table only had positions 0..31, so it had to keep
     the last 16 tokens and re-run them through the model with new positions.
     This is Mistral's "rolling buffer cache"; what it computes is exactly
     sliding-window attention (every token, at every layer, sees the last `window`
     tokens), and the script proves that against a full forward pass with a band mask.

It also prints the KV-cache memory for our model (MHA vs GQA vs MQA) and for real
models, from their published configs, and measures four context strategies on the
whole text (sliding cache, full recompute, GPT-style re-prefill, unlimited context).

Run (trains llama/model.npz first if needed, ~6 s):
    python llama/generate.py                          # interactive: type prompts
    python llama/generate.py --prompt "The keeper"    # one-shot
    python llama/generate.py --prompt "The sea" --temperature 0.5 --top-k 5
    python llama/generate.py --benchmark              # time cache vs no cache
"""
import argparse
import time
import warnings

import numpy as np

from llama_numpy import (DEFAULT_TEXT, LLaMA, get_trained_model, load_text, rope_angles,
                         rope_rotate, sample_next, softmax)

warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)

GiB, KiB = 2 ** 30, 2 ** 10

# Published configs (Meta's params.json / Hugging Face config.json of each model).
REAL_MODELS = [
    # name,              vocab,   d,     L,  heads, kv heads, ffn hidden, sliding window
    ("LLaMA-2-7B",       32000, 4096,  32,  32, 32, 11008, None),
    ("LLaMA-2-70B",      32000, 8192,  80,  64,  8, 28672, None),
    ("(70B if it were MHA)", 32000, 8192, 80, 64, 64, 28672, None),
    ("Mistral-7B v0.1",  32000, 4096,  32,  32,  8, 14336, 4096),
    ("Llama-3-8B",      128256, 4096,  32,  32,  8, 14336, None),
    ("Qwen2.5-7B",      152064, 3584,  28,  28,  4, 18944, None),  # + 129k q/k/v biases
]


# ----------------------------------------------------------------------------
# The cache
# ----------------------------------------------------------------------------
class KVCache:
    """Rolling buffer with the K and V of the last `window` tokens of every layer.

    k, v: (n_layer, n_kv_head, window, head_size). The token at absolute position p
    lives in slot p % window. Slots are NOT in time order once the buffer has wrapped,
    and that's fine: the position is baked into each key by RoPE, and attention is a
    weighted SUM over the cached keys, which doesn't care about their order."""

    def __init__(self, cfg, window, dtype=np.float64):
        hs = cfg["d_model"] // cfg["n_head"]
        shape = (cfg["n_layer"], cfg["n_kv_head"], window, hs)
        self.k, self.v = np.zeros(shape, dtype), np.zeros(shape, dtype)
        self.window = window
        self.pos = 0  # tokens seen so far = absolute position of the next token

    def __len__(self):
        return min(self.pos, self.window)

    def nbytes(self):
        return self.k.nbytes + self.v.nbytes


def prefill(model, ids, window=None):
    """One parallel forward over the prompt (sliding-window mask if it is longer than the
    window). Returns (last logits, cache holding the last `window` tokens' K, V)."""
    window = window or model.config["block_size"]
    T = len(ids)
    logits = model.forward(np.array(ids)[None], window=window)  # (1, T, V)
    cache = KVCache(model.config, window, logits.dtype)
    for p in range(max(0, T - window), T):
        for l, blk in enumerate(model.blocks):
            cache.k[l, :, p % window] = blk.attention.k[0, :, p]  # already rotated at p
            cache.v[l, :, p % window] = blk.attention.v[0, :, p]
    cache.pos = T
    return logits[0, -1], cache


def decode_step(model, token, cache):
    """Feed ONE new token at absolute position cache.pos; returns its logits.
    Nothing is ever re-computed, however long the generation runs."""
    p, W = cache.pos, cache.window
    a0 = model.blocks[0].attention
    nh, nkv, g, hs = a0.n_head, a0.n_kv_head, a0.groups, a0.hs
    cos, sin = rope_angles([p], hs, model.config["rope_base"], cache.k.dtype)  # angle p·θ_i
    x = model.tok_embeddings.params["weight"][token][None, None]  # (1, 1, C), no position!
    n = min(p + 1, W)  # valid slots after writing this token
    for l, blk in enumerate(model.blocks):
        a = blk.attention
        h = blk.attention_norm.forward(x)
        q = rope_rotate(a.split_heads(a.wq.forward(h), nh)[0], cos, sin)  # (nh, 1, hs)
        k = rope_rotate(a.split_heads(a.wk.forward(h), nkv)[0], cos, sin)  # (nkv, 1, hs)
        v = a.split_heads(a.wv.forward(h), nkv)[0]
        cache.k[l, :, p % W] = k[:, 0]  # overwrite the oldest token once the buffer is full
        cache.v[l, :, p % W] = v[:, 0]
        K, Vc = cache.k[l, :, :n], cache.v[l, :, :n]  # (nkv, n, hs)
        qg = q.reshape(nkv, g, hs)  # query heads grouped by the KV head they share
        att = softmax(qg @ K.transpose(0, 2, 1) * a.scale)  # (nkv, g, n): no mask needed
        y = (att @ Vc).reshape(1, 1, nh * hs)
        x = x + a.wo.forward(y)
        x = x + blk.feed_forward.forward(blk.ffn_norm.forward(x))
    cache.pos += 1
    return (model.norm.forward(x) @ model.head_weight())[0, 0]


def generate_ids(model, ids, n, temperature=0.8, top_k=None, rng=None, use_cache=True):
    rng = rng or np.random.default_rng()
    T = model.config["block_size"]
    if not use_cache:  # recompute the last T tokens for every new token
        return model.generate(ids, n, temperature, top_k, rng)[len(ids):]
    logits, cache = prefill(model, list(ids), T)
    out = []
    for _ in range(n):
        nxt = sample_next(logits, temperature, top_k, rng)
        out.append(nxt)
        logits = decode_step(model, nxt, cache)  # never re-prefills
    return out


# ----------------------------------------------------------------------------
# Proofs
# ----------------------------------------------------------------------------
def decode_all(model, ids, window):
    """Logits for every position, feeding the tokens one by one through the cache."""
    logits, cache = prefill(model, ids[:1], window)
    out = [logits]
    for tok in ids[1:]:
        out.append(decode_step(model, tok, cache))
    return np.stack(out), cache


def check_in_window(model, ids):
    T = model.config["block_size"]
    ids = ids[:T]
    cached, _ = decode_all(model, ids, T)
    full = model.forward(np.array(ids)[None])[0]
    return float(np.abs(cached - full).max()), len(ids)


def check_sliding(model, ids, window):
    """Beyond the window: cache decode == one forward with a sliding-window (band) mask."""
    cached, cache = decode_all(model, ids, window)
    full = model.forward(np.array(ids)[None], window=window)[0]
    return float(np.abs(cached - full).max()), cache


def context_strategies(model, data, W):
    """Loss per char on every target at position >= W of the whole text, with 4 ways
    of handling a text longer than the training context W."""
    N = len(data)
    tgt = data[W:]
    res = {}

    def nll(logits, targets):
        z = logits - logits.max(-1, keepdims=True)
        lp = z - np.log(np.exp(z).sum(-1, keepdims=True))
        return -lp[np.arange(len(targets)), targets]

    # (a) the sliding cache: each token sees the last W tokens at EVERY layer.
    logits, _ = decode_all(model, list(data[:-1]), W)
    res["sliding KV cache (this script)"] = nll(logits[W - 1:], tgt).mean()
    # (b) full recompute of the last W tokens (positions 0..W-1) for every new token.
    starts = np.arange(0, N - W)
    x = np.stack([data[s:s + W] for s in starts])
    res["recompute last W tokens (--no-cache)"] = nll(model.forward(x)[:, -1], tgt).mean()
    # (c) GPT-style: when full, keep the last W/2 tokens and re-prefill them. So every
    #     target is predicted inside a causal window that starts at a multiple of W/2,
    #     with W/2+1 .. W tokens of context.
    h = W // 2
    starts = np.arange(0, N - W, h)
    chunks = np.stack([data[s:s + W + 1] for s in starts])
    lg = model.forward(chunks[:, :-1])[:, h:W]  # logits that predict chunk[h+1 .. W]
    t = starts[:, None] + np.arange(h + 1, W + 1)[None]  # absolute target positions
    keep = t >= W
    res["re-prefill last W/2 (GPT generate.py)"] = nll(lg[keep], chunks[:, h + 1:][keep]).mean()
    # (d) no window at all: offsets up to N - 1, never seen in training.
    logits, _ = decode_all(model, list(data[:-1]), N)
    res["unlimited context (offsets > W unseen)"] = nll(logits[W - 1:], tgt).mean()
    return res


# ----------------------------------------------------------------------------
# Memory tables and benchmark
# ----------------------------------------------------------------------------
def kv_bytes_per_token(n_layer, n_kv_head, head_size, bytes_per=2):
    return 2 * n_layer * n_kv_head * head_size * bytes_per  # 2 = K and V


def llama_params(V, d, L, nh, nkv, ffn):
    hs = d // nh
    return L * (2 * d * d + 2 * d * nkv * hs + 3 * d * ffn + 2 * d) + 2 * V * d + d


def flops_per_token(V, d, L, nh, nkv, ffn, T):
    """Forward FLOPs (1 multiply-add = 2 FLOPs) for one token that attends over T tokens:
    (weight matmuls, attention scores + weighted sum). Embedding lookup: free."""
    hs = d // nh
    weights = 2 * (L * (2 * d * d + 2 * d * nkv * hs + 3 * d * ffn) + d * V)
    attention = 2 * L * 2 * nh * hs * T  # q·k for T keys and A·v, every query head
    return weights, attention


def print_memory_tables(cfg):
    L, nh, d, T = cfg["n_layer"], cfg["n_head"], cfg["d_model"], cfg["block_size"]
    hs = d // nh
    print(f"KV cache of OUR model (float32, {T}-token window, {L} layers, head size {hs}):")
    for nkv in (nh, nh // 2, 1):
        kind = {nh: "MHA", 1: "MQA"}.get(nkv, "GQA")
        per = kv_bytes_per_token(L, nkv, hs, 4)
        mark = "   <- this model" if nkv == cfg["n_kv_head"] else ""
        print(f"  {kind} n_kv_head={nkv}: 2 × {L} × {nkv} × {hs} × 4 B = {per:5,d} B/token, "
              f"{per * T / KiB:4.0f} KiB for the full window{mark}")

    print("\nKV cache of real models, fp16 (2 bytes), batch 1:  "
          "bytes/token = 2 (K,V) × n_layer × n_kv_head × head_size × 2")
    print(f"  {'model':22s}{'params':>8s}{'L':>4s}{'heads':>6s}{'kv':>4s}{'hs':>5s}"
          f"{'KiB/token':>11s}{'4k ctx':>10s}{'32k ctx':>10s}")
    for name, V, dm, Lr, nhr, nkvr, ffn, sw in REAL_MODELS:
        hsr = dm // nhr
        per = kv_bytes_per_token(Lr, nkvr, hsr)
        n32 = 32768 if sw is None else min(32768, sw)
        note = "" if sw is None else f"  (rolling {sw}-token window)"
        params = llama_params(V, dm, Lr, nhr, nkvr, ffn) / 1e9
        print(f"  {name:22s}{params:7.2f}B{Lr:4d}{nhr:6d}{nkvr:4d}{hsr:5d}{per / KiB:11.0f}"
              f"{per * 4096 / GiB:8.2f} G{per * n32 / GiB:8.2f} G{note}")
    print("  Multiply by the batch size: 32 users × 4k tokens of LLaMA-2-7B = 64 GiB of cache,\n"
          "  more than the 13.5 GB of fp16 weights. That is why GQA (8 KV heads) is everywhere.\n")

    print("Forward FLOPs per generated token (2 per multiply-add), attending over T tokens:")
    print(f"  {'model':22s}{'T':>7s}{'weights':>12s}{'attention':>12s}{'total':>11s}"
          f"{'≈ 2N':>9s}")
    rows = [("ours", cfg["vocab_size"], d, L, nh, cfg["n_kv_head"], cfg["ffn_hidden"], T)]
    rows += [(r[0], *r[1:7], ctx) for r in REAL_MODELS[:2] for ctx in (4096, 32768)]
    for name, V, dm, Lr, nhr, nkvr, ffn, ctx in rows:
        w, a = flops_per_token(V, dm, Lr, nhr, nkvr, ffn, ctx)
        n = llama_params(V, dm, Lr, nhr, nkvr, ffn)
        unit, div = ("MFLOP", 1e6) if n < 1e9 else ("GFLOP", 1e9)
        print(f"  {name:22s}{ctx:7d}{w / div:9.2f} {unit[0]}{a / div:10.2f} {unit[0]}"
              f"{(w + a) / div:9.2f} {unit[0]}{2 * n / div:7.2f} {unit[0]}   ({unit})")
    print("  Attention's share grows with T: the weights cost the same for every token.\n")


def time_generation(model, n, repeats=1):
    """Seconds per generated token: no cache (recompute last block_size), cache."""
    out = []
    for use_cache in (False, True):
        rng = np.random.default_rng(0)
        t0 = time.perf_counter()
        for _ in range(repeats):
            generate_ids(model, [0], n, rng=rng, use_cache=use_cache)
        out.append((time.perf_counter() - t0) / (repeats * n))
    return out


def benchmark(model):
    T = model.config["block_size"]
    no, yes = time_generation(model, 4 * T, repeats=3)
    print(f"Our trained model (window {T}), generating {4 * T} tokens (4 windows' worth):")
    print(f"  without KV cache: {no * 1e3:.2f} ms/token   with: {yes * 1e3:.2f} ms/token   "
          f"({no / yes:.1f}x faster)")
    big = LLaMA(model.config["vocab_size"], n_layer=2, n_head=4, n_kv_head=2, d_model=128,
                block_size=256)
    no, yes = time_generation(big, 511)
    print("Random-weight model with a longer window (256, d_model 128), generating 511 tokens:")
    print(f"  without KV cache: {no * 1e3:.2f} ms/token   with: {yes * 1e3:.2f} ms/token   "
          f"({no / yes:.1f}x faster)")
    # flat cost per token, even after the window is full
    cache = prefill(big, [0], 256)[1]
    times = []
    for i in range(511):
        t0 = time.perf_counter()
        decode_step(big, i % 44, cache)
        times.append(time.perf_counter() - t0)
    print(f"  per-token decode time, tokens 100-255: {np.median(times[100:255]) * 1e3:.2f} ms, "
          f"tokens 300-511 (window full, sliding): {np.median(times[300:]) * 1e3:.2f} ms")


def encode(chars, prompt):
    stoi = {c: i for i, c in enumerate(chars)}
    unknown = sorted({c for c in prompt if c not in stoi})
    if unknown:
        print(f"(skipping characters never seen in training: {unknown})")
    ids = [stoi[c] for c in prompt if c in stoi]
    if not ids:
        raise ValueError("the prompt has no characters the model knows")
    return ids


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompt", help="starting text; omit for interactive mode")
    ap.add_argument("--length", type=int, default=300, help="characters to generate")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None, help="for reproducible output")
    ap.add_argument("--no-cache", action="store_true",
                    help="recompute the last block_size tokens for every new token")
    ap.add_argument("--benchmark", action="store_true", help="time cache vs no cache and exit")
    args = ap.parse_args()

    model, chars, _ = get_trained_model(np.float64)  # float64: the checks are exact
    cfg = model.config
    W = cfg["block_size"]
    _, _, data = load_text(DEFAULT_TEXT)

    diff, n = check_in_window(model, encode(chars, "The keeper opened the door and sa"))
    print(f"Check 1 (inside the window): {n} tokens decoded one at a time with the cache vs one "
          f"full forward pass: max |logit diff| = {diff:.1e} ✔")
    assert diff < 1e-9
    long_ids = list(data[:200])
    diff, cache = check_sliding(model, long_ids, W)
    print(f"Check 2 (beyond the window): {len(long_ids)} tokens through a {W}-slot rolling cache, "
          f"never re-prefilled, vs one forward\n  pass with a sliding-window mask "
          f"(token i sees i-{W - 1}..i): max |logit diff| = {diff:.1e} ✔")
    assert diff < 1e-9
    print(f"  cache K, V shape (n_layer, n_kv_head, window, head_size) = {cache.k.shape}; "
          f"the newest token (position {cache.pos - 1}) sits in slot {(cache.pos - 1) % W}\n")

    print_memory_tables(cfg)

    if args.benchmark:
        print(f"Loss per char on the {len(data) - W} targets at position >= {W} of the whole "
              f"text, by context strategy (float64):")
        for name, v in context_strategies(model, data, W).items():
            print(f"  {name:42s} {v:.3f}")
        print()
        benchmark(get_trained_model(np.float32, verbose=False)[0])
        return

    rng = np.random.default_rng(args.seed)
    model32 = get_trained_model(np.float32, verbose=False)[0]

    def run(prompt):
        out = generate_ids(model32, encode(chars, prompt), args.length, args.temperature,
                           args.top_k, rng, use_cache=not args.no_cache)
        print(prompt + "".join(chars[i] for i in out))

    if args.prompt is not None:
        run(args.prompt)
        return

    print("Type the start of some text and press Enter. Empty line or Ctrl+C to quit.\n")
    while True:
        try:
            prompt = input("prompt> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not prompt:
            break
        try:
            run(prompt)
        except ValueError as e:
            print(f"({e})")
        print()


if __name__ == "__main__":
    main()
