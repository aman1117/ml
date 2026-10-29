"""
Give the trained mini-GPT some starting text (a prompt) and let it continue,
using a KV CACHE: the trick every real LLM server uses to generate quickly.

This is a character-level language model: it does NOT answer questions. It
only predicts "what character comes next" in the style of its training text,
so the best prompts look like the start of a sentence from it, e.g.
"The keeper" or "Good morning".

How generation works:
  1. PREFILL: run the whole prompt through the model in ONE parallel forward
     pass, and keep every layer's keys K and values V (the "KV cache").
  2. DECODE, one token at a time: for the newest token only, compute its
     q, k, v; append k, v to the cache; attend with q over ALL cached keys;
     sample the next token from the logits; repeat.
  Without the cache, step t recomputes keys/values for all t previous tokens
  (O(t²) work per new token across the layers' attention, O(t) for the rest);
  with it, each new token costs O(t) for attention and O(1) for everything else.

Our model learned positions 0..block_size-1 only (absolute position
embeddings). When the window is full, the cached keys would all need new
positions, so we keep the last block_size/2 tokens, prefill them again, and
continue. (Models with RoPE / relative positions, or a long context, avoid this.)

Run (after training with transformer_numpy.py):
    python transformer/generate.py                         # interactive: type prompts
    python transformer/generate.py --prompt "The keeper"   # one-shot
    python transformer/generate.py --prompt "The sea" --temperature 0.5 --top-k 5
    python transformer/generate.py --benchmark             # time cache vs no cache
"""
import argparse
import os
import time

import numpy as np

from transformer_numpy import GPT, load_model, sample_next, softmax

HERE = os.path.dirname(os.path.abspath(__file__))


class KVCache:
    """Per layer: K and V of every token seen so far, each (nh, t, hs)."""

    def __init__(self, n_layer):
        self.k = [None] * n_layer
        self.v = [None] * n_layer

    def __len__(self):
        return 0 if self.k[0] is None else self.k[0].shape[1]

    def nbytes(self):
        return sum(a.nbytes for a in self.k + self.v if a is not None)


def prefill(model, ids):
    """One parallel forward over the prompt. Returns (last logits, filled cache)."""
    logits = model.forward(np.array(ids)[None])  # (1, T, V)
    cache = KVCache(len(model.blocks))
    for l, blk in enumerate(model.blocks):
        cache.k[l] = blk.attn.k[0]  # the attention layer kept K and V: (nh, T, hs)
        cache.v[l] = blk.attn.v[0]
    return logits[0, -1], cache


def decode_step(model, token, cache):
    """Feed ONE new token (at position len(cache)), update the cache, return its logits."""
    pos = len(cache)
    x = model.wte.params["weight"][token] + model.wpe.params["weight"][pos]
    x = x[None, None]  # (1, 1, C): a batch with one sequence of one token
    for l, blk in enumerate(model.blocks):
        a = blk.attn
        q, k, v = np.split(a.c_attn.forward(blk.ln_1.forward(x)), 3, axis=-1)
        q, k, v = a.split_heads(q)[0], a.split_heads(k)[0], a.split_heads(v)[0]  # (nh, 1, hs)
        cache.k[l] = np.concatenate([cache.k[l], k], axis=1)  # (nh, t+1, hs)
        cache.v[l] = np.concatenate([cache.v[l], v], axis=1)
        # No mask needed: the cache only holds the past and the token itself.
        att = softmax(q @ cache.k[l].transpose(0, 2, 1) * a.scale)  # (nh, 1, t+1)
        y = (att @ cache.v[l])[None]  # (1, nh, 1, hs)
        x = x + a.c_proj.forward(a.merge_heads(y))
        x = x + blk.mlp.forward(blk.ln_2.forward(x))
    x = model.ln_f.forward(x)
    W = model.wte.params["weight"].T if model.lm_head is None else model.lm_head.params["W"]
    return (x @ W)[0, 0]


def generate_ids(model, ids, n, temperature=0.8, top_k=None, rng=None, use_cache=True):
    rng = rng or np.random.default_rng()
    T = model.config["block_size"]
    ids = list(ids)
    if not use_cache:  # the simple way: recompute everything for every new token
        return model.generate(ids, n, temperature, top_k, rng)[len(ids):]
    logits, cache = prefill(model, ids[-T:])
    out = []
    for _ in range(n):
        nxt = sample_next(logits, temperature, top_k, rng)
        out.append(nxt)
        ids.append(nxt)
        if len(cache) == T:  # window full: re-prefill the most recent half
            logits, cache = prefill(model, ids[-(T // 2):])
        else:
            logits = decode_step(model, nxt, cache)
    return out


def check_cache(model, ids):
    """Logits from prefill(1 token) + decode steps must equal one full forward."""
    T = min(len(ids), model.config["block_size"])
    ids = ids[:T]
    full = model.forward(np.array(ids)[None])[0]  # (T, V): every position at once
    logits, cache = prefill(model, ids[:1])
    cached = [logits]
    for tok in ids[1:]:
        cached.append(decode_step(model, tok, cache))
    diff = float(np.abs(np.stack(cached) - full).max())
    return diff, T, cache


def encode(chars, prompt):
    stoi = {c: i for i, c in enumerate(chars)}
    unknown = sorted({c for c in prompt if c not in stoi})
    if unknown:
        print(f"(skipping characters never seen in training: {unknown})")
    ids = [stoi[c] for c in prompt if c in stoi]
    if not ids:
        raise ValueError("the prompt has no characters the model knows")
    return ids


def time_generation(model, n, repeats=1):
    """Seconds per generated token, without and with the KV cache."""
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
    no, yes = time_generation(model, T - 1, repeats=5)
    print(f"Our trained model (block_size {T}), generating {T - 1} tokens:")
    print(f"  without KV cache: {no * 1e3:.2f} ms/token   with: {yes * 1e3:.2f} ms/token   "
          f"({no / yes:.1f}x faster)")
    # The speed-up depends on the context length, not on the trained weights,
    # so use random weights for a longer context.
    big = GPT(model.config["vocab_size"], n_layer=2, n_head=4, d_model=128, block_size=256)
    no, yes = time_generation(big, 255)
    print("Random-weight model with a longer context (block_size 256, d_model 128), "
          "generating 255 tokens:")
    print(f"  without KV cache: {no * 1e3:.2f} ms/token   with: {yes * 1e3:.2f} ms/token   "
          f"({no / yes:.1f}x faster)")
    print("Without the cache, token t re-processes all t earlier tokens, so the cost per token\n"
          "grows with t. With it, only the newest token goes through the layers.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=os.path.join(HERE, "model.npz"))
    ap.add_argument("--prompt", help="starting text; omit for interactive mode")
    ap.add_argument("--length", type=int, default=300, help="characters to generate")
    ap.add_argument("--temperature", type=float, default=0.8,
                    help="<1 = safer/repetitive, >1 = more random")
    ap.add_argument("--top-k", type=int, default=None,
                    help="only sample from the k most likely characters")
    ap.add_argument("--seed", type=int, default=None, help="for reproducible output")
    ap.add_argument("--no-cache", action="store_true", help="recompute everything each step")
    ap.add_argument("--benchmark", action="store_true", help="time cache vs no cache and exit")
    args = ap.parse_args()

    if not os.path.exists(args.model):
        raise SystemExit(f"No model at {args.model}. Train first: "
                         "python transformer/transformer_numpy.py")
    model, chars, _ = load_model(args.model)
    cfg = model.config

    # Always prove the cache is exact before using it (in float64, so round-off
    # can't hide a bug).
    model64 = load_model(args.model, dtype=np.float64)[0]
    diff, T, cache = check_cache(model64, encode(chars, "The keeper opened the door and sa"))
    print(f"KV cache check: {T} tokens decoded one at a time with the cache vs one full "
          f"forward pass: max |logit diff| = {diff:.1e} ✔")
    assert diff < 1e-9
    print(f"KV cache size at {T} tokens: 2 (K,V) × {cfg['n_layer']} layers × {T} tokens × "
          f"{cfg['d_model']} dims × 4 bytes = {cache.nbytes() // 2:,} bytes (float32)\n")

    if args.benchmark:
        benchmark(model)
        return

    rng = np.random.default_rng(args.seed)

    def run(prompt):
        out = generate_ids(model, encode(chars, prompt), args.length, args.temperature,
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
