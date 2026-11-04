"""
BPE in PyTorch, and the question that matters: does a GPT learn better from characters
or from BPE tokens? Compared fairly, in bits per byte.

1. The same BPE trainer written with tensor ops (the way you would count pairs on a
   GPU or in a data pipeline): the corpus is one flat id tensor with -1 between chunks,
   pair counts come from torch.unique + scatter_add, and a merge is a masked rewrite
   (with the "aaa" overlap rule done by run-length parity). It must learn EXACTLY the
   same 3,840 merges as bpe.train_bpe on the whole training corpus.

2. Train the mini-GPT of ../transformer/transformer_pytorch.py (imported, unchanged) on
   the same text tokenized three ways: characters, BPE-512 and BPE-4096. Same model
   shape, same batch size, same number of steps and the same context length IN TOKENS,
   so roughly the same compute. Held-out split: the last 10% of the corpus.

   Per-token loss can't be compared across tokenizers: a BPE token carries ~3 bytes,
   so predicting it is harder than predicting one character, but you need fewer of
   them. The fair unit is the cost of the whole text:
       bits per byte = Σ_tokens NLL (nats) / (ln 2 · number of UTF-8 bytes predicted)
   which is independent of how the text was cut up. Lower is better; a uniform guess
   over 256 byte values is 8 bits per byte.

Results are saved to tokenizer/bpe_pytorch.npz (make_figures.py draws them).

Run (~2 min on a laptop CPU):
    python tokenizer/bpe_pytorch.py
    python tokenizer/bpe_pytorch.py --steps 1500 --vocabs 1024
"""
import argparse
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from bpe import GPT4_PATTERN, get_tokenizer, load_corpus, split_text, train_bpe

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "transformer"))
from transformer_numpy import lr_at  # noqa: E402  (warmup + cosine, as in ../transformer/)
from transformer_pytorch import GPT, make_optimizer  # noqa: E402


# ----------------------------------------------------------------------------
# Part 1: BPE training with tensor ops
# ----------------------------------------------------------------------------
def train_bpe_torch(text, vocab_size, pattern=GPT4_PATTERN):
    chunk_freq = {}
    for c in split_text(text, pattern):
        chunk_freq[c] = chunk_freq.get(c, 0) + 1
    flat, weight = [], []
    for c, f in chunk_freq.items():  # unique chunks, -1 between them: no pair crosses a chunk
        b = list(c.encode("utf-8"))
        flat += b + [-1]
        weight += [f] * (len(b) + 1)
    ids = torch.tensor(flat, dtype=torch.long)
    w = torch.tensor(weight, dtype=torch.long)
    merges = []
    for k in range(vocab_size - 256):
        left, right = ids[:-1], ids[1:]
        valid = (left >= 0) & (right >= 0)
        if not valid.any():
            break
        codes = (left * vocab_size + right)[valid]  # one integer per pair: a·V + b
        uniq, inv = torch.unique(codes, return_inverse=True)  # sorted, so ties -> smallest (a, b)
        counts = torch.zeros(len(uniq), dtype=torch.long).scatter_add_(0, inv, w[:-1][valid])
        code = uniq[torch.argmax(counts)].item()  # argmax returns the FIRST maximum
        a, b = divmod(code, vocab_size)
        idx = 256 + k
        merges.append((a, b))
        # positions where the pair starts; for a == b ("aaa"), keep every other one in a run
        m = (left == a) & (right == b)
        pos = torch.arange(len(m))
        starts = m & ~torch.cat([torch.tensor([False]), m[:-1]])
        run_start = torch.cummax(torch.where(starts, pos, torch.zeros_like(pos)), 0).values
        m &= (pos - run_start) % 2 == 0
        keep = torch.ones(len(ids), dtype=torch.bool)
        keep[1:][m] = False  # drop the right half of each merged pair
        ids = torch.where(torch.cat([m, torch.tensor([False])]), torch.tensor(idx), ids)
        ids, w = ids[keep], w[keep]
    return merges


# ----------------------------------------------------------------------------
# Part 2: char vs BPE language models, in bits per byte
# ----------------------------------------------------------------------------
def encode_all(kind, train, val, chars=None, tok=None):
    """Returns train ids, val ids, bytes per id (for bits per byte), vocabulary size."""
    if kind == "chars":
        stoi = {c: i for i, c in enumerate(chars)}
        nbytes = torch.tensor([len(c.encode("utf-8")) for c in chars])
        enc = lambda s: torch.tensor([stoi[c] for c in s])  # noqa: E731
        return enc(train), enc(val), nbytes, len(chars)
    nbytes = torch.tensor([len(tok.vocab[i]) for i in range(tok.n_vocab)])
    enc = lambda s: torch.tensor(tok.encode_ordinary(s))  # noqa: E731
    return enc(train), enc(val), nbytes, tok.n_vocab


@torch.no_grad()
def bits_per_byte(model, data, nbytes, T):
    """Σ NLL over every predicted token / (ln 2 · bytes of those tokens), in T-token windows.
    Also returns the plain mean loss per token, to show why that number misleads."""
    model.eval()
    starts = range(0, len(data) - T - 1, T)
    x = torch.stack([data[i:i + T] for i in starts])
    y = torch.stack([data[i + 1:i + T + 1] for i in starts])
    nll = 0.0
    for i in range(0, len(x), 64):
        logits = model(x[i:i + 64])
        nll += F.cross_entropy(logits.flatten(0, 1), y[i:i + 64].flatten(), reduction="sum").item()
    model.train()
    n_bytes = nbytes[y].sum().item()
    return nll / (math.log(2) * n_bytes), nll / y.numel(), n_bytes


def run(label, train_ids, val_ids, nbytes, V, args, train_eval_ids):
    torch.manual_seed(0)
    model = GPT(V, args.n_layer, args.n_head, args.d_model, args.block_size, dropout=args.dropout)
    n_params = sum(p.numel() for p in model.parameters())
    n_emb = model.wte.weight.numel() + model.lm_head.weight.numel()
    bpt = nbytes[train_ids].sum().item() / len(train_ids)
    print(f"\n{label}: V = {V}, {len(train_ids):,} training tokens ({bpt:.2f} bytes/token), "
          f"{n_params:,} params ({n_emb:,} = {n_emb / n_params:.0%} in wte + lm_head). "
          f"Context {args.block_size} tokens ≈ {args.block_size * bpt:.0f} bytes")
    opt = make_optimizer(model, args.lr, args.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_at(s, 1.0, args.warmup, args.steps))
    g = torch.Generator().manual_seed(0)
    T, hist = args.block_size, []
    t0 = time.time()
    for step in range(args.steps + 1):
        if step % args.eval_every == 0 or step == args.steps:
            va, va_tok, _ = bits_per_byte(model, val_ids, nbytes, T)
            tr, _, _ = bits_per_byte(model, train_eval_ids, nbytes, T)
            seen = step * args.batch_size * T * bpt
            hist.append((step, seen, time.time() - t0, tr, va, va_tok))
            print(f"  step {step:5d} | train {tr:.3f} | val {va:.3f} bits/byte | val loss "
                  f"{va_tok:.3f} nats/token | {seen / 1e6:4.1f} MB seen | {time.time() - t0:5.1f} s")
        if step == args.steps:
            break
        ix = torch.randint(len(train_ids) - T - 1, (args.batch_size,), generator=g)
        x = torch.stack([train_ids[i:i + T] for i in ix])
        y = torch.stack([train_ids[i + 1:i + T + 1] for i in ix])
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
    return model, np.array(hist), n_params, n_emb, bpt


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vocabs", type=int, nargs="+", default=[512, 4096],
                    help="BPE vocabulary sizes to compare with characters")
    ap.add_argument("--n-layer", type=int, default=3)
    ap.add_argument("--n-head", type=int, default=4)
    ap.add_argument("--d-model", type=int, default=96)
    ap.add_argument("--block-size", type=int, default=128, help="context length in TOKENS")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--warmup", type=int, default=60)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--skip-part1", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(min(8, torch.get_num_threads()))
    train, val = load_corpus("train"), load_corpus("val")

    if not args.skip_part1:
        print("Part 1: BPE training with tensor ops == bpe.train_bpe (pure Python, incremental)")
        t0 = time.time()
        ref = train_bpe(train, 4096)
        t1 = time.time()
        mine = train_bpe_torch(train, 4096)
        t2 = time.time()
        assert mine == ref, "tensor trainer disagrees with bpe.train_bpe"
        print(f"  all {len(ref)} merges identical on the {len(train.encode()):,}-byte training text "
              f"(Python incremental {t1 - t0:.2f} s, torch full recount per merge {t2 - t1:.1f} s) ✔")
        print("  The tensor version recounts every pair at every merge (simple, parallel, O(N) per")
        print("  merge); the Python one only touches the chunks that changed. On a GPU with huge")
        print("  corpora you would combine both ideas; production trainers (HF tokenizers,")
        print("  SentencePiece) are incremental, in Rust/C++.\n")

    print("Part 2: the same mini-GPT on characters vs BPE tokens, compared in bits per byte")
    print(f"  model: GPT from ../transformer/transformer_pytorch.py, {args.n_layer} layers, "
          f"d_model {args.d_model}, {args.n_head} heads, context {args.block_size} tokens, "
          f"batch {args.batch_size}, {args.steps} steps, dropout {args.dropout}")
    print(f"  train {len(train.encode()):,} bytes, held-out val {len(val.encode()):,} bytes "
          "(the last 10% of the corpus)")
    chars = sorted(set(train + val))  # a real char model would need <unk> for unseen chars
    eval_slice = train[-len(val):]    # same size as val, to measure the train/val gap
    results = {}
    setups = [("chars", None)] + [(f"BPE-{V}", V) for V in args.vocabs]
    tok_full = get_tokenizer(max(args.vocabs + [4096]))
    for label, V in setups:
        tok = tok_full.truncated(V) if V else None
        tr_ids, va_ids, nbytes, nv = encode_all("chars" if V is None else "bpe", train, val,
                                                chars, tok)
        ev_ids = encode_all("chars" if V is None else "bpe", eval_slice, "", chars, tok)[0]
        model, hist, n_params, n_emb, bpt = run(label, tr_ids, va_ids, nbytes, nv, args, ev_ids)
        results[label] = dict(hist=hist, n_params=n_params, n_emb=n_emb, bpt=bpt, V=nv,
                              n_train=len(tr_ids), model=model, tok=tok)

    print(f"\n{'':10s}{'V':>6s}{'params':>11s}{'bytes/tok':>10s}{'ctx bytes':>10s}"
          f"{'val nats/tok':>13s}{'train bpb':>10s}{'val bpb':>9s}{'best val':>9s}{'time':>7s}")
    for label, r in results.items():
        h = r["hist"]
        print(f"{label:10s}{r['V']:6d}{r['n_params']:11,d}{r['bpt']:10.2f}"
              f"{args.block_size * r['bpt']:10.0f}{h[-1, 5]:13.3f}{h[-1, 3]:10.3f}{h[-1, 4]:9.3f}"
              f"{h[:, 4].min():9.3f}{h[-1, 2]:6.0f}s")
    print(f"{'uniform':10s}{'':6s}{'':11s}{'':10s}{'':10s}{'':13s}{'8.000':>10s}{'8.000':>9s}")
    print("\n  nats/tok: the per-token loss each run reports. It ranks the models in the WRONG")
    print("  order of difficulty: a BPE token carries several bytes, so its loss is naturally")
    print("  larger. bits per byte (bpb) divides the total cost of the text by its size in bytes,")
    print("  which is the same for every tokenizer, so it is the number to compare.")

    for label, r in results.items():  # a short sample from each model, for the eyeball test
        model = r["model"]
        model.eval()
        torch.manual_seed(0)
        prompt = "The keeper"
        if r["tok"] is None:
            stoi = {c: i for i, c in enumerate(chars)}
            idx = torch.tensor([[stoi[c] for c in prompt]])
            out = model.generate(idx, 160, temperature=0.8, top_k=20)[0].tolist()
            text = "".join(chars[i] for i in out)
        else:
            idx = torch.tensor([r["tok"].encode_ordinary(prompt)])
            n = int(160 / r["bpt"])  # about the same number of characters
            out = model.generate(idx, n, temperature=0.8, top_k=20)[0].tolist()
            text = r["tok"].decode(out)
        print(f"\n===== {label} sample =====\n{text[:220]}")

    np.savez(os.path.join(HERE, "bpe_pytorch.npz"),
             labels=np.array(list(results)),
             **{f"hist_{k}": r["hist"] for k, r in results.items()},
             **{f"bpt_{k}": r["bpt"] for k, r in results.items()},
             **{f"V_{k}": r["V"] for k, r in results.items()},
             **{f"n_params_{k}": r["n_params"] for k, r in results.items()},
             block_size=args.block_size)
    print(f"\nsaved {os.path.relpath(os.path.join(HERE, 'bpe_pytorch.npz'))}")


if __name__ == "__main__":
    main()
