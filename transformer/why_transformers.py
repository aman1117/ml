"""
Why Transformers? Two experiments that make the key insight tangible.

1. MEMORY: an RNN squeezes the whole past into ONE fixed-size vector, so the
   more it has to remember, the more it forgets. Attention keeps every past
   token and LOOKS UP what it needs (query · key = "is this the one?").

   Task: associative recall, a tiny "phone book". There are 16 names (keys) and
   8 numbers (values). Each sequence first lists N entries (N = 1..16, random),
   one token per entry ("name k has number v"), then asks for the number of
   random listed names, one name per token, until the sequence is 48 tokens long:

       [k3=5] [k11=0] [k7=2] ... [k9=6] | k7? k3? k7? k9? ...
        └────── N entries ──────┘         targets: 2   5   2   6

   Chance = 1/8 = 12.5%. We train a mini-GPT, a GRU and a vanilla RNN with the
   same budget (same batches, steps, optimizer; ~120k parameters each) and
   measure accuracy vs the number of entries N and vs the distance between the
   query and the entry it asks about.

2. SPEED: an RNN must process the T tokens one after another (T sequential
   steps). A Transformer processes all T tokens of a layer at once with big
   matrix multiplies (n_layer sequential steps), but attention builds a T × T
   score matrix, so its cost grows like T². We time forward passes over T.

Results are saved to transformer/why_transformers.npz (make_figures.py draws them).

Run:
    python transformer/why_transformers.py              (~45 s)
    python transformer/why_transformers.py --steps 3000 (more training for the RNNs)
"""
import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer_pytorch import GPT

HERE = os.path.dirname(os.path.abspath(__file__))
N_KEYS, N_VALS, T, N_MAX = 16, 8, 48, 16
VOCAB = N_KEYS * N_VALS + N_KEYS  # entry tokens (key, value) + query tokens (key)


def make_batch(B, g, N=None):
    """Returns x (B,T) tokens, y (B,T) targets (-100 = no target), and for every
    position: the distance to the asked entry, and the number of entries N."""
    Ns = torch.randint(1, N_MAX + 1, (B,), generator=g) if N is None else torch.full((B,), N)
    keys = torch.argsort(torch.rand(B, N_KEYS, generator=g), dim=1)  # distinct names
    vals = torch.randint(N_VALS, (B, N_KEYS), generator=g)
    pos = torch.arange(T)[None].expand(B, T)
    is_entry = pos < Ns[:, None]
    slot = pos.clamp(max=N_KEYS - 1)
    entry_tok = torch.gather(keys, 1, slot) * N_VALS + torch.gather(vals, 1, slot)
    asked = (torch.rand(B, T, generator=g) * Ns[:, None]).long()  # which entry each query asks
    query_tok = N_KEYS * N_VALS + torch.gather(keys, 1, asked)
    x = torch.where(is_entry, entry_tok, query_tok)
    y = torch.where(is_entry, torch.full_like(x, -100), torch.gather(vals, 1, asked))
    return x, y, pos - asked, Ns[:, None].expand(B, T)


class Recurrent(nn.Module):
    """Embedding -> nn.GRU or nn.RNN (tanh) -> Linear, like ../rnn/rnn_pytorch.py."""

    def __init__(self, cell, hidden):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, hidden)
        self.rnn = {"GRU": nn.GRU, "RNN": nn.RNN}[cell](hidden, hidden, batch_first=True)
        self.head = nn.Linear(hidden, N_VALS)

    def forward(self, x):
        return self.head(self.rnn(self.embed(x))[0])


def logits_of(model, x):
    out = model(x)
    return out[..., :N_VALS] if isinstance(model, GPT) else out  # answers are values 0..7


def train(model, steps, batch_size=32, lr=3e-3):
    g = torch.Generator().manual_seed(0)  # identical batches for every model
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    for _ in range(steps):
        x, y, _, _ = make_batch(batch_size, g)
        out = logits_of(model, x)
        mask = y >= 0
        loss = F.cross_entropy(out[mask], y[mask])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, n_eval=2000):
    model.eval()
    x, y, dist, Ns = make_batch(n_eval, torch.Generator().manual_seed(123))
    correct = (logits_of(model, x).argmax(-1) == y)
    mask = y >= 0
    by_n = np.array([correct[mask & (Ns == n)].float().mean().item() for n in range(1, N_MAX + 1)])
    bins = np.arange(1, T + 1, 8)  # distance bins 1-8, 9-16, ...
    by_d = np.array([correct[mask & (dist >= lo) & (dist < lo + 8)].float().mean().item()
                     for lo in bins])
    return correct[mask].float().mean().item(), by_n, by_d, bins


@torch.no_grad()
def lookup_attention(model):
    """For one example, the attention of every head from the queries to the entries.
    Returns the per-head fraction of attention a query puts on the entry it asks about."""
    x, y, dist, _ = make_batch(1, torch.Generator().manual_seed(7), N=8)
    maps = []

    def hook(mod, inp, out):
        q, k, v = mod.c_attn(inp[0]).split(inp[0].shape[-1], dim=2)
        B, Tq, C = q.shape
        q, k = (z.view(B, Tq, mod.n_head, -1).transpose(1, 2) for z in (q, k))
        s = q @ k.transpose(-1, -2) / np.sqrt(k.shape[-1])
        s = s.masked_fill(torch.triu(torch.ones(Tq, Tq, dtype=torch.bool), 1), float("-inf"))
        maps.append(F.softmax(s, -1)[0].numpy())

    handles = [blk.attn.register_forward_hook(hook) for blk in model.blocks]
    model(x)
    for h in handles:
        h.remove()
    att = np.stack(maps)  # (n_layer, n_head, T, T)
    qpos = np.arange(8, T)
    target = (qpos - dist[0, 8:].numpy())  # position of the asked entry
    on_target = att[:, :, qpos, target].mean(-1)  # (n_layer, n_head)
    return att, x[0].numpy(), y[0].numpy(), on_target


def token_name(t):
    if t < N_KEYS * N_VALS:
        return f"k{t // N_VALS}={t % N_VALS}"
    return f"k{t - N_KEYS * N_VALS}?"


# ----------------------------------------------------------------------------
# Experiment 2: forward-pass time vs sequence length
# ----------------------------------------------------------------------------
@torch.no_grad()
def time_forward(model, x, repeats=3):
    model(x)  # warm-up
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        model(x)
        best = min(best, time.perf_counter() - t0)
    return best


def same_work_two_ways(L, d=128, batch=8):
    """The SAME multiply (every token's d-vector @ a (d, 4d) matrix) done one token at
    a time, as an RNN is forced to, vs all T tokens in one matmul, as a Transformer can."""
    W = torch.randn(d, 4 * d)
    x = torch.randn(batch, L, d)
    seq = time_forward(lambda z: [z[:, t] @ W for t in range(L)], x)
    par = time_forward(lambda z: z @ W, x)
    return seq, par


def speed_experiment(lengths, d=128, batch=8):
    gru = nn.GRU(d, d, batch_first=True).eval()
    gpt = GPT(64, n_layer=2, n_head=4, d_model=d, block_size=max(lengths)).eval()
    rows = []
    for L in lengths:
        seq, par = same_work_two_ways(L, d, batch)
        t_gru = time_forward(lambda z: gru(z), torch.randn(batch, L, d))
        t_gpt = time_forward(gpt, torch.randint(64, (batch, L)))
        # Multiply-adds per layer: projections + MLP = 12·d² per token; attention
        # scores q·k and the weighted sum of v = 2·L·d per token (for the full L×L matrix).
        flops_lin = 12 * d * d * L
        flops_att = 2 * L * L * d
        rows.append((L, seq, par, t_gru, t_gpt, flops_att / (flops_lin + flops_att)))
    return np.array(rows)


def run(steps=1500, save=os.path.join(HERE, "why_transformers.npz")):
    """Run both experiments, print the results, save them to `save`."""
    torch.set_num_threads(min(4, torch.get_num_threads()))

    print("EXPERIMENT 1: associative recall (a tiny phone book)")
    x, y, _, _ = make_batch(1, torch.Generator().manual_seed(3), N=5)
    print("  example: " + " ".join(token_name(t) for t in x[0, :12].tolist()) + " ...")
    print("  targets: " + " ".join("·" if t < 0 else str(t) for t in y[0, :12].tolist()) + " ...")
    print(f"  {N_KEYS} names, {N_VALS} numbers, sequences of {T} tokens, chance = "
          f"{1 / N_VALS:.1%}. Each model: {steps} steps of 32 sequences.\n")

    results = {}
    models = {}
    for name in ["Transformer", "GRU", "RNN"]:
        torch.manual_seed(0)
        model = (GPT(VOCAB, n_layer=2, n_head=4, d_model=64, block_size=T) if name == "Transformer"
                 else Recurrent(name, hidden=128 if name == "GRU" else 208))  # ~equal params
        n_params = sum(p.numel() for p in model.parameters())
        t0 = time.time()
        loss = train(model, steps)
        acc, by_n, by_d, bins = evaluate(model)
        results[name] = (acc, by_n, by_d, n_params, time.time() - t0)
        models[name] = model
        print(f"  {name:11s} {n_params:7,} params | final train loss {loss:.3f} | "
              f"test accuracy {acc:6.1%} | {time.time() - t0:4.1f} s")

    print("\n  Accuracy vs number of entries N it had to remember:")
    print("  " + "N".rjust(11) + "".join(f"{n:>6d}" for n in range(1, N_MAX + 1, 3)))
    for name, (_, by_n, _, _, _) in results.items():
        print(f"  {name:>11s}" + "".join(f"{by_n[n - 1]:6.0%}" for n in range(1, N_MAX + 1, 3)))
    print("\n  Accuracy vs distance (query position - position of the entry it asks about):")
    print("  " + "distance".rjust(11) + "".join(f"{f'{lo}-{lo + 7}':>7s}" for lo in bins))
    for name, (_, _, by_d, _, _) in results.items():
        print(f"  {name:>11s}" + "".join(f"{a:7.0%}" for a in by_d))

    att, ex_x, ex_y, on_target = lookup_attention(models["Transformer"])
    li, hi = np.unravel_index(on_target.argmax(), on_target.shape)
    print(f"\n  Inside the Transformer: layer {li} head {hi} puts {on_target[li, hi]:.0%} of a "
          "query's attention on exactly the entry it asks about")
    print("  (averaged over the queries of one example): attention IS a lookup by content.")
    print("  The recurrent nets have no such option: everything must fit in one hidden vector,")
    print("  so their accuracy falls as N grows, wherever the entry sits in the sequence.")

    print("\nEXPERIMENT 2: time vs sequence length T (batch 8, d = 128, this CPU)")
    lengths = [64, 128, 256, 512, 1024, 2048]
    speed = speed_experiment(lengths)
    print("  (a) the SAME matmul work: token by token (RNN-style) vs all tokens at once")
    print("  (b) real forward passes: nn.GRU (T sequential steps) vs a 2-layer GPT (2 steps)\n")
    print(f"  {'T':>6s} | {'(a) one by one':>14s} {'all at once':>12s} {'speed-up':>9s} | "
          f"{'(b) nn.GRU':>10s} {'GPT':>8s} {'GPT µs/token':>13s} {'attention FLOPs':>16s}")
    for L, seq, par, t_gru, t_gpt, share in speed:
        print(f"  {int(L):6d} | {seq * 1e3:12.2f}ms {par * 1e3:10.2f}ms {seq / par:8.0f}x | "
              f"{t_gru * 1e3:8.1f}ms {t_gpt * 1e3:6.1f}ms {t_gpt / (8 * L) * 1e6:13.2f} "
              f"{share:15.0%}")
    print("  (a) Identical arithmetic, but one big matmul keeps every core busy: that is the")
    print("      parallelism a Transformer gets during training and prefill. An RNN can't use")
    print("      it over time, because step t needs h_{t-1}.")
    print("  (b) Honest caveat: on a CPU this small GRU is still faster than the GPT, which does")
    print("      ~12x more arithmetic per token (MLP + attention). The GPT's cost per token grows")
    print("      with T because attention is T × T: at T = 6·d = 768 attention equals the rest.")

    np.savez(save, steps=steps, n_max=N_MAX, bins=bins, lengths=speed[:, 0],
             speed=speed, att=att, ex_x=ex_x, ex_y=ex_y, on_target=on_target,
             **{f"{k}.by_n": v[1] for k, v in results.items()},
             **{f"{k}.by_d": v[2] for k, v in results.items()},
             **{f"{k}.acc": v[0] for k, v in results.items()},
             **{f"{k}.params": v[3] for k, v in results.items()})
    print(f"\nSaved results to {os.path.relpath(save)} (used by make_figures.py).")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--save", default=os.path.join(HERE, "why_transformers.npz"))
    args = ap.parse_args()
    run(args.steps, args.save)


if __name__ == "__main__":
    main()
