"""
The same mini-GPT in PyTorch.

1. Builds a PyTorch module with nn.Embedding, nn.LayerNorm, nn.Linear,
   nn.GELU(approximate="tanh") and F.scaled_dot_product_attention(is_causal=True),
   copies the NumPy model's weights into it, and checks that
       - the logits are identical, and
       - autograd's gradients are identical to our hand-written backward()
   for every single parameter (float64, separate and tied embeddings).
2. Trains a mini-GPT the idiomatic PyTorch way on the same text with a
   train/validation split: AdamW (weight decay on matrices only), linear
   warmup + cosine learning-rate decay, clip_grad_norm_, dropout. Prints train /
   val loss per character and val perplexity, then samples.
3. Trains a char-RNN (embedding -> nn.RNN -> linear, like ../rnn/rnn_pytorch.py)
   on exactly the same batches for the same number of steps, for comparison.

Run:
    python transformer/transformer_pytorch.py
    python transformer/transformer_pytorch.py --steps 3000 --dropout 0.2 --no-rnn
"""
import argparse
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer_numpy import DEFAULT_TEXT, GPT as NumpyGPT, load_text, lr_at


# ----------------------------------------------------------------------------
# The model, written the way you would in a real codebase (cf. nanoGPT)
# ----------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.0):
        super().__init__()
        self.n_head = n_head
        self.c_attn = nn.Linear(d_model, 3 * d_model)  # W_q, W_k, W_v in one matmul
        self.c_proj = nn.Linear(d_model, d_model)  # W_o
        self.dropout = dropout
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(C, dim=2)
        # (B, T, C) -> (B, nh, T, hs): each head attends independently
        q, k, v = (z.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for z in (q, k, v))
        # softmax(q kᵀ/√hs + causal mask) v, fused (FlashAttention kernels on GPU)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                           dropout_p=self.dropout if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # concat heads
        return self.resid_drop(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, d_model, dropout=0.0):
        super().__init__()
        self.c_fc = nn.Linear(d_model, 4 * d_model)
        self.gelu = nn.GELU(approximate="tanh")  # the GPT-2 GELU
        self.c_proj = nn.Linear(4 * d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.0):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head, dropout)
        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))  # pre-LN residual blocks
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, n_layer=2, n_head=4, d_model=64, block_size=32,
                 tie_weights=False, dropout=0.0):
        super().__init__()
        self.block_size = block_size
        self.wte = nn.Embedding(vocab_size, d_model)
        self.wpe = nn.Embedding(block_size, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(d_model, n_head, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        if tie_weights:
            self.lm_head.weight = self.wte.weight  # one tensor, used twice
        # GPT-2 init (same as the NumPy model)
        for name, p in self.named_parameters():
            if p.dim() >= 2:
                std = 0.02 / math.sqrt(2 * n_layer) if name.endswith("c_proj.weight") else 0.02
                nn.init.normal_(p, 0.0, std)
            elif name.endswith("bias"):
                nn.init.zeros_(p)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.wte(idx) + self.wpe(pos))
        for blk in self.blocks:
            x = blk(x)
        logits = self.lm_head(self.ln_f(x))
        if targets is None:
            return logits
        return logits, F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

    @torch.no_grad()
    def generate(self, idx, n, temperature=1.0, top_k=None):
        for _ in range(n):
            logits = self(idx[:, -self.block_size:])[:, -1] / temperature
            if top_k is not None:
                kth = torch.topk(logits, top_k).values[:, -1:]
                logits[logits < kth] = -float("inf")
            idx = torch.cat([idx, torch.multinomial(F.softmax(logits, -1), 1)], dim=1)
        return idx


# ----------------------------------------------------------------------------
# Part 1: identical to our NumPy model, forward AND backward
# ----------------------------------------------------------------------------
def copy_numpy_weights(np_model, pt_model):
    """Our Linear stores W as (in, out); nn.Linear stores (out, in): transpose."""
    P = {k: torch.from_numpy(v) for k, v in np_model.params().items()}
    with torch.no_grad():
        pt_model.wte.weight.copy_(P["wte.weight"])
        pt_model.wpe.weight.copy_(P["wpe.weight"])
        for i, blk in enumerate(pt_model.blocks):
            for ln in ("ln_1", "ln_2"):
                getattr(blk, ln).weight.copy_(P[f"h.{i}.{ln}.gamma"])
                getattr(blk, ln).bias.copy_(P[f"h.{i}.{ln}.beta"])
            for mod, sub in (("attn", "c_attn"), ("attn", "c_proj"),
                             ("mlp", "c_fc"), ("mlp", "c_proj")):
                lin = getattr(getattr(blk, mod), sub)
                lin.weight.copy_(P[f"h.{i}.{mod}.{sub}.W"].T)
                lin.bias.copy_(P[f"h.{i}.{mod}.{sub}.b"])
        pt_model.ln_f.weight.copy_(P["ln_f.gamma"])
        pt_model.ln_f.bias.copy_(P["ln_f.beta"])
        if "lm_head.W" in P:
            pt_model.lm_head.weight.copy_(P["lm_head.W"].T)


def torch_grad_for(name, pt_model):
    """The PyTorch gradient matching one of our NumPy parameter names, in our layout."""
    parts = name.split(".")
    if name.startswith(("wte", "wpe")):
        return getattr(pt_model, parts[0]).weight.grad
    if name.startswith("ln_f"):
        m = pt_model.ln_f
        return (m.weight if parts[1] == "gamma" else m.bias).grad
    if name == "lm_head.W":
        return pt_model.lm_head.weight.grad.T
    blk = pt_model.blocks[int(parts[1])]
    if parts[2] in ("ln_1", "ln_2"):
        m = getattr(blk, parts[2])
        return (m.weight if parts[3] == "gamma" else m.bias).grad
    lin = getattr(getattr(blk, parts[2]), parts[3])
    return lin.weight.grad.T if parts[4] == "W" else lin.bias.grad


def check_against_numpy(tie):
    V, B, T = 11, 3, 8
    rng = np.random.default_rng(0)
    np_model = NumpyGPT(V, n_layer=2, n_head=4, d_model=16, block_size=T,
                        tie_weights=tie, seed=0, dtype=np.float64)
    for name, p in np_model.params().items():  # non-trivial values everywhere
        p[...] = (1.0 + 0.3 * rng.standard_normal(p.shape) if name.endswith("gamma")
                  else 0.3 * rng.standard_normal(p.shape))
    idx = rng.integers(0, V, (B, T))
    targets = rng.integers(0, V, (B, T))
    np_logits = np_model.forward(idx)
    np_loss = np_model.forward(idx, targets)
    np_grads = np_model.backward()

    pt_model = GPT(V, n_layer=2, n_head=4, d_model=16, block_size=T, tie_weights=tie).double()
    copy_numpy_weights(np_model, pt_model)
    logits, loss = pt_model(torch.from_numpy(idx), torch.from_numpy(targets))
    loss.backward()

    d_logits = np.abs(logits.detach().numpy() - np_logits).max()
    assert d_logits < 1e-10, d_logits
    assert abs(loss.item() - np_loss) < 1e-12
    worst = 0.0
    for name, g in np_grads.items():
        diff = np.abs(torch_grad_for(name, pt_model).numpy() - g).max()
        assert diff < 1e-10, (name, diff)
        worst = max(worst, diff)
    kind = "tied" if tie else "separate"
    print(f"  {kind:8s} embeddings: logits max |diff| = {d_logits:.1e}, loss {loss.item():.6f} "
          f"vs {np_loss:.6f}, all {len(np_grads)} gradient tensors max |diff| = {worst:.1e}")


# ----------------------------------------------------------------------------
# Part 2: idiomatic training with a validation split
# ----------------------------------------------------------------------------
class CharRNN(nn.Module):
    """The ../rnn/rnn_pytorch.py model: Embedding -> nn.RNN(tanh) -> Linear."""

    def __init__(self, vocab_size, embed=32, hidden=128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed)
        self.rnn = nn.RNN(embed, hidden, batch_first=True)
        self.head = nn.Linear(hidden, vocab_size)

    def forward(self, idx, targets):
        logits = self.head(self.rnn(self.embed(idx))[0])
        return logits, F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))


def batches(data, B, T, g):
    ix = torch.randint(len(data) - T - 1, (B,), generator=g)
    return (torch.stack([data[i:i + T] for i in ix]),
            torch.stack([data[i + 1:i + T + 1] for i in ix]))


@torch.no_grad()
def eval_loss(model, data, T):
    """Mean loss per char over all non-overlapping T-windows of data."""
    model.eval()
    starts = range(0, len(data) - T - 1, T)
    x = torch.stack([data[i:i + T] for i in starts])
    y = torch.stack([data[i + 1:i + T + 1] for i in starts])
    loss = model(x, y)[1].item()
    model.train()
    return loss


def make_optimizer(model, lr, weight_decay):
    """AdamW, with weight decay only on matrices (not biases / LayerNorm)."""
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    return torch.optim.AdamW([{"params": decay, "weight_decay": weight_decay},
                              {"params": no_decay, "weight_decay": 0.0}],
                             lr=lr, betas=(0.9, 0.95))


def fit(model, train_data, val_data, args, label, log=True):
    g = torch.Generator().manual_seed(0)  # same batches for every model
    opt = make_optimizer(model, args.lr, args.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: lr_at(s, 1.0, args.warmup, args.steps))  # warmup + cosine
    t0 = time.time()
    best = (float("inf"), 0)
    for step in range(args.steps + 1):
        if step % args.eval_every == 0:
            tr = eval_loss(model, train_data, args.block_size)
            va = eval_loss(model, val_data, args.block_size)
            best = min(best, (va, step))
            if log:
                print(f"  {label} step {step:5d} | train {tr:.3f} | val {va:.3f} "
                      f"(ppl {math.exp(va):5.2f}) | lr {sched.get_last_lr()[0]:.1e} | "
                      f"{time.time() - t0:4.1f} s")
        if step == args.steps:
            break
        x, y = batches(train_data, args.batch_size, args.block_size, g)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
    return tr, va, best, time.time() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", default=DEFAULT_TEXT)
    ap.add_argument("--n-layer", type=int, default=2)
    ap.add_argument("--n-head", type=int, default=4)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--block-size", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--no-rnn", action="store_true", help="skip the char-RNN comparison")
    args = ap.parse_args()

    print("Part 1: PyTorch == our NumPy model (float64, random weights)")
    check_against_numpy(tie=False)
    check_against_numpy(tie=True)
    print("PyTorch logits and autograd gradients == our hand-written forward/backward ✔\n")

    torch.manual_seed(0)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    text, chars, data = load_text(args.file)
    data = torch.from_numpy(data)
    V = len(chars)
    n_val = int(len(data) * args.val_frac)
    train_data, val_data = data[:-n_val], data[-n_val:]  # the END of the story is held out
    print(f"Part 2: train on {len(train_data)} chars, validate on the last {n_val} "
          f"(never trained on). Uniform guessing: ln({V}) = {math.log(V):.3f}")

    model = GPT(V, args.n_layer, args.n_head, args.d_model, args.block_size,
                dropout=args.dropout)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"mini-GPT: {args.n_layer} layers, {args.n_head} heads, d_model {args.d_model}, "
          f"block {args.block_size}, dropout {args.dropout}: {n_params:,} params")
    tr, va, (best_va, best_step), secs = fit(model, train_data, val_data, args, "GPT")
    print(f"GPT final: train {tr:.3f}, val {va:.3f} per char (val perplexity "
          f"{math.exp(va):.2f}); best val {best_va:.3f} at step {best_step}; {secs:.1f} s")

    model.eval()
    start = torch.tensor([[chars.index(text[0])]])
    out = model.generate(start, 400, args.temperature, args.top_k)[0].tolist()
    print(f"\n===== GPT sample (temperature {args.temperature}, top-k {args.top_k}) =====")
    print("".join(chars[i] for i in out) + "\n")

    if not args.no_rnn:
        torch.manual_seed(0)
        rnn = CharRNN(V)
        print(f"char-RNN (../rnn/rnn_pytorch.py model, hidden 128): "
              f"{sum(p.numel() for p in rnn.parameters()):,} params, same batches and schedule")
        rtr, rva, (rbest, rstep), rsecs = fit(rnn, train_data, val_data, args, "RNN")
        print(f"RNN final: train {rtr:.3f}, val {rva:.3f} per char (val perplexity "
              f"{math.exp(rva):.2f}); best val {rbest:.3f} at step {rstep}; {rsecs:.1f} s\n")
        print(f"{'':10s}{'train loss':>12s}{'val loss':>10s}{'best val':>10s}{'val ppl':>9s}")
        print(f"{'mini-GPT':10s}{tr:12.3f}{va:10.3f}{best_va:10.3f}{math.exp(va):9.2f}")
        print(f"{'char-RNN':10s}{rtr:12.3f}{rva:10.3f}{rbest:10.3f}{math.exp(rva):9.2f}")
        print(f"{'uniform':10s}{math.log(V):12.3f}{math.log(V):10.3f}{'':10s}{V:9d}")


if __name__ == "__main__":
    main()
