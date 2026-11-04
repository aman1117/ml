"""
A tiny mixture-of-experts (MoE) LLaMA in PyTorch: top-2 routing over 8 SwiGLU
experts, the load-balancing auxiliary loss, and expert capacity.

The idea (Shazeer et al. 2017; Switch Transformer, Fedus et al. 2021; Mixtral,
Jiang et al. 2024): replace the feed-forward of each block by E expert
feed-forwards and a ROUTER (one linear layer, d -> E). Each token goes to its
top-k experts only, and their outputs are mixed with the router's weights:

    logits = x W_router                      (N, E)        N = B·T tokens
    top-k:  (g_1..g_k, e_1..e_k) = softmax over the k largest logits   (Mixtral)
    y = Σ_j g_j · Expert_{e_j}(x)

So the model has E× the feed-forward parameters, but each token only pays for k
experts: total params >> active params per token.

The catch: nothing stops the router from sending everything to a few experts
(the others then get no gradient and never improve: a rich-get-richer collapse).
The Switch load-balancing loss fixes it:

    aux = E · Σ_e f_e · P_e      f_e = fraction of token->expert assignments to e
                                 P_e = mean router probability of e (differentiable)

It is 1 when the load is perfectly uniform and up to E when one expert takes all.

This script:
  1. checks the sparse dispatch against a brute-force dense computation,
  2. counts total vs active parameters,
  3. trains (same batches, same steps) a dense LLaMA, an MoE without the aux loss,
     and an MoE with it, then measures expert utilization, how many tokens a
     capacity limit would drop, val loss, and what characters each expert gets.

Run (~1 min):
    python llama/moe.py
    python llama/moe.py --experts 16 --top-k 1 --aux 0.02
"""
import argparse
import math
import os
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from llama_numpy import DEFAULT_TEXT, load_text, lr_at
from llama_pytorch import LLaMA, FeedForward, n_params
from transformer_pytorch import batches, eval_loss, make_optimizer

warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)
HERE = os.path.dirname(os.path.abspath(__file__))


class MoE(nn.Module):
    """Top-k mixture of SwiGLU experts.  x: (B, T, C) -> (B, T, C).

    capacity_factor: if set, each expert processes at most
        capacity = ceil(capacity_factor · N · k / E)
    tokens per batch; the rest are DROPPED for that expert (they contribute 0 and
    only the residual connection carries them through). None = dropless."""

    def __init__(self, d, hidden, n_experts=8, top_k=2, capacity_factor=None):
        super().__init__()
        self.E, self.k, self.capacity_factor = n_experts, top_k, capacity_factor
        self.router = nn.Linear(d, n_experts, bias=False)
        self.experts = nn.ModuleList([FeedForward(d, hidden) for _ in range(n_experts)])
        self.aux_loss = torch.tensor(0.0)

    def route(self, xf):
        logits = self.router(xf)  # (N, E)
        top_logits, top_idx = logits.topk(self.k, dim=-1)  # (N, k)
        gates = F.softmax(top_logits, dim=-1)  # renormalize over the chosen experts
        return logits, gates, top_idx

    def forward(self, x):
        B, T, C = x.shape
        xf = x.reshape(-1, C)
        N = xf.shape[0]
        logits, gates, top_idx = self.route(xf)
        # Switch load-balancing loss: f_e is a count (no gradient), P_e is differentiable,
        # so the gradient lowers the probability of the experts that are overloaded.
        counts = torch.bincount(top_idx.flatten(), minlength=self.E)
        f = counts.float() / (N * self.k)
        P = F.softmax(logits, dim=-1).mean(0)
        self.aux_loss = self.E * (f * P).sum()
        self.counts = counts.detach()

        out = torch.zeros_like(xf)
        cap = (N if self.capacity_factor is None else
               math.ceil(self.capacity_factor * N * self.k / self.E))
        self.dropped = 0
        for e, expert in enumerate(self.experts):
            tok, slot = (top_idx == e).nonzero(as_tuple=True)  # tokens that chose e
            if len(tok) > cap:  # over capacity: keep the first `cap` (in token order)
                self.dropped += len(tok) - cap
                tok, slot = tok[:cap], slot[:cap]
            if len(tok):
                out.index_add_(0, tok, gates[tok, slot, None] * expert(xf[tok]))
        return out.view(B, T, C)


def dense_reference(moe, x):
    """Brute force: run EVERY expert on EVERY token, weight by the (mostly zero) gates."""
    xf = x.reshape(-1, x.shape[-1])
    _, gates, top_idx = moe.route(xf)
    G = torch.zeros(xf.shape[0], moe.E, dtype=x.dtype).scatter(1, top_idx, gates)  # (N, E)
    all_out = torch.stack([ex(xf) for ex in moe.experts], dim=1)  # (N, E, C)
    return (G[..., None] * all_out).sum(1).view_as(x)


def make_model(V, args, moe):
    torch.manual_seed(0)
    if not moe:
        return LLaMA(V, ffn_hidden=args.dense_hidden, dropout=args.dropout)
    m = LLaMA(V, ffn_hidden=args.dense_hidden, dropout=args.dropout)
    for blk in m.layers:  # swap each dense feed-forward for an MoE with the same active size
        blk.feed_forward = MoE(64, args.expert_hidden, args.experts, args.top_k)
        for name, p in blk.feed_forward.named_parameters():
            std = 0.02 / math.sqrt(4) if name.endswith("w2.weight") else 0.02
            nn.init.normal_(p, 0.0, std)
    return m


def moe_layers(model):
    return [blk.feed_forward for blk in model.layers if isinstance(blk.feed_forward, MoE)]


def train(model, train_data, val_data, args, aux_coef):
    g = torch.Generator().manual_seed(0)  # identical batches for every model
    torch.manual_seed(1)  # and identical dropout masks
    opt = make_optimizer(model, args.lr, 0.1)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_at(s, 1.0, 100, args.steps))
    max_load = []  # per step: the busiest expert's share of the assignments (layer 0)
    t0 = time.time()
    for step in range(args.steps):
        x, y = batches(train_data, args.batch_size, 32, g)
        _, loss = model(x, y)
        layers = moe_layers(model)
        if layers:
            loss = loss + aux_coef * sum(m.aux_loss for m in layers)
            c = layers[0].counts.float()
            max_load.append((c.max() / c.sum()).item())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
    return eval_loss(model, val_data, 32), np.array(max_load), time.time() - t0


@torch.no_grad()
def utilization(model, data, chars, cfs=(1.0, 1.25, 2.0)):
    """Over all 32-char windows of `data`: share of assignments per expert and layer,
    the aux loss, the fraction of assignments a capacity limit would drop, and the
    characters each layer-0 expert receives as a first choice."""
    model.eval()
    starts = range(0, len(data) - 33, 32)
    x = torch.stack([data[i:i + 32] for i in starts])
    out = {"share": [], "aux": [], "drop": {cf: [] for cf in cfs}}
    captured = []
    hooks = [m.register_forward_pre_hook(lambda mod, inp: captured.append(inp[0]))
             for m in moe_layers(model)]
    model(x)
    for h in hooks:
        h.remove()
    for m, inp in zip(moe_layers(model), captured):
        xf = inp.reshape(-1, inp.shape[-1])
        logits, _, top_idx = m.route(xf)
        counts = torch.bincount(top_idx.flatten(), minlength=m.E).float()
        out["share"].append((counts / counts.sum()).numpy())
        P = F.softmax(logits, -1).mean(0)
        out["aux"].append((m.E * (counts / counts.sum()) * P).sum().item())
        N = xf.shape[0]
        for cf in cfs:
            cap = math.ceil(cf * N * m.k / m.E)
            out["drop"][cf].append((counts - cap).clamp(min=0).sum().item() / (N * m.k))
    first = moe_layers(model)[0].route(captured[0].reshape(-1, captured[0].shape[-1]))[2][:, 0]
    ids = x.reshape(-1).numpy()
    top_chars = []
    for e in range(moe_layers(model)[0].E):
        sel = ids[first.numpy() == e]
        vals, cnt = np.unique(sel, return_counts=True)
        order = np.argsort(-cnt)[:6]
        top_chars.append(("".join({" ": "␣", "\n": "↵"}.get(chars[vals[i]], chars[vals[i]])
                                  for i in order), len(sel) / len(ids)))
    model.train()
    out["share"] = np.array(out["share"])
    return out, top_chars


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experts", type=int, default=8)
    ap.add_argument("--top-k", type=int, default=2)
    ap.add_argument("--expert-hidden", type=int, default=88,
                    help="SwiGLU hidden size of each expert (88 = half of the dense 176)")
    ap.add_argument("--dense-hidden", type=int, default=176)
    ap.add_argument("--aux", type=float, default=0.01, help="aux loss coefficient (Switch: 0.01)")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--save", default=os.path.join(HERE, "moe.npz"))
    args = ap.parse_args()
    torch.set_num_threads(min(4, torch.get_num_threads()))

    print("1. Sparse dispatch == brute-force dense mixture (float64, random weights)")
    torch.manual_seed(0)
    moe = MoE(16, 24, n_experts=8, top_k=2).double()
    x = torch.randn(3, 10, 16, dtype=torch.float64)
    diff = (moe(x) - dense_reference(moe, x)).abs().max().item()
    print(f"   max |difference| = {diff:.1e}  {'OK' if diff < 1e-12 else 'FAIL'}")
    assert diff < 1e-12
    moe.capacity_factor = 1.0
    moe(x)
    print(f"   with capacity factor 1.0 (capacity = ceil(1.0·30·2/8) = 8 per expert): "
          f"{moe.dropped} of 60 assignments dropped\n")

    text, chars, data = load_text(DEFAULT_TEXT)
    data = torch.from_numpy(data)
    V = len(chars)
    n_val = int(len(data) * 0.1)
    train_data, val_data = data[:-n_val], data[-n_val:]

    dense = make_model(V, args, moe=False)
    sparse = make_model(V, args, moe=True)
    d, L, E, k, H = 64, 2, args.experts, args.top_k, args.expert_hidden
    expert = 3 * d * H
    total, total_dense = n_params(sparse), n_params(dense)
    active = total - L * (E - k) * expert
    print(f"2. Parameters (d=64, 2 layers, {E} experts of SwiGLU hidden {H}, top-{k}):")
    print(f"   one expert: 3 × {d} × {H} = {expert:,}   router: {d} × {E} = {d * E} per layer")
    print(f"   dense LLaMA (hidden {args.dense_hidden}):  {total_dense:9,d} params, "
          f"all active")
    print(f"   MoE LLaMA:                       {total:9,d} params total, {active:,} active "
          f"per token ({active / total:.0%})")
    print(f"   Mixtral 8x7B: 46.7B total, 12.9B active (2 of 8 experts; attention is shared)\n")

    print(f"3. Training on the same batches, {args.steps} steps each "
          f"(aux coefficient {args.aux} when on):")
    res = {}
    for label, model, coef in (("dense", dense, 0.0), ("MoE, no aux loss", sparse, 0.0),
                               ("MoE + aux loss", None, args.aux)):
        if model is None:
            model = make_model(V, args, moe=True)
        val, max_load, secs = train(model, train_data, val_data, args, coef)
        res[label] = dict(val=val, max_load=max_load, secs=secs, model=model)
        print(f"   {label:18s} val loss {val:.3f}   {secs:5.1f} s")

    print("\n   Expert utilization over the training text (share of the top-2 assignments, "
          f"uniform = {1 / E:.1%}):")
    stats = {}
    for label in ("MoE, no aux loss", "MoE + aux loss"):
        u, top_chars = utilization(res[label]["model"], train_data, chars)
        stats[label] = (u, top_chars)
        for layer, share in enumerate(u["share"]):
            cv = share.std() / share.mean()
            print(f"   {label:18s} layer {layer}: " + " ".join(f"{s:5.1%}" for s in share)
                  + f"   max {share.max():.1%}, CV {cv:.2f}, aux {u['aux'][layer]:.2f}")
    print("   (aux = E·Σ f_e·P_e: 1.00 is perfectly balanced)\n")

    print("   Assignments a capacity limit would DROP (capacity = ceil(cf · N · k / E)):")
    for label in ("MoE, no aux loss", "MoE + aux loss"):
        u = stats[label][0]
        print(f"   {label:18s} " + "   ".join(
            f"cf {cf}: {np.mean(v):5.1%}" for cf, v in u["drop"].items()))

    print("\n   What layer 0's experts receive as FIRST choice (MoE + aux loss): "
          "most frequent input chars")
    for e, (s, frac) in enumerate(stats["MoE + aux loss"][1]):
        print(f"     expert {e}: {frac:5.1%} of tokens   {s}")

    np.savez(args.save,
             share_noaux=stats["MoE, no aux loss"][0]["share"],
             share_aux=stats["MoE + aux loss"][0]["share"],
             maxload_noaux=res["MoE, no aux loss"]["max_load"],
             maxload_aux=res["MoE + aux loss"]["max_load"],
             val=np.array([res[k]["val"] for k in ("dense", "MoE, no aux loss",
                                                   "MoE + aux loss")]),
             params=np.array([total_dense, total, active]),
             drop_noaux=np.array([np.mean(v) for v in stats["MoE, no aux loss"][0]["drop"].values()]),
             drop_aux=np.array([np.mean(v) for v in stats["MoE + aux loss"][0]["drop"].values()]))
    print(f"\nSaved results to {os.path.relpath(args.save)} (used by make_figures.py)")


if __name__ == "__main__":
    main()
