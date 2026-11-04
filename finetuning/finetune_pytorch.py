"""
Fine-tuning a GPT end to end in PyTorch: SFT (full and LoRA) and DPO, with no
wrapper library: our own LoRALinear, our own masked loss, our own DPO loss,
our own training loops.

1. Equivalence: our PyTorch LoRALinear == the NumPy one in lora_numpy.py.
   A NumPy mini-GPT with LoRA on q,k,v,o,fc,proj and the PyTorch GPT from
   ../transformer/transformer_pytorch.py with our LoRALinear wrappers get the
   same weights; we compare logits, the masked SFT loss, the DPO loss and the
   gradients of every A and B (float64), and the merged forward pass.
2. The experiment (README sections 7-11), on the lighthouse chats from domain_data.py:
     base     pretrain a 4-layer GPT on ../rnn/input.txt (90% train / 10% held out),
              or load finetuning/base_gpt.pt; evaluate it with a 1-shot prompt
     SFT full every weight trains (masked loss on the assistant tokens only)
     SFT LoRA the base is frozen; rank-8 adapters on q,k,v,o and both MLP matrices
     DPO      on top of the SFT LoRA model (merged): a fresh LoRA is the policy, a
              frozen copy of the SFT model is the reference
   For each: exact-match accuracy and format adherence on held-out questions,
   the refusal rate on unknown lighthouses, and CATASTROPHIC FORGETTING (loss on
   the story before/after), plus trainable parameters and wall-clock time.
   Results (curves, the singular values of the full fine-tune's ΔW, ...) are
   saved to finetuning/finetune_results.npz for make_figures.py.

Run (~2 min the first time, ~70 s once base_gpt.pt exists):
    python finetuning/finetune_pytorch.py
    python finetuning/finetune_pytorch.py --rank 2 --targets q,v --beta 0.1
"""
import argparse
import copy
import json
import math
import os
import sys
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "transformer"))
from transformer_numpy import GPT as NumpyGPT, lr_at  # noqa: E402
from transformer_pytorch import GPT, copy_numpy_weights  # noqa: E402

import domain_data as dd  # noqa: E402
import lora_numpy as ln  # noqa: E402

warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)

BASE_CFG = dict(n_layer=4, n_head=4, d_model=128, block_size=128, dropout=0.1)
BASE_PATH = os.path.join(HERE, "base_gpt.pt")
RESULTS_PATH = os.path.join(HERE, "finetune_results.npz")
THREADS = 4


# ----------------------------------------------------------------------------
# LoRA as an nn.Module
# ----------------------------------------------------------------------------
class LoRALinear(nn.Module):
    """y = base(x) + (alpha/r) · (x A) B, base frozen.  Works on any nn.Linear
    (our GPT's or a Hugging Face model's).  A: (in, r), B: (r, out) per output
    slice, as in lora_numpy.py (PEFT stores the transposes, lora_A (r, in) and
    lora_B (out, r): same math)."""

    def __init__(self, base, r, alpha, slices=None, generator=None):
        super().__init__()
        self.base = base
        for p in base.parameters():
            p.requires_grad_(False)
        self.r, self.alpha, self.scale = r, alpha, alpha / r
        n_out, n_in = base.weight.shape
        self.n_out = n_out
        self.slices = slices or {"": (0, n_out)}
        w = base.weight
        for sfx, (s, e) in self.slices.items():
            A = torch.randn(n_in, r, generator=generator, dtype=torch.float32) / math.sqrt(n_in)
            self.register_parameter("A" + sfx, nn.Parameter(A.to(w.dtype)))
            self.register_parameter("B" + sfx, nn.Parameter(torch.zeros(r, e - s, dtype=w.dtype)))
        self.merged, self.enabled = False, True

    def forward(self, x):
        y = self.base(x)
        if self.merged or not self.enabled:
            return y
        for sfx, (s, e) in self.slices.items():
            d = (x @ getattr(self, "A" + sfx)) @ getattr(self, "B" + sfx) * self.scale
            y = y + (d if e - s == self.n_out else F.pad(d, (s, self.n_out - e)))
        return y

    @torch.no_grad()
    def delta_W(self):
        """ΔW in nn.Linear's (out, in) layout."""
        dW = torch.zeros_like(self.base.weight)
        for sfx, (s, e) in self.slices.items():
            dW[s:e] = (self.scale * getattr(self, "A" + sfx) @ getattr(self, "B" + sfx)).T
        return dW

    @torch.no_grad()
    def merge(self):
        if not self.merged:
            self.base.weight += self.delta_W()
            self.merged = True

    @torch.no_grad()
    def unmerge(self):
        if self.merged:
            self.base.weight -= self.delta_W()
            self.merged = False


def apply_lora(model, r, alpha, targets=ln.ALL_TARGETS, seed=0):
    """Same wrapping as lora_numpy.apply_lora, for transformer_pytorch.GPT.
    Afterwards ONLY the adapters' A and B have requires_grad=True."""
    g = torch.Generator().manual_seed(seed)
    for p in model.parameters():
        p.requires_grad_(False)
    C = model.wte.weight.shape[1]
    for blk in model.blocks:
        qkv = {f"_{n}": (i * C, (i + 1) * C) for i, n in enumerate("qkv") if n in targets}
        if qkv:
            blk.attn.c_attn = LoRALinear(blk.attn.c_attn, r, alpha, qkv, g)
        if "o" in targets:
            blk.attn.c_proj = LoRALinear(blk.attn.c_proj, r, alpha, generator=g)
        if "fc" in targets:
            blk.mlp.c_fc = LoRALinear(blk.mlp.c_fc, r, alpha, generator=g)
        if "proj" in targets:
            blk.mlp.c_proj = LoRALinear(blk.mlp.c_proj, r, alpha, generator=g)
    return model


def inject_lora(model, target_names, r, alpha, seed=0):
    """Generic version for any model (e.g. a Hugging Face Llama): wrap every
    nn.Linear whose attribute name is in target_names (e.g. {"q_proj", "v_proj"})."""
    g = torch.Generator().manual_seed(seed)
    for p in model.parameters():
        p.requires_grad_(False)
    n = 0
    for module in list(model.modules()):
        for name, child in list(module.named_children()):
            if name in target_names and isinstance(child, nn.Linear):
                setattr(module, name, LoRALinear(child, r, alpha, generator=g))
                n += 1
    return n


def lora_modules(model):
    return [m for m in model.modules() if isinstance(m, LoRALinear)]


def merge_and_unload(model):
    """Fold every adapter into its base weight and put the plain nn.Linear back."""
    for module in list(model.modules()):
        for name, child in list(module.named_children()):
            if isinstance(child, LoRALinear):
                child.merge()
                setattr(module, name, child.base)
    return model


def trainable(model):
    return [p for p in model.parameters() if p.requires_grad]


def n_trainable(model):
    return sum(p.numel() for p in trainable(model))


# ----------------------------------------------------------------------------
# Losses (the autograd twins of lora_numpy.py)
# ----------------------------------------------------------------------------
def token_logprobs(logits, y):
    """log p_t[y_t] for every position: (B, T)."""
    return -F.cross_entropy(logits.flatten(0, 1), y.flatten(), reduction="none").view(y.shape)


def masked_ce(logits, y, m):
    return -(token_logprobs(logits, y) * m).sum() / m.sum().clamp(min=1)


def seq_logprob(logits, y, m):
    return (token_logprobs(logits, y) * m).sum(1)


def dpo_loss(pi_w, pi_l, ref_w, ref_l, beta):
    z = beta * ((pi_w - ref_w) - (pi_l - ref_l))
    return -F.logsigmoid(z).mean(), z


# ----------------------------------------------------------------------------
# Part 1: PyTorch LoRA == NumPy LoRA
# ----------------------------------------------------------------------------
def check_against_numpy():
    V, B, T, C = 11, 3, 9, 16
    rng = np.random.default_rng(0)
    npm = NumpyGPT(V, n_layer=2, n_head=4, d_model=C, block_size=T, seed=0, dtype=np.float64)
    for name, p in npm.params().items():
        p[...] = (1 + 0.3 * rng.standard_normal(p.shape) if name.endswith("gamma")
                  else 0.3 * rng.standard_normal(p.shape))
    ptm = GPT(V, n_layer=2, n_head=4, d_model=C, block_size=T).double()
    copy_numpy_weights(npm, ptm)  # base weights first, then wrap both
    ln.apply_lora(npm, r=3, alpha=6.0)
    apply_lora(ptm, r=3, alpha=6.0)
    np_lora = ln.lora_params(npm)
    for k, p in np_lora.items():  # B = 0 at init would hide bugs in dA: randomize
        p[...] = 0.3 * rng.standard_normal(p.shape)
    pt_lora = dict(ptm.named_parameters())
    name_map = {}
    for k in np_lora:  # h.0.attn.c_attn.A_q -> blocks.0.attn.c_attn.A_q
        tk = "blocks." + k[2:]
        name_map[k] = tk
        with torch.no_grad():
            pt_lora[tk].copy_(torch.from_numpy(np_lora[k]))
    assert n_trainable(ptm) == sum(p.size for p in np_lora.values())
    ptm.eval()

    idx = rng.integers(0, V, (B, T))
    y = rng.integers(0, V, (B, T))
    m = (rng.random((B, T)) < 0.5).astype(np.float64)
    worst = {}
    # masked SFT loss
    loss_np, dlog = ln.masked_cross_entropy(npm.forward(idx), y, m)
    npm.dlogits = dlog
    npm.backward()
    g_np = ln.lora_grads(npm)
    ptm.zero_grad()
    logits = ptm(torch.from_numpy(idx))
    loss_pt = masked_ce(logits, torch.from_numpy(y), torch.from_numpy(m))
    loss_pt.backward()
    worst["logits"] = np.abs(logits.detach().numpy() - npm.forward(idx)).max()
    worst["SFT loss"] = abs(loss_pt.item() - loss_np)
    worst["SFT dA,dB"] = max(np.abs(pt_lora[name_map[k]].grad.numpy() - g).max()
                             for k, g in g_np.items())
    # DPO loss: rows 0..1 = "chosen", rows 1..2 = "rejected" (any sequences will do)
    ref = rng.standard_normal((2, 2))
    lp, probs = ln.sequence_logprob(npm.forward(idx[:2]), y[:2], m[:2])
    lq, probs_q = ln.sequence_logprob(npm.forward(idx[1:]), y[1:], m[1:])
    loss_np, d_w, d_l, _ = ln.dpo_loss(lp, lq, ref[0], ref[1], 0.5)
    npm.forward(np.concatenate([idx[:2], idx[1:]]))
    npm.dlogits = ln.sequence_logprob_backward(
        np.concatenate([d_w, d_l]), np.concatenate([probs, probs_q]),
        np.concatenate([y[:2], y[1:]]), np.concatenate([m[:2], m[1:]]))
    npm.backward()
    g_np = ln.lora_grads(npm)
    ptm.zero_grad()
    t = lambda a: torch.from_numpy(a)
    pw = seq_logprob(ptm(t(idx[:2])), t(y[:2]), t(m[:2]))
    pl = seq_logprob(ptm(t(idx[1:])), t(y[1:]), t(m[1:]))
    loss_pt, _ = dpo_loss(pw, pl, t(ref[0]), t(ref[1]), 0.5)
    loss_pt.backward()
    worst["DPO loss"] = abs(loss_pt.item() - loss_np)
    worst["DPO dA,dB"] = max(np.abs(pt_lora[name_map[k]].grad.numpy() - g).max()
                             for k, g in g_np.items())
    # merge: same function, no adapters left
    before = ptm(t(idx)).detach()
    merge_and_unload(ptm)
    worst["merged logits"] = (ptm(t(idx)).detach() - before).abs().max().item()
    for k, v in worst.items():
        print(f"  {k:14s} max |PyTorch - NumPy| = {v:.1e}")
        assert v < 1e-10, (k, v)
    print("PyTorch LoRALinear + losses == lora_numpy.py (forward, SFT/DPO gradients, merge) ✔\n")


# ----------------------------------------------------------------------------
# Part 2: the experiment
# ----------------------------------------------------------------------------
def setup():
    torch.set_num_threads(THREADS)
    story = dd.load_story()
    tok = ln.CharTokenizer(dd.build_vocab(story))
    ids = torch.tensor(tok.encode(story))
    n_val = len(ids) // 10
    train_rows, test_rows, prefs = dd.load_all()
    return dict(tok=tok, story_train=ids[:-n_val], story_val=ids[-n_val:],
                train=train_rows, test=test_rows, prefs=prefs)


@torch.no_grad()
def story_loss(model, data, T=None):
    """Next-char loss over all non-overlapping windows (the forgetting probe)."""
    model.eval()
    T = T or BASE_CFG["block_size"]
    starts = range(0, len(data) - T - 1, T)
    x = torch.stack([data[i:i + T] for i in starts])
    y = torch.stack([data[i + 1:i + T + 1] for i in starts])
    return F.cross_entropy(model(x).flatten(0, 1), y.flatten()).item()


def pretrain(tok, data, steps=1000, batch_size=16, lr=2e-3, seed=0, log=True):
    torch.manual_seed(seed)
    cfg = dict(BASE_CFG)
    model = GPT(len(tok.chars), **cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    g = torch.Generator().manual_seed(seed)
    T = cfg["block_size"]
    model.train()
    for step in range(steps):
        ix = torch.randint(len(data) - T - 1, (batch_size,), generator=g)
        x = torch.stack([data[i:i + T] for i in ix])
        y = torch.stack([data[i + 1:i + T + 1] for i in ix])
        for grp in opt.param_groups:
            grp["lr"] = lr_at(step, lr, 100, steps)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if log and step % 250 == 0:
            print(f"  pretrain step {step:4d} | loss {loss.item():.3f}")
    return model


def get_base(env, log=True):
    """Load finetuning/base_gpt.pt or pretrain the base model (~40 s) and save it."""
    tok = env["tok"]
    if os.path.exists(BASE_PATH):
        ck = torch.load(BASE_PATH)
        if ck["chars"] == tok.chars and ck["cfg"] == BASE_CFG:
            model = GPT(len(tok.chars), **BASE_CFG)
            model.load_state_dict(ck["state"])
            return model, ck["secs"]
    if log:
        print(f"pretraining the base GPT on {len(env['story_train'])} chars of ../rnn/input.txt "
              f"(the last {len(env['story_val'])} are held out)...")
    t0 = time.time()
    model = pretrain(tok, env["story_train"], log=log)
    secs = time.time() - t0
    torch.save(dict(state=model.state_dict(), chars=tok.chars, cfg=BASE_CFG, secs=secs), BASE_PATH)
    return model, secs


def batches_of(n, batch_size, g):
    """Shuffle every epoch, yield index batches forever."""
    while True:
        perm = torch.randperm(n, generator=g).tolist()
        for i in range(0, n - batch_size + 1, batch_size):
            yield perm[i:i + batch_size]


def to_t(*arrs):
    return [torch.from_numpy(a) for a in arrs]


def train_sft(model, data, steps, lr, batch_size=16, seed=0, weight_decay=0.0, log_every=0,
              replay=None):
    """data: [(ids, mask)]. Trains whatever has requires_grad=True. Returns loss curve.
    replay = (story ids, n windows per step): also train on some PRETRAINING text
    every step (plain next-char loss), the simplest defence against forgetting."""
    opt = torch.optim.AdamW(trainable(model), lr=lr, betas=(0.9, 0.95), weight_decay=weight_decay)
    g = torch.Generator().manual_seed(seed)
    it = batches_of(len(data), batch_size, g)
    losses = []
    model.train()
    T = BASE_CFG["block_size"]
    for step in range(steps):
        b = [data[i] for i in next(it)]
        x, y, m = to_t(*ln.pad_batch([s for s, _ in b], [k for _, k in b], 0))
        for grp in opt.param_groups:
            grp["lr"] = lr_at(step, lr, steps // 10, steps)
        loss = masked_ce(model(x), y, m.float())
        if replay is not None:
            story, n = replay
            ix = torch.randint(len(story) - T - 1, (n,), generator=g)
            xs = torch.stack([story[i:i + T] for i in ix])
            ys = torch.stack([story[i + 1:i + T + 1] for i in ix])
            loss = loss + model(xs, ys)[1]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable(model), 1.0)
        opt.step()
        losses.append(loss.item())
        if log_every and (step % log_every == 0 or step == steps - 1):
            print(f"    step {step:4d} | masked loss {loss.item():.4f}")
    return losses


@torch.no_grad()
def pair_logprobs(model, pairs, ix):
    seqs = [pairs[i][0] for i in ix] + [pairs[i][2] for i in ix]
    masks = [pairs[i][1] for i in ix] + [pairs[i][3] for i in ix]
    x, y, m = to_t(*ln.pad_batch(seqs, masks, 0))
    lp = seq_logprob(model(x), y, m.float())
    return lp[:len(ix)], lp[len(ix):]


def train_dpo(policy, ref_model, pairs, steps, lr, beta, batch_size=16, seed=0,
              nll_coef=0.0, log_every=0):
    """DPO: the reference model is frozen; only the policy's trainable params move.
    Returns per-step stats (loss, margin, implicit rewards, log-probs of chosen/rejected)."""
    ref_model.eval()
    all_ix = list(range(len(pairs)))
    ref_w, ref_l = [], []
    for i in range(0, len(pairs), 64):  # the reference never changes: compute once
        w, l_ = pair_logprobs(ref_model, pairs, all_ix[i:i + 64])
        ref_w.append(w)
        ref_l.append(l_)
    ref_w, ref_l = torch.cat(ref_w), torch.cat(ref_l)
    opt = torch.optim.AdamW(trainable(policy), lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
    g = torch.Generator().manual_seed(seed)
    it = batches_of(len(pairs), batch_size, g)
    hist = []
    # Dropout OFF (as TRL does by default): with dropout the policy and the
    # reference would disagree even before the first update, and the log-prob
    # ratios would be noise.
    policy.eval()
    for step in range(steps):
        ix = next(it)
        seqs = [pairs[i][0] for i in ix] + [pairs[i][2] for i in ix]
        masks = [pairs[i][1] for i in ix] + [pairs[i][3] for i in ix]
        x, y, m = to_t(*ln.pad_batch(seqs, masks, 0))
        m = m.float()
        tlp = token_logprobs(policy(x), y) * m
        lp = tlp.sum(1)
        pw, pl = lp[:len(ix)], lp[len(ix):]
        loss, z = dpo_loss(pw, pl, ref_w[ix], ref_l[ix], beta)
        total = loss
        if nll_coef:  # RPO-style: also keep the chosen answers likely (per-token NLL)
            total = loss - nll_coef * (tlp[:len(ix)].sum() / m[:len(ix)].sum())
        for grp in opt.param_groups:
            grp["lr"] = lr_at(step, lr, steps // 10, steps)
        opt.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(trainable(policy), 1.0)
        opt.step()
        hist.append(dict(loss=loss.item(), margin=z.mean().item(),
                         accuracy=(z > 0).float().mean().item(),
                         chosen=(pw - ref_w[ix]).mean().item(),
                         rejected=(pl - ref_l[ix]).mean().item()))
        if log_every and (step % log_every == 0 or step == steps - 1):
            h = hist[-1]
            print(f"    step {step:4d} | DPO loss {h['loss']:.3f} | margin {h['margin']:+6.2f} | "
                  f"Δlog π chosen {h['chosen']:+6.2f} rejected {h['rejected']:+7.2f} | "
                  f"pref. acc {h['accuracy']:.2f}")
    return hist


def torch_logits_fn(model):
    def f(ids):
        model.eval()
        with torch.no_grad():
            return model(torch.from_numpy(ids)).numpy()
    return f


def evaluate(model, env):
    """Greedy answers to train + test questions; story loss on train and held-out text."""
    fn = torch_logits_fn(model)
    s_train, _ = ln.evaluate_chat(fn, env["train"], env["tok"])
    s_test, answers = ln.evaluate_chat(fn, env["test"], env["tok"])
    return dict(scores={**s_train, **s_test}, answers=answers,
                story_train=story_loss(model, env["story_train"]),
                story_val=story_loss(model, env["story_val"]))


def few_shot_eval(model, env):
    """The base model has never seen a chat or our reserved tokens, so give it the
    task in the plain-text form it knows (quoted dialogue, as in the story), with
    one solved example of the same kind (all that fits in 128 characters)."""
    tok, fn = env["tok"], torch_logits_fn(model)
    quote = tok.stoi['"']
    by_attr = {}
    for r in env["train"]:
        by_attr.setdefault(r["attribute"], []).append(r)
    rows = env["train"] + env["test"]
    prompts = []
    for i, r in enumerate(rows):
        shots = [s for s in by_attr[r["attribute"]] if s["entity"] != r["entity"]]
        shot = shots[i % len(shots)]
        text = (f'"{shot["messages"][0]["content"]}"\n"{shot["messages"][1]["content"]}"\n'
                f'"{r["messages"][0]["content"]}"\n"')
        prompts.append(tok.encode(text))
    gen = ln.greedy_generate(fn, prompts, quote, max_new=40)
    answers = [(tok.decode(g[:g.index(quote)]), True) if quote in g else (tok.decode(g), False)
               for g in gen]
    scores = ln.score(rows, answers)
    return dict(scores=scores, answers=answers[len(env["train"]):],
                story_train=story_loss(model, env["story_train"]),
                story_val=story_loss(model, env["story_val"]))


def delta_spectra(base, tuned):
    """Singular values of ΔW = W_tuned - W_base for every Linear matrix of the blocks."""
    out = {}
    sb, st = base.state_dict(), tuned.state_dict()
    for k in sb:
        if k.startswith("blocks.") and k.endswith("weight") and sb[k].dim() == 2:
            out[k] = torch.linalg.svdvals((st[k] - sb[k]).double()).numpy()
    return out


def energy_rank(s, frac=0.9):
    """How many singular directions hold `frac` of ΔW's squared Frobenius norm."""
    e = np.cumsum(s ** 2) / np.sum(s ** 2)
    return int(np.searchsorted(e, frac) + 1)


def pct(x):
    return f"{100 * x:5.1f}%"


def print_table(rows):
    hdr = (f"{'':22s}{'trainable':>10s}{'time':>7s}{'train':>8s}{'para.':>8s}{'para.':>8s}"
           f"{'unseen':>8s}{'unseen':>8s}{'out of':>8s}{'unknown':>9s}{'story':>8s}{'story':>8s}")
    sub = (f"{'':22s}{'params':>10s}{'(s)':>7s}{'exact':>8s}{'exact':>8s}{'format':>8s}"
           f"{'exact':>8s}{'format':>8s}{'scope':>8s}{'place':>9s}{'train':>8s}{'val':>8s}")
    print(hdr + "\n" + sub)
    for name, r in rows:
        s = r["scores"]
        print(f"{name:22s}{r['params']:>10,}{r['secs']:7.1f}{pct(s['train']['exact']):>8s}"
              f"{pct(s['paraphrase']['exact']):>8s}{pct(s['paraphrase']['format']):>8s}"
              f"{pct(s['unseen_fact']['exact']):>8s}{pct(s['unseen_fact']['format']):>8s}"
              f"{pct(s['out_of_scope']['exact']):>8s}{pct(s['unknown_place']['exact']):>9s}"
              f"{r['story_train']:8.3f}{r['story_val']:8.3f}")


def sft_row(name, model, env, data, args, lr, replay=None, log=True):
    if log:
        print(f"\n{name}: {n_trainable(model):,} trainable params, {args.sft_steps} steps x "
              f"{args.batch_size} chats, lr {lr}"
              + (f", + {replay[1]} story windows per step" if replay else ""))
    t0 = time.time()
    curve = train_sft(model, data, args.sft_steps, lr, args.batch_size, replay=replay,
                      log_every=args.sft_steps // 2 if log else 0)
    return dict(evaluate(model, env), params=n_trainable(model), secs=time.time() - t0,
                curve=curve)


def dpo_row(name, sft_model, pairs, env, args, nll, log=True):
    """policy = SFT model + fresh LoRA; reference = a frozen copy of the SFT model."""
    ref_model = copy.deepcopy(sft_model)
    for p in ref_model.parameters():
        p.requires_grad_(False)
    policy = apply_lora(copy.deepcopy(sft_model), args.rank, args.alpha, args.targets, seed=1)
    if log:
        print(f"\n{name}: {len(pairs)} pairs, beta {args.beta}, lr {args.dpo_lr}, "
              f"{args.dpo_steps} steps, NLL weight {nll:g}; policy = SFT LoRA model (merged) "
              f"+ fresh LoRA ({n_trainable(policy):,} params), reference = frozen copy")
    t0 = time.time()
    hist = train_dpo(policy, ref_model, pairs, args.dpo_steps, args.dpo_lr, args.beta,
                     args.batch_size, nll_coef=nll, log_every=args.dpo_steps // 3 if log else 0)
    return dict(evaluate(policy, env), params=n_trainable(policy), secs=time.time() - t0,
                dpo=hist)


def run_experiment(args, log=True):
    """All rows of the results table. Returns (env, base, full-FT model, results)."""
    env = setup()
    tok = env["tok"]
    sft_data = ln.sft_arrays(env["train"], tok)
    pairs = ln.pref_arrays(env["prefs"], tok)
    base, base_secs = get_base(env, log)
    n_all = sum(p.numel() for p in base.parameters())
    results = {}

    t0 = time.time()
    r = few_shot_eval(base, env)
    results["base, 1-shot prompt"] = dict(r, params=0, secs=time.time() - t0)
    if log:
        print(f"base model: {n_all:,} params ({BASE_CFG['n_layer']} layers, d_model "
              f"{BASE_CFG['d_model']}, block {BASE_CFG['block_size']}), pretrained in "
              f"{base_secs:.0f} s; story loss train {r['story_train']:.3f} / held-out "
              f"{r['story_val']:.3f} (uniform: ln({len(tok.chars)}) = "
              f"{math.log(len(tok.chars)):.3f})")
        for row, (a, _) in list(zip(env["test"], r["answers"]))[:2]:
            print(f"  1-shot: {row['messages'][0]['content']!r} -> {a[:40]!r}")

    torch.manual_seed(0)
    full = copy.deepcopy(base)
    results["SFT full"] = sft_row("(a) SFT, full fine-tuning", full, env, sft_data, args,
                                  args.full_lr, log=log)
    torch.manual_seed(0)
    results["SFT full + replay"] = sft_row(
        "(a') SFT, full fine-tuning + replay of pretraining text", copy.deepcopy(base), env,
        sft_data, args, args.full_lr, replay=(env["story_train"], args.replay), log=log)
    torch.manual_seed(0)
    lora = apply_lora(copy.deepcopy(base), args.rank, args.alpha, args.targets)
    name = f"SFT LoRA r={args.rank}"
    results[name] = sft_row(f"(b) SFT, LoRA r={args.rank} alpha={args.alpha:g} on "
                            f"{','.join(args.targets)} ({n_trainable(lora) / n_all:.1%} of the "
                            f"weights)", lora, env, sft_data, args, args.lora_lr, log=log)
    sft_model = merge_and_unload(lora)
    results["LoRA SFT + DPO"] = dpo_row("(c) DPO", sft_model, pairs, env, args, 0.0, log)
    results["LoRA SFT + DPO+NLL"] = dpo_row("(c') DPO + NLL on the chosen answers (RPO)",
                                            sft_model, pairs, env, args, args.nll, log)
    return env, base, full, sft_model, results


def refusal_rate(answers, rows, split):
    sel = [a for (a, _), r in zip(answers, rows) if r["split"] == split]
    return sum(a in (dd.REFUSAL, dd.NOT_IN_HANDBOOK) for a in sel) / len(sel)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=16.0)
    ap.add_argument("--targets", default=",".join(ln.ALL_TARGETS))
    ap.add_argument("--sft-steps", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--full-lr", type=float, default=1e-3)
    ap.add_argument("--lora-lr", type=float, default=3e-3)
    ap.add_argument("--replay", type=int, default=4,
                    help="story windows mixed into every SFT step of the replay row")
    ap.add_argument("--dpo-steps", type=int, default=150)
    ap.add_argument("--dpo-lr", type=float, default=1e-3)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--nll", type=float, default=1.0,
                    help="weight of the NLL loss on the chosen answers in the DPO+NLL row")
    ap.add_argument("--no-check", action="store_true", help="skip part 1")
    args = ap.parse_args()
    args.targets = tuple(args.targets.split(","))

    t_start = time.time()
    torch.set_num_threads(THREADS)
    if not args.no_check:
        print("Part 1: our PyTorch LoRA vs lora_numpy.py (float64, random A and B)")
        check_against_numpy()
    print("Part 2: fine-tuning on the lighthouse handbook (domain_data.py)\n")
    env, base, full, sft_lora, results = run_experiment(args)

    print(f"\nResults on held-out questions (greedy decoding). exact = the whole answer is "
          f"right; format = the right\nkind of answer, then <|end|>; out of scope / unknown "
          f"place = gives the right refusal; story = loss\nper char on the pretraining text "
          f"(train part / held-out part): the forgetting probe.\n")
    print_table([(k, v) for k, v in results.items()])
    print("\nHow often each model refuses (any refusal), per test split:")
    for k, v in results.items():
        print(f"  {k:22s} " + "  ".join(
            f"{sp} {pct(refusal_rate(v['answers'], env['test'], sp))}"
            for sp in ("paraphrase", "unseen_fact", "out_of_scope", "unknown_place")))

    spectra_full = delta_spectra(base, full)
    spectra_lora = delta_spectra(base, sft_lora)
    k0 = "blocks.0.attn.c_attn.weight"
    s = spectra_full[k0]
    ranks = [energy_rank(v) for v in spectra_full.values()]
    rnd = torch.linalg.svdvals(torch.randn(full.state_dict()[k0].shape, dtype=torch.float64,
                                           generator=torch.Generator().manual_seed(0))).numpy()
    print(f"\nIs the full fine-tune's update low-rank? ΔW = W_after - W_before of {k0} "
          f"{tuple(full.state_dict()[k0].shape)}:\n  top singular values "
          + " ".join(f"{x:.3f}" for x in s[:6]) + f" ... smallest {s[-1]:.4f}\n"
          f"  {energy_rank(s)} of {len(s)} directions hold 90% of ‖ΔW‖² (a random Gaussian "
          f"matrix of that shape needs {energy_rank(rnd)}); median over all {len(ranks)} block "
          f"matrices: {int(np.median(ranks))}. LoRA r={args.rank}'s ΔW has rank <= "
          f"{args.rank * (3 if k0.endswith('c_attn.weight') else 1)} by construction "
          f"(one rank-{args.rank} adapter per q, k, v).")

    print("\nSome held-out answers:")
    names = list(results)
    shown = set()
    for i, row in enumerate(env["test"]):
        if row["split"] in shown:
            continue
        shown.add(row["split"])
        print(f"  [{row['split']}] {row['messages'][0]['content']!r} (gold: "
              f"{row['messages'][1]['content']!r})")
        for n in names:
            a, ok = results[n]["answers"][i]
            print(f"      {n:22s} -> {a[:44]!r}{'' if ok else ' (no stop)'}")

    keep = ("scores", "params", "secs", "curve", "dpo", "story_train", "story_val")
    save = {"results": json.dumps({k: {kk: vv for kk, vv in v.items() if kk in keep}
                                   for k, v in results.items()}),
            "args": json.dumps(vars(args)), "random_svals": rnd}
    save.update({f"spectra_full.{k}": v for k, v in spectra_full.items()})
    save.update({f"spectra_lora.{k}": v for k, v in spectra_lora.items()})
    np.savez(RESULTS_PATH, **save)
    print(f"\nsaved curves and spectra to {os.path.relpath(RESULTS_PATH)}; "
          f"total {time.time() - t_start:.0f} s")


if __name__ == "__main__":
    main()
