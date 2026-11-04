"""
LoRA, the masked SFT loss and the DPO loss from scratch in NumPy, with
HAND-WRITTEN backward passes, applied to the NumPy mini-GPT of ../transformer/.

LoRA (Hu et al. 2021). Keep the pretrained weight W0 FROZEN and learn a low-rank
correction next to it:

    y = x W0 + b + (alpha / r) · (x A) B          x: (N, n_in)   W0: (n_in, n_out)
                                                  A: (n_in, r)   B: (r, n_out),  r << n_in
    ΔW = (alpha / r) · A B                        rank(ΔW) <= r

    init: A ~ N(0, 1/n_in) (random), B = 0  ->  ΔW = 0: the adapted model starts
          EXACTLY equal to the base model, and still gets a gradient (dB != 0).
    backward (dy = dL/dy, s = alpha / r, h = x A):
          dB = s · hᵀ dy          dA = s · xᵀ (dy Bᵀ)          dx = dy W0ᵀ + s · (dy Bᵀ) Aᵀ
          no dW0 at all: nothing to store for the frozen weight, no optimizer state.
    merge:  W0 <- W0 + s · A B  (zero extra inference cost);  unmerge: W0 <- W0 - s · A B

The GPT fuses W_q, W_k, W_v into one (C, 3C) matrix `c_attn`. LoRALinear therefore
supports column SLICES: one (A, B) pair per slice, e.g. q and v only (the original
LoRA paper's choice) adapt columns [0, C) and [2C, 3C) of c_attn.

Masked SFT loss (supervised fine-tuning on chats). Cross-entropy only where the
target is an assistant token (mask m_t = 1):
    L = Σ_t m_t · (-log p_t[y_t]) / Σ_t m_t           dlogits_t = m_t (p_t - onehot(y_t)) / Σ m

DPO (Rafailov et al. 2023). For a prompt x with a chosen answer y_w and a
rejected one y_l, with log π(y|x) = Σ_{t in answer} log p_t[y_t] (sequence log-prob):
    z = β · [(log π(y_w) - log π_ref(y_w)) - (log π(y_l) - log π_ref(y_l))]
    L = -log σ(z)                 dL/dlog π(y_w) = -β σ(-z),   dL/dlog π(y_l) = +β σ(-z)
    d log π(y) / dlogits_t = m_t (onehot(y_t) - p_t)

    optional: + nll · (masked CE of the chosen answers), so the chosen answers
    stay likely while the margin grows (README section 10)

main() runs the whole pipeline in NumPy: pretrain a mini-GPT on ../rnn/input.txt
(saved as finetuning/base_numpy.npz), SFT it on the lighthouse chats with LoRA
(only A and B train), merge, then DPO with a fresh LoRA whose reference model is
"the same weights with LoRA switched off". finetune_pytorch.py does the bigger
version (and full fine-tuning, replay, plain DPO vs DPO + NLL).

Run (~35 s the first time, ~20 s once base_numpy.npz exists):
    python finetuning/lora_numpy.py
    python finetuning/lora_numpy.py --rank 2 --targets q,v
"""
import argparse
import contextlib
import io
import json
import math
import os
import sys
import time
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "transformer"))
from transformer_numpy import (GPT, AdamW, Layer, Linear, clip_grad_norm,  # noqa: E402
                               cross_entropy, get_batch, load_model, lr_at, save_model,
                               softmax, train)

import domain_data as dd  # noqa: E402

warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)

ALL_TARGETS = ("q", "k", "v", "o", "fc", "proj")  # attention q/k/v/o, MLP in/out
BASE_CONFIG = dict(n_layer=2, n_head=4, d_model=64, block_size=128)
BASE_PATH = os.path.join(HERE, "base_numpy.npz")
LORA_PATH = os.path.join(HERE, "lora_numpy.npz")


# ----------------------------------------------------------------------------
# LoRA
# ----------------------------------------------------------------------------
class LoRALinear(Layer):
    """Wraps a frozen transformer_numpy.Linear. Trainable params: A, B per slice.

    slices: {suffix: (start, end)} output columns that get their own (A, B).
    Default: one slice over all columns, params named "A" and "B"."""

    def __init__(self, base, r, alpha, rng, slices=None):
        super().__init__()
        self.base, self.r, self.alpha = base, r, alpha
        self.scale = alpha / r
        n_in, n_out = base.params["W"].shape
        self.slices = slices or {"": (0, n_out)}
        dtype = base.params["W"].dtype
        for sfx, (s, e) in self.slices.items():
            # A random so dB is non-zero at step 0; B = 0 so the model starts unchanged.
            self.params["A" + sfx] = (rng.standard_normal((n_in, r)) / math.sqrt(n_in)).astype(dtype)
            self.params["B" + sfx] = np.zeros((r, e - s), dtype=dtype)
        self.merged, self.enabled = False, True

    def forward(self, x):
        self.x = x
        y = self.base.forward(x)  # x W0 + b (W0 already contains s·A B if merged)
        if self.merged or not self.enabled:
            return y
        self.h = {}
        for sfx, (s, e) in self.slices.items():
            h = x @ self.params["A" + sfx]  # (..., r): the low-rank bottleneck
            self.h[sfx] = h
            y[..., s:e] += self.scale * (h @ self.params["B" + sfx])
        return y

    def backward(self, dy):
        assert not self.merged, "unmerge before training"
        W0 = self.base.params["W"]
        dx = dy @ W0.T  # frozen W0: only its input gradient is needed, never dW0
        if not self.enabled:
            return dx
        n_in = W0.shape[0]
        x2 = self.x.reshape(-1, n_in)
        for sfx, (s, e) in self.slices.items():
            A, B = self.params["A" + sfx], self.params["B" + sfx]
            dys = dy[..., s:e].reshape(-1, e - s)
            h2 = self.h[sfx].reshape(-1, self.r)
            self.grads["B" + sfx] = self.scale * (h2.T @ dys)  # (r, N) @ (N, out)
            dh = self.scale * (dys @ B.T)  # (N, r)
            self.grads["A" + sfx] = x2.T @ dh  # (in, N) @ (N, r)
            dx += (dh @ A.T).reshape(dx.shape)
        return dx

    def delta_W(self):
        """The full (n_in, n_out) update this adapter adds to W0."""
        n_in, n_out = self.base.params["W"].shape
        dW = np.zeros((n_in, n_out), dtype=self.base.params["W"].dtype)
        for sfx, (s, e) in self.slices.items():
            dW[:, s:e] = self.scale * (self.params["A" + sfx] @ self.params["B" + sfx])
        return dW

    def merge(self):
        if not self.merged:
            self.base.params["W"] += self.delta_W()
            self.merged = True

    def unmerge(self):
        if self.merged:
            self.base.params["W"] -= self.delta_W()
            self.merged = False


def apply_lora(model, r, alpha, targets=ALL_TARGETS, seed=0):
    """Replace the chosen Linears of every block by LoRALinear wrappers (in place).
    targets: subset of q, k, v (slices of c_attn), o (attn c_proj), fc, proj (MLP)."""
    rng = np.random.default_rng(seed)
    C = model.config["d_model"]
    wrapped = []
    for blk in model.blocks:
        qkv = {f"_{n}": (i * C, (i + 1) * C) for i, n in enumerate("qkv") if n in targets}
        if qkv:
            blk.attn.c_attn = LoRALinear(blk.attn.c_attn, r, alpha, rng, qkv)
            wrapped.append(blk.attn.c_attn)
        if "o" in targets:
            blk.attn.c_proj = LoRALinear(blk.attn.c_proj, r, alpha, rng)
            wrapped.append(blk.attn.c_proj)
        if "fc" in targets:
            blk.mlp.c_fc = LoRALinear(blk.mlp.c_fc, r, alpha, rng)
            wrapped.append(blk.mlp.c_fc)
        if "proj" in targets:
            blk.mlp.c_proj = LoRALinear(blk.mlp.c_proj, r, alpha, rng)
            wrapped.append(blk.mlp.c_proj)
    return wrapped


def lora_layers(model):
    return [l for l in model.layers().values() if isinstance(l, LoRALinear)]


def lora_params(model):
    """Only the adapters' A and B: the ONLY thing the optimizer sees."""
    return {k: v for k, v in model.params().items()
            if k.rsplit(".", 1)[1].startswith(("A", "B"))}


def lora_grads(model):
    g = {}
    for name, layer in model.layers().items():
        if isinstance(layer, LoRALinear):
            g.update({f"{name}.{k}": layer.grads[k] for k in layer.params})
    return g


def set_lora_enabled(model, on):
    for l in lora_layers(model):
        l.enabled = on


def merge_and_unload(model):
    """Fold every adapter into its W0 and put the plain Linear back."""
    for blk in model.blocks:
        for mod, name in ((blk.attn, "c_attn"), (blk.attn, "c_proj"),
                          (blk.mlp, "c_fc"), (blk.mlp, "c_proj")):
            layer = getattr(mod, name)
            if isinstance(layer, LoRALinear):
                layer.merge()
                setattr(mod, name, layer.base)


def lora_param_count(n_in, n_out, r):
    """A (n_in, r) + B (r, n_out) vs the full n_in·n_out."""
    return r * (n_in + n_out)


# ----------------------------------------------------------------------------
# Losses
# ----------------------------------------------------------------------------
def masked_cross_entropy(logits, targets, mask):
    """Mean CE over positions with mask = 1 only. Returns (loss, dlogits)."""
    V = logits.shape[-1]
    z = logits.reshape(-1, V)
    t = targets.reshape(-1)
    m = mask.reshape(-1).astype(z.dtype)
    n = max(m.sum(), 1.0)
    p = softmax(z)
    logp = np.log(p[np.arange(len(t)), t])
    loss = -(m * logp).sum() / n
    dz = p
    dz[np.arange(len(t)), t] -= 1.0
    dz *= (m / n)[:, None]  # masked positions get exactly zero gradient
    return float(loss), dz.reshape(logits.shape)


def sequence_logprob(logits, targets, mask):
    """log π(y|x) = Σ_t mask_t log p_t[y_t] per sequence. Returns (logp (B,), probs)."""
    B, T, V = logits.shape
    p = softmax(logits)
    tok = np.take_along_axis(p, targets[..., None], axis=-1)[..., 0]  # (B, T)
    return (np.log(tok) * mask).sum(axis=1), p


def sequence_logprob_backward(dlogp, probs, targets, mask):
    """dL/dlogits from dL/dlog π (B,): d log p_t[y_t] / d z_t = onehot(y_t) - p_t."""
    d = -probs.copy()
    np.put_along_axis(d, targets[..., None],
                      np.take_along_axis(d, targets[..., None], axis=-1) + 1.0, axis=-1)
    return d * (mask * dlogp[:, None])[..., None]


def log_sigmoid(z):
    return -np.logaddexp(0.0, -z)


def dpo_loss(pi_w, pi_l, ref_w, ref_l, beta):
    """Mean DPO loss over pairs and its gradient w.r.t. the policy log-probs.

    The implicit reward of an answer is β·(log π - log π_ref): how much MORE likely
    the policy makes it than the reference does."""
    r_w, r_l = beta * (pi_w - ref_w), beta * (pi_l - ref_l)
    z = r_w - r_l
    loss = float(-log_sigmoid(z).mean())
    dz = -np.exp(log_sigmoid(-z)) / len(z)  # d(-log σ(z))/dz = -σ(-z)
    stats = dict(reward_chosen=float(r_w.mean()), reward_rejected=float(r_l.mean()),
                 margin=float(z.mean()), accuracy=float((z > 0).mean()))
    return loss, beta * dz, -beta * dz, stats


# ----------------------------------------------------------------------------
# Data -> arrays, generation, evaluation (shared with finetune_pytorch.py)
# ----------------------------------------------------------------------------
class CharTokenizer:
    def __init__(self, chars):
        self.chars = list(chars)
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.end_id = self.stoi[dd.END]

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, ids):
        return "".join(self.chars[i] for i in ids)


def pad_batch(seqs, masks, pad_id):
    """Right-pad to the longest: idx = seq[:-1], targets = seq[1:], mask on targets."""
    L = max(len(s) for s in seqs)
    ids = np.full((len(seqs), L), pad_id, dtype=np.int64)
    m = np.zeros((len(seqs), L), dtype=np.float64)
    for i, (s, mk) in enumerate(zip(seqs, masks)):
        ids[i, :len(s)] = s
        m[i, :len(s)] = mk
    return ids[:, :-1], ids[:, 1:], m[:, 1:]  # pads have mask 0 -> never trained


def sft_arrays(rows, tok):
    """Every chat -> (ids, mask) with mask = 1 on the assistant characters + <|end|>."""
    out = []
    for r in rows:
        s, m = dd.render_with_mask(r["messages"])
        out.append((tok.encode(s), m))
    return out


def pref_arrays(prefs, tok):
    """Each pair -> (chosen ids, chosen mask, rejected ids, rejected mask)."""
    out = []
    for p in prefs:
        pair = []
        for ans in (p["chosen"], p["rejected"]):
            s, m = dd.render_with_mask(p["prompt"] + [{"role": "assistant", "content": ans}])
            pair += [tok.encode(s), m]
        out.append(tuple(pair))
    return out


def greedy_generate(logits_fn, prompts, end_id, max_new=40, block_size=128):
    """Batched greedy decoding of many prompts of different lengths.

    Right-padded buffer: because attention is causal, the garbage after a row's
    current length never influences the logits we read at its last position.
    logits_fn(ids (N, T) int array) -> (N, T, V) numpy logits."""
    N = len(prompts)
    lens = np.array([len(p) for p in prompts])
    buf = np.full((N, min(block_size, lens.max() + max_new)), end_id, dtype=np.int64)
    for i, p in enumerate(prompts):
        buf[i, :len(p)] = p
    out = [[] for _ in range(N)]
    done = np.zeros(N, dtype=bool)
    for _ in range(max_new):
        active = np.where(~done & (lens < buf.shape[1]))[0]
        if len(active) == 0:
            break
        T = int(lens[active].max())
        logits = logits_fn(buf[active, :T])
        nxt = logits[np.arange(len(active)), lens[active] - 1].argmax(-1)
        for i, t in zip(active, nxt):
            out[i].append(int(t))
            buf[i, lens[i]] = t
            lens[i] += 1
            if t == end_id:
                done[i] = True
    return out


def answers_from(gen_ids, tok):
    """Generated ids -> (answer text without <|end|>, stopped properly?)."""
    res = []
    for ids in gen_ids:
        if tok.end_id in ids:
            res.append((tok.decode(ids[:ids.index(tok.end_id)]), True))
        else:
            res.append((tok.decode(ids), False))
    return res


def score(rows, answers):
    """Exact match and format adherence per split. answers: [(text, stopped)].
    format = stopped with <|end|> and the answer has the right SHAPE for the question
    (any valid value; for the refusal splits, the refusal)."""
    by = {}
    for r, (text, stopped) in zip(rows, answers):
        gold = r["messages"][1]["content"]
        fmt = stopped and dd.format_ok(r, text)
        d = by.setdefault(r["split"], {"n": 0, "exact": 0, "format": 0})
        d["n"] += 1
        d["exact"] += int(stopped and text == gold)
        d["format"] += int(fmt)
    return {k: {"exact": v["exact"] / v["n"], "format": v["format"] / v["n"], "n": v["n"]}
            for k, v in by.items()}


def evaluate_chat(logits_fn, rows, tok, block_size=128):
    prompts = [tok.encode(dd.render(r["messages"][:1], add_generation_prompt=True)) for r in rows]
    answers = answers_from(greedy_generate(logits_fn, prompts, tok.end_id,
                                           block_size=block_size), tok)
    return score(rows, answers), answers


def story_loss(model, data, T=None):
    """Mean next-char loss over non-overlapping windows of the story (forgetting probe)."""
    T = T or model.config["block_size"]
    starts = np.arange(0, len(data) - T - 1, T)
    x = np.stack([data[i:i + T] for i in starts])
    y = np.stack([data[i + 1:i + T + 1] for i in starts])
    return cross_entropy(model.forward(x), y)[0]


# ----------------------------------------------------------------------------
# The NumPy pipeline: pretrain -> LoRA SFT -> merge -> LoRA DPO
# ----------------------------------------------------------------------------
def get_base(tok, story_ids, steps=1500, verbose=True):
    """The pretrained base mini-GPT (V = story chars + 3 reserved chat tokens).
    Loaded from finetuning/base_numpy.npz, or pretrained here (~15 s) and saved."""
    if os.path.exists(BASE_PATH):
        model, chars, _ = load_model(BASE_PATH)
        if chars == tok.chars and all(model.config[k] == v for k, v in BASE_CONFIG.items()):
            return model
    if verbose:
        print(f"pretraining the NumPy base model on ../rnn/input.txt ({steps} steps)...")
    with contextlib.redirect_stdout(io.StringIO()):
        model, hist = train(story_ids, len(tok.chars), batch_size=8, steps=steps,
                            verbose=False, **BASE_CONFIG)
    save_model(BASE_PATH, model, tok.chars)
    return model


def train_sft_lora(model, data, steps, lr, batch_size, seed, log_every=0):
    params = lora_params(model)
    opt = AdamW(params, weight_decay=0.0)
    rng = np.random.default_rng(seed)
    losses = []
    for step in range(steps):
        batch = [data[i] for i in rng.integers(0, len(data), batch_size)]
        x, y, m = pad_batch([b[0] for b in batch], [b[1] for b in batch], 0)
        loss, dlogits = masked_cross_entropy(model.forward(x), y, m)
        model.dlogits = dlogits
        model.backward()
        grads = lora_grads(model)
        clip_grad_norm(grads, 1.0)
        opt.step(grads, lr_at(step, lr, steps // 10, steps))
        losses.append(loss)
        if log_every and (step % log_every == 0 or step == steps - 1):
            print(f"  SFT step {step:4d} | masked loss {loss:.3f}")
    return losses


def ref_logprobs(model, pairs):
    """log π_ref(chosen), log π_ref(rejected) for every pair, computed ONCE."""
    out = []
    for p in pairs:
        x, y, m = pad_batch([p[0], p[2]], [p[1], p[3]], 0)
        lp, _ = sequence_logprob(model.forward(x), y, m)
        out.append(lp)
    return np.array(out)  # (n_pairs, 2)


def train_dpo(model, pairs, ref, steps, lr, beta, batch_size, seed, nll_coef=0.0, log_every=0):
    params = lora_params(model)
    opt = AdamW(params, weight_decay=0.0)
    rng = np.random.default_rng(seed)
    hist = []
    for step in range(steps):
        ix = rng.integers(0, len(pairs), batch_size)
        seqs = [pairs[i][0] for i in ix] + [pairs[i][2] for i in ix]  # chosen..., rejected...
        masks = [pairs[i][1] for i in ix] + [pairs[i][3] for i in ix]
        x, y, m = pad_batch(seqs, masks, 0)
        logp, probs = sequence_logprob(model.forward(x), y, m)
        loss, d_w, d_l, st = dpo_loss(logp[:batch_size], logp[batch_size:],
                                      ref[ix, 0], ref[ix, 1], beta)
        dlogits = sequence_logprob_backward(np.concatenate([d_w, d_l]), probs, y, m)
        if nll_coef:  # + nll_coef · (masked CE of the chosen answers): keeps them likely
            logits_w = np.log(probs[:batch_size])  # log-softmax, up to a per-row constant
            _, d_nll = masked_cross_entropy(logits_w, y[:batch_size], m[:batch_size])
            dlogits[:batch_size] += nll_coef * d_nll
        model.dlogits = dlogits
        model.backward()
        grads = lora_grads(model)
        clip_grad_norm(grads, 1.0)
        opt.step(grads, lr_at(step, lr, steps // 10, steps))
        hist.append(dict(loss=loss, **st))
        if log_every and (step % log_every == 0 or step == steps - 1):
            print(f"  DPO step {step:4d} | loss {loss:.3f} | margin {st['margin']:+.2f} | "
                  f"pref. accuracy {st['accuracy']:.2f}")
    return hist


SPLITS = ("train", "paraphrase", "unseen_fact", "out_of_scope", "unknown_place")


def table_row(name, model, rows, tok, story_ids):
    fwd = lambda ids: model.forward(ids)
    s = evaluate_chat(fwd, rows, tok)[0]
    cells = "".join(f"{100 * s[k]['exact']:9.1f}%{100 * s[k]['format']:6.1f}%" for k in SPLITS)
    print(f"{name:16s}{cells}{story_loss(model, story_ids):8.3f}")


def table_header():
    print(f"\n{'':16s}" + "".join(f"{k:>17s}" for k in SPLITS) + f"{'story':>8s}")
    print(f"{'':16s}" + "".join(f"{'exact':>10s}{'format':>7s}" for _ in SPLITS) + f"{'loss':>8s}")


DEFAULTS = dict(rank=8, alpha=16.0, targets=ALL_TARGETS, sft_steps=600, sft_lr=3e-3,
                dpo_steps=150, dpo_lr=1e-3, beta=0.5, nll=1.0)


def pipeline(rank=8, alpha=16.0, targets=ALL_TARGETS, sft_steps=600, sft_lr=3e-3,
             dpo_steps=150, dpo_lr=1e-3, beta=0.5, nll=1.0, log=True):
    """base -> LoRA SFT -> merge -> LoRA DPO. Saves both adapters to lora_numpy.npz.
    Returns (tok, story_ids, base model, SFT model, DPO model)."""
    say = print if log else (lambda *a, **k: None)
    story = dd.load_story()
    tok = CharTokenizer(dd.build_vocab(story))
    story_ids = np.array(tok.encode(story), dtype=np.int64)
    train_rows, _, prefs = dd.load_all()
    t0 = time.time()
    model = get_base(tok, story_ids, verbose=log)
    n_base = model.n_params()
    say(f"base model: {n_base:,} params, V = {len(tok.chars)} (44 story chars + 3 reserved), "
        f"story loss {story_loss(model, story_ids):.3f} ({time.time() - t0:.1f} s)")

    apply_lora(model, rank, alpha, targets)
    sft_adapter = lora_params(model)
    n_lora = sum(p.size for p in sft_adapter.values())
    say(f"\nLoRA r={rank} alpha={alpha:g} on {','.join(targets)}: "
        f"{n_lora:,} trainable params = {n_lora / n_base:.1%} of {n_base:,}")
    t0 = time.time()
    train_sft_lora(model, sft_arrays(train_rows, tok), sft_steps, sft_lr, 16, 0,
                   log_every=sft_steps // 3 if log else 0)
    say(f"  ({time.time() - t0:.1f} s)")
    probe = np.array([tok.encode(story[:64])])
    logits_before = model.forward(probe)
    merge_and_unload(model)
    diff = np.abs(model.forward(probe) - logits_before).max()
    say(f"merged the adapters into W0: max |logit diff| = {diff:.1e} (float32), "
        f"params back to {model.n_params():,}")
    sft = GPT(len(tok.chars), **BASE_CONFIG)
    for k, p in sft.params().items():
        p[...] = model.params()[k]

    apply_lora(model, rank, alpha, targets, seed=1)
    pairs = pref_arrays(prefs, tok)
    set_lora_enabled(model, False)  # adapters off = the SFT model = the reference
    ref = ref_logprobs(model, pairs)
    set_lora_enabled(model, True)
    say(f"\nDPO (+ {nll:g} x NLL on the chosen answers) with a fresh LoRA, beta={beta}, on "
        f"{len(pairs)} pairs.\nReference = the SFT model = the policy with its adapters "
        f"switched off: no second copy of the model needed.")
    t0 = time.time()
    train_dpo(model, pairs, ref, dpo_steps, dpo_lr, beta, 16, 0, nll_coef=nll,
              log_every=dpo_steps // 3 if log else 0)
    say(f"  ({time.time() - t0:.1f} s)")
    cfg = dict(r=rank, alpha=alpha, targets=list(targets), beta=beta)
    np.savez(LORA_PATH, config=np.array(json.dumps(cfg)),
             **{f"sft.{k}": v for k, v in sft_adapter.items()},
             **{f"dpo.{k}": v for k, v in lora_params(model).items()})
    say(f"\nsaved the SFT and DPO adapters ({n_lora:,} numbers each, vs {n_base:,} for a "
        f"full copy of the model) to {os.path.relpath(LORA_PATH)}")
    return tok, story_ids, get_base(tok, story_ids, verbose=False), sft, model


def load_pipeline():
    """(tok, story_ids, base, SFT model, DPO model) rebuilt from base_numpy.npz +
    lora_numpy.npz, or trained now with the defaults (~20-35 s)."""
    if os.path.exists(LORA_PATH) and os.path.exists(BASE_PATH):
        d = np.load(LORA_PATH)
        cfg = json.loads(str(d["config"]))
        story = dd.load_story()
        tok = CharTokenizer(dd.build_vocab(story))
        story_ids = np.array(tok.encode(story), dtype=np.int64)
        base = get_base(tok, story_ids, verbose=False)
        model = get_base(tok, story_ids, verbose=False)
        apply_lora(model, cfg["r"], cfg["alpha"], tuple(cfg["targets"]), seed=0)
        for k, p in lora_params(model).items():
            p[...] = d[f"sft.{k}"]
        merge_and_unload(model)
        sft = GPT(len(tok.chars), **BASE_CONFIG)  # the merged SFT weights (= DPO reference)
        for k, p in sft.params().items():
            p[...] = model.params()[k]
        apply_lora(model, cfg["r"], cfg["alpha"], tuple(cfg["targets"]), seed=1)
        for k, p in lora_params(model).items():
            p[...] = d[f"dpo.{k}"]
        return tok, story_ids, base, sft, model
    with contextlib.redirect_stdout(io.StringIO()):
        return pipeline(log=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rank", type=int, default=DEFAULTS["rank"])
    ap.add_argument("--alpha", type=float, default=DEFAULTS["alpha"])
    ap.add_argument("--targets", default=",".join(ALL_TARGETS),
                    help="comma list from q,k,v,o,fc,proj")
    ap.add_argument("--sft-steps", type=int, default=DEFAULTS["sft_steps"])
    ap.add_argument("--sft-lr", type=float, default=DEFAULTS["sft_lr"])
    ap.add_argument("--dpo-steps", type=int, default=DEFAULTS["dpo_steps"])
    ap.add_argument("--dpo-lr", type=float, default=DEFAULTS["dpo_lr"])
    ap.add_argument("--beta", type=float, default=DEFAULTS["beta"])
    ap.add_argument("--nll", type=float, default=DEFAULTS["nll"],
                    help="NLL weight on the chosen answers")
    args = ap.parse_args()
    tok, story_ids, base, sft, dpo = pipeline(
        args.rank, args.alpha, tuple(args.targets.split(",")), args.sft_steps, args.sft_lr,
        args.dpo_steps, args.dpo_lr, args.beta, args.nll)
    train_rows, test_rows, _ = dd.load_all()
    rows = train_rows + test_rows
    table_header()
    table_row("base (no chat)", base, rows, tok, story_ids)
    table_row("LoRA SFT", sft, rows, tok, story_ids)
    table_row("LoRA SFT + DPO", dpo, rows, tok, story_ids)


if __name__ == "__main__":
    main()
