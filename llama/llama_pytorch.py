"""
The same mini-LLaMA in PyTorch, then the experiments.

(a) A PyTorch module written the way real code does it: nn.RMSNorm, RoPE as a
    complex multiplication (torch.view_as_complex, as in Meta's reference code),
    grouped-query attention through F.scaled_dot_product_attention(enable_gqa=True)
    (falls back to repeat_interleave on older PyTorch), SwiGLU with F.silu, no
    biases. Copies the NumPy model's weights in and asserts identical logits AND
    autograd gradients == our hand-written backward, for MHA, GQA and MQA.
(b) Idiomatic training with a validation split: the GPT of
    ../transformer/transformer_pytorch.py vs this LLaMA with the SAME parameter
    count (the SwiGLU hidden size is chosen to match), the same batches, optimizer,
    schedule and dropout, 3 init/dropout seeds each. Prints a val-loss table.
(c) Ablation, one change at a time:
        GPT -> +RMSNorm -> +RoPE (no position table) -> +SwiGLU -> -biases -> +GQA
    with the val loss (mean ± std of 3 seeds) and the parameter count of every step.
    Row 0 is verified to be exactly ../transformer's GPT (same logits).
The 24 training runs of (b) and (c) are independent, so they run in parallel worker
processes (1 thread each, the fastest setting for a model this small).
(d) bf16 autocast on CPU: a smoke test of mixed precision (what dtype comes out,
    how close the losses stay, how fast it is here).

Results go to llama/llama_pytorch.npz for make_figures.py.

Run (~50 s with 9 workers):
    python llama/llama_pytorch.py
    python llama/llama_pytorch.py --skip-ablation --steps 3000
    python llama/llama_pytorch.py --workers 1      # everything in one process (~3 min)
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

from llama_numpy import DEFAULT_TEXT, LLaMA as NumpyLLaMA, ffn_hidden_size, load_text, lr_at
from transformer_pytorch import GPT as TorchGPT, batches, eval_loss, make_optimizer

warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)
HERE = os.path.dirname(os.path.abspath(__file__))


def _sdpa_has_gqa():
    try:
        q, k = torch.zeros(1, 2, 1, 2), torch.zeros(1, 1, 1, 2)
        F.scaled_dot_product_attention(q, k, k, enable_gqa=True)
        return True
    except (TypeError, RuntimeError):
        return False


HAS_GQA = _sdpa_has_gqa()  # PyTorch >= 2.5


# ----------------------------------------------------------------------------
# RoPE, the complex-number way (Meta's llama/model.py does exactly this)
# ----------------------------------------------------------------------------
def precompute_freqs_cis(hs, T, base=10000.0):
    """e^{j·m·θ_i} for m < T: a (T, hs/2) complex table."""
    theta = 1.0 / base ** (torch.arange(0, hs, 2, dtype=torch.float64) / hs)
    angles = torch.outer(torch.arange(T, dtype=torch.float64), theta)
    return torch.polar(torch.ones_like(angles), angles)


def apply_rotary_emb(x, freqs_cis):
    """x: (B, n, T, hs). View pairs (x[2i], x[2i+1]) as complex numbers, multiply by
    e^{j·m·θ_i}, view back. Computed in at least float32 (complex bf16 doesn't exist)."""
    T = x.shape[-2]
    xc = torch.view_as_complex(
        x.to(torch.promote_types(x.dtype, torch.float32)).reshape(*x.shape[:-1], -1, 2))
    out = torch.view_as_real(xc * freqs_cis[:T].to(xc.dtype)).flatten(-2)
    return out.type_as(x)


# ----------------------------------------------------------------------------
# One configurable model: LLaMA by default, GPT-2 with the right flags
# ----------------------------------------------------------------------------
class Attention(nn.Module):
    def __init__(self, d, n_head, n_kv_head, bias=False, dropout=0.0):
        super().__init__()
        self.n_head, self.n_kv_head, self.hs = n_head, n_kv_head, d // n_head
        self.wq = nn.Linear(d, n_head * self.hs, bias=bias)
        self.wk = nn.Linear(d, n_kv_head * self.hs, bias=bias)
        self.wv = nn.Linear(d, n_kv_head * self.hs, bias=bias)
        self.wo = nn.Linear(n_head * self.hs, d, bias=bias)
        self.dropout = dropout
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x, freqs_cis=None):
        B, T, C = x.shape
        q = self.wq(x).view(B, T, self.n_head, self.hs).transpose(1, 2)  # (B, nh, T, hs)
        k = self.wk(x).view(B, T, self.n_kv_head, self.hs).transpose(1, 2)  # (B, nkv, T, hs)
        v = self.wv(x).view(B, T, self.n_kv_head, self.hs).transpose(1, 2)
        if freqs_cis is not None:
            q, k = apply_rotary_emb(q, freqs_cis), apply_rotary_emb(k, freqs_cis)
        p = self.dropout if self.training else 0.0
        if self.n_kv_head == self.n_head:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=p)
        elif HAS_GQA:  # the kernel reads each K/V head for its whole group of query heads
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=p,
                                               enable_gqa=True)
        else:  # older PyTorch: materialize the copies (query head h -> KV head h // g)
            g = self.n_head // self.n_kv_head
            y = F.scaled_dot_product_attention(q, k.repeat_interleave(g, dim=1),
                                               v.repeat_interleave(g, dim=1),
                                               is_causal=True, dropout_p=p)
        return self.resid_drop(self.wo(y.transpose(1, 2).reshape(B, T, C)))


class FeedForward(nn.Module):
    def __init__(self, d, hidden, kind="swiglu", bias=False, dropout=0.0):
        super().__init__()
        self.kind = kind
        self.w1 = nn.Linear(d, hidden, bias=bias)
        self.w2 = nn.Linear(hidden, d, bias=bias)
        self.w3 = nn.Linear(d, hidden, bias=bias) if kind == "swiglu" else None
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        if self.kind == "swiglu":
            return self.drop(self.w2(F.silu(self.w1(x)) * self.w3(x)))
        return self.drop(self.w2(F.gelu(self.w1(x), approximate="tanh")))


def make_norm(kind, d, eps=1e-5):
    return nn.RMSNorm(d, eps=eps) if kind == "rms" else nn.LayerNorm(d, eps=eps)


class Block(nn.Module):
    def __init__(self, d, n_head, n_kv_head, ffn_hidden, norm, mlp, bias, dropout):
        super().__init__()
        self.attention_norm = make_norm(norm, d)
        self.attention = Attention(d, n_head, n_kv_head, bias, dropout)
        self.ffn_norm = make_norm(norm, d)
        self.feed_forward = FeedForward(d, ffn_hidden, mlp, bias, dropout)

    def forward(self, x, freqs_cis=None):
        x = x + self.attention(self.attention_norm(x), freqs_cis)
        return x + self.feed_forward(self.ffn_norm(x))


class LLaMA(nn.Module):
    """Defaults = LLaMA. GPT-2 = norm="layer", pos="learned", mlp="gelu", bias=True,
    n_kv_head=n_head, ffn_hidden=4·d (verified against ../transformer in part (c))."""

    def __init__(self, vocab_size, n_layer=2, n_head=4, n_kv_head=2, d_model=64, block_size=32,
                 ffn_hidden=None, rope_base=10000.0, tie_weights=False, dropout=0.0,
                 norm="rms", pos="rope", mlp="swiglu", bias=False):
        super().__init__()
        if ffn_hidden is None:
            ffn_hidden = ffn_hidden_size(d_model) if mlp == "swiglu" else 4 * d_model
        self.block_size = block_size
        self.tok_embeddings = nn.Embedding(vocab_size, d_model)
        self.wpe = nn.Embedding(block_size, d_model) if pos == "learned" else None
        self.drop = nn.Dropout(dropout)
        self.layers = nn.ModuleList([Block(d_model, n_head, n_kv_head, ffn_hidden, norm, mlp,
                                           bias, dropout) for _ in range(n_layer)])
        self.norm = make_norm(norm, d_model)
        self.output = nn.Linear(d_model, vocab_size, bias=False)
        if tie_weights:
            self.output.weight = self.tok_embeddings.weight
        freqs = precompute_freqs_cis(d_model // n_head, block_size, rope_base)
        self.register_buffer("freqs_cis", freqs if pos == "rope" else None, persistent=False)
        for name, p in self.named_parameters():  # same init as the NumPy models
            if p.dim() >= 2:
                resid = name.endswith(("wo.weight", "w2.weight"))
                nn.init.normal_(p, 0.0, 0.02 / math.sqrt(2 * n_layer) if resid else 0.02)
            elif name.endswith("bias"):
                nn.init.zeros_(p)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok_embeddings(idx)
        if self.wpe is not None:
            x = x + self.wpe(torch.arange(T, device=idx.device))
        x = self.drop(x)
        for blk in self.layers:
            x = blk(x, self.freqs_cis)
        logits = self.output(self.norm(x))
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


def n_params(model):
    return sum(p.numel() for p in model.parameters())


# ----------------------------------------------------------------------------
# (a) identical to our NumPy model, forward AND backward
# ----------------------------------------------------------------------------
def copy_numpy_weights(np_model, pt_model):
    """Our Linear stores W as (in, out); nn.Linear stores (out, in): transpose."""
    P = {k: torch.from_numpy(v) for k, v in np_model.params().items()}
    with torch.no_grad():
        for name, p in pt_model.named_parameters():
            key = name.replace(".weight", ".W") if P.get(name) is None else name
            src = P[key]
            p.copy_(src.T if key.endswith(".W") else src)


def torch_grad_for(name, pt_model):
    """The PyTorch gradient for one of our NumPy parameter names, in our layout."""
    params = dict(pt_model.named_parameters())
    if name.endswith(".W"):
        return params[name[:-2] + ".weight"].grad.T
    return params[name].grad


def check_against_numpy(n_kv_head, tie):
    V, B, T, d, nh = 11, 3, 8, 32, 4
    rng = np.random.default_rng(0)
    np_model = NumpyLLaMA(V, n_layer=2, n_head=nh, n_kv_head=n_kv_head, d_model=d,
                          block_size=T, tie_weights=tie, seed=0, dtype=np.float64)
    for name, p in np_model.params().items():  # non-trivial values everywhere
        p[...] = (1.0 + 0.3 * rng.standard_normal(p.shape) if p.ndim == 1
                  else 0.3 * rng.standard_normal(p.shape))
    idx = rng.integers(0, V, (B, T))
    targets = rng.integers(0, V, (B, T))
    np_logits = np_model.forward(idx)
    np_loss = np_model.forward(idx, targets)
    np_grads = np_model.backward()

    pt_model = LLaMA(V, n_layer=2, n_head=nh, n_kv_head=n_kv_head, d_model=d, block_size=T,
                     tie_weights=tie).double()
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
    kind = {nh: "MHA", 1: "MQA"}.get(n_kv_head, "GQA")
    print(f"  {kind} n_kv_head={n_kv_head} {'tied' if tie else 'untied'}: logits max |diff| = "
          f"{d_logits:.1e}, loss {loss.item():.6f} vs {np_loss:.6f}, all {len(np_grads)} "
          f"gradient tensors max |diff| = {worst:.1e}")


# ----------------------------------------------------------------------------
# (b) training
# ----------------------------------------------------------------------------
def fit(model, train_data, val_data, args, steps, eval_every, label=None, seed=0):
    """../transformer/transformer_pytorch.py's fit(), recording the val curve.
    The batch generator is seeded the same for every model: identical batches."""
    g = torch.Generator().manual_seed(seed)
    opt = make_optimizer(model, args.lr, args.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_at(s, 1.0, args.warmup, steps))
    t0 = time.time()
    hist = {"step": [], "train": [], "val": [], "time": []}
    for step in range(steps + 1):
        if step % eval_every == 0 or step == steps:
            tr = eval_loss(model, train_data, args.block_size)
            va = eval_loss(model, val_data, args.block_size)
            for k, v in zip(hist, (step, tr, va, time.time() - t0)):
                hist[k].append(v)
            if label:
                print(f"  {label} step {step:5d} | train {tr:.3f} | val {va:.3f} "
                      f"(ppl {math.exp(va):5.2f}) | {time.time() - t0:4.1f} s")
        if step == steps:
            break
        x, y = batches(train_data, args.batch_size, args.block_size, g)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
    return {k: np.array(v) for k, v in hist.items()}


def matched_ffn_hidden(target, V, d, L, nh, nkv, multiple=8):
    """SwiGLU hidden size that brings the LLaMA's parameter count closest to `target`."""
    base = n_params(LLaMA(V, L, nh, nkv, d, ffn_hidden=multiple)) - L * 3 * d * multiple
    return int(round((target - base) / (3 * d * L) / multiple)) * multiple


# ----------------------------------------------------------------------------
# (c) ablation
# ----------------------------------------------------------------------------
GPT_FLAGS = dict(norm="layer", pos="learned", mlp="gelu", bias=True)
ABLATION = [
    ("GPT-2 (../transformer)", dict(GPT_FLAGS, n_kv_head=4)),
    ("+ RMSNorm", dict(GPT_FLAGS, norm="rms", n_kv_head=4)),
    ("+ RoPE (no wpe)", dict(GPT_FLAGS, norm="rms", pos="rope", n_kv_head=4)),
    ("+ SwiGLU (8/3·d)", dict(norm="rms", pos="rope", mlp="swiglu", bias=True, n_kv_head=4)),
    ("- biases", dict(norm="rms", pos="rope", mlp="swiglu", bias=False, n_kv_head=4)),
    ("+ GQA (2 KV heads)", dict(norm="rms", pos="rope", mlp="swiglu", bias=False, n_kv_head=2)),
]


def check_row0_is_gpt(V):
    """Our configurable class with GPT flags computes exactly ../transformer's GPT."""
    torch.manual_seed(0)
    gpt = TorchGPT(V).double().eval()
    ours = LLaMA(V, n_kv_head=4, **GPT_FLAGS).double().eval()
    C = 64
    with torch.no_grad():
        ours.tok_embeddings.weight.copy_(gpt.wte.weight)
        ours.wpe.weight.copy_(gpt.wpe.weight)
        for a, b in zip(ours.layers, gpt.blocks):
            for dst, src in ((a.attention_norm, b.ln_1), (a.ffn_norm, b.ln_2)):
                dst.load_state_dict(src.state_dict())
            for i, lin in enumerate((a.attention.wq, a.attention.wk, a.attention.wv)):
                lin.weight.copy_(b.attn.c_attn.weight[i * C:(i + 1) * C])
                lin.bias.copy_(b.attn.c_attn.bias[i * C:(i + 1) * C])
            a.attention.wo.load_state_dict(b.attn.c_proj.state_dict())
            a.feed_forward.w1.load_state_dict(b.mlp.c_fc.state_dict())
            a.feed_forward.w2.load_state_dict(b.mlp.c_proj.state_dict())
        ours.norm.load_state_dict(gpt.ln_f.state_dict())
        ours.output.load_state_dict(gpt.lm_head.state_dict())
        idx = torch.randint(V, (2, 32))
        diff = (ours(idx) - gpt(idx)).abs().max().item()
    assert diff < 1e-12 and n_params(ours) == n_params(gpt), diff
    return diff, n_params(gpt)


def split_data(path, val_frac):
    _, chars, data = load_text(path)
    data = torch.from_numpy(data)
    n_val = int(len(data) * val_frac)
    return chars, data[:-n_val], data[-n_val:]  # the END of the story is held out


def run_job(job):
    """One training run (in a worker process, 1 thread: fastest for a model this small).
    job: model ("gpt" or "llama"), flags, seed (init + dropout), steps, eval_every, args."""
    torch.set_num_threads(1)
    args = job["args"]
    chars, train_data, val_data = split_data(args.file, args.val_frac)
    torch.manual_seed(job["seed"])  # right before building the model, as ../transformer does
    if job["model"] == "gpt":
        m = TorchGPT(len(chars), dropout=args.dropout)
    else:
        m = LLaMA(len(chars), dropout=args.dropout, **job["flags"])
    hist = fit(m, train_data, val_data, args, job["steps"], job["eval_every"])
    return dict(hist=hist, n_params=n_params(m),
                state=m.state_dict() if job.get("keep") else None)


def run_jobs(jobs, workers):
    """Independent runs in parallel processes; each is seeded, so results don't depend
    on the scheduling."""
    if workers <= 1:
        return [run_job(j) for j in jobs]
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(min(workers, len(jobs)), mp_context=mp.get_context("spawn")) as ex:
        return list(ex.map(run_job, jobs))


def run_ablation(V, args):
    print(f"\nPart (c): ablation, one change at a time ({args.ablation_steps} steps, "
          f"seeds {list(range(args.seeds))}, same batches, dropout {args.dropout})")
    diff, n = check_row0_is_gpt(V)
    print(f"  row 0 check: our class with GPT flags == ../transformer GPT ({n:,} params), "
          f"logits max |diff| = {diff:.1e}")
    jobs = [dict(model="llama", flags=flags, seed=seed, steps=args.ablation_steps,
                 eval_every=args.ablation_steps, args=args)
            for _, flags in ABLATION for seed in range(args.seeds)]
    t0 = time.time()
    out = run_jobs(jobs, args.workers)
    rows = []
    prev = None
    print(f"  {'':24s}{'params':>9s}{'val loss (mean ± std)':>24s}{'change':>9s}   per seed")
    for r, (name, _) in enumerate(ABLATION):
        res = out[r * args.seeds:(r + 1) * args.seeds]
        vals = np.array([x["hist"]["val"][-1] for x in res])
        rows.append((name, res[0]["n_params"], float(vals.mean()), float(vals.std())))
        change = "" if prev is None else f"{vals.mean() - prev:+.3f}"
        prev = vals.mean()
        print(f"  {name:24s}{rows[-1][1]:9,d}{rows[-1][2]:15.3f} ± {rows[-1][3]:.3f}{change:>9s}"
              f"   " + " ".join(f"{v:.3f}" for v in vals))
    print(f"  ({len(jobs)} runs in {time.time() - t0:.0f} s)")
    return rows


# ----------------------------------------------------------------------------
# (d) bf16 autocast
# ----------------------------------------------------------------------------
def bf16_demo(V, train_data, args, steps=150):
    print(f"\nPart (d): bf16 autocast on CPU ({steps} steps from the same init, same batches)")
    out = {}
    for use_bf16 in (False, True):
        torch.manual_seed(0)
        m = LLaMA(V)
        g = torch.Generator().manual_seed(0)
        opt = make_optimizer(m, args.lr, args.weight_decay)
        t0 = time.time()
        for step in range(steps):
            x, y = batches(train_data, args.batch_size, args.block_size, g)
            # autocast runs matmuls in bf16 and keeps weights, softmax/losses in fp32
            with torch.autocast("cpu", dtype=torch.bfloat16, enabled=use_bf16):
                logits, loss = m(x, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        dt = (time.time() - t0) / steps
        out[use_bf16] = (loss.item(), dt, logits.dtype, next(m.parameters()).grad.dtype)
    for k, label in ((False, "fp32"), (True, "bf16 autocast")):
        loss, dt, ldt, gdt = out[k]
        print(f"  {label:14s} final train loss {loss:.3f}   {dt * 1e3:5.1f} ms/step   "
              f"logits {str(ldt):15s} weight grads {gdt}")
    torch.manual_seed(0)
    m = LLaMA(V).eval()
    x, _ = batches(train_data, 8, args.block_size, torch.Generator().manual_seed(1))
    with torch.no_grad():
        ref = m(x)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            low = m(x).float()
    rel = ((low - ref).norm() / ref.norm()).item()
    print(f"  same weights, one forward: relative logit error of bf16 = {rel:.1e} "
          f"(bf16 keeps 8 bits of mantissa: ~3e-3 per number)")
    return out[False][0], out[True][0], out[False][1], out[True][1], rel


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", default=DEFAULT_TEXT)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--block-size", type=int, default=32)
    ap.add_argument("--steps", type=int, default=1500, help="part (b)")
    ap.add_argument("--ablation-steps", type=int, default=1000, help="part (c)")
    ap.add_argument("--seeds", type=int, default=3, help="init/dropout seeds per model")
    ap.add_argument("--workers", type=int, default=min(9, os.cpu_count() or 1),
                    help="parallel training processes (1 = run everything in this process)")
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--eval-every", type=int, default=100, help="part (b) val curve")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--skip-ablation", action="store_true")
    ap.add_argument("--save", default=os.path.join(HERE, "llama_pytorch.npz"))
    args = ap.parse_args()

    print(f"Part (a): PyTorch == our NumPy model (float64, random weights; "
          f"SDPA enable_gqa available: {HAS_GQA})")
    for nkv, tie in ((4, False), (2, False), (1, True)):
        check_against_numpy(nkv, tie)
    print("PyTorch logits and autograd gradients == our hand-written forward/backward ✔\n")

    chars, train_data, val_data = split_data(args.file, args.val_frac)
    text = load_text(args.file)[0]
    V = len(chars)
    target = n_params(TorchGPT(V))
    H = matched_ffn_hidden(target, V, 64, 2, 4, 2)
    print(f"Part (b): train on {len(train_data)} chars, validate on the last {len(val_data)}. "
          f"{args.steps} steps, batch {args.batch_size}, dropout {args.dropout}, same batches,\n"
          f"  init/dropout seeds {list(range(args.seeds))}, {args.workers} worker processes")
    jobs = [dict(model=kind, flags=dict(ffn_hidden=H), seed=seed, steps=args.steps,
                 eval_every=args.eval_every, args=args, keep=(kind == "llama" and seed == 0))
            for kind in ("gpt", "llama") for seed in range(args.seeds)]
    t0 = time.time()
    out = run_jobs(jobs, args.workers)
    runs = {"GPT": out[:args.seeds], "LLaMA": out[args.seeds:]}
    print(f"  GPT   (../transformer): {runs['GPT'][0]['n_params']:,} params")
    print(f"  LLaMA (GQA 4q/2kv, SwiGLU hidden {H} to match): "
          f"{runs['LLaMA'][0]['n_params']:,} params   ({len(jobs)} runs in "
          f"{time.time() - t0:.0f} s)\n")
    print(f"{'':8s}{'params':>9s}{'train loss':>12s}{'val loss':>16s}{'best val':>16s}"
          f"{'val ppl':>9s}   val per seed")
    for label, rs in runs.items():
        tr = np.array([r["hist"]["train"][-1] for r in rs])
        va = np.array([r["hist"]["val"][-1] for r in rs])
        best = np.array([r["hist"]["val"].min() for r in rs])
        print(f"{label:8s}{rs[0]['n_params']:9,d}{tr.mean():12.3f}{va.mean():9.3f} ± "
              f"{va.std():.3f}{best.mean():9.3f} ± {best.std():.3f}{math.exp(va.mean()):9.2f}"
              f"   " + " ".join(f"{v:.3f}" for v in va))
    print(f"{'uniform':8s}{'':9s}{math.log(V):12.3f}{math.log(V):9.3f}{'':23s}{V:9d}")

    llama = LLaMA(V, ffn_hidden=H)
    llama.load_state_dict(runs["LLaMA"][0]["state"])
    llama.eval()
    start = torch.tensor([[chars.index(text[0])]])
    torch.manual_seed(0)
    out_ids = llama.generate(start, 300, 0.8, 10)[0].tolist()
    print("\n===== LLaMA (seed 0) sample (temperature 0.8, top-k 10) =====")
    print("".join(chars[i] for i in out_ids))

    save = {f"{k}.{kk}": np.stack([r["hist"][kk] for r in rs])
            for k, rs in runs.items() for kk in rs[0]["hist"]}
    save["n_params"] = np.array([runs["GPT"][0]["n_params"], runs["LLaMA"][0]["n_params"]])
    if not args.skip_ablation:
        rows = run_ablation(V, args)
        save["ablation.names"] = np.array([r[0] for r in rows])
        save["ablation.params"] = np.array([r[1] for r in rows])
        save["ablation.val"] = np.array([r[2] for r in rows])
        save["ablation.spread"] = np.array([r[3] for r in rows])
        save["ablation.steps"] = np.array(args.ablation_steps)
    torch.set_num_threads(1)
    save["bf16"] = np.array(bf16_demo(V, train_data, args))
    np.savez(args.save, **save)
    print(f"\nSaved results to {os.path.relpath(args.save)} (used by make_figures.py)")


if __name__ == "__main__":
    main()
