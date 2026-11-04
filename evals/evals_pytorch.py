"""
Batched log-likelihood evaluation in PyTorch: the "loglikelihood" request of
lm-evaluation-harness (Gao et al.), behind every multiple-choice benchmark
(MMLU, HellaSwag, ARC, ...): no generation, just "how likely is each answer?".
(README section 8.)

1. EQUIVALENCE. Copy the NumPy mini-GPT (models.py "gpt") into the PyTorch GPT of
   ../transformer/transformer_pytorch.py (float64). Score every (context, choice)
   of tasks/mc.jsonl with ONE padded batch, and check that every per-character
   log-prob equals the NumPy scoring used by the harness (models.GPTModel.logprobs).
2. THE lm-eval WAY. One row per request: context + continuation, left-truncated to
   the model's window, right-padded, a mask selects the continuation positions:
       inp = (ctx + cont)[-(T+1):][:-1]     targets = (ctx + cont)[-(T+1):][1:]
   T times fewer rows than one-window-per-character, at the price of LESS context
   for the first continuation characters. We measure both effects.
3. SCORING RULES for multiple choice, on the same log-probs:
       sum      Σ log p(choice chars)          lm-eval "acc"; penalizes long choices
       mean     Σ log p / len(choice)          lm-eval "acc_norm" (length in characters)
       pmi      Σ log p(choice | ctx) - Σ log p(choice | neutral ctx)
                (how much the CONTEXT raised the choice's probability)
       greedy   is the choice exactly what greedy decoding would produce?
4. PERPLEXITY WINDOWS. bits/char of the held-out text with non-overlapping
   windows (what ../transformer/transformer_numpy.py's evaluate() does) vs a
   sliding window with stride s: same model, same text, different number.

Run (~10 s; trains the models first if evals/runs/models/ is empty):
    python evals/evals_pytorch.py
"""
from __future__ import annotations

import json
import math
import os
import time
import warnings

import numpy as np
import torch
import torch.nn.functional as F

from models import get_model, load_story, split_offset
from transformer_pytorch import GPT as TorchGPT, copy_numpy_weights

warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)

HERE = os.path.dirname(os.path.abspath(__file__))
NEUTRAL = "\n\n"  # "no context": the start of a paragraph


def to_torch(np_model):
    """The PyTorch twin of a models.GPTModel, float64, eval mode."""
    cfg = np_model.gpt.config
    pt = TorchGPT(cfg["vocab_size"], cfg["n_layer"], cfg["n_head"], cfg["d_model"],
                  cfg["block_size"], cfg["tie_weights"]).double().eval()
    copy_numpy_weights(np_model.gpt, pt)
    return pt


def load_mc():
    with open(os.path.join(HERE, "tasks", "mc.jsonl"), encoding="utf-8") as f:
        return [json.loads(line) for line in f]


@torch.no_grad()
def loglik_rolling(pt, stoi, requests, T):
    """Every continuation char gets its OWN window: the T chars right before it.
    All windows of all requests go in one right-padded batch; with a causal mask,
    padding AFTER a window's last token cannot change that token's logits.
    Returns a list of per-char log-prob arrays (same as models.GPTModel.logprobs)."""
    windows, targets, owner = [], [], []
    for r, (ctx, cont) in enumerate(requests):
        ids = [stoi[c] for c in ctx + cont]
        for i in range(len(ctx), len(ids)):
            windows.append(ids[max(0, i - T):i])
            targets.append(ids[i])
            owner.append(r)
    L = torch.tensor([len(w) for w in windows])
    x = torch.zeros(len(windows), T, dtype=torch.long)
    for j, w in enumerate(windows):
        x[j, :len(w)] = torch.tensor(w)
    logp = F.log_softmax(pt(x), dim=-1)  # (N, T, V)
    lp = logp[torch.arange(len(windows)), L - 1, torch.tensor(targets)].numpy()
    out = [[] for _ in requests]
    for j, r in enumerate(owner):
        out[r].append(lp[j])
    return [np.array(o) for o in out]


@torch.no_grad()
def loglik_single_window(pt, stoi, requests, T):
    """lm-evaluation-harness style: ONE row per request. Returns (per-char log-probs,
    is_greedy) for each request."""
    rows, conts = [], []
    for ctx, cont in requests:
        whole = [stoi[c] for c in ctx + cont][-(T + 1):]
        assert len(cont) <= T, "continuation longer than the window"
        rows.append(whole)
        conts.append(len(cont))
    order = sorted(range(len(rows)), key=lambda i: -len(rows[i]))  # longest first, as lm-eval
    x = torch.zeros(len(rows), T, dtype=torch.long)
    y = torch.zeros(len(rows), T, dtype=torch.long)
    mask = torch.zeros(len(rows), T, dtype=torch.bool)
    for b, i in enumerate(order):
        w = rows[i]
        x[b, :len(w) - 1] = torch.tensor(w[:-1])
        y[b, :len(w) - 1] = torch.tensor(w[1:])
        mask[b, len(w) - 1 - conts[i]:len(w) - 1] = True  # only the continuation is scored
    logp = F.log_softmax(pt(x), dim=-1)
    tok = logp.gather(-1, y[..., None])[..., 0]  # (B, T) log p of each target
    greedy = logp.argmax(-1) == y
    out, is_greedy = [None] * len(rows), [None] * len(rows)
    for b, i in enumerate(order):
        out[i] = tok[b][mask[b]].numpy()
        is_greedy[i] = bool(greedy[b][mask[b]].all())
    return out, is_greedy


def mc_requests(items):
    reqs, owner = [], []
    for k, it in enumerate(items):
        for c in it["choices"]:
            reqs.append((it["context"], c))
            owner.append(k)
    return reqs, owner


def score_rules(items, lp, lp_neutral, greedy):
    """Accuracy of each scoring rule, and which items each one gets right."""
    right = {name: [] for name in ("sum", "mean", "pmi", "greedy")}
    j = 0
    for it in items:
        n = len(it["choices"])
        s = np.array([lp[j + c].sum() for c in range(n)])
        m = np.array([lp[j + c].mean() for c in range(n)])
        u = np.array([lp_neutral[j + c].sum() for c in range(n)])
        g = [greedy[j + c] for c in range(n)]
        a = it["answer"]
        right["sum"].append(int(s.argmax()) == a)
        right["mean"].append(int(m.argmax()) == a)
        right["pmi"].append(int((s - u).argmax()) == a)
        right["greedy"].append(g[a] and sum(g) == 1)
        j += n
    return right


@torch.no_grad()
def bpc_with_stride(pt, ids, first, T, stride):
    """bits/char of ids[first:] scored with windows of T tokens that start every
    `stride` tokens (HF-style strided perplexity): each target is scored once, by
    the first window that contains it, so it sees between T - stride + 1 and T
    tokens of context. stride = T is non-overlapping chunks (the first char of
    each chunk sees 1 token); stride = 1 gives every target the full window."""
    windows, targets = [], []
    for t in range(first, len(ids)):
        b = max(0, stride * math.ceil((t - T) / stride))
        windows.append(ids[b:t])
        targets.append(ids[t])
    L = torch.tensor([len(w) for w in windows])
    x = torch.zeros(len(windows), T, dtype=torch.long)
    for j, w in enumerate(windows):
        x[j, :len(w)] = torch.tensor(w)
    logp = F.log_softmax(pt(x), dim=-1)
    lp = logp[torch.arange(len(windows)), L - 1, torch.tensor(targets)]
    return -float(lp.sum()) / math.log(2) / len(targets), float(L.float().mean())


def main():
    torch.set_num_threads(min(4, torch.get_num_threads()))
    items = load_mc()
    reqs, owner = mc_requests(items)

    print("Part 1: PyTorch batched scoring == the NumPy harness scoring (float64)")
    np_model = get_model("gpt")
    pt = to_torch(np_model)
    T, stoi = np_model.T, np_model.stoi
    t0 = time.perf_counter()
    lp_pt = loglik_rolling(pt, stoi, reqs, T)
    t_pt = time.perf_counter() - t0
    t0 = time.perf_counter()
    lp_np = [np.array(np_model.logprobs(c, x)) for c, x in reqs]
    t_np = time.perf_counter() - t0
    diff = max(np.abs(a - b).max() for a, b in zip(lp_pt, lp_np))
    n_chars = sum(len(x) for x in lp_np)
    print(f"  {len(items)} questions x 4 choices = {len(reqs)} requests, {n_chars} scored chars")
    print(f"  max |log p_torch - log p_numpy| over all {n_chars} chars = {diff:.1e}")
    assert diff < 1e-10, diff
    print(f"  one padded PyTorch batch {t_pt * 1e3:.0f} ms vs NumPy one request at a time "
          f"{t_np * 1e3:.0f} ms ✔\n")

    print("Part 2: the lm-eval request layout (one row per request, left-truncated)")
    t0 = time.perf_counter()
    lp_sw, greedy = loglik_single_window(pt, stoi, reqs, T)
    t_sw = time.perf_counter() - t0
    short = [(c[-(T + 1 - len(x)):], x) for c, x in reqs]  # cut so ctx + cont fits in T + 1
    a_sw, _ = loglik_single_window(pt, stoi, short, T)
    a_roll = loglik_rolling(pt, stoi, short, T)
    d_fit = max(np.abs(u - v).max() for u, v in zip(a_sw, a_roll))
    assert d_fit < 1e-10, d_fit
    print(f"  {len(reqs)} rows instead of {n_chars}: {t_sw * 1e3:.0f} ms. When context + "
          f"continuation fit in T + 1 = {T + 1}")
    print(f"  chars, both layouts give every char the same context: max |diff| {d_fit:.1e} ✔")
    d = [abs(u.sum() - v.sum()) for u, v in zip(lp_sw, lp_pt)]
    print(f"  Our 48-char contexts do NOT fit: in one row, the k-th continuation char from "
          f"the end sees only")
    print(f"  T + 1 - k chars, so the choice's sum log p differs from per-char windows by up "
          f"to {max(d):.2f} nats")
    r_roll = score_rules(items, lp_pt, lp_pt, greedy)["sum"]
    r_sw = score_rules(items, lp_sw, lp_sw, greedy)["sum"]
    flips = sum(u != v for u, v in zip(r_roll, r_sw))
    print(f"  (mean {np.mean(d):.2f}); multiple-choice decisions that change: {flips} of "
          f"{len(items)}.\n")

    print("Part 3: scoring rules on tasks/mc.jsonl (accuracy, 95% CI half-width ±1.96·SE)")
    neutral = [(NEUTRAL, x) for _, x in reqs]
    header = f"  {'model':16s}" + "".join(f"{r:>16s}" for r in ("sum (acc)", "mean (acc_norm)",
                                                                 "pmi", "greedy"))
    print(header)
    per_model = {}
    for spec in ("gpt", "gpt@steps=300", "gpt-full"):
        m = np_model if spec == "gpt" else get_model(spec)
        p = pt if spec == "gpt" else to_torch(m)
        lp = loglik_rolling(p, stoi, reqs, T)
        lpn = loglik_rolling(p, stoi, neutral, T)
        _, g = loglik_single_window(p, stoi, reqs, T)
        right = score_rules(items, lp, lpn, g)
        per_model[spec] = right
        cells = []
        for rule in ("sum", "mean", "pmi", "greedy"):
            acc = float(np.mean(right[rule]))
            cells.append(f"{acc:6.3f} ±{1.96 * math.sqrt(acc * (1 - acc) / len(items)):.2f}")
        print(f"  {spec:16s}" + "".join(f"{c:>16s}" for c in cells))
    right = per_model["gpt@steps=300"]
    long_correct = [len(it["choices"][it["answer"]]) > max(len(c) for i, c in enumerate(it["choices"])
                                                           if i != it["answer"]) for it in items]
    print(f"\n  gpt@steps=300 on the {sum(long_correct)} questions whose right answer is the "
          f"LONGEST choice:")
    for rule in ("sum", "mean", "pmi"):
        acc_long = np.mean([r for r, lc in zip(right[rule], long_correct) if lc])
        acc_rest = np.mean([r for r, lc in zip(right[rule], long_correct) if not lc])
        print(f"    {rule:5s} {acc_long:.2f} on those, {acc_rest:.2f} on the other "
              f"{len(items) - sum(long_correct)}")
    print("  Sum log p charges every extra character, so long right answers lose; the mean")
    print("  removes that penalty but can favour long fluent wrong answers. No rule is free.\n")

    print("Part 4: one model, one text, different bits/char (window T = 32)")
    text = load_story()
    off = split_offset(text)
    ids = [stoi[c] for c in text]
    print(f"  {'model':16s} {'text':9s}" + "".join(f"{f'stride {st}':>20s}" for st in (T, T // 2, 1)))
    for spec in ("gpt", "gpt@steps=300"):
        p = pt if spec == "gpt" else to_torch(get_model(spec, verbose=False))
        for name, first, last in (("held-out", off, len(ids)), ("train", T, off)):
            cells = []
            for stride in (T, T // 2, 1):
                bpc, ctx = bpc_with_stride(p, ids[:last], first, T, stride)
                cells.append(f"{bpc:.3f} (ctx {ctx:4.1f})")
            print(f"  {spec:16s} {name:9s}" + "".join(f"{c:>20s}" for c in cells))
    print("  ctx = mean number of context chars per scored char. Same weights, same text:")
    print("  the number moves with the stride, so always report the stride you used.")


if __name__ == "__main__":
    main()
