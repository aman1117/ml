"""
Speculative decoding from scratch: a small DRAFT model guesses γ tokens, the big
TARGET model checks all of them in ONE forward pass, and a rejection-sampling rule
keeps the output distribution EXACTLY the target's (Leviathan et al. 2023,
Chen et al. 2023).

Why it works: a decode step is memory-bound (batching.py), so the target scoring
γ+1 tokens in one pass costs about the same as scoring 1. If the draft is often
right, we get several tokens per target pass.

The rule, for each drafted token x ~ q (draft) with target distribution p:
    accept x with probability min(1, p(x) / q(x));
    on the first rejection, sample from the residual  norm(max(0, p - q))  and stop;
    if all γ are accepted, sample one bonus token from the target's next p.
Then P(output = x) = q(x)·min(1, p(x)/q(x)) + P(reject)·residual(x) = p(x).

This file:
  1. trains a tiny draft GPT (1 layer, d = 32, ~5 s) on the same text as the target
     (the 2-layer mini-GPT from ../transformer/);
  2. checks the rule numerically (and shows that a naive rule is biased);
  3. proves greedy speculative == greedy target, token for token;
  4. a statistical test: the 1st-3rd generated characters after "The " over 20,000
     runs match the target's exact marginals (chi-square + TV distance);
  5. measures the acceptance rate and wall-clock speed-up vs γ.

Run (~60 s):
    python serving/speculative.py
"""
from __future__ import annotations

import argparse
import contextlib
import io
import math
import os
import time

import numpy as np

from common import (HERE, CharTokenizer, KVCache, decode_step, evaluate, generate_ids,
                    get_model, heldout_ids, load_model, lm_logits, prefill, save_model,
                    softmax, train, train_ids)

DRAFT_PATH = os.path.join(HERE, "draft_cache.npz")
DRAFT_CONFIG = dict(n_layer=1, n_head=2, d_model=32, block_size=32)


def get_draft(target, chars, verbose=True):
    """The draft model: loaded from serving/draft_cache.npz or trained (~5 s)."""
    if os.path.exists(DRAFT_PATH):
        m, c, _ = load_model(DRAFT_PATH, dtype=target.wte.params["weight"].dtype)
        if c == chars:
            return m
    tok = CharTokenizer(chars)
    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        m, hist = train(train_ids(tok), len(chars), steps=1500, verbose=False, **DRAFT_CONFIG)
    save_model(DRAFT_PATH, m, chars, hist)
    if verbose:
        print(f"(trained the draft model in {time.time() - t0:.1f} s -> "
              f"{os.path.relpath(DRAFT_PATH)})")
    return load_model(DRAFT_PATH, dtype=target.wte.params["weight"].dtype)[0]


def probs(logits, temperature, top_k=None):
    """The distribution sample_next() samples from (temperature 0 = one-hot argmax)."""
    z = np.asarray(logits, dtype=np.float64)
    if temperature == 0:
        p = np.zeros_like(z)
        p[int(np.argmax(z))] = 1.0
        return p
    z = z / temperature
    if top_k is not None and top_k < len(z):
        z = np.where(z < np.sort(z)[-top_k], -np.inf, z)
    return softmax(z)


def extend(model, tokens, cache):
    """Feed several tokens at once on top of a KV cache (one forward pass).
    Token i sits at position len(cache)+i and sees the cache and tokens 0..i.
    Returns logits (len(tokens), V); the cache grows by len(tokens)."""
    t0, k = len(cache), len(tokens)
    x = (model.wte.params["weight"][np.asarray(tokens)]
         + model.wpe.params["weight"][t0:t0 + k])[None]  # (1, k, C)
    mask = np.tril(np.ones((k, t0 + k), dtype=bool), k=t0)  # query i sees keys <= t0+i
    for l, blk in enumerate(model.blocks):
        a = blk.attn
        q, kk, v = np.split(a.c_attn.forward(blk.ln_1.forward(x)), 3, axis=-1)
        q, kk, v = a.split_heads(q)[0], a.split_heads(kk)[0], a.split_heads(v)[0]
        cache.k[l] = kk if cache.k[l] is None else np.concatenate([cache.k[l], kk], axis=1)
        cache.v[l] = v if cache.v[l] is None else np.concatenate([cache.v[l], v], axis=1)
        s = (q @ cache.k[l].transpose(0, 2, 1)) / math.sqrt(a.hs)  # (nh, k, t0+k)
        y = softmax(np.where(mask, s, -np.inf)) @ cache.v[l]
        x = x + a.c_proj.forward(a.merge_heads(y[None]))
        x = x + blk.mlp.forward(blk.ln_2.forward(x))
    return lm_logits(model, model.ln_f.forward(x))[0]


def truncate(cache, n):
    """Roll the cache back to its first n tokens (drop rejected drafts)."""
    cache.k = [k[:, :n] for k in cache.k]
    cache.v = [v[:, :n] for v in cache.v]


def accept_reject(ps, qs, drafts, rng):
    """The exact rule. ps: g+1 target distributions, qs: g draft distributions,
    drafts: g tokens with drafts[i] ~ qs[i]. Returns (k accepted, next token)."""
    for i, x in enumerate(drafts):
        if rng.random() < min(1.0, ps[i][x] / qs[i][x]):
            continue
        residual = np.maximum(ps[i] - qs[i], 0.0)
        total = residual.sum()
        residual = residual / total if total > 0 else ps[i]
        return i, int(rng.choice(len(residual), p=residual))
    return len(drafts), int(rng.choice(len(ps[-1]), p=ps[-1]))  # bonus token


class GPTDraft:
    """A small GPT as the draft, with its own KV cache."""

    def __init__(self, model):
        self.model, self.cache, self.calls = model, None, 0

    def reset(self, ids):
        self.cache = prefill(self.model, ids)[1]
        self.calls += 1

    def feed(self, tokens):
        """Append tokens; return the logits for the next one."""
        self.calls += 1
        return extend(self.model, tokens, self.cache)[-1]

    def truncate(self, n):
        truncate(self.cache, n)

    def __len__(self):
        return len(self.cache)


class NGramDraft:
    """A draft with no neural network: next-char counts of the training text after the
    longest matching context (up to n-1 chars), lightly smoothed. Costs microseconds."""

    def __init__(self, data, V, n=6, smooth=0.01):
        self.V, self.n, self.smooth = V, n, smooth
        self.table = {}
        for k in range(n):
            for i in range(k, len(data)):
                ctx = tuple(data[i - k:i])
                self.table.setdefault(ctx, np.zeros(V))[data[i]] += 1
        self.ctx, self.calls = [], 0

    def reset(self, ids):
        self.ctx = list(ids)

    def feed(self, tokens):
        self.calls += 1
        self.ctx += list(tokens)
        for k in range(self.n - 1, -1, -1):
            c = self.table.get(tuple(self.ctx[len(self.ctx) - k:]) if k else ())
            if c is not None:
                return np.log((c + self.smooth) / (c.sum() + self.smooth * self.V))

    def truncate(self, n):
        del self.ctx[n:]

    def __len__(self):
        return len(self.ctx)


def speculative_generate(target, draft, ids, n, gamma=4, temperature=0.8, top_k=None,
                         rng=None, stats=None):
    """n new tokens from the TARGET's distribution, with exactly generate.py's window
    semantics (prefill the last T tokens; when the window is full, keep the last T/2).
    `draft` is a GPT (wrapped in GPTDraft) or any object with reset/feed/truncate.

    State between iterations: the target cache holds ids[ws:-1] and the newest
    token ("pending") is not fed yet. One iteration = the draft feeds pending and
    proposes g tokens; ONE target forward over [pending, x_1..x_g] gives the g+1
    distributions p_1..p_{g+1}; accept/reject; roll the caches back."""
    rng = rng or np.random.default_rng()
    stats = stats if stats is not None else {}
    for key in ("target_calls", "draft_calls", "proposed", "accepted", "iterations"):
        stats.setdefault(key, 0)
    if not hasattr(draft, "feed"):
        draft = GPTDraft(draft)
    calls0 = draft.calls
    T = target.config["block_size"]
    ids = list(ids)
    logits, t_cache = prefill(target, ids[-T:])
    draft.reset(ids[-T:])
    stats["target_calls"] += 1
    out = [int(rng.choice(len(logits), p=probs(logits, temperature, top_k)))]
    ids.append(out[-1])
    d_feed = [out[-1]]  # tokens the draft has not seen yet
    while len(out) < n:
        c = len(t_cache)
        if c == T:  # the window is full: same slide as generate.py
            t_cache = prefill(target, ids[-(T // 2):-1])[1]
            draft.reset(ids[-(T // 2):-1])
            d_feed = [ids[-1]]
            stats["target_calls"] += 1
            c = len(t_cache)
        g = min(gamma, T - c - 1, n - len(out))  # never draft across a window slide
        drafts, qs = [], []
        dl = draft.feed(d_feed)
        for i in range(g):
            qs.append(probs(dl, temperature, top_k))
            drafts.append(int(rng.choice(len(qs[-1]), p=qs[-1])))
            if i < g - 1:
                dl = draft.feed([drafts[-1]])
        vl = extend(target, [ids[-1]] + drafts, t_cache)  # ONE target pass: g+1 positions
        stats["target_calls"] += 1
        ps = [probs(v, temperature, top_k) for v in vl]
        k, y = accept_reject(ps, qs, drafts, rng)
        stats["iterations"] += 1
        stats["proposed"] += g
        stats["accepted"] += k
        new = drafts[:k] + [y]
        out += new
        ids += new
        truncate(t_cache, c + 1 + k)  # keep pending + accepted drafts
        kd = min(len(draft), c + 1 + k)
        draft.truncate(kd)
        d_feed = ids[len(ids) - 1 - (c + 1 + k - kd):]  # what the draft still has to see
    stats["draft_calls"] += draft.calls - calls0
    return out[:n]


# ----------------------------------------------------------------------------
# Tests of the claim "same distribution as the target"
# ----------------------------------------------------------------------------
def chi2_critical(dof, z=3.09):
    """Wilson-Hilferty approximation of the chi-square quantile (z = 3.09: p = 0.001)."""
    return dof * (1 - 2 / (9 * dof) + z * math.sqrt(2 / (9 * dof))) ** 3


def chi_square(counts, expected_p, min_expected=5.0):
    """Pearson chi-square, merging bins with expected count < min_expected."""
    N = counts.sum()
    E = expected_p * N
    big = E >= min_expected
    O = np.append(counts[big], counts[~big].sum())
    Ee = np.append(E[big], E[~big].sum())
    keep = Ee > 0
    stat = float(((O[keep] - Ee[keep]) ** 2 / Ee[keep]).sum())
    dof = int(keep.sum()) - 1
    return stat, dof


def rule_test(V=10, N=200_000, seed=0):
    """Vectorized: draft x ~ q, apply the rule, histogram the result vs p.
    Also a NAIVE rule (accept if p(x) >= q(x) else resample from p) to see the bias."""
    rng = np.random.default_rng(seed)
    p = rng.dirichlet(np.ones(V) * 0.5)
    q = rng.dirichlet(np.ones(V) * 0.5)
    x = rng.choice(V, size=N, p=q)
    accept = rng.random(N) < np.minimum(1.0, p[x] / q[x])
    residual = np.maximum(p - q, 0)
    residual /= residual.sum()
    y = np.where(accept, x, rng.choice(V, size=N, p=residual))
    naive = np.where(p[x] >= q[x], x, rng.choice(V, size=N, p=p))
    out = {}
    for name, s in (("exact rule", y), ("naive rule", naive)):
        c = np.bincount(s, minlength=V).astype(float)
        stat, dof = chi_square(c, p)
        out[name] = dict(tv=0.5 * np.abs(c / N - p).sum(), chi2=stat, dof=dof,
                         crit=chi2_critical(dof), accept_rate=float(accept.mean()))
    out["expected_accept"] = float(np.minimum(p, q).sum())  # = 1 - TV(p, q)
    return out


def exact_marginals(model, prompt, temperature, top_k=None, depth=3):
    """Exact distribution of the 1st, 2nd, 3rd generated token under the target alone,
    by enumerating every prefix (V + V² forward rows in one batched pass each)."""
    V = model.config["vocab_size"]
    T = model.config["block_size"]
    assert len(prompt) + depth - 1 <= T, "keep the test inside one window"
    p1 = probs(model.forward(np.array(prompt)[None])[0, -1], temperature, top_k)
    rows = np.array([prompt + [a] for a in range(V)])
    p2_given = np.array([probs(z, temperature, top_k) for z in model.forward(rows)[:, -1]])
    m2 = p1 @ p2_given
    out = [p1, m2]
    if depth >= 3:
        rows = np.array([prompt + [a, b] for a in range(V) for b in range(V)])
        p3_given = np.array([probs(z, temperature, top_k) for z in model.forward(rows)[:, -1]])
        joint12 = (p1[:, None] * p2_given).ravel()
        out.append(joint12 @ p3_given)
    return out


def distribution_test(target, draft, prompt, N, gamma=3, temperature=1.0, seed=0):
    """N independent speculative runs of 3 tokens; compare the empirical marginal of
    tokens 2 and 3 (the drafted ones) with the exact target marginals."""
    exact = exact_marginals(target, prompt, temperature)
    V = len(exact[0])
    rng = np.random.default_rng(seed)
    counts = np.zeros((3, V))
    stats = {}
    for _ in range(N):
        out = speculative_generate(target, draft, prompt, 3, gamma, temperature, rng=rng,
                                   stats=stats)
        for i, t in enumerate(out):
            counts[i, t] += 1
    res = []
    for i in range(3):
        stat, dof = chi_square(counts[i], exact[i])
        res.append(dict(pos=i + 1, tv=0.5 * np.abs(counts[i] / N - exact[i]).sum(),
                        chi2=stat, dof=dof, crit=chi2_critical(dof)))
    return res, stats


# ----------------------------------------------------------------------------
def sweep(target, draft, prompts, n, gammas, temperature, repeats=3):
    """Acceptance rate and speed-up vs γ (best of `repeats` timings)."""
    def run_plain():
        for i, p in enumerate(prompts):
            generate_ids(target, p, n, temperature, rng=np.random.default_rng(i))

    t_plain = min(timed(run_plain) for _ in range(repeats))
    rows = []
    for g in gammas:
        st = {}

        def run_spec(st=st, g=g):
            st.clear()
            for i, p in enumerate(prompts):
                speculative_generate(target, draft, p, n, g, temperature,
                                     rng=np.random.default_rng(i), stats=st)

        t = min(timed(run_spec) for _ in range(repeats))
        alpha = st["accepted"] / max(1, st["proposed"])
        rows.append(dict(gamma=g, alpha=alpha, speedup=t_plain / t,
                         tokens_per_call=len(prompts) * n / st["target_calls"],
                         target_calls=st["target_calls"], draft_calls=st["draft_calls"],
                         ms_per_token=t / (len(prompts) * n) * 1e3))
    return t_plain / (len(prompts) * n) * 1e3, rows


def timed(f):
    t0 = time.perf_counter()
    f()
    return time.perf_counter() - t0


def expected_tokens(alpha, gamma):
    """E[tokens per target pass] = (1 - α^{γ+1}) / (1 - α) (Leviathan et al., eq. 1)."""
    return (1 - alpha ** (gamma + 1)) / (1 - alpha) if alpha < 1 else gamma + 1


def call_cost_ratio(target, draft):
    """c = time of a draft decode step / time of a target pass over γ+1 tokens."""
    _, tc = prefill(target, list(range(10)))
    _, dc = prefill(draft, list(range(10)))
    td = min(timed(lambda: (decode_step(draft, 1, dc), truncate(dc, 10))) for _ in range(300))
    tt = min(timed(lambda: (extend(target, [1, 2, 3, 4, 5], tc), truncate(tc, 10)))
             for _ in range(300))
    t1 = min(timed(lambda: (decode_step(target, 1, tc), truncate(tc, 10))) for _ in range(300))
    return td, tt, t1


DRAFTS = (("gpt", 0.8), ("gpt", 0.0), ("ngram", 0.8), ("ngram", 0.0))
TEST_PROMPT = "The "  # high-entropy next characters: a demanding test


def run_all(target, draft, tok, n=150, gammas=(1, 2, 3, 4, 5, 6, 8), N=20_000):
    prompts = [tok.encode(p)[0] for p in ("The keeper", "One morning", "The girl saw",
                                          "\"Good morning,\"", "The sea was")]
    ngram = NGramDraft(train_ids(tok), len(tok))
    out = dict(rule=rule_test(), sweeps={})
    out["dist"], out["dist_stats"] = distribution_test(target, draft, tok.encode(TEST_PROMPT)[0],
                                                       N)
    for name, temp in DRAFTS:
        d = GPTDraft(draft) if name == "gpt" else ngram
        out["sweeps"][(name, temp)] = sweep(target, d, prompts, n, gammas, temp)
    out["cost"] = call_cost_ratio(target, draft)
    out["ngram_ms"] = min(timed(lambda: (ngram.feed([1]), ngram.truncate(10)))
                          for _ in range(300)) * 1e3
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples", type=int, default=20_000, help="runs for the statistical test")
    args = ap.parse_args()

    target, chars, _ = get_model()
    tok = CharTokenizer(chars)
    draft = get_draft(target, chars)
    held = heldout_ids(tok)
    print(f"\ntarget: {target.n_params():,} params ({target.config['n_layer']} layers, "
          f"d={target.config['d_model']}) held-out loss {evaluate(target, held):.3f}")
    print(f"draft:  {draft.n_params():,} params ({draft.config['n_layer']} layer, "
          f"d={draft.config['d_model']}) held-out loss {evaluate(draft, held):.3f}")

    print("\n== 1. The rule, numerically (V = 10, random p and q, 200,000 draws) ==")
    r = rule_test()
    print(f"  accept rate {r['exact rule']['accept_rate']:.4f} (theory Σ min(p, q) = "
          f"{r['expected_accept']:.4f})")
    for name in ("exact rule", "naive rule"):
        x = r[name]
        verdict = "matches p" if x["chi2"] < x["crit"] else "does NOT match p"
        print(f"  {name}: TV(empirical, p) = {x['tv']:.4f}, chi² = {x['chi2']:.1f} "
              f"(dof {x['dof']}, p=0.001 critical {x['crit']:.1f}) -> {verdict}")

    print("\n== 2. Greedy speculative == greedy target (float64, 300 tokens, γ = 4) ==")
    t64 = get_model(np.float64, verbose=False)[0]
    d64 = get_draft(t64, chars, verbose=False)
    for p in ("The keeper", "She cleaned"):
        ids = tok.encode(p)[0]
        a = speculative_generate(t64, d64, ids, 300, 4, temperature=0.0)
        b = generate_ids(t64, ids, 300, temperature=0.0, rng=np.random.default_rng(0))
        print(f"  {p!r}: identical = {a == b}  ({tok.decode(a)[:50]!r}...)")

    res = run_all(target, draft, tok, N=args.samples)
    print(f"\n== 3. Distribution test: prompt {TEST_PROMPT!r}, temperature 1, γ = 3, "
          f"{args.samples:,} runs ==")
    for d in res["dist"]:
        ok = "OK" if d["chi2"] < d["crit"] else "MISMATCH"
        print(f"  token {d['pos']}: TV = {d['tv']:.4f}, chi² = {d['chi2']:.1f} (dof {d['dof']}, "
              f"critical {d['crit']:.1f}) {ok}")
    st = res["dist_stats"]
    print(f"  (token 1 comes from the prefill; tokens 2-3 went through accept/reject: "
          f"{st['accepted']}/{st['proposed']} drafts accepted)")

    td, tt, t1 = res["cost"]
    print(f"\n== 4. Speed: 5 prompts x 150 tokens ==")
    print(f"  one target decode step {t1 * 1e3:.3f} ms; one target pass over 5 tokens "
          f"{tt * 1e3:.3f} ms ({tt / t1:.2f}x a single step)")
    print(f"  one GPT-draft step {td * 1e3:.3f} ms ({td / t1:.2f}x the target); one n-gram "
          f"draft lookup {res['ngram_ms']:.3f} ms ({res['ngram_ms'] / (t1 * 1e3):.2f}x)")
    for (name, temp), (plain_ms, rows) in res["sweeps"].items():
        print(f"  {name} draft, temperature {temp} (plain target decoding: "
              f"{plain_ms:.3f} ms/token)")
        print(f"  {'γ':>5s} {'accept α':>9s} {'tokens/target pass':>19s} {'theory':>7s} "
              f"{'ms/token':>9s} {'speed-up':>9s}")
        for r in rows:
            print(f"  {r['gamma']:5d} {r['alpha']:9.3f} {r['tokens_per_call']:19.2f} "
                  f"{expected_tokens(r['alpha'], r['gamma']):7.2f} {r['ms_per_token']:9.3f} "
                  f"{r['speedup']:8.2f}x")
    print("  'theory' assumes every draft is accepted independently with prob α and that the\n"
          "  window never cuts a draft short; our 32-token window caps γ near its end.\n"
          "  The GPT draft costs half a target step in NumPy (Python overhead per layer), so\n"
          "  it LOSES time even at α ≈ 0.6-0.8; the n-gram draft is nearly free and wins.\n"
          "  With a 70B target and a 1B draft on a GPU the draft costs a few % of a target\n"
          "  pass, which is why 2-3x speed-ups are typical there.")


if __name__ == "__main__":
    main()
