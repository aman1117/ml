"""
Gradient check for everything trainable in lora_numpy.py: compare each
hand-written backward pass with central finite differences in float64,

    dL/dw ≈ (L(w + eps) - L(w - eps)) / (2 eps)

Part 1  LoRALinear on its own (L = Σ y ⊙ G for a fixed random G, so dL/dy = G):
        gradients of x, A, B for a plain Linear and for q/v slices of a fused
        (C, 3C) qkv matrix; the B = 0 init (output unchanged, dA = 0, dB != 0);
        merge == unmerged forward; unmerge restores W0; the parameter count.
Part 2  The losses: masked cross-entropy (and that masked positions get exactly
        zero gradient), the sequence log-prob, and the DPO loss w.r.t. the logits
        of the chosen and rejected sequences.
Part 3  LoRA inside a full tiny GPT (from ../transformer/transformer_numpy.py):
        every A and B through the masked SFT loss and through the DPO loss; only
        A and B are trainable, and an optimizer step leaves every base weight
        bit-for-bit unchanged.

Try breaking a line (e.g. drop `self.scale *` in LoRALinear.backward, or flip
the sign of `-beta * dz` in dpo_loss) and re-run: the check must FAIL.

Run (~3 s):
    python finetuning/gradient_check.py
"""
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "transformer"))
from transformer_numpy import GPT, AdamW, Linear  # noqa: E402

import lora_numpy as ln  # noqa: E402

warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)

F64 = np.float64
EPS = 1e-5
TOL = 1e-6


def numerical_grad(f, arr, eps=EPS):
    grad = np.zeros_like(arr)
    it = np.nditer(arr, flags=["multi_index"])
    for _ in it:
        i = it.multi_index
        old = arr[i]
        arr[i] = old + eps
        lp = f()
        arr[i] = old - eps
        lm = f()
        arr[i] = old
        grad[i] = (lp - lm) / (2 * eps)
    return grad


def rel_error(a, b):
    """‖a - b‖ / (‖a‖ + ‖b‖) over the whole tensor (see ../transformer/gradient_check.py)."""
    return float(np.linalg.norm(a - b) / max(1e-20, np.linalg.norm(a) + np.linalg.norm(b)))


def report(name, shape, err, tol=TOL):
    ok = err < tol
    print(f"  {name:38s} shape={str(shape):10s} relative error = {err:.2e}  "
          f"{'OK' if ok else 'FAIL'}")
    return ok


def fact(name, value, ok, note=""):
    print(f"  {name:38s} {value:<38s} {'OK' if ok else 'FAIL'}  {note}")
    return ok


def check_lora_layer(name, layer, x, rng):
    G = rng.standard_normal(layer.forward(x).shape)
    f = lambda: float(np.sum(layer.forward(x) * G))
    layer.forward(x)
    dx = layer.backward(G)
    ok = report(f"{name}  x", x.shape, rel_error(dx, numerical_grad(f, x)))
    for k, p in layer.params.items():
        ok &= report(f"{name}  {k}", p.shape, rel_error(layer.grads[k], numerical_grad(f, p)))
    return ok


def part1(rng):
    print("Part 1: LoRALinear on its own (N=2x5 tokens, n_in=6, n_out=7, r=3, alpha=6)\n")
    ok = True
    base = Linear(6, 7, rng, std=0.5, dtype=F64)
    base.params["b"][...] = rng.standard_normal(7)
    lora = ln.LoRALinear(base, r=3, alpha=6.0, rng=rng)
    x = rng.standard_normal((2, 5, 6))

    # The init: B = 0, so the wrapped layer is EXACTLY the base layer, and
    # dA = s·xᵀ(dy Bᵀ) = 0 while dB = s·(xA)ᵀ dy != 0: B moves first, then A follows.
    y0 = x @ base.params["W"] + base.params["b"]
    same = np.array_equal(lora.forward(x), y0)
    ok &= fact("init: output == base output", f"max |diff| = {np.abs(lora.forward(x) - y0).max():.0e}",
               same, "(B = 0)")
    lora.backward(rng.standard_normal((2, 5, 7)))
    dA0, dB0 = np.abs(lora.grads["A"]).max(), np.abs(lora.grads["B"]).max()
    ok &= fact("init: dA = 0, dB != 0", f"max|dA| = {dA0:.0e}, max|dB| = {dB0:.2f}",
               dA0 == 0 and dB0 > 0)

    lora.params["B"][...] = rng.standard_normal(lora.params["B"].shape) * 0.5
    ok &= check_lora_layer("LoRA (all 7 columns)", lora, x, rng)

    # q and v slices of a fused qkv projection (C = 4 -> 3C = 12 columns), k untouched.
    C = 4
    qkv = Linear(C, 3 * C, rng, std=0.5, dtype=F64)
    sl = ln.LoRALinear(qkv, r=2, alpha=4.0, rng=rng, slices={"_q": (0, C), "_v": (2 * C, 3 * C)})
    for k in ("B_q", "B_v"):
        sl.params[k][...] = rng.standard_normal(sl.params[k].shape) * 0.5
    ok &= check_lora_layer("LoRA q,v slices of qkv", sl, rng.standard_normal((2, 5, C)), rng)
    dW = sl.delta_W()
    ok &= fact("slices: ΔW[:, k columns] == 0", f"max |ΔW_k| = {np.abs(dW[:, C:2 * C]).max():.0e}",
               not dW[:, C:2 * C].any())

    # merge: same function with zero extra cost; unmerge gives W0 back
    W0 = base.params["W"].copy()
    y = lora.forward(x)
    lora.merge()
    d_merge = np.abs(lora.forward(x) - y).max()
    ok &= fact("merge: x(W0 + sAB) == xW0 + s(xA)B", f"max |diff| = {d_merge:.1e}", d_merge < 1e-12)
    lora.unmerge()
    d_un = np.abs(base.params["W"] - W0).max()
    ok &= fact("unmerge restores W0", f"max |diff| = {d_un:.1e}", d_un < 1e-12)
    rank = np.linalg.matrix_rank(lora.delta_W())
    ok &= fact("rank(ΔW) <= r", f"rank = {rank} (r = 3, ΔW is 6x7)", rank <= 3)
    n = sum(p.size for p in lora.params.values())
    ok &= fact("params = r·(n_in + n_out)", f"{n} = 3·(6 + 7), vs 6·7 = 42 for W0",
               n == ln.lora_param_count(6, 7, 3))
    return ok


def part2(rng):
    print("\nPart 2: the losses (B=3 sequences, T=6, V=5)\n")
    ok = True
    B, T, V = 3, 6, 5
    logits = rng.standard_normal((B, T, V)) * 2
    y = rng.integers(0, V, (B, T))
    m = np.zeros((B, T))
    m[:, 3:] = 1.0  # e.g. 3 prompt tokens, then 3 assistant tokens
    m[2, 5] = 0.0  # a padded position

    _, d = ln.masked_cross_entropy(logits.copy(), y, m)
    f = lambda: ln.masked_cross_entropy(logits.copy(), y, m)[0]
    ok &= report("masked cross-entropy", logits.shape, rel_error(d, numerical_grad(f, logits)))
    ok &= fact("masked positions: zero gradient", f"max |dlogits| = {np.abs(d[m == 0]).max():.0e}",
               not d[m == 0].any())
    loss1 = f()
    logits[m == 0] += rng.standard_normal(logits[m == 0].shape) * 10  # change the prompt logits
    ok &= fact("masked positions: loss unchanged", f"|Δloss| = {abs(f() - loss1):.0e}",
               f() == loss1, "(prompt logits x10 noise)")

    c = rng.standard_normal(B)  # L = Σ_b c_b log π(y_b): checks the per-sequence backward
    lp, probs = ln.sequence_logprob(logits, y, m)
    d = ln.sequence_logprob_backward(c, probs, y, m)
    f = lambda: float(c @ ln.sequence_logprob(logits, y, m)[0])
    ok &= report("sequence log-prob Σ_t m_t log p_t[y_t]", logits.shape,
                 rel_error(d, numerical_grad(f, logits)))

    # DPO: rows 0..B-1 of `both` are the chosen sequences, rows B..2B-1 the rejected ones.
    both = rng.standard_normal((2 * B, T, V)) * 2
    yy = rng.integers(0, V, (2 * B, T))
    mm = np.concatenate([m, m[::-1]])
    ref = rng.standard_normal((2, B)) * 3 - 8
    beta = 0.5

    def dpo_value():
        lp, _ = ln.sequence_logprob(both, yy, mm)
        return ln.dpo_loss(lp[:B], lp[B:], ref[0], ref[1], beta)[0]

    lp, probs = ln.sequence_logprob(both, yy, mm)
    loss, d_w, d_l, _ = ln.dpo_loss(lp[:B], lp[B:], ref[0], ref[1], beta)
    d = ln.sequence_logprob_backward(np.concatenate([d_w, d_l]), probs, yy, mm)
    ok &= report("DPO loss -> logits (chosen+rejected)", both.shape,
                 rel_error(d, numerical_grad(dpo_value, both)))
    l0 = ln.dpo_loss(lp[:B], lp[B:], lp[:B], lp[B:], beta)[0]
    ok &= fact("DPO at policy == reference", f"loss = {l0:.6f} = log 2", abs(l0 - np.log(2)) < 1e-12)
    return ok


def tiny_gpt(rng):
    model = GPT(7, n_layer=2, n_head=2, d_model=8, block_size=6, seed=0, dtype=F64)
    for name, p in model.params().items():
        p[...] = (1 + 0.3 * rng.standard_normal(p.shape) if name.endswith("gamma")
                  else 0.3 * rng.standard_normal(p.shape))
    ln.apply_lora(model, r=2, alpha=4.0, targets=ln.ALL_TARGETS)
    for p in ln.lora_params(model).values():
        p[...] = rng.standard_normal(p.shape) * 0.3  # non-zero B, so dA is non-trivial
    return model


def part3(rng):
    model = tiny_gpt(rng)
    lp = ln.lora_params(model)
    n_all = sum(p.size for p in model.params().values()) + sum(
        l.base.params["W"].size + l.base.params["b"].size for l in ln.lora_layers(model))
    print(f"\nPart 3: LoRA (r=2, q,k,v,o,fc,proj) inside a tiny GPT (V=7, L=2, d=8, "
          f"T=5): {sum(p.size for p in lp.values())} trainable of {n_all} numbers\n")
    ok = True
    B, T = 2, 5
    idx = rng.integers(0, 7, (B, T))
    y = rng.integers(0, 7, (B, T))
    m = np.zeros((B, T))
    m[:, 2:] = 1.0

    def sft_loss():
        return ln.masked_cross_entropy(model.forward(idx), y, m)[0]

    _, d = ln.masked_cross_entropy(model.forward(idx), y, m)
    model.dlogits = d
    model.backward()
    grads = ln.lora_grads(model)
    worst = 0.0
    for k, p in lp.items():
        err = rel_error(grads[k], numerical_grad(sft_loss, p))
        worst = max(worst, err)
        if k.startswith("h.0."):
            ok &= report(f"SFT  {k}", p.shape, err)
    ok &= report(f"SFT  all {len(lp)} A/B tensors, both layers", "(worst)", worst)

    ref = rng.standard_normal((2, B)) - 6
    pairs_idx = np.concatenate([idx, idx[:, ::-1]])  # chosen = idx, rejected = reversed
    pairs_y = np.concatenate([y, y[:, ::-1]])
    pairs_m = np.concatenate([m, m])

    def dpo_value():
        lp_, _ = ln.sequence_logprob(model.forward(pairs_idx), pairs_y, pairs_m)
        return ln.dpo_loss(lp_[:B], lp_[B:], ref[0], ref[1], 0.5)[0]

    logp, probs = ln.sequence_logprob(model.forward(pairs_idx), pairs_y, pairs_m)
    _, d_w, d_l, _ = ln.dpo_loss(logp[:B], logp[B:], ref[0], ref[1], 0.5)
    model.dlogits = ln.sequence_logprob_backward(np.concatenate([d_w, d_l]), probs,
                                                 pairs_y, pairs_m)
    model.backward()
    grads = ln.lora_grads(model)
    worst = max(rel_error(grads[k], numerical_grad(dpo_value, p)) for k, p in lp.items())
    ok &= report(f"DPO  all {len(lp)} A/B tensors, both layers", "(worst)", worst)

    # Frozen base: the optimizer only ever sees A and B.
    frozen = {k: v.copy() for k, v in model.params().items() if k not in lp}
    frozen.update({f"{i}.base.{k}": v.copy() for i, l in enumerate(ln.lora_layers(model))
                   for k, v in l.base.params.items()})
    before = {k: v.copy() for k, v in lp.items()}
    opt = AdamW(lp, weight_decay=0.0)
    opt.step(grads, lr=1e-2)
    moved = all(not np.array_equal(before[k], lp[k]) for k in lp)
    now = {k: v for k, v in model.params().items() if k not in lp}
    now.update({f"{i}.base.{k}": v for i, l in enumerate(ln.lora_layers(model))
                for k, v in l.base.params.items()})
    unchanged = all(np.array_equal(frozen[k], now[k]) for k in frozen)
    ok &= fact("one AdamW step on A, B only", f"{len(lp)} A/B moved, {len(frozen)} base "
               "tensors bit-identical", moved and unchanged)
    return ok


def main():
    rng = np.random.default_rng(0)
    results = [part1(rng), part2(rng), part3(rng)]
    ok = all(results)
    print("\nGradient check", "PASSED" if ok else "FAILED")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
