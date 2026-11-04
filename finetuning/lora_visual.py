"""
Watch LoRA, the SFT loss mask and one DPO pair, step by step, in your terminal.

With the real NumPy models of lora_numpy.py (base, LoRA SFT, LoRA DPO):
  1. a frozen W0 (d x d) next to the adapter A (d x r) and B (r x d): shapes,
     parameter counts, and why B starts at zero,
  2. one token through a LoRA layer: x W0, the r-number bottleneck x A, and the
     correction (alpha/r)(x A) B,
  3. the trained update ΔW = (alpha/r) A B of a real layer as a heatmap, and its
     singular values: exactly r of them are non-zero,
  4. merging: W0 + ΔW gives the same outputs with zero extra cost,
  5. the SFT loss mask on a chat: which characters are trained, and the loss on
     each of them before and after fine-tuning,
  6. one DPO preference pair: log-probs under the policy and the frozen
     reference, the implicit rewards, the margin and the loss.

Uses finetuning/base_numpy.npz + lora_numpy.npz if they exist (made by
`python finetuning/lora_numpy.py`), otherwise trains them first (~20-35 s).

Run:
    python finetuning/lora_visual.py             (press Enter for the next step)
    python finetuning/lora_visual.py --no-pause  (or with stdin from /dev/null)
    python finetuning/lora_visual.py --layer h.1.mlp.c_proj
"""
import argparse
import json
import os
import sys
import warnings

import numpy as np

import domain_data as dd
import lora_numpy as ln

warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)

TTY = sys.stdout.isatty()
SHADE = " ░▒▓█"
NO_PAUSE = False


def color(code, s):
    return f"\033[{code}m{s}\033[0m" if TTY else s


def cyan(s): return color("36;1", s)      # inputs, frozen things
def yellow(s): return color("33;1", s)    # the adapter A, B
def green(s): return color("32;1", s)     # trained / chosen / results
def red(s): return color("31;1", s)       # rejected
def dim(s): return color("2", s)


def pause():
    if sys.stdin.isatty() and not NO_PAUSE:
        try:
            input(dim("\n  [Enter] next step "))
        except EOFError:
            print()
    print("\n" + "=" * 78 + "\n")


def vec(v, n=6):
    v = np.ravel(v)
    return "[" + " ".join(f"{x:+.2f}" for x in v[:n]) + (" …]" if len(v) > n else "]")


def heatmap(M, title, max_rows=16, max_cols=64):
    """|M| as shades, block-averaged down to at most max_rows x max_cols cells."""
    r = max(1, int(np.ceil(M.shape[0] / max_rows)))
    c = max(1, int(np.ceil(M.shape[1] / max_cols)))
    H, W = M.shape[0] // r, M.shape[1] // c
    A = np.abs(M[:H * r, :W * c]).reshape(H, r, W, c).mean(axis=(1, 3))
    scale = A.max() + 1e-12
    print(f"  {title}  ({M.shape[0]}x{M.shape[1]}, |values| averaged over {r}x{c} blocks; "
          f"█ = {scale:.3f})")
    for row in A:
        print("   │" + "".join(SHADE[min(4, int(v / scale * 4.999))] for v in row) + "│")


def show_chat(s):
    return dd.show(s)


def get_layer(model, name):
    layers = model.layers()
    if name not in layers or not isinstance(layers[name], ln.LoRALinear):
        names = [k for k, v in layers.items() if isinstance(v, ln.LoRALinear)]
        sys.exit(f"--layer must be one of: {', '.join(names)}")
    return layers[name]


def main():
    global NO_PAUSE
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layer", default="h.0.attn.c_proj",
                    help="which adapted layer to show (default h.0.attn.c_proj, 64x64)")
    ap.add_argument("--no-pause", action="store_true")
    args = ap.parse_args()
    NO_PAUSE = args.no_pause

    have = os.path.exists(ln.LORA_PATH) and os.path.exists(ln.BASE_PATH)
    if not have:
        print("No finetuning/lora_numpy.npz yet: training base -> LoRA SFT -> DPO first "
              "(~20-35 s).\nTo keep them, run: python finetuning/lora_numpy.py\n")
    tok, story_ids, base, sft, dpo = ln.load_pipeline()
    if have:
        print(dim("(loaded base_numpy.npz and the SFT + DPO adapters from lora_numpy.npz)\n"))
    # Rebuild the SFT model with its adapter still separate, to look at A and B.
    d = np.load(ln.LORA_PATH)
    cfg = json.loads(str(d["config"]))
    sft_lora = ln.get_base(tok, story_ids, verbose=False)
    ln.apply_lora(sft_lora, cfg["r"], cfg["alpha"], tuple(cfg["targets"]))
    for k, p in ln.lora_params(sft_lora).items():
        p[...] = d[f"sft.{k}"]
    layer = get_layer(sft_lora, args.layer)
    W0 = layer.base.params["W"]
    n_in, n_out = W0.shape
    r, s = layer.r, layer.scale
    sfx = next(iter(layer.slices))
    A, B = layer.params["A" + sfx], layer.params["B" + sfx]
    s0, s1 = layer.slices[sfx]

    # ------------------------------------------------------------------ step 1
    print(cyan("Step 1: a frozen weight and a tiny adapter next to it"))
    print(f"""
  Layer {args.layer} of the trained NumPy mini-GPT. Full fine-tuning would change
  every entry of W0. LoRA freezes W0 and learns ΔW = (alpha/r)·A·B instead:

      {cyan('W0')}  ({n_in} x {n_out}) = {W0.size:6,} numbers   frozen
      {yellow('A')}   ({A.shape[0]} x {r})  = {A.size:6,} numbers   trained, starts random  N(0, 1/{n_in})
      {yellow('B')}   ({r} x {B.shape[1]})  = {B.size:6,} numbers   trained, starts at ZERO
      alpha = {layer.alpha:g}, r = {r}  ->  scale alpha/r = {s:g}

  Why B = 0: then ΔW = A·B = 0, so the adapted model starts EXACTLY equal to the
  pretrained one (no random damage), while dL/dB = s·(xA)ᵀ·dy is not zero, so
  training can start. (If both were 0, both gradients would be 0 forever.)
""")
    print("  parameter count for one d x d matrix (LoRA: r·(d + d)):")
    print(f"  {'d':>6s}{'full d²':>14s}" + "".join(f"{'r=' + str(k):>12s}" for k in (1, 8, 64)))
    for dd_ in (64, 768, 4096, 8192):
        print(f"  {dd_:6d}{dd_ * dd_:14,}" + "".join(
            f"{ln.lora_param_count(dd_, dd_, k):12,}" for k in (1, 8, 64)))
    n_lora = sum(p.size for p in ln.lora_params(sft_lora).values())
    n_all = base.n_params()
    print(f"\n  The whole model: {n_lora:,} adapter numbers on q,k,v,o,fc,proj of both "
          f"layers = {n_lora / n_all:.1%} of {n_all:,}.\n  (At d = 4096, r = 8 is 0.4% "
          f"per matrix: that is why a 7B model's LoRA file is ~10-100 MB, not 14 GB.)")
    pause()

    # ------------------------------------------------------------------ step 2
    print(cyan("Step 2: one token through the LoRA layer"))
    chat = dd.render(dd.load_all()[0][0]["messages"])
    ids = np.array([tok.encode(chat)])
    sft_lora.forward(ids)
    x = layer.x[0, -1]  # the input this layer saw at the last position (<|end|>)
    base_out = x @ W0[:, s0:s1]
    h = x @ A
    delta = s * (h @ B)
    print(f"""
  x (the input at the last character of a training chat), {n_in} numbers:
      {cyan(vec(x))}
  frozen path   x·W0               {n_out} numbers: {vec(base_out)}   ‖·‖ = {np.linalg.norm(base_out):.2f}
  adapter path  h = x·A            {yellow(vec(h, 8))}   <- only {r} numbers!
                (alpha/r)·h·B      {n_out} numbers: {yellow(vec(delta))}   ‖·‖ = {np.linalg.norm(delta):.2f}
  output        x·W0 + b + (alpha/r)·(x·A)·B

  Everything the adapter adds has to squeeze through {r} numbers per token. The
  correction is {np.linalg.norm(delta) / np.linalg.norm(base_out):.0%} of the size of the frozen output here.
  Backward (lora_numpy.py, checked by gradient_check.py):
      dB = s·hᵀ·dy     dA = s·xᵀ·(dy·Bᵀ)     dx = dy·W0ᵀ + s·(dy·Bᵀ)·Aᵀ     and no dW0.""")
    pause()

    # ------------------------------------------------------------------ step 3
    print(cyan("Step 3: the trained update ΔW = (alpha/r)·A·B is low-rank"))
    dW = layer.delta_W()[:, s0:s1]
    print()
    heatmap(W0[:, s0:s1], "W0 (frozen, pretrained)")
    print()
    heatmap(dW, "ΔW after LoRA SFT")
    sv0 = np.linalg.svd(W0[:, s0:s1].astype(np.float64), compute_uv=False)
    sv = np.linalg.svd(dW.astype(np.float64), compute_uv=False)
    print(f"\n  singular values of W0: {' '.join(f'{v:.2f}' for v in sv0[:10])} …  "
          f"(rank {np.linalg.matrix_rank(W0)})")
    print(f"  singular values of ΔW: {yellow(' '.join(f'{v:.2f}' for v in sv[:r]))}  "
          f"then {' '.join(f'{v:.0e}' for v in sv[r:r + 3])} …")
    print(f"\n  ΔW is a {n_in}x{n_out} matrix but has exactly rank {np.linalg.matrix_rank(dW, tol=1e-5)}: "
          f"every column is a mix of the {r} columns of A.\n"
          f"  ‖ΔW‖/‖W0‖ = {np.linalg.norm(dW) / np.linalg.norm(W0[:, s0:s1]):.2f}: a small, "
          f"structured nudge. The bet of LoRA (Hu et al. 2021): the CHANGE a\n"
          f"  fine-tune needs has low 'intrinsic rank', even when W0 itself is full rank.")
    pause()

    # ------------------------------------------------------------------ step 4
    print(cyan("Step 4: merging for inference"))
    before = sft_lora.forward(ids)
    layer.merge()
    after = sft_lora.forward(ids)
    layer.unmerge()
    print(f"""
  W_merged = W0 + (alpha/r)·A·B  (one {n_in}x{n_out} add, once)
  max |logits(merged) - logits(adapter)| over the whole chat = {np.abs(after - before).max():.1e}
  (float32 rounding). After merging, inference costs exactly what the base model
  costs: no extra matmuls, no extra latency. Unmerge (W0 -= ΔW) to swap in a
  different adapter: one base model in memory, many cheap task adapters.""")
    pause()

    # ------------------------------------------------------------------ step 5
    print(cyan("Step 5: the SFT loss mask: train on the answer, not on the question"))
    row = dd.load_all()[0][0]
    text, mask = dd.render_with_mask(row["messages"])
    x_ids = np.array([tok.encode(text)])
    targets = x_ids[:, 1:]
    m = np.array([mask[1:]], dtype=np.float64)

    def per_char_loss(model):
        p = ln.softmax(model.forward(x_ids[:, :-1]).astype(np.float64))
        return -np.log(np.take_along_axis(p, targets[..., None], -1)[0, :, 0])

    lb, ls = per_char_loss(base), per_char_loss(sft)
    print(f"\n  {dd.show(text)}\n")
    print("  Each character is a target for the position before it. mask = 1 (green) only")
    print("  for the assistant's answer and its <|end|>: the prompt is context, not a target.\n")
    print(f"  {'target':>13s} {'mask':>5s} {'loss base':>10s} {'loss SFT':>9s}")
    shown = 0
    for t in range(len(targets[0])):
        c = text[t + 1]
        name = dd.SPECIAL_NAMES.get(c, repr(c)[1:-1] if c != " " else "␣")
        trained = m[0, t] > 0
        if not trained and t not in (0, 1, len(row["messages"][0]["content"]) - 1,
                                     len(row["messages"][0]["content"])):
            continue
        line = f"  {name:>13s} {int(m[0, t]):5d} {lb[t]:10.2f} {ls[t]:9.3f}"
        print(green(line) if trained else dim(line))
        shown += 1
    mb = (lb * m[0]).sum() / m.sum()
    msft = (ls * m[0]).sum() / m.sum()
    print(f"\n  masked loss = mean over the {int(m.sum())} green rows: base {mb:.2f} -> SFT "
          f"{msft:.3f}  (uniform guessing: ln({len(tok.chars)}) = {np.log(len(tok.chars)):.2f})")
    print(f"  The base model has NEVER seen <|end|> ({lb[-1]:.1f} nats on it): SFT must teach "
          f"it to STOP.\n  (Rows of the prompt are skipped above except a few; their loss is "
          f"computed but multiplied by 0.)")
    pause()

    # ------------------------------------------------------------------ step 6
    print(cyan("Step 6: one DPO preference pair"))
    prefs = dd.load_all()[2]
    # an out-of-scope question about a lighthouse that had NO refusal demo in SFT
    pair = next(p for p in prefs if p["kind"] == "out_of_scope"
                and not any(e in p["prompt"][0]["content"] for e in dd.LIGHTHOUSES[:3]))
    beta = cfg.get("beta", 0.5)

    def logp(model, answer, enabled=True):
        ln.set_lora_enabled(model, enabled)
        s_, mk = dd.render_with_mask(pair["prompt"] + [{"role": "assistant", "content": answer}])
        x_, y_, m_ = ln.pad_batch([tok.encode(s_)], [mk], 0)
        out = float(ln.sequence_logprob(model.forward(x_), y_, m_)[0][0])
        ln.set_lora_enabled(model, True)
        return out

    w, l_ = pair["chosen"], pair["rejected"]
    ref_w, ref_l = logp(dpo, w, False), logp(dpo, l_, False)
    pi_w, pi_l = logp(dpo, w), logp(dpo, l_)
    z = beta * ((pi_w - ref_w) - (pi_l - ref_l))
    print(f"""
  prompt    {pair['prompt'][0]['content']!r}
  chosen    {green(repr(w))}
  rejected  {red(repr(l_))}   (a confident made-up answer)
  (SFT only saw refusal demos for {', '.join(dd.LIGHTHOUSES[:3])}; this lighthouse is not one of them)

  log π(answer | prompt) = sum of the log-probs of the answer's characters + <|end|>
                              reference (SFT, frozen)   policy (after DPO)   difference
  chosen                      {ref_w:12.2f}           {pi_w:14.2f}     {pi_w - ref_w:+8.2f}
  rejected                    {ref_l:12.2f}           {pi_l:14.2f}     {pi_l - ref_l:+8.2f}

  implicit rewards  β·(log π - log π_ref), β = {beta}:
      chosen {green(f'{beta * (pi_w - ref_w):+.2f}')}    rejected {red(f'{beta * (pi_l - ref_l):+.2f}')}
  margin  z = {z:+.2f}      DPO loss = -log σ(z) = {float(-ln.log_sigmoid(np.array(z))):.4f}   (log 2 = 0.693 at the start)

  The reference is the SAME network with its DPO adapter switched off, so DPO
  needed no second model in memory. The gradient pushes log π(chosen) up and
  log π(rejected) down, weighted by σ(-z): pairs the policy already gets right
  (z >> 0) stop contributing. Without the reference term, nothing would stop the
  policy from drifting arbitrarily far from the SFT model; β sets how far.""")
    pause()
    print(green("Done.") + " Next: python finetuning/finetune_pytorch.py (full FT vs LoRA, "
          "forgetting, DPO at a bigger scale).")


if __name__ == "__main__":
    main()
