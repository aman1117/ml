"""
RoPE and grouped-query attention, step by step in your terminal.

You will see:
  1. the frequencies θ_i = 10000^(-2i/hs): one per PAIR of dimensions,
  2. each pair of a query vector drawn as a 2D "clock hand" that turns by θ_i per
     position (fast hands for the first pairs, slow ones for the last),
  3. the rotation in numbers: (a, b) -> (a cos - b sin, a sin + b cos), length kept,
  4. the key property: the SAME q and k placed at different absolute positions but
     at the same offset give the SAME score, so the score matrix is constant along
     its diagonals; and on the trained model, shifting a whole prompt by +1000
     positions leaves every attention weight unchanged,
  5. long-range decay: how the score of identical q and k falls with the distance,
  6. grouped-query attention: which query heads share which K/V head, what that
     does to the KV cache, and that heads sharing K/V still attend differently.

Uses llama/model.npz (trains it first if missing, ~6 s).

Run:
    python llama/rope_visual.py               (press Enter for the next step)
    python llama/rope_visual.py --no-pause    (or with stdin not a terminal)
"""
import argparse
import math
import sys
import warnings

import numpy as np

from llama_numpy import (DEFAULT_TEXT, get_trained_model, load_text, rope_angles, rope_freqs,
                         rope_rotate, softmax)

warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)

TTY = sys.stdout.isatty()
SPARK = "▁▂▃▄▅▆▇█"
PAUSE = True


def color(code, s):
    return f"\033[{code}m{s}\033[0m" if TTY else s


def cyan(s): return color("36;1", s)      # inputs, positions
def yellow(s): return color("33;1", s)    # angles, rotations
def green(s): return color("32;1", s)     # results
def red(s): return color("31;1", s)
def dim(s): return color("2", s)


def pause():
    if PAUSE and sys.stdin.isatty():
        try:
            input(dim("\n  [Enter] next step "))
        except EOFError:
            print()
    print("\n" + "=" * 78 + "\n")


def title(s):
    print(cyan(s) + "\n")


def at(v, m, base=10000.0):
    """Vector v (hs,) placed at position m: every pair rotated by m·θ_i."""
    cos, sin = rope_angles([m], len(v), base, np.float64)
    return rope_rotate(v[None], cos, sin)[0]


def clock(x, y, rows=7, cols=13):
    """A 2D vector (x, y) as an ASCII clock hand. Terminal cells are ~2x taller than wide,
    so the horizontal axis is stretched by 2."""
    grid = [[" "] * cols for _ in range(rows)]
    cy, cx, R = rows // 2, cols // 2, rows // 2
    for k in range(48):  # the dial
        a = 2 * math.pi * k / 48
        r_, c_ = cy - round(R * math.sin(a)), cx + round(2 * R * math.cos(a))
        grid[r_][c_] = "·"
    ang, length = math.atan2(y, x), 1.0
    for s in np.linspace(0, length, 12)[1:]:
        r_, c_ = cy - round(R * s * math.sin(ang)), cx + round(2 * R * s * math.cos(ang))
        grid[r_][c_] = "*"
    r_, c_ = cy - round(R * math.sin(ang)), cx + round(2 * R * math.cos(ang))
    grid[r_][c_] = "O"
    grid[cy][cx] = "+"
    return ["".join(row) for row in grid]


def step_frequencies(hs=16):
    title(f"Step 1. One rotation speed per PAIR of dimensions (head size {hs} -> {hs // 2} pairs)")
    print("GPT added a learned position vector to each token. RoPE adds nothing: it ROTATES\n"
          "each pair of dimensions (q[2i], q[2i+1]) of every query and key by the angle\n"
          "position · θ_i, with   θ_i = 10000^(-2i/hs)   (Su et al. 2021).\n")
    th = rope_freqs(hs)
    print(f"  {'pair i':>7s}  {'dims':>7s}  {'θ_i (rad/position)':>19s}  "
          f"{'one full turn every':>20s}")
    for i, t in enumerate(th):
        print(f"  {i:7d}  {f'{2 * i},{2 * i + 1}':>7s}  {yellow(f'{t:19.5f}')}  "
              f"{2 * math.pi / t:14.1f} positions")
    print("\nThe first pairs spin fast (they tell neighbours apart), the last ones barely move\n"
          "(they still differ between positions thousands apart). Like the hands of a clock:\n"
          "seconds, minutes, hours.")
    pause()


def step_clocks(rng, hs=8):
    title(f"Step 2. A query vector of size {hs} = {hs // 2} clock hands, at positions 0..5")
    q = rng.standard_normal(hs)
    th = rope_freqs(hs)
    print(f"q = {np.round(q, 2)}  (the same content at every position)\n")
    positions = [0, 1, 2, 3, 4, 5]
    print("          " + "".join(cyan(f"{'m = ' + str(m):^15s}") for m in positions))
    for i in range(hs // 2):
        rows = [clock(*at(q, m)[2 * i:2 * i + 2]) for m in positions]
        for r in range(len(rows[0])):
            label = f"pair {i}" if r == 2 else (f"θ={th[i]:.3f}" if r == 4 else "")
            print(f"{label:>9s} " + "".join(f" {rows[j][r]} " for j in range(len(positions))))
        print()
    print("Pair 0 turns 1 radian (57°) per position; pair 3 turns 0.001 rad: it looks frozen\n"
          "here but has moved 57° by position 1000. Every hand keeps its length: RoPE never\n"
          "changes how big q is, only which way its pairs point.")
    pause()


def step_numbers(rng):
    title("Step 3. The rotation in numbers (pair 0, θ_0 = 1 rad/position)")
    a, b = rng.standard_normal(2)
    print("In complex numbers: z = a + j·b  becomes  z · e^{j·m·θ}. As a 2×2 matrix:\n"
          "   [a']   [cos mθ  -sin mθ] [a]\n"
          "   [b'] = [sin mθ   cos mθ] [b]\n")
    print(f"  {'m':>3s} {'angle':>8s} {'a′':>8s} {'b′':>8s} {'length':>8s}")
    for m in range(5):
        c, s = math.cos(m), math.sin(m)
        a2, b2 = a * c - b * s, a * s + b * c
        print(f"  {m:3d} {yellow(f'{m * 57.3:7.1f}°')} {a2:8.3f} {b2:8.3f} "
              f"{green(f'{math.hypot(a2, b2):8.3f}')}")
    print("\nBackward pass: the transpose of a rotation is the rotation by -mθ, so\n"
          "RoPE.backward(dy) = rotate dy by -m·θ_i. No parameters, nothing to learn.")
    pause()


def step_relative(model, chars, rng, hs=16):
    title("Step 4. Why rotate? Scores only depend on the OFFSET m - n")
    q, k = rng.standard_normal(hs), rng.standard_normal(hs)
    print("Rotations compose: R(mθ)ᵀ R(nθ) = R((n-m)θ). So for every pair\n"
          "   (R_m q)·(R_n k) = q·R_(n-m) k     ->  the score sees only m - n.\n")
    print(f"  same q and k (size {hs}), placed at query position m and key position n:\n")
    print(f"  {'m':>6s} {'n':>6s} {'m - n':>6s} {'score q_m·k_n':>15s}")
    for m, n in [(5, 3), (105, 103), (5005, 5003), (3, 5), (1003, 1005), (9, 3), (909, 903)]:
        s = at(q, m) @ at(k, n)
        print(f"  {m:6d} {n:6d} {yellow(f'{m - n:+6d}')} {green(f'{s:15.6f}')}")
    print("\nSame offset -> same score, to 1e-14 (gradient_check.py checks this). Note +2 and\n"
          "-2 differ: RoPE knows which token came first.\n")
    print("Score matrix S[m, n] for the same q, k at positions 0..7 (constant along diagonals):")
    S = np.array([[at(q, m) @ at(k, n) for n in range(8)] for m in range(8)])
    print("        n →" + "".join(f"{n:7d}" for n in range(8)))
    for m in range(8):
        print(f"    m = {m}  " + "".join(
            (yellow if (m - n) % 3 == 0 else str)(f"{S[m, n]:7.2f}") for n in range(8)))

    # the real model: shift the prompt by +1000 positions
    prompt = "The keeper"
    stoi = {c: i for i, c in enumerate(chars)}
    ids = np.array([stoi[c] for c in prompt])
    blk = model.blocks[0]
    a = blk.attention
    h = blk.attention_norm.forward(model.tok_embeddings.params["weight"][ids][None])
    qr = a.split_heads(a.wq.forward(h), a.n_head)[0]  # (nh, T, hs) before RoPE
    kr = a.split_heads(a.wk.forward(h), a.n_kv_head)[0]
    T = len(ids)
    mask = np.tril(np.ones((T, T), bool))

    def attn(offset):
        cos, sin = rope_angles(np.arange(T) + offset, a.hs, model.config["rope_base"],
                               np.float64)
        qq, kk = rope_rotate(qr, cos, sin), rope_rotate(kr, cos, sin)
        kk = np.repeat(kk, a.groups, axis=0)
        return softmax(np.where(mask, qq @ kk.transpose(0, 2, 1) * a.scale, -np.inf))

    d = np.abs(attn(0) - attn(1000)).max()
    print(f"\nThe trained model, layer 0, reading {prompt!r} at positions 0..{T - 1} and again at "
          f"1000..{999 + T}:\n  max |attention weight difference| over all {a.n_head} heads = "
          + green(f"{d:.1e}") + ". GPT's learned table has no position 1000 at all.")
    pause()


def step_decay(hs=64):
    title(f"Step 5. Long-range decay (head size {hs}, base 10000)")
    th = rope_freqs(hs)
    print("Take q = k (a token looking for itself). The score at distance Δ is\n"
          "   Σ_i |q_i|² cos(Δ·θ_i):  all terms agree at Δ = 0, then the fast pairs fall out\n"
          "   of phase, and the average over the pairs shrinks (not monotonically) as Δ grows:\n")
    dists = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    vals = [np.cos(dd * th).mean() for dd in dists]
    for dd, v in zip(dists, vals):
        n = int(round(max(v, 0) * 40))
        print(f"  Δ = {dd:5d}  {v:+.3f}  " + green("█" * n) + (red(" (negative)") if v < 0 else ""))
    print("\nNearby tokens get a head start, far ones are not forbidden (unlike a hard window).\n"
          "Context extension (README section 3.4) changes θ_i so that long distances look like\n"
          "the ones seen in training: position interpolation divides every angle by s,\n"
          "NTK/YaRN raise the base so only the slow pairs are stretched.")
    pause()


def step_gqa(model):
    title("Step 6. Grouped-query attention: query heads share K/V heads")
    cfg = model.config
    nh, nkv = cfg["n_head"], cfg["n_kv_head"]
    g = nh // nkv
    print("Every query head still has its own W_q. But K and V are computed for only\n"
          "n_kv_head heads; query head h reads K/V head h // (n_head / n_kv_head).\n")
    for n_head, n_kv in [(8, 8), (8, 2), (8, 1)]:
        kind = {n_head: "MHA", 1: "MQA"}.get(n_kv, "GQA")
        gg = n_head // n_kv
        qs = "".join(cyan(f"q{h}".ljust(4)) for h in range(n_head))
        kvs = "".join(yellow(f"kv{kv}".center(4 * gg)) for kv in range(n_kv))
        print(f"  {kind} n_head={n_head} n_kv_head={n_kv}   (cache = {n_kv}/{n_head} of MHA)\n"
              f"     query heads  {qs}\n     K/V heads    {kvs}\n")
    L, hs, T = cfg["n_layer"], cfg["d_model"] // nh, cfg["block_size"]
    print(f"Our model: {nh} query heads, {nkv} K/V heads ({g} query heads each).\n"
          f"  KV cache = 2 × {L} layers × {nkv} × {hs} × {T} tokens × 4 B = "
          + green(f"{2 * L * nkv * hs * T * 4 // 1024} KiB")
          + f" (MHA: {2 * L * nh * hs * T * 4 // 1024} KiB).")
    print("LLaMA-2-70B: 64 query heads, 8 K/V heads, 80 layers, head size 128, fp16, 4k tokens:\n"
          "  MHA would need 2 × 80 × 64 × 128 × 4096 × 2 B = 10 GiB per sequence; GQA needs "
          + green("1.25 GiB") + ".\n")

    # do heads that share K/V attend the same way? (no: their queries differ)
    _, _, data = load_text(DEFAULT_TEXT)
    starts = np.arange(0, len(data) - T - 1, T)
    model.forward(np.stack([data[i:i + T] for i in starts]))
    A = model.attention_maps()[0]  # (B, nh, T, T), layer 0
    print("Trained model, layer 0, whole text: mean |A_h1 - A_h2| (total variation per row)")
    for h1 in range(nh):
        row = []
        for h2 in range(nh):
            tv = 0.5 * np.abs(A[:, h1] - A[:, h2]).sum(-1).mean()
            share = h1 // g == h2 // g
            s = f"{tv:5.2f}" + ("*" if share and h1 != h2 else " ")
            row.append(yellow(s) if share and h1 != h2 else s)
        print(f"    q{h1} (kv{h1 // g}): " + "  ".join(row))
    print("\n* = two heads that read the SAME K and V. Their attention patterns still differ\n"
          "(0 = identical, 1 = no overlap), only a little less than unrelated heads, because\n"
          "each has its own queries. GQA shares the memory of the past, not the question\n"
          "asked of it.")
    pause()


def main():
    global PAUSE
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-pause", action="store_true", help="don't wait for Enter")
    args = ap.parse_args()
    PAUSE = not args.no_pause
    model, chars, _ = get_trained_model(np.float64)
    rng = np.random.default_rng(0)
    step_frequencies()
    step_clocks(rng)
    step_numbers(rng)
    step_relative(model, chars, rng)
    step_decay()
    step_gqa(model)
    print(green("Done.") + " Next: python llama/gradient_check.py, then python llama/generate.py")


if __name__ == "__main__":
    main()
