"""
Watch the trained mini-GPT read a prompt, step by step, in your terminal.

You will see, with the real numbers of the trained model:
  - characters -> token ids -> token embedding + position embedding,
  - ONE attention head of layer 0: the query, key and value of every token,
    the score matrix q·k/√hs with the causal mask (· = masked, -inf),
    the softmax weights as a heatmap, and the last token's output as a
    weighted mix of the value vectors,
  - all heads concatenated -> W_o -> residual add, then LayerNorm -> MLP -> add,
  - what the heads of the next layer look at,
  - the final top-5 next-character predictions,
  - and how ONE parallel forward pass gives a prediction after EVERY prefix.

Uses transformer/model.npz if it exists (train it with
`python transformer/transformer_numpy.py`), otherwise trains one first (~10 s).

Run:
    python transformer/attention_step_visual.py            (press Enter for the next step)
    python transformer/attention_step_visual.py --prompt "said the girl" --head 2
"""
import argparse
import contextlib
import io
import os
import sys

import numpy as np

from transformer_numpy import DEFAULT_TEXT, GELU_C, GPT, load_model, load_text, softmax, train

HERE = os.path.dirname(os.path.abspath(__file__))
TTY = sys.stdout.isatty()
SPARK = "▁▂▃▄▅▆▇█"
SHADE = " ░▒▓█"
SHOW = 4  # how many numbers of a vector to print
SPARK_DIMS = 24  # how many embedding dims to draw as bars


def color(code, s):
    return f"\033[{code}m{s}\033[0m" if TTY else s


def cyan(s): return color("36;1", s)      # tokens / inputs
def yellow(s): return color("33;1", s)    # attention: queries, keys, weights
def green(s): return color("32;1", s)     # results / predictions
def dim(s): return color("2", s)


def pause():
    if sys.stdin.isatty():
        try:
            input(dim("\n  [Enter] next step "))
        except EOFError:  # Ctrl+D: just keep going without pausing
            print()
    print("\n" + "=" * 78 + "\n")


def ch(c):
    """Make spaces and newlines visible."""
    return {" ": "␣", "\n": "↵"}.get(c, c)


def vec(v, n=SHOW):
    v = np.ravel(v)
    return "[" + " ".join(f"{x:+5.2f}" for x in v[:n]) + (" …]" if len(v) > n else "]")


def spark(v, scale=None):
    """All numbers of v as tiny bars: ▁ = most negative, █ = most positive."""
    v = np.ravel(v)
    scale = scale or (np.abs(v).max() + 1e-12)
    idx = np.clip(((v / scale + 1) / 2 * 8).astype(int), 0, 7)
    return "".join(SPARK[i] for i in idx)


def bar(p, width=24):
    n = int(round(p * width))
    return "█" * n + "·" * (width - n)


def layer_norm(x, ln):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return ln.params["gamma"] * (x - mu) / np.sqrt(var + ln.eps) + ln.params["beta"]


def gelu(x):
    return 0.5 * x * (1 + np.tanh(GELU_C * (x + 0.044715 * x ** 3)))


def get_model():
    path = os.path.join(HERE, "model.npz")
    if os.path.exists(path):
        model, chars, _ = load_model(path, dtype=np.float64)
        print(dim(f"(loaded the trained model from {os.path.relpath(path)})\n"))
        return model, chars
    print("No transformer/model.npz yet, so training one first (~10 s).")
    print("To keep it, run: python transformer/transformer_numpy.py\n")
    _, chars, data = load_text(DEFAULT_TEXT)
    with contextlib.redirect_stdout(io.StringIO()):  # hide the training printout
        model32, _ = train(data, len(chars), verbose=False)
    model = GPT(dtype=np.float64, **model32.config)
    for (_, p), (_, p32) in zip(model.params().items(), model32.params().items()):
        p[...] = p32
    return model, chars


def cell(w):
    """One attention weight as a 5-char cell: a shade block and a percentage."""
    return f"{SHADE[min(4, int(w * 5))]}{w * 100:3.0f}%"


def attention_grid(A, labels, masked_char="·"):
    """Print a (T, T) matrix of weights as a shaded heatmap with percentages."""
    T = len(labels)
    print("        keys →  " + "".join(f"{ch(c):>6s}" for c in labels))
    for i in range(T):
        cells = []
        for j in range(T):
            if j > i:
                cells.append(dim(f"{masked_char:>6s}"))
            else:
                w = A[i, j]
                s = f" {cell(w)}"
                cells.append(yellow(s) if w == A[i, :i + 1].max() else s)
        print(f"  query {ch(labels[i]):>2s} (t={i:<2d})" + "".join(cells))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("text", nargs="?", help="same as --prompt")
    ap.add_argument("--prompt", default=None, help='text to read (default "The keeper")')
    ap.add_argument("--head", type=int, default=None,
                    help="which head of layer 0 to show (default: the most 'previous-token' one)")
    args = ap.parse_args()
    prompt = args.prompt or args.text or "The keeper"

    model, chars = get_model()
    stoi = {c: i for i, c in enumerate(chars)}
    unknown = sorted({c for c in prompt if c not in stoi})
    if unknown:
        print(f"(skipping characters never seen in training: {unknown})\n")
    prompt = "".join(c for c in prompt if c in stoi)
    cfg = model.config
    prompt = prompt[-cfg["block_size"]:]
    if not prompt:
        raise SystemExit("The prompt has no characters the model knows.")
    T, C, nh, L, V = len(prompt), cfg["d_model"], cfg["n_head"], cfg["n_layer"], len(chars)
    hs = C // nh
    ids = np.array([stoi[c] for c in prompt])

    # ------------------------------------------------------------------ STEP 0
    print("STEP 0: the setup. A decoder-only Transformer (a mini-GPT):\n")
    print(f"   {cyan('embeddings')}  token table wte ({V}, {C}) + position table wpe "
          f"({cfg['block_size']}, {C})")
    for l in range(L):
        print(f"   block {l}     x = x + {yellow('Attention')}(LN(x))   "
              f"{nh} heads × {hs} dims = {C}")
        print(f"               x = x + MLP(LN(x))         {C} → {4 * C} → {C}")
    print(f"   {green('output')}      logits = LN(x) @ W_head  ({C}, {V})")
    print(f"\n   {model.n_params():,} parameters. Vocabulary: V = {V} characters. "
          f"Context: up to {cfg['block_size']} characters.")
    print(f"\n   We will read {cyan(repr(prompt))}: all {T} characters AT ONCE (no loop "
          "over time, unlike the RNN).")
    pause()

    # ------------------------------------------------------------------ STEP 1
    print("STEP 1: characters → token ids (one id per character, its index in the vocabulary)\n")
    print("   char  " + "".join(f"{ch(c):>5s}" for c in prompt))
    print("   id    " + cyan("".join(f"{i:>5d}" for i in ids)))
    print("   pos   " + "".join(f"{t:>5d}" for t in range(T)))
    print(dim(f"\n   Real LLMs use sub-word tokens (BPE, ~50k-200k of them); the idea is the same."))
    pause()

    # ------------------------------------------------------------------ STEP 2
    wte, wpe = model.wte.params["weight"], model.wpe.params["weight"]
    tok, pos = wte[ids], wpe[:T]
    x = tok + pos  # (T, C)
    print(f"STEP 2: look up two vectors of {C} numbers per token and ADD them\n")
    print("   x_t = wte[id_t] (WHAT the token is) + wpe[t] (WHERE it is)\n")
    print(f"   {'':6s} {'token vector wte[id]':<{SPARK_DIMS}s}   {'position wpe[t]':<{SPARK_DIMS}s}"
          f"   x")
    scale = np.abs(x[:, :SPARK_DIMS]).max()
    n = SPARK_DIMS
    for t in range(T):
        print(f"   {cyan(ch(prompt[t])):>2s} t={t:<2d} {spark(tok[t, :n], scale)} + "
              f"{spark(pos[t, :n], scale)} = {green(spark(x[t, :n], scale))}")
    repeated = [c for c in prompt if prompt.count(c) > 1]
    if repeated:
        a, b = [t for t in range(T) if prompt[t] == repeated[0]][:2]
        print(dim(f"\n   '{ch(prompt[a])}' at t={a} and t={b}: SAME token vector, different "
                  "position vector, so x differs."))
    print(dim(f"   Bars: the first {SPARK_DIMS} of the {C} numbers, ▁ = most negative … "
              "█ = most positive."))
    print(dim("   Without wpe, attention would see a bag of characters with no order."))
    pause()

    # ------------------------------------------------------------------ STEP 3
    blk = model.blocks[0]
    h = layer_norm(x, blk.ln_1)
    print("STEP 3: LayerNorm. Each token's vector is rescaled to mean 0, std 1 "
          "(then × γ + β, learned)\n")
    for t in range(min(T, 4)):
        xhat = (x[t] - x[t].mean()) / np.sqrt(x[t].var() + blk.ln_1.eps)
        print(f"   {ch(prompt[t]):>2s}  before: mean {x[t].mean():+.3f} std {x[t].std():.3f}   "
              f"normalized: mean {xhat.mean():+.3f} std {xhat.std():.3f}")
    print(dim("   (… same for every token.) Keeps the numbers in a healthy range "
              "before the attention matmuls."))
    pause()

    # ------------------------------------------------------------------ STEP 4
    W, b = blk.attn.c_attn.params["W"], blk.attn.c_attn.params["b"]
    qkv = h @ W + b  # (T, 3C)
    q_all, k_all, v_all = np.split(qkv, 3, axis=-1)
    split = lambda z: z.reshape(T, nh, hs).transpose(1, 0, 2)  # (nh, T, hs)
    q_all, k_all, v_all = split(q_all), split(k_all), split(v_all)
    mask = np.tril(np.ones((T, T), dtype=bool))
    scores_all = np.where(mask, q_all @ k_all.transpose(0, 2, 1) / np.sqrt(hs), -np.inf)
    att_all = softmax(scores_all)
    if args.head is None:
        prev = [np.mean([att_all[i, t, t - 1] for t in range(1, T)]) for i in range(nh)]
        hd = int(np.argmax(prev))
    else:
        hd = args.head % nh
    q, k, v, S, A = q_all[hd], k_all[hd], v_all[hd], scores_all[hd], att_all[hd]

    print(f"STEP 4: self-attention, layer 0, head {hd} (of {nh}). Q, K and V all come from the "
          "SAME tokens:\n")
    print(f"   q = x W_q   \"what am I looking for?\"     ({T}, {C}) @ ({C}, {hs}) → ({T}, {hs})")
    print(f"   k = x W_k   \"what do I contain?\"")
    print(f"   v = x W_v   \"what do I hand over if you pick me?\"\n")
    w = len(vec(q[0]))
    print(f"   {'':7s}{'q (first ' + str(SHOW) + ' of ' + str(hs) + ')':<{w + 2}s}"
          f"{'k':<{w + 2}s}v")
    for t in range(T):
        print(f"   {ch(prompt[t]):>2s} t={t:<2d} {yellow(vec(q[t]))}  {yellow(vec(k[t]))}  "
              f"{vec(v[t])}")
    print(dim(f"\n   All {nh} heads do this in parallel with their own W_q, W_k, W_v "
              f"(one ({C}, {3 * C}) matmul computes them all)."))
    pause()

    # ------------------------------------------------------------------ STEP 5
    print(f"STEP 5: scores = q · k / √{hs} for every (query, key) pair, plus the CAUSAL MASK\n")
    print("        keys →  " + "".join(f"{ch(c):>6s}" for c in prompt))
    for i in range(T):
        cells = "".join(dim(f"{'·':>6s}") if j > i else f"{S[i, j]:+6.1f}" for j in range(T))
        print(f"  query {ch(prompt[i]):>2s} (t={i:<2d})" + cells)
    print(dim("\n   · = -inf: a token may NOT look at later tokens (they don't exist yet when "
              "generating)."))
    print(dim(f"   Dividing by √{hs} keeps the scores' spread ~1 however big the head is "
              "(README section 3)."))
    pause()

    # ------------------------------------------------------------------ STEP 6
    print("STEP 6: softmax along each row → attention weights (each row sums to 100%)\n")
    attention_grid(A, prompt)
    print(dim("\n   Read a ROW: where that token looks. Brightest cell per row in yellow. "
              "exp(-inf) = 0, so the"))
    print(dim("   future gets exactly 0%. Row 0 can only look at itself."))
    off1 = np.mean([A[t, t - 1] for t in range(1, T)])
    print(f"\n   This head puts on average {off1:.0%} of its attention on the PREVIOUS token "
          f"(uniform over the visible\n   tokens would give {np.mean([1 / (t + 1) for t in range(1, T)]):.0%}). "
          "Try --head 0..3 to see the others.")
    pause()

    # ------------------------------------------------------------------ STEP 7
    t = T - 1
    print(f"STEP 7: the output of the LAST token {cyan(repr(prompt[t]))} = weighted mix of the "
          "value vectors\n")
    for j in range(T):
        print(f"   {A[t, j]:5.1%} × v[{ch(prompt[j])}] {vec(v[j])}  {yellow(bar(A[t, j], 16))}")
    y_last = A[t] @ v
    print(f"   {'= y':>13s} {green(vec(y_last))}")
    print(dim(f"\n   y is {hs} numbers: what head {hd} brings to '{ch(prompt[t])}' from the "
              "tokens it attended to."))
    pause()

    # ------------------------------------------------------------------ STEP 8
    Y = (att_all @ v_all).transpose(1, 0, 2).reshape(T, C)  # concat the heads
    attn_out = Y @ blk.attn.c_proj.params["W"] + blk.attn.c_proj.params["b"]
    x1 = x + attn_out
    h2 = layer_norm(x1, blk.ln_2)
    hid = gelu(h2 @ blk.mlp.c_fc.params["W"] + blk.mlp.c_fc.params["b"])
    mlp_out = hid @ blk.mlp.c_proj.params["W"] + blk.mlp.c_proj.params["b"]
    x2 = x1 + mlp_out
    print("STEP 8: finish block 0 (shown for the last token)\n")
    print(f"   concat the {nh} heads' y → ({C},), then × W_o ({C}, {C}) = attn_out:")
    print(f"       {spark(attn_out[t])}")
    print(f"   residual add:  x = x + attn_out         "
          f"‖x‖ {np.linalg.norm(x[t]):.2f} → {np.linalg.norm(x1[t]):.2f}")
    print(f"   LayerNorm, then MLP: {C} → {4 * C} (GELU) → {C}")
    print(f"     hidden GELU units ({4 * C}), {np.mean(hid[t] > 0.05):.0%} clearly active:")
    for r in range(0, 4 * C, 64):
        print(f"       {spark(hid[t, r:r + 64], np.abs(hid[t]).max())}")
    print(f"   residual add:  x = x + mlp_out          "
          f"‖x‖ {np.linalg.norm(x1[t]):.2f} → {np.linalg.norm(x2[t]):.2f}")
    print(dim("\n   Attention MOVES information between tokens; the MLP then processes each "
              "token on its own."))
    print(dim("   The '+' (residual stream) means each sublayer only ADDS an update to x."))
    pause()

    # ------------------------------------------------------------------ STEP 9
    x = x2
    for l in range(1, L):
        blk = model.blocks[l]
        h = layer_norm(x, blk.ln_1)
        qq, kk, vv = (split(z) for z in np.split(h @ blk.attn.c_attn.params["W"]
                                                  + blk.attn.c_attn.params["b"], 3, axis=-1))
        att = softmax(np.where(mask, qq @ kk.transpose(0, 2, 1) / np.sqrt(hs), -np.inf))
        Y = (att @ vv).transpose(1, 0, 2).reshape(T, C)
        x = x + Y @ blk.attn.c_proj.params["W"] + blk.attn.c_proj.params["b"]
        h2 = layer_norm(x, blk.ln_2)
        x = x + (gelu(h2 @ blk.mlp.c_fc.params["W"] + blk.mlp.c_fc.params["b"])
                 @ blk.mlp.c_proj.params["W"] + blk.mlp.c_proj.params["b"])
        print(f"STEP 9: block {l} does the same. Where do its {nh} heads look from the last "
              f"token {cyan(repr(prompt[t]))}?\n")
        print("          " + "".join(f"{ch(c):>6s}" for c in prompt))
        for i in range(nh):
            cells = "".join(f" {cell(w)}" for w in att[i, t])
            top = int(att[i, t].argmax())
            print(f"   head {i}  {cells}   → mostly '{ch(prompt[top])}' (t={top}, "
                  f"{att[i, t, top]:.0%})")
        print(dim("\n   Later layers read a residual stream that already mixes in earlier tokens, "
                  "so their"))
        print(dim("   patterns are about what those tokens now CONTAIN, not just where they are."))

        pause()

    # ------------------------------------------------------------------ STEP 10
    xf = layer_norm(x, model.ln_f)
    Wout = model.wte.params["weight"].T if model.lm_head is None else model.lm_head.params["W"]
    logits = xf @ Wout  # (T, V)
    check = model.forward(ids[None])[0]
    assert np.allclose(logits, check), "visual recomputation must match model.forward"
    p = softmax(logits[t])
    print(f"STEP 10: final LayerNorm, then logits = x @ W_head ({C}, {V}), softmax → "
          "next-char probabilities\n")
    print(f"   After {cyan(repr(prompt))} the model predicts:\n")
    for i in np.argsort(p)[::-1][:5]:
        print(f"     '{ch(chars[i])}'  {green(bar(p[i], 30))}  {p[i]:6.1%}")
    print(dim("\n   (These logits match model.forward() exactly: the steps above ARE the model.)"))
    pause()

    # ------------------------------------------------------------------ STEP 11
    print("STEP 11: one forward pass gave a prediction after EVERY prefix, not just the last\n")
    print("   Because of the causal mask, row t only saw characters 0..t, so it is exactly what")
    print("   the model would predict if the text stopped there. Training scores all of them at")
    print("   once (T predictions per sequence, in parallel); generation uses only the last.\n")
    for s in range(T):
        ps = softmax(logits[s])
        top = np.argsort(ps)[::-1][:3]
        nxt = prompt[s + 1] if s + 1 < T else None
        guesses = "  ".join(
            (green if nxt is not None and chars[i] == nxt else str)(f"'{ch(chars[i])}' {ps[i]:4.0%}")
            for i in top)
        actual = f"actual: '{ch(nxt)}'" if nxt is not None else "(next: generate.py samples here)"
        print(f"   {prompt[:s + 1]!r:>{T + 3}s} → {guesses}   {dim(actual)}")
    pause()

    print("RECAP\n")
    print(f"  * The {T} tokens were processed TOGETHER: every step above is a matmul over all "
          "tokens.")
    print("  * Each token = token embedding + position embedding.")
    print("  * Attention: q·k scores say who looks at whom, the causal mask hides the future,")
    print("    softmax turns scores into weights, and the output is a weighted mix of the v's.")
    print("  * Many heads look for different things; W_o mixes their results back into x.")
    print("  * The residual stream x only gets UPDATES added (attention, then MLP), block by block.")
    print("  * The last row's logits give the next-character probabilities; generate.py samples")
    print("    one, appends it, and repeats (with a KV cache so old tokens aren't recomputed).")
    print("\nNext: README sections 3-5, and the heatmaps in images/ "
          "(python transformer/make_figures.py redraws them).")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n(stopped)")
