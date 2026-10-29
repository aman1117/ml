"""
Attention, step by step, in your terminal, with numbers small enough to check by hand.

Part A (steps 1-7): ONE attention lookup with 4-number vectors you can read.
  The decoder of a date translator has read "wed 3 march 2021" and is about to
  write the MONTH. It asks a question (the query), every input token offers a
  label (its key) and some content (its value):

      scores  = q · k_j             how well does token j match what I need?
      scaled  = scores / √d_k       keep the numbers in a range softmax likes
      weights = softmax(scaled)     turn them into percentages that sum to 1
      context = Σ_j weights_j v_j   blend the values by those percentages

  Then the same thing for several queries at once as matrices,
  softmax(Q Kᵀ / √d_k) V, and what a padding mask does.
  (Real models learn their 64..12288-number vectors; these 4 are hand-made so
  each number has a meaning.)

Part B (steps 8+): the REAL trained date model (attention/date_model.pt, trained
  by seq2seq_pytorch.py, or trained here first in ~15 s) writes its answer one
  character at a time; you see where it looks for every character.

Run:
    python attention/attention_visual.py                      (press Enter to go on)
    python attention/attention_visual.py --date "3rd of july 1999"
    python attention/attention_visual.py --part a             (only the hand-made part)
"""
import argparse
import sys

import numpy as np

from attention_numpy import scaled_dot_product_attention

TTY = sys.stdout.isatty()
SHADES = " ·░▒▓█"

TOKENS = ["wed", "3", "march", "2021"]
KEY_FEATURES = ["weekday?", "day?", "month?", "year?"]
VALUE_FEATURES = ["day", "month", "year", "weekday"]
# Keys: "what kind of thing am I?"  (one row per input token)
K = np.array([[2.0, 0.0, 0.0, 0.0],    # wed
              [0.0, 2.0, 0.0, 0.5],    # 3      (a small number: a bit year-like too)
              [0.0, 0.0, 2.0, 0.0],    # march
              [0.0, 0.5, 0.0, 2.0]])   # 2021   (digits: a bit day-like too)
# Values: "what do I hand over if you pick me?"
V = np.array([[0.0, 0.0, 0.0, 3.0],    # wed   -> weekday 3
              [3.0, 0.0, 0.0, 0.0],    # 3     -> day 3
              [0.0, 3.0, 0.0, 0.0],    # march -> month 3
              [0.0, 0.0, 21.0, 0.0]])  # 2021  -> year 21
# Queries: "what am I looking for right now?"
QUERIES = {"write the DAY": [0.0, 4.0, 0.0, 1.0],
           "write the MONTH": [0.0, 1.0, 4.0, 0.0],
           "write the YEAR": [0.0, 1.0, 0.0, 4.0]}
PAD_KEY = np.array([0.0, 1.0, 1.0, 1.0])  # garbage: padding vectors are whatever the net computes
PAD_VALUE = np.array([9.0, 9.0, 9.0, 9.0])


def color(code, s):
    return f"\033[{code}m{s}\033[0m" if TTY else s


def cyan(s): return color("36;1", s)      # queries
def yellow(s): return color("33;1", s)    # keys / scores
def green(s): return color("32;1", s)     # results (weights, context)
def magenta(s): return color("35;1", s)   # values
def dim(s): return color("2", s)


def pause():
    if sys.stdin.isatty():
        try:
            input(dim("\n  [Enter] next step "))
        except EOFError:
            print()
    print("\n" + "=" * 72 + "\n")


def vec(v, fmt="{:5.2f}"):
    return "[" + " ".join(fmt.format(x) for x in v) + "]"


def bar(p, width=30):
    n = int(round(p * width))
    return "█" * n + "·" * (width - n)


def shade(w):
    return SHADES[min(len(SHADES) - 1, int(np.ceil(w * (len(SHADES) - 1) - 1e-9)))]


def grid(M, rows, cols, fmt="{:7.2f}", w=8):
    print(" " * 18 + "".join(f"{c:>{w}}" for c in cols))
    for r, row in zip(rows, M):
        print(f"  {r:>15} " + "".join(fmt.format(x).rjust(w) for x in row))


# ======================================================================= Part A

def part_a():
    q = np.array(QUERIES["write the MONTH"])
    d = len(q)

    print("STEP 1: the setup. A soft dictionary lookup with three kinds of vectors.\n")
    print(f"  The decoder has read {cyan(repr(' '.join(TOKENS)))} and must now "
          f"{cyan('write the MONTH')}.\n")
    print(f"  {cyan('query  q')}   what am I looking for?          (from the decoder, 1 vector)")
    print(f"  {yellow('keys   k_j')} what kind of thing is token j?  (from the encoder, 1 per token)")
    print(f"  {magenta('values v_j')} what does token j hand over?    (from the encoder, 1 per token)\n")
    print(f"  Here d_k = {d} and every number has a meaning (real models learn theirs):\n")
    print(f"     key features   = [{', '.join(KEY_FEATURES)}]")
    print(f"     value features = [{', '.join(VALUE_FEATURES)}]\n")
    print(f"  {cyan('q')} = {cyan(vec(q, '{:4.1f}'))}   \"I need a month. A day would be a weak "
          "second choice.\"\n")
    print(f"  {'token':>8}   {'key k_j':<22} {'value v_j'}")
    for t, k, v in zip(TOKENS, K, V):
        print(f"  {t:>8}   {yellow(vec(k, '{:4.1f}'))}  {magenta(vec(v, '{:4.1f}'))}")
    print(dim("\n  Like a Python dict, but the lookup is SOFT: instead of 'exact key match or"))
    print(dim("  KeyError', every key matches a bit, and we blend all values by how well."))
    pause()

    print(f"STEP 2: scores. Dot product of the query with EVERY key: q · k_j.\n")
    raw = K @ q
    for t, k, s in zip(TOKENS, K, raw):
        terms = " + ".join(f"{a:g}·{b:g}" for a, b in zip(q, k))
        print(f"  {t:>6}:  {terms:<34} = {yellow(f'{s:5.2f}')}")
    print(f"\n  Big score = the key points the same way as the query. "
          f"{yellow('march')} wins clearly ({raw[2]:g}).")
    print("  The dot product is a similarity measure: it's how attention compares CONTENT.")
    pause()

    print(f"STEP 3: scale by √d_k = √{d} = {np.sqrt(d):g}.\n")
    scaled = raw / np.sqrt(d)
    for t, s, sc in zip(TOKENS, raw, scaled):
        print(f"  {t:>6}:  {s:5.2f} / {np.sqrt(d):g} = {yellow(f'{sc:5.2f}')}")
    rng = np.random.default_rng(0)
    stds = {dd: np.std(np.sum(rng.standard_normal((20000, dd)) * rng.standard_normal((20000, dd)),
                              axis=1)) for dd in (4, 64, 512)}
    print("\n  Why? A dot product sums d_k terms, so for random vectors it grows like √d_k:")
    for dd, sd in stds.items():
        print(f"     d_k = {dd:>3}: std of q·k = {sd:5.1f}   (√d_k = {np.sqrt(dd):5.1f})")
    print("  Huge scores make softmax put ~100% on one key and its gradient ≈ 0 (no learning).")
    print("  Dividing by √d_k keeps the scores' spread ≈ 1 whatever the size. (README section 5.)")
    pause()

    print("STEP 4: softmax turns the scores into weights: positive, summing to 1.\n")
    e = np.exp(scaled)
    w = e / e.sum()
    print(f"  {'token':>6}   scaled   exp(scaled)   / {e.sum():.2f} = weight")
    for t, sc, ee, ww in zip(TOKENS, scaled, e, w):
        print(f"  {t:>6}   {sc:6.2f}   {ee:9.2f}       {green(f'{ww:6.3f}')}  {green(bar(ww))}")
    print(f"\n  {green(f'{w[2]:.0%}')} of the attention goes to {yellow('march')}. "
          "Nothing gets exactly 0: the lookup is soft,")
    print("  which is what makes it differentiable (gradients reach every key and value).")
    pause()

    print("STEP 5: the context = weighted sum of the VALUES.\n")
    print(f"  {'':>6}   weight × value{' ' * 20}features: {VALUE_FEATURES}")
    for t, ww, v in zip(TOKENS, w, V):
        print(f"  {t:>6}   {ww:5.3f} × {magenta(vec(v, '{:4.1f}'))}  = {vec(ww * v)}")
    ctx = w @ V
    print(f"  {'':>6}   {'─' * 50}")
    print(f"  {'sum':>6}   context = {green(vec(ctx))}")
    print(f"\n  Compare with march's value {magenta(vec(V[2], '{:4.1f}'))}: the context is MOSTLY")
    print(f"  the value of the top-weighted token (month ≈ {ctx[1]:.2f}), plus a little of the")
    print(f"  others (the year's 21 leaks in as {ctx[2]:.2f}). The decoder reads 'month 3' from it.")
    pause()

    print("STEP 6: all queries at once = matrices.  Out = softmax(Q Kᵀ / √d_k) V\n")
    names = list(QUERIES)
    Q = np.array([QUERIES[n] for n in names])
    print(f"  Q {Q.shape}: one ROW per query (3 output steps)     "
          f"K {K.shape}, V {V.shape}: one ROW per token\n")
    print(f"  {yellow('Q Kᵀ / √d_k')}  (3, 4): row i = query i's scores for every token")
    grid(Q @ K.T / np.sqrt(d), names, TOKENS)
    Out, A, _ = scaled_dot_product_attention(Q, K, V)
    print(f"\n  {green('A = softmax(...)')} row by row  (3, 4): every row sums to 1")
    grid(A, names, TOKENS, fmt="{:7.3f}")
    print("\n  as a heatmap (darker = more weight):")
    print(" " * 18 + "".join(f"{t:>8}" for t in TOKENS))
    for n, row in zip(names, A):
        print(f"  {n:>15} " + "".join(f"{shade(x) * 3:>8}" for x in row))
    print(f"\n  {green('Out = A V')}  (3, 4) = one context vector per query")
    grid(Out, names, VALUE_FEATURES)
    print("\n  Each query picked its own token: day → '3', month → 'march', year → '2021'.")
    print("  One matrix multiply does all lookups in parallel. A Transformer does exactly this")
    print("  with thousands of queries (one per token) at once.")
    pause()

    print("STEP 7: masking. Batches pad short inputs with <pad>, and <pad> must be ignored.\n")
    Kp, Vp = np.vstack([K, PAD_KEY]), np.vstack([V, PAD_VALUE])
    toks = TOKENS + ["<pad>"]
    print(f"  Add a <pad> token. Its key {yellow(vec(PAD_KEY, '{:4.1f}'))} and value "
          f"{magenta(vec(PAD_VALUE, '{:4.1f}'))} are junk.\n")
    out_nm, a_nm, _ = scaled_dot_product_attention(q[None], Kp, Vp)
    mask = np.array([True, True, True, True, False])
    out_m, a_m, _ = scaled_dot_product_attention(q[None], Kp, Vp, mask[None])
    print(f"  {'token':>6}   no mask{' ' * 27}with mask (score of <pad> := -∞)")
    for j, t in enumerate(toks):
        print(f"  {t:>6}   {a_nm[0, j]:5.3f} {bar(a_nm[0, j], 20)}     "
              f"{green(f'{a_m[0, j]:5.3f}')} {green(bar(a_m[0, j], 20))}")
    print(f"\n  context without mask: {vec(out_nm[0])}   ← <pad>'s 9s pollute every feature")
    print(f"  context with mask:    {green(vec(out_m[0]))}   ← back to ≈ march's value")
    print("\n  A mask sets forbidden scores to -∞ BEFORE the softmax, so exp(-∞) = 0: exactly zero")
    print("  weight and zero gradient. Preview: a CAUSAL mask (query i may only see keys ≤ i) is")
    print("  how GPT stops a token from peeking at the future (../transformer/).")
    pause()


# ======================================================================= Part B

def part_b(text):
    from seq2seq_pytorch import load_or_train_date_model, translate_date
    from tasks import clean_date_text

    model = load_or_train_date_model()
    text = clean_date_text(text)
    out, alpha = translate_date(model, text)
    steps = list(out) + ["<eos>"]
    n = len(steps)

    print("STEP 8: now a REAL trained model: the date translator from seq2seq_pytorch.py.\n")
    print(f"  input:  {cyan(text)}   ({len(text)} chars)")
    print(f"  The encoder turns each input char into a 128-number vector h_j (keys AND values).")
    print(f"  The decoder writes ONE char per step. At each step its state s_t is the query:")
    print(f"     α_t = softmax((s_t W_q) · (h_j W_k) / √64)   then   c_t = Σ_j α_tj h_j")
    print(f"  We'll watch all {n} steps. Each row of shades is one α_t (it sums to 1).")
    print(dim("  (h_j comes from a bidirectional GRU, so it also knows its neighbours: looking at"))
    print(dim("  any letter of 'march' can be enough to read the month.)"))
    pause()

    header = "              " + "".join(c if c != " " else "␣" for c in text)
    for t, (o, row) in enumerate(zip(steps, alpha)):
        print(f"STEP {9 + t}: output step t = {t + 1} of {n}, written so far: "
              f"{green(repr(out[:t]))}\n")
        print(header)
        print(f"  α_{t + 1:<2} {o:>6}  " + "".join(shade(w) for w in row))
        top = np.argsort(row)[::-1][:3]
        print()
        for j in top:
            print(f"     {row[j]:5.1%} on {text[j]!r:5} (position {j:2d})  {bar(row[j], 25)}")
        print(f"\n  → writes {green(repr(o))}")
        pause()

    print("The whole alignment at once (rows = output chars, columns = input chars):\n")
    print(header)
    for o, row in zip(steps, alpha):
        print(f"       {o:>6}  " + "".join(shade(w) for w in row))
    print(f"\n  result: {cyan(text)}  →  {green(out)}")
    pause()


def recap():
    print("RECAP\n")
    print("  * Attention = a soft, differentiable dictionary lookup:")
    print("      scores = q·k_j / √d_k → weights = softmax(scores) → context = Σ weights_j · v_j")
    print("  * Many queries at once: Out = softmax(Q Kᵀ / √d_k) V. Shapes (n,d)(d,m)→(n,m), (n,m)(m,d_v)→(n,d_v).")
    print("  * √d_k stops the scores from growing with the vector size (softmax would saturate).")
    print("  * Masks put -∞ on forbidden scores (padding now, the future in GPT) → weight exactly 0.")
    print("  * In seq2seq, the decoder state is the query and the encoder states are keys and values,")
    print("    so the decoder re-reads the input at every step instead of relying on ONE vector.")
    print("\nNext: README sections 5-7 (√d_k, masking, the backward pass), then "
          "`python attention/gradient_check.py`.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", default="wednesday 3 march 2021",
                    help="input for Part B (the trained date model)")
    ap.add_argument("--part", choices=["a", "b", "all"], default="all")
    args = ap.parse_args()
    if args.part in ("a", "all"):
        part_a()
    if args.part in ("b", "all"):
        part_b(args.date)
    recap()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n(stopped)")
