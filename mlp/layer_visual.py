"""
See one layer of neurons computed step by step, as pictures in your terminal.

Story (the "talent show" analogy):
  - EXAMPLES (rows of X) = contestants. Each has 3 FEATURES: sing, dance, act.
  - NEURONS (columns of W) = judges. Each judge has their own opinion of how
    much each skill matters (weights) and a mood (bias).
  - Z = the scoreboard: one score per contestant (row) per judge (column).

Run:
    python mlp/layer_visual.py        (press Enter to go to the next step)
"""
import sys

import numpy as np

FEATURES = ["sing", "dance", "act"]
EXAMPLES = ["Riya", "Arjun"]
NEURONS = ["Judge A", "Judge B"]

X = np.array([[1.0, 2.0, 3.0],     # Riya's skills
              [4.0, 5.0, 6.0]])    # Arjun's skills
W = np.array([[1.0, 0.5],          # how much each judge values "sing"
              [0.0, 0.5],          # ... "dance"
              [-1.0, 0.5]])        # ... "act"
b = np.array([0.0, 1.0])           # Judge A is neutral, Judge B is generous (+1)

TTY = sys.stdout.isatty()


def color(code, s):
    return f"\033[{code}m{s}\033[0m" if TTY else s


def cyan(s): return color("36;1", s)      # examples / rows
def yellow(s): return color("33;1", s)    # neurons / columns
def green(s): return color("32;1", s)     # results
def dim(s): return color("2", s)


def n(v):
    return f"{v:g}"


def p(v):
    """Number for use inside a sum: wrap negatives in parentheses."""
    return f"({n(v)})" if v < 0 else n(v)


def pause():
    if sys.stdin.isatty():
        input(dim("\n  [Enter] next step "))
    print("\n" + "=" * 70 + "\n")


def table(M, row_labels, col_labels, hl_row=None, hl_col=None, w=8):
    """Print a labelled matrix, highlighting one row (cyan) and one column (yellow)."""
    head = " " * 10 + "".join(
        (yellow if j == hl_col else str)(f"{c:>{w}}") for j, c in enumerate(col_labels))
    print(head)
    for i, (label, row) in enumerate(zip(row_labels, M)):
        cells = []
        for j, v in enumerate(row):
            s = f"{n(v):>{w}}"
            if i == hl_row and j == hl_col:
                s = green(s)
            elif i == hl_row:
                s = cyan(s)
            elif j == hl_col:
                s = yellow(s)
            cells.append(s)
        lab = cyan(f"{label:>8}") if i == hl_row else f"{label:>8}"
        print(f"{lab}  " + "".join(cells))


print("STEP 1: the inputs X. One ROW per example (contestant), one COLUMN per feature.\n")
table(X, EXAMPLES, FEATURES)
print(f"\n  X has shape (N, D_in) = {X.shape}: {X.shape[0]} examples, {X.shape[1]} features each.")
pause()

print("STEP 2: ONE neuron = one judge. It multiplies each feature by its own weight,")
print("adds them up, then adds its bias.\n")
x, w = X[0], W[:, 0]
z = x @ w + b[0]
print(f"  {cyan('Riya')}'s skills       {yellow('Judge A')}'s weights")
ends = ["──┐", f"──┼──▶ sum + bias({n(b[0])}) ──▶ {green(n(z))}", "──┘"]
for f, xv, wv, end in zip(FEATURES, x, w, ends):
    print(f"   {f:>5} = {cyan(f'{n(xv):>3}')}  ──── × {yellow(f'{n(wv):>4}')} ──{end}")
print(f"\n  {' + '.join(f'{n(xv)}·{p(wv)}' for xv, wv in zip(x, w))} + {n(b[0])} = {green(n(z))}")
print(f"\n  Judge A likes singing (+1), ignores dancing (0), dislikes acting (-1).")
pause()

print("STEP 3: a LAYER = several judges. Stand each judge's weights up as a COLUMN,")
print("put the columns side by side, and that grid is the weight matrix W.\n")
print(f"   {yellow('Judge A')}   {yellow('Judge B')}            W  (one column per neuron)")
for i, f in enumerate(FEATURES):
    a, bb = n(W[i, 0]), n(W[i, 1])
    print(f"    [{a:>4} ]    [{bb:>4} ]    {'──▶' if i == 1 else '   '}    {f:>5} │ {a:>4}  {bb:>4} │")
print(f"\n  W has shape (D_in, D_out) = {W.shape}: one ROW per feature, one COLUMN per neuron.")
print(f"  b = {b.tolist()}: one bias per neuron.")
pause()

print("STEP 4: fill the scoreboard. Every cell = (a ROW of X) · (a COLUMN of W) + bias.\n")
Z = np.full((2, 2), np.nan)
for i in range(X.shape[0]):
    for j in range(W.shape[1]):
        Z[i, j] = X[i] @ W[:, j] + b[j]
        print(f"  {cyan(EXAMPLES[i])} × {yellow(NEURONS[j])}:\n")
        print(f"     row of X    {cyan('[' + '  '.join(f'{n(v):>3}' for v in X[i]) + ' ]')}")
        print(f"     column of W {yellow('[' + '  '.join(f'{n(v):>3}' for v in W[:, j]) + ' ]')}")
        print(f"{'':20}{'    '.join('↓' for _ in FEATURES)}   multiply pairs, add up, add bias")
        terms = " + ".join(p(xv * wv) for xv, wv in zip(X[i], W[:, j]))
        print(f"     {terms} + {n(b[j])} = {green(n(Z[i, j]))}\n")
        shown = np.where(np.isnan(Z), 0, Z)
        print("  scoreboard Z so far:")
        table(shown, EXAMPLES, NEURONS, hl_row=i, hl_col=j)
        pause()

print("STEP 5: the whole scoreboard in ONE line of code:  Z = X @ W + b\n")
Z_np = X @ W + b
assert np.allclose(Z, Z_np)
print(f"   X {X.shape}        @      W {W.shape}      +  b {b.shape}   =   Z {Z_np.shape}")
print(f"   {cyan('examples')} × features     features × {yellow('neurons')}              {cyan('examples')} × {yellow('neurons')}")
print(f"\n   The two inner numbers must match: X has {X.shape[1]} features per example, W has {W.shape[0]} weights")
print("   per neuron (every judge needs exactly one weight per skill).\n")
table(Z_np, EXAMPLES, NEURONS)
print(f"\n  {green('Same numbers as the cell-by-cell loop')}. NumPy just does all cells at once (much faster).")
pause()

print("STEP 6: activation. Apply relu = max(0, z) to EVERY cell separately.\n")
A = np.maximum(0, Z_np)
print("        Z                         A = relu(Z)")
for i, label in enumerate(EXAMPLES):
    z_row = "".join(f"{n(v):>6}" for v in Z_np[i])
    a_row = "".join(f"{n(v):>6}" for v in A[i])
    print(f"  {label:>6} {z_row}     ──▶     {label:>6} {a_row}")
print(f"\n  A has the same shape {A.shape}. A becomes the input X of the NEXT layer:")
print("  its 2 columns (Judge A, Judge B) are the next layer's 2 features.")
print("\nThat's a whole layer: matrix multiply, add bias, apply activation. Stack these = an MLP.")
