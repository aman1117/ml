"""
Generate the pictures used in mlp/README.md (saved to mlp/images/).

Every figure teaches ONE idea, and every number in them is computed here
(nothing is made up). Re-run any time:

    python mlp/make_figures.py
"""
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

from mlp_numpy import ACTIVATIONS, MLP, make_spiral, train, train_val_split  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "images")
CLASS_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c"]
plt.rcParams.update({"font.size": 12, "axes.titlesize": 14, "axes.titleweight": "bold"})


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", os.path.relpath(path, os.path.dirname(HERE)))


def arrow(ax, a, b, color="black", lw=1.5, style="-|>", ls="-", rad=0.0, ms=14):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, color=color, lw=lw, ls=ls,
                                 mutation_scale=ms, connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=0, shrinkB=0))


def node(ax, xy, text, r=0.32, fc="white", ec="black", fs=12):
    ax.add_patch(plt.Circle(xy, r, fc=fc, ec=ec, lw=1.8, zorder=3))
    ax.text(*xy, text, ha="center", va="center", fontsize=fs, zorder=4)


def box(ax, xy, w, h, text, fc="white", ec="black", fs=12):
    x, y = xy
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.05",
                                fc=fc, ec=ec, lw=1.8, zorder=3))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, zorder=4)


def blank(ax, xlim, ylim):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.axis("off")


# ---------------------------------------------------------------------------
# 1. One neuron
# ---------------------------------------------------------------------------
def fig_neuron():
    x = np.array([1.0, 2.0, 3.0])
    w = np.array([0.5, 0.5, 0.5])
    b = 1.0
    z = x @ w + b
    a = max(0.0, z)

    fig, ax = plt.subplots(figsize=(11, 4.6))
    blank(ax, (-1.2, 10.2), (-1.4, 3.4))
    names = ["sing", "dance", "act"]
    ys = [2.4, 1.0, -0.4]
    s = (4.2, 1.0)
    for name, xv, wv, y in zip(names, x, w, ys):
        node(ax, (0, y), f"{xv:g}", fc="#d8ecff")
        ax.text(-0.55, y, f"x ({name})", ha="right", va="center", fontsize=11)
        arrow(ax, (0.32, y), (s[0] - 0.55, s[1] + (y - 1.0) * 0.25))
        ax.text(2.1, (y + s[1]) / 2 + 0.15, f"× w = {wv:g}", fontsize=11, color="#b35900",
                ha="center", bbox=dict(fc="white", ec="none", pad=1))
    node(ax, s, "Σ + b", r=0.55, fc="#fff3d6")
    ax.text(s[0], s[1] + 1.05, f"bias b = {b:g}", ha="center", fontsize=11, color="#b35900")
    arrow(ax, (s[0], s[1] + 0.85), (s[0], s[1] + 0.57))
    arrow(ax, (s[0] + 0.55, s[1]), (6.2, s[1]))
    ax.text(5.35, s[1] + 0.2, f"z = {z:g}", ha="center", fontsize=12)
    box(ax, (6.9, s[1]), 1.3, 0.8, "ReLU\nmax(0, z)", fc="#e7f7e7")
    arrow(ax, (7.6, s[1]), (8.9, s[1]))
    node(ax, (9.3, s[1]), f"{a:g}", fc="#e7f7e7")
    ax.text(9.3, s[1] - 0.6, "output a", ha="center", fontsize=11)
    calc = " + ".join(f"{xv:g}·{wv:g}" for xv, wv in zip(x, w))
    ax.text(4.5, -1.15, f"z = {calc} + {b:g} = {z:g}        a = ReLU({z:g}) = {a:g}",
            ha="center", fontsize=12, bbox=dict(fc="#f4f4f4", ec="#bbbbbb"))
    ax.set_title("One neuron: multiply each input by its weight, add them up, add the bias, then apply an activation")
    save(fig, "neuron.png")


# ---------------------------------------------------------------------------
# 2. A layer = matrix multiplication
# ---------------------------------------------------------------------------
def draw_matrix(ax, M, x0, y0, row_labels, col_labels, title, hl_rows=(), hl_cols=(),
                hl_cells=(), color="#ffe08a", cell=1.0, fmt="{:g}"):
    R, C = M.shape
    for i in range(R):
        for j in range(C):
            fc = "white"
            if (i, j) in hl_cells:
                fc = "#9be29b"
            elif i in hl_rows or j in hl_cols:
                fc = color
            ax.add_patch(Rectangle((x0 + j * cell, y0 - (i + 1) * cell), cell, cell,
                                   fc=fc, ec="black", lw=1.2))
            ax.text(x0 + (j + 0.5) * cell, y0 - (i + 0.5) * cell, fmt.format(M[i, j]),
                    ha="center", va="center", fontsize=13)
    for i, lab in enumerate(row_labels):
        ax.text(x0 - 0.12, y0 - (i + 0.5) * cell, lab, ha="right", va="center", fontsize=11)
    for j, lab in enumerate(col_labels):
        ax.text(x0 + (j + 0.5) * cell, y0 + 0.15, lab, ha="center", va="bottom", fontsize=10,
                rotation=0)
    ax.text(x0 + C * cell / 2, y0 + 0.95, title, ha="center", va="bottom", fontsize=13,
            fontweight="bold")
    ax.text(x0 + C * cell / 2, y0 - R * cell - 0.25, f"shape {M.shape}", ha="center", va="top",
            fontsize=11, color="#555555")


def fig_layer_matmul():
    X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    W = np.array([[1.0, 0.5], [0.0, 0.5], [-1.0, 0.5]])
    b = np.array([0.0, 1.0])
    Z = X @ W + b
    i, j = 0, 1

    fig, ax = plt.subplots(figsize=(13, 5.6))
    blank(ax, (-1.6, 17.2), (-5.4, 2.0))
    draw_matrix(ax, X, 0, 0, ["Riya", "Arjun"], ["sing", "dance", "act"],
                "X  (inputs)\none ROW per example", hl_rows=[i], color="#bfe3ff")
    ax.text(3.75, -1.0, "@", fontsize=26, ha="center", va="center")
    draw_matrix(ax, W, 5.4, 0.5, ["sing", "dance", "act"], ["Judge\nA", "Judge\nB"],
                "W  (weights)\none COLUMN per neuron", hl_cols=[j], color="#ffd59e")
    ax.text(8.0, -1.0, "+", fontsize=26, ha="center", va="center")
    draw_matrix(ax, b[None, :], 8.7, -0.5, [""], ["A", "B"], "b (bias)", hl_cols=[j],
                color="#ffd59e")
    ax.text(11.2, -1.0, "=", fontsize=26, ha="center", va="center")
    draw_matrix(ax, Z, 12.7, 0, ["Riya", "Arjun"], ["Judge\nA", "Judge\nB"],
                "Z  (outputs)\nexample × neuron", hl_cells=[(i, j)])
    terms = " + ".join(f"{xv:g}·{wv:g}" for xv, wv in zip(X[i], W[:, j]))
    ax.text(7.8, -4.3,
            f"Green cell = blue ROW of X  ·  orange COLUMN of W  +  bias\n"
            f"{terms} + {b[j]:g} = {Z[i, j]:g}",
            ha="center", va="center", fontsize=13, bbox=dict(fc="#f4f4f4", ec="#bbbbbb"))
    ax.text(7.8, -5.3, "(N, D_in) @ (D_in, D_out) → (N, D_out):   (2, 3) @ (3, 2) → (2, 2)"
            "   the inner sizes must match",
            ha="center", va="center", fontsize=12, color="#444444")
    ax.set_title("A layer of neurons = one matrix multiplication (every example × every neuron at once)",
                 pad=28)
    save(fig, "layer_matmul.png")


# ---------------------------------------------------------------------------
# 3. Forward and backward pass through a network
# ---------------------------------------------------------------------------
def fig_network():
    sizes = [2, 4, 3]
    xs = [0, 4, 8]
    fig, ax = plt.subplots(figsize=(12, 6.8))
    blank(ax, (-2.2, 11.2), (-4.6, 4.2))
    pos = []
    for x, n in zip(xs, sizes):
        ys = np.linspace(1.4 * (n - 1) / 2, -1.4 * (n - 1) / 2, n)
        pos.append([(x, y) for y in ys])
    for a_layer, b_layer in zip(pos[:-1], pos[1:]):
        for p in a_layer:
            for q in b_layer:
                ax.plot([p[0], q[0]], [p[1], q[1]], color="#c8c8c8", lw=1, zorder=1)
    fills = ["#d8ecff", "#fff3d6", "#e7f7e7"]
    labels = [["x1", "x2"], ["h1", "h2", "h3", "h4"], ["class 0", "class 1", "class 2"]]
    for layer, fc, labs in zip(pos, fills, labels):
        for p, lab in zip(layer, labs):
            node(ax, p, lab, r=0.42, fc=fc, fs=10)
    under = ["input X\n(N, 2)", "hidden A1 = ReLU(Z1)\n(N, 4)", "logits Z2 → softmax\n(N, 3)"]
    for x, t in zip(xs, under):
        ax.text(x, -2.75, t, ha="center", va="top", fontsize=11)
    ax.text(2, 2.75, "W1 (2, 4), b1 (4,)", ha="center", fontsize=11, color="#b35900")
    ax.text(6, 2.75, "W2 (4, 3), b2 (3,)", ha="center", fontsize=11, color="#b35900")
    ax.text(10.6, 0, "loss\nL", ha="center", va="center", fontsize=12, fontweight="bold")
    arrow(ax, (8.5, 0), (10.1, 0), color="black")

    arrow(ax, (-0.5, 3.6), (10.6, 3.6), color="#1f5fbf", lw=3, ms=22)
    ax.text(5, 3.75, "FORWARD PASS: compute the prediction   X → Z1 = X·W1 + b1 → A1 = ReLU(Z1) → Z2 = A1·W2 + b2",
            ha="center", va="bottom", fontsize=10.5, color="#1f5fbf", fontweight="bold")
    arrow(ax, (10.6, -3.9), (-0.5, -3.9), color="#c62828", lw=3, ms=22)
    ax.text(5, -4.05, "BACKWARD PASS (backprop): send the error back   dZ2 = p − y → dW2, dA1 → dZ1 → dW1",
            ha="center", va="top", fontsize=10.5, color="#c62828", fontweight="bold")
    ax.set_title("A 2 → 4 → 3 network: data flows forward, gradients flow backward", pad=28)
    save(fig, "network.png")


# ---------------------------------------------------------------------------
# 4. Activation functions and their slopes
# ---------------------------------------------------------------------------
def fig_activations():
    z = np.linspace(-6, 6, 400)
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.6), sharex=True)
    for col, name in enumerate(["sigmoid", "tanh", "relu"]):
        f, df = ACTIVATIONS[name]
        a = f(z)
        d = df(z, a)
        ax, axd = axes[0, col], axes[1, col]
        ax.plot(z, a, lw=3, color="#1f5fbf")
        axd.plot(z, d, lw=3, color="#c62828")
        ax.set_title(name)
        for axx in (ax, axd):
            axx.axhline(0, color="#999999", lw=0.8)
            axx.axvline(0, color="#999999", lw=0.8)
            axx.grid(alpha=0.3)
        axd.set_ylim(-0.1, 1.15)
        axd.set_xlabel("z (weighted sum)")
        if name in ("sigmoid", "tanh"):
            for side in (-1, 1):
                axd.axvspan(3.5 * side, 6 * side, color="#ffcdd2", alpha=0.5)
            axd.text(0, 0.95 if name == "sigmoid" else 1.05, "", ha="center")
            axd.text(-4.75, 0.55, "slope ≈ 0\n(saturated:\nlearning\nstalls)",
                     ha="center", fontsize=9, color="#8e0000")
            axd.text(4.75, 0.55, "slope ≈ 0", ha="center", fontsize=9, color="#8e0000")
            peak = 0.25 if name == "sigmoid" else 1.0
            axd.annotate(f"max slope {peak:g}", (0, peak), (1.2, peak + 0.15 if peak < 1 else 0.7),
                         fontsize=10, arrowprops=dict(arrowstyle="->"))
        else:
            axd.axvspan(-6, 0, color="#ffcdd2", alpha=0.5)
            axd.text(-3, 0.5, "slope = 0\n(neuron 'dead'\nif stuck here)", ha="center",
                     fontsize=9, color="#8e0000")
            axd.text(3, 0.75, "slope = 1\n(gradient passes\nthrough fully)", ha="center", fontsize=9)
    axes[0, 0].set_ylabel("activation f(z)", color="#1f5fbf")
    axes[1, 0].set_ylabel("slope f'(z)\n(used in backprop)", color="#c62828")
    fig.suptitle("Activation functions (top) and their slopes (bottom). Slope ≈ 0 means almost no gradient flows back.",
                 fontweight="bold", y=1.0)
    fig.tight_layout()
    save(fig, "activations.png")


# ---------------------------------------------------------------------------
# helpers for decision-boundary plots
# ---------------------------------------------------------------------------
def boundary(ax, model, X, y, title):
    g = np.linspace(-1.15, 1.15, 250)
    gx, gy = np.meshgrid(g, g)
    pred = model.predict(np.c_[gx.ravel(), gy.ravel()]).reshape(gx.shape)
    from matplotlib.colors import ListedColormap
    ax.contourf(gx, gy, pred, levels=[-0.5, 0.5, 1.5, 2.5],
                cmap=ListedColormap(["#cfe3f5", "#ffe0c2", "#d3efd3"]))
    for c in range(3):
        m = y == c
        ax.scatter(X[m, 0], X[m, 1], s=16, color=CLASS_COLORS[c], edgecolor="black",
                   linewidth=0.4, label=f"class {c}")
    ax.set_title(title, fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")


# ---------------------------------------------------------------------------
# 5. Why we need hidden layers + non-linearity
# ---------------------------------------------------------------------------
def fig_why_nonlinearity():
    X, y = make_spiral()
    Xtr, ytr, Xval, yval = train_val_split(X, y)
    models = [("No hidden layer (linear model)", [2, 3], "relu"),
              ("Hidden layer, NO activation\n(linear layers stacked)", [2, 64, 3], "linear"),
              ("Hidden layer + ReLU (MLP)", [2, 64, 3], "relu")]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, (title, sizes, act) in zip(axes, models):
        if act == "linear":
            ACTIVATIONS["linear"] = (lambda z: z, lambda z, a: np.ones_like(z))
        m = MLP(sizes, act)
        train(m, Xtr, ytr, Xval, yval, lr=0.5 if act == "relu" else 0.1, epochs=300,
              verbose=False)
        acc = (m.predict(Xval) == yval).mean()
        boundary(ax, m, X, y, f"{title}\nvalidation accuracy {acc:.0%}")
    ACTIVATIONS.pop("linear", None)
    fig.suptitle("Spirals can't be split by straight lines. Only the non-linearity (ReLU) lets the network bend its boundaries.",
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "why_nonlinearity.png")


# ---------------------------------------------------------------------------
# 6. Gradient descent and the learning rate
# ---------------------------------------------------------------------------
def fig_gradient_descent():
    # A simple bowl-shaped loss: steep in w2, shallow in w1.
    def loss(w1, w2):
        return 0.5 * (w1 ** 2 + 10 * w2 ** 2)

    def grad(w):
        return np.array([w[0], 10 * w[1]])

    g1, g2 = np.meshgrid(np.linspace(-5, 5, 200), np.linspace(-3, 3, 200))
    settings = [(0.02, "lr = 0.02: too small\n(barely moves)"),
                (0.15, "lr = 0.15: good\n(reaches the bottom)"),
                (0.205, "lr = 0.205: too big\n(bounces and blows up)")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, (lr, title) in zip(axes, settings):
        ax.contour(g1, g2, loss(g1, g2), levels=np.geomspace(0.05, 60, 14), cmap="viridis",
                   linewidths=1)
        w = np.array([-4.5, 1.2])
        path = [w.copy()]
        for _ in range(25):
            w = w - lr * grad(w)
            path.append(w.copy())
        path = np.array(path)
        ax.plot(path[:, 0], path[:, 1], "o-", color="#c62828", ms=4, lw=1.5)
        ax.plot(*path[0], "s", color="black", ms=8)
        ax.plot(0, 0, "*", color="gold", ms=18, mec="black")
        ax.set_xlim(-5, 5)
        ax.set_ylim(-3, 3)
        ax.set_xlabel("weight w1")
        ax.set_ylabel("weight w2")
        ax.set_title(f"{title}\nloss after 25 steps: {loss(*path[-1]):.3g}", fontsize=12)
    fig.suptitle("Gradient descent: each step moves the weights downhill (w = w − lr·gradient). ■ = start, ★ = lowest loss",
                 fontweight="bold")
    fig.tight_layout()
    save(fig, "gradient_descent.png")


# ---------------------------------------------------------------------------
# 7. Backprop on a computational graph (same numbers as backprop_by_hand.py)
# ---------------------------------------------------------------------------
def fig_backprop_graph():
    x1, x2, y = 1.0, -2.0, 0.5
    w1, w2, b1, w3, b2 = 0.3, -0.1, 0.0, 0.8, 0.1
    z1 = w1 * x1 + w2 * x2 + b1
    h = math.tanh(z1)
    yhat = w3 * h + b2
    L = 0.5 * (yhat - y) ** 2
    d_yhat = yhat - y
    d_h = d_yhat * w3
    d_z1 = d_h * (1 - h ** 2)
    grads = {"w1": d_z1 * x1, "w2": d_z1 * x2, "b1": d_z1, "w3": d_yhat * h, "b2": d_yhat}

    fig, ax = plt.subplots(figsize=(14, 7.4))
    blank(ax, (-1.4, 15.2), (-5.0, 5.0))

    def gnode(xy, top, val, g=None, fc="white", w=2.1):
        txt = f"{top}\n= {val:.4g}"
        box(ax, xy, w, 0.95, txt, fc=fc, fs=11)
        if g is not None:
            ax.text(xy[0], xy[1] - 0.72, f"∂L/∂{top.split()[0]} = {g:.4f}", ha="center",
                    va="top", fontsize=10.5, color="#c62828", fontweight="bold")

    params = {"w1": (0, 4.2), "x1": (0, 2.4), "w2": (0, 0.6), "x2": (0, -1.2), "b1": (0, -3.0)}
    vals = {"w1": w1, "x1": x1, "w2": w2, "x2": x2, "b1": b1}
    for k, p in params.items():
        is_param = k[0] in "wb"
        gnode(p, k, vals[k], grads.get(k), fc="#fff3d6" if is_param else "#d8ecff", w=1.5)
        arrow(ax, (p[0] + 0.8, p[1]), (3.0, 0.6))
    gnode((4.0, 0.6), "z1 = w1·x1 + w2·x2 + b1", z1, d_z1, w=3.9)
    arrow(ax, (5.95, 0.6), (6.9, 0.6))
    gnode((7.9, 0.6), "h = tanh(z1)", h, d_h, w=2.1)
    gnode((7.9, 3.0), "w3", w3, grads["w3"], fc="#fff3d6", w=1.5)
    gnode((7.9, -2.0), "b2", b2, grads["b2"], fc="#fff3d6", w=1.5)
    arrow(ax, (8.95, 0.6), (9.9, 0.6))
    arrow(ax, (8.65, 3.0), (10.2, 1.1))
    arrow(ax, (8.65, -2.0), (10.2, 0.1))
    gnode((11.1, 0.6), "ŷ = w3·h + b2", yhat, d_yhat, w=2.2)
    arrow(ax, (12.2, 0.6), (13.1, 0.6))
    gnode((14.0, 0.6), "L = ½(ŷ − y)²", L, None, fc="#ffd6d6", w=2.0)
    ax.text(14.0, -0.3, f"target y = {y:g}", ha="center", va="top", fontsize=10)

    ax.text(7.0, -4.55,
            "Black = forward values (left → right).   Red = gradients, computed right → left with the chain rule:\n"
            "each node's red number = (red number of the node it feeds into) × (local slope).  "
            f"e.g. ∂L/∂z1 = ∂L/∂h × (1 − h²) = {d_h:.4f} × {1 - h ** 2:.4f} = {d_z1:.4f}",
            ha="center", va="center", fontsize=11, bbox=dict(fc="#f4f4f4", ec="#bbbbbb"))
    ax.set_title("Backpropagation on the tiny network from backprop_by_hand.py", pad=8)
    save(fig, "backprop_graph.png")


# ---------------------------------------------------------------------------
# 8. Overfitting
# ---------------------------------------------------------------------------
def fig_overfitting():
    X, y = make_spiral(n_per_class=200, noise=0.6, seed=1)
    perm = np.random.default_rng(0).permutation(len(X))
    Xtr, ytr = X[perm[:60]], y[perm[:60]]
    Xval, yval = X[perm[-300:]], y[perm[-300:]]
    runs = {}
    for reg in (0.0, 1e-2):
        m = MLP([2, 256, 256, 3], "relu")
        h = train(m, Xtr, ytr, Xval, yval, lr=0.1, epochs=2000, batch_size=60, reg=reg,
                  verbose=False)
        runs[reg] = (m, h)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    h = runs[0.0][1]
    ep = [r["epoch"] for r in h]
    ax = axes[0]
    ax.plot(ep, [r["train_loss"] for r in h], lw=2.5, label="training loss")
    ax.plot(ep, [r["val_loss"] for r in h], lw=2.5, label="validation loss (new data)")
    best = min(h, key=lambda r: r["val_loss"])
    ax.axvline(best["epoch"], color="black", ls="--", lw=1)
    ax.annotate(f"best val loss {best['val_loss']:.2f}\n(epoch {best['epoch']}):\nstop here",
                (best["epoch"], best["val_loss"]), (best["epoch"] + 250, 0.25),
                arrowprops=dict(arrowstyle="->"), fontsize=10)
    ax.annotate(f"memorizing:\nval loss climbs to {h[-1]['val_loss']:.2f}",
                (2000, h[-1]["val_loss"]), (1150, 1.03), arrowprops=dict(arrowstyle="->"),
                fontsize=10)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_ylim(0, 1.2)
    ax.legend(loc="center right")
    ax.grid(alpha=0.3)
    ax.set_title("No regularization: train ↓ but val ↑")

    for ax, reg, title in [(axes[1], 0.0, "No regularization"), (axes[2], 1e-2, "L2 regularization (reg = 0.01)")]:
        m, hh = runs[reg]
        boundary(ax, m, Xtr, ytr,
                 f"{title}\ntrain acc {hh[-1]['train_acc']:.0%}, val loss {hh[-1]['val_loss']:.2f}")
    axes[1].text(0, -1.3, "boundary twists around individual\nnoisy points = memorized noise", ha="center",
                 va="top", fontsize=10, color="#8e0000")
    axes[2].text(0, -1.3, "simpler, smoother boundary\n= better on new data", ha="center", va="top",
                 fontsize=10, color="#1b5e20")
    fig.suptitle("Overfitting: 60 noisy training points, a 67k-parameter network (dots = the training points)",
                 fontweight="bold")
    fig.tight_layout()
    save(fig, "overfitting.png")


if __name__ == "__main__":
    fig_neuron()
    fig_layer_matmul()
    fig_network()
    fig_activations()
    fig_why_nonlinearity()
    fig_gradient_descent()
    fig_backprop_graph()
    fig_overfitting()
