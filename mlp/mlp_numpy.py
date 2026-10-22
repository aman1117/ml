"""
Multi-Layer Perceptron (MLP) from scratch in NumPy.

Task: classify 2D points from 3 interleaved spirals. A straight line can't
separate them, so the network must learn a curved (non-linear) boundary.

Convention: one example per ROW. X has shape (N, D) (N examples, D features).
    Z = A_prev @ W + b        W: (fan_in, fan_out), b: (fan_out,)
    A = activation(Z)
(The RNN folder uses column vectors instead; both are just transposes.)

Run:
    python mlp/mlp_numpy.py
    python mlp/mlp_numpy.py --hidden 64 64 --activation tanh --epochs 500
"""
import argparse
import os
import warnings

import numpy as np

# NumPy 2.0 + Apple Accelerate raises spurious "... encountered in matmul"
# warnings even when results are correct. Only silence those.
warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def make_spiral(n_per_class=100, n_classes=3, noise=0.2, seed=0):
    """Points on n_classes spiral arms in roughly [-1, 1]^2. Returns X (N, 2), y (N,)."""
    rng = np.random.default_rng(seed)
    N = n_per_class * n_classes
    X = np.zeros((N, 2))
    y = np.zeros(N, dtype=int)
    for c in range(n_classes):
        ix = slice(n_per_class * c, n_per_class * (c + 1))
        r = np.linspace(0.05, 1.0, n_per_class)  # radius grows along the arm
        t = np.linspace(c * 4, (c + 1) * 4, n_per_class)  # angle
        t = t + rng.standard_normal(n_per_class) * noise
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = c
    return X, y


def train_val_split(X, y, val_frac=0.2, seed=0):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X))
    n_val = int(len(X) * val_frac)
    val, tr = perm[:n_val], perm[n_val:]
    return X[tr], y[tr], X[val], y[val]


# ----------------------------------------------------------------------------
# Building blocks
# ----------------------------------------------------------------------------
# Each activation: (function, derivative). The derivative gets both Z (input)
# and A (output) because some derivatives are easiest to write using the output.
ACTIVATIONS = {
    "relu": (lambda z: np.maximum(0.0, z),
             lambda z, a: (z > 0).astype(z.dtype)),
    "tanh": (np.tanh,
             lambda z, a: 1.0 - a ** 2),
    "sigmoid": (lambda z: 1.0 / (1.0 + np.exp(-z)),
                lambda z, a: a * (1.0 - a)),
}


def softmax(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)  # numerical stability
    e = np.exp(shifted)
    return e / e.sum(axis=1, keepdims=True)


def softmax_cross_entropy(logits, y):
    """Mean cross-entropy loss and its gradient w.r.t. the logits."""
    N = len(y)
    probs = softmax(logits)
    loss = -np.log(probs[np.arange(N), y] + 1e-12).mean()
    dlogits = probs.copy()
    dlogits[np.arange(N), y] -= 1.0  # p - onehot(y)
    dlogits /= N  # because the loss is a mean over the batch
    return loss, dlogits


# ----------------------------------------------------------------------------
# The model
# ----------------------------------------------------------------------------
class MLP:
    def __init__(self, sizes, activation="relu", seed=0):
        """sizes = [input_dim, hidden1, ..., hiddenK, n_classes]"""
        rng = np.random.default_rng(seed)
        self.sizes = list(sizes)
        self.activation = activation
        self.act, self.act_grad = ACTIVATIONS[activation]
        # He init for ReLU, Xavier-style init for tanh/sigmoid: keeps the
        # activations' spread roughly constant from layer to layer.
        gain = 2.0 if activation == "relu" else 1.0
        self.W, self.b = [], []
        for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
            self.W.append(rng.standard_normal((fan_in, fan_out)) * np.sqrt(gain / fan_in))
            self.b.append(np.zeros(fan_out))

    @property
    def n_layers(self):
        return len(self.W)

    def forward(self, X):
        """Returns logits (N, C) and a cache of what backward() needs."""
        cache = []
        A = X
        for i in range(self.n_layers):
            Z = A @ self.W[i] + self.b[i]
            # No activation on the last layer: it outputs raw scores (logits).
            A_out = Z if i == self.n_layers - 1 else self.act(Z)
            cache.append((A, Z, A_out))
            A = A_out
        return A, cache

    def backward(self, dlogits, cache, reg=0.0):
        """Backpropagation: apply the chain rule layer by layer, last to first."""
        dW = [None] * self.n_layers
        db = [None] * self.n_layers
        dA = dlogits  # gradient of the loss w.r.t. this layer's output
        for i in reversed(range(self.n_layers)):
            A_prev, Z, A_out = cache[i]
            if i == self.n_layers - 1:
                dZ = dA
            else:
                dZ = dA * self.act_grad(Z, A_out)  # through the activation
            dW[i] = A_prev.T @ dZ + reg * self.W[i]  # Z = A_prev @ W + b
            db[i] = dZ.sum(axis=0)
            dA = dZ @ self.W[i].T  # pass the gradient to the previous layer
        return dW, db

    def loss(self, X, y, reg=0.0):
        logits, _ = self.forward(X)
        data_loss, _ = softmax_cross_entropy(logits, y)
        reg_loss = 0.5 * reg * sum(np.sum(W * W) for W in self.W)
        return data_loss + reg_loss

    def loss_and_grads(self, X, y, reg=0.0):
        logits, cache = self.forward(X)
        data_loss, dlogits = softmax_cross_entropy(logits, y)
        reg_loss = 0.5 * reg * sum(np.sum(W * W) for W in self.W)
        dW, db = self.backward(dlogits, cache, reg)
        return data_loss + reg_loss, dW, db

    def sgd_step(self, dW, db, lr):
        """Gradient descent: move every parameter a small step downhill."""
        for i in range(self.n_layers):
            self.W[i] -= lr * dW[i]
            self.b[i] -= lr * db[i]

    def predict_proba(self, X):
        return softmax(self.forward(X)[0])

    def predict(self, X):
        return self.forward(X)[0].argmax(axis=1)


def accuracy(model, X, y):
    return (model.predict(X) == y).mean()


def train(model, Xtr, ytr, Xval, yval, lr=0.5, epochs=300, batch_size=32,
          reg=1e-3, print_every=50, seed=0, verbose=True):
    """Mini-batch stochastic gradient descent. Returns per-epoch history."""
    rng = np.random.default_rng(seed)
    history = []
    for epoch in range(1, epochs + 1):
        perm = rng.permutation(len(Xtr))  # shuffle every epoch
        for start in range(0, len(Xtr), batch_size):
            idx = perm[start:start + batch_size]
            _, dW, db = model.loss_and_grads(Xtr[idx], ytr[idx], reg)
            model.sgd_step(dW, db, lr)

        h = {
            "epoch": epoch,
            "train_loss": model.loss(Xtr, ytr),  # data loss only (reg=0)
            "val_loss": model.loss(Xval, yval),
            "train_acc": accuracy(model, Xtr, ytr),
            "val_acc": accuracy(model, Xval, yval),
        }
        history.append(h)
        if verbose and (epoch == 1 or epoch % print_every == 0):
            print(f"epoch {epoch:4d} | train loss {h['train_loss']:.3f} acc {h['train_acc']:.1%}"
                  f" | val loss {h['val_loss']:.3f} acc {h['val_acc']:.1%}")
    return history


# ----------------------------------------------------------------------------
# Visualisation and saving
# ----------------------------------------------------------------------------
def ascii_plot(model, X, y, width=64, height=28):
    """Text picture: background = predicted class, digits = actual data points."""
    regions = ".:-"  # background character for predicted class 0, 1, 2
    xs = np.linspace(-1.1, 1.1, width)
    ys = np.linspace(1.1, -1.1, height)  # top row = largest y
    gx, gy = np.meshgrid(xs, ys)
    pred = model.predict(np.c_[gx.ravel(), gy.ravel()]).reshape(height, width)
    canvas = [[regions[p % len(regions)] for p in row] for row in pred]
    for (px, py), label in zip(X, y):
        col = int(round((px + 1.1) / 2.2 * (width - 1)))
        row = int(round((1.1 - py) / 2.2 * (height - 1)))
        if 0 <= row < height and 0 <= col < width:
            canvas[row][col] = str(label)
    print("\n".join("".join(r) for r in canvas))
    print("background '.' ':' '-' = what the model predicts for class 0 / 1 / 2;"
          " digits = real data points")


def save_model(path, model):
    arrays = {f"W{i}": W for i, W in enumerate(model.W)}
    arrays.update({f"b{i}": b for i, b in enumerate(model.b)})
    np.savez(path, sizes=np.array(model.sizes), activation=np.array(model.activation), **arrays)


def load_model(path):
    d = np.load(path)
    model = MLP([int(s) for s in d["sizes"]], str(d["activation"]))
    model.W = [d[f"W{i}"] for i in range(model.n_layers)]
    model.b = [d[f"b{i}"] for i in range(model.n_layers)]
    return model


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hidden", type=int, nargs="*", default=[64],
                    help="hidden layer sizes, e.g. --hidden 64 64 (none = linear model)")
    ap.add_argument("--activation", choices=ACTIVATIONS, default="relu")
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--reg", type=float, default=1e-3, help="L2 regularization strength")
    ap.add_argument("--noise", type=float, default=0.2, help="how messy the spirals are")
    ap.add_argument("--save", default=os.path.join(here, "model.npz"))
    args = ap.parse_args()

    X, y = make_spiral(noise=args.noise)
    Xtr, ytr, Xval, yval = train_val_split(X, y)
    sizes = [2] + args.hidden + [3]
    model = MLP(sizes, args.activation)
    n_params = sum(W.size + b.size for W, b in zip(model.W, model.b))
    print(f"layers {sizes}, activation {args.activation}, {n_params:,} parameters")
    print(f"{len(Xtr)} training points, {len(Xval)} validation points\n")

    train(model, Xtr, ytr, Xval, yval, args.lr, args.epochs, args.batch_size, args.reg)

    print()
    ascii_plot(model, X, y)
    save_model(args.save, model)
    print(f"\nSaved model to {args.save}. Try: python mlp/predict.py")


if __name__ == "__main__":
    main()
