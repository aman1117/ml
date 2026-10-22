"""
Overfitting, made visible.

Overfitting = the model memorizes the training data (including its noise)
instead of learning the general pattern, so it does worse on new data.

Setup: messy spirals (noise 0.6), only 60 training points, and a big network
(2 hidden layers x 256 units, ~67k parameters: about 1000x more parameters
than training points).

Part 1: watch train loss keep falling while validation loss turns around.
Part 2: L2 regularization (weight decay) reduces overfitting.
Part 3: more data is the best cure.

Run:
    python mlp/overfitting.py
"""
import numpy as np

from mlp_numpy import MLP, make_spiral, train

NOISE, N_TRAIN, EPOCHS = 0.6, 60, 2000
SIZES = [2, 256, 256, 3]

X, y = make_spiral(n_per_class=200, noise=NOISE, seed=1)
perm = np.random.default_rng(0).permutation(len(X))
X_val, y_val = X[perm[-300:]], y[perm[-300:]]  # 300 held-out points, never trained on


def run(n_train, reg):
    idx = perm[:n_train]
    model = MLP(SIZES, "relu")
    return train(model, X[idx], y[idx], X_val, y_val, lr=0.1, epochs=EPOCHS,
                 batch_size=60, reg=reg, verbose=False)


def summary(h):
    best = min(h, key=lambda r: r["val_loss"])
    last = h[-1]
    return (f"train loss {last['train_loss']:.2f} acc {last['train_acc']:4.0%} | "
            f"val loss {last['val_loss']:.2f} acc {last['val_acc']:4.0%} | "
            f"best val loss {best['val_loss']:.2f} at epoch {best['epoch']}")


print(f"PART 1: big network, {N_TRAIN} training points, no regularization\n")
h = run(N_TRAIN, reg=0.0)
best = min(h, key=lambda r: r["val_loss"])
print(" epoch | train loss | val loss | train acc | val acc")
for r in h:
    if r["epoch"] in (1, 50, 100, best["epoch"], 500, 1000, 1500, 2000):
        mark = "  <- best val loss: stop here (early stopping)" if r is best else ""
        print(f" {r['epoch']:5d} |   {r['train_loss']:.3f}    |  {r['val_loss']:.3f}   |"
              f"   {r['train_acc']:4.0%}    |  {r['val_acc']:4.0%}{mark}")
print("\n -> Train loss heads to ~0 (memorized), but val loss rises after the best epoch.")
print("    The model grows over-confident on points it hasn't seen.\n")

print("PART 2: same setup, different L2 regularization strengths\n")
for reg in [0.0, 1e-3, 3e-3, 1e-2]:
    print(f"  reg={reg:<6} {summary(run(N_TRAIN, reg))}")
print("\n -> More regularization keeps the weights small: worse fit on the training set,")
print("    much better val loss. Too much would underfit.\n")

print("PART 3: no regularization, but more training data\n")
for n in [60, 150, 300]:
    print(f"  {n:3d} points  {summary(run(n, 0.0))}")
print("\n -> With more data there is less noise to memorize, so the gap between train and val shrinks.")
