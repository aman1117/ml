"""
The key insight of CNNs, as an experiment: CNN vs MLP on shapes that move around.

Part 1  Shapes anywhere. Same data, same optimizer (SGD + momentum), same epochs.
        The MLP (../mlp/mlp_numpy.py, reused as-is) sees 256 unrelated pixels and must
        learn every shape at every position separately. The CNN learns one small
        detector per pattern and slides it everywhere (weight sharing).

Part 2  Train ONLY on shapes in the LEFT half of the image, test on shapes in the
        RIGHT half (positions never seen in training).
          MLP:                fails. Right-half pixels never carried a shape in training,
                              so their weights never learned anything.
          CNN (flatten head): ALSO fails! The conv layers detect the shape on the right
                              just fine (convolution is translation-EQUIVARIANT), but the
                              dense layer after flatten has separate weights per position.
          CNN (gap head):     works. Global average pooling throws away "where" and
                              keeps "what", so the output is translation-INVARIANT.

Run:
    python cnn/cnn_vs_mlp.py
    python cnn/cnn_vs_mlp.py --epochs 30
"""
import argparse
import os
import sys
import time

import numpy as np

from cnn_numpy import CLASSES, CNN, ascii_image, default_data, make_shapes, side_by_side, train

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "mlp"))
from mlp_numpy import MLP  # noqa: E402  (the MLP class from ../mlp/, unchanged)

SIZE, HALF = 16, 8


def train_mlp(mlp, X, y, epochs=15, lr=0.05, momentum=0.9, batch_size=32, seed=0):
    """Same optimizer and schedule as cnn_numpy.train, for the ../mlp/ MLP class.
    X: (N, 1, 16, 16) images, flattened to (N, 256) rows: the MLP has no idea that
    pixel 17 sits right below pixel 1."""
    X = X.reshape(len(X), -1)
    rng = np.random.default_rng(seed)
    vW = [np.zeros_like(W) for W in mlp.W]
    vb = [np.zeros_like(b) for b in mlp.b]
    for _ in range(epochs):
        perm = rng.permutation(len(X))
        for start in range(0, len(X), batch_size):
            idx = perm[start:start + batch_size]
            _, dW, db = mlp.loss_and_grads(X[idx], y[idx])
            for i in range(mlp.n_layers):
                vW[i] = momentum * vW[i] - lr * dW[i]
                vb[i] = momentum * vb[i] - lr * db[i]
                mlp.W[i] += vW[i]
                mlp.b[i] += vb[i]


def mlp_acc(mlp, X, y):
    return float((mlp.predict(X.reshape(len(X), -1)) == y).mean())


def cnn_acc(model, X, y):
    return float(model.evaluate(X, y)[1])


def n_params_mlp(mlp):
    return sum(W.size + b.size for W, b in zip(mlp.W, mlp.b))


def part1(epochs, hidden, verbose=True):
    Xtr, ytr, Xval, yval = default_data()
    res = {}
    t0 = time.time()
    mlp = MLP([SIZE * SIZE, hidden, len(CLASSES)], "relu", seed=0)
    train_mlp(mlp, Xtr, ytr, epochs)
    res["MLP"] = (n_params_mlp(mlp), mlp_acc(mlp, Xtr, ytr), mlp_acc(mlp, Xval, yval),
                  time.time() - t0)
    t0 = time.time()
    cnn = CNN(head="flatten")
    train(cnn, Xtr, ytr, Xval, yval, epochs, verbose=False)
    res["CNN"] = (cnn.n_params(), cnn_acc(cnn, Xtr, ytr), cnn_acc(cnn, Xval, yval),
                  time.time() - t0)
    if verbose:
        print(f"PART 1: shapes anywhere in the image. {len(Xtr)} training images, "
              f"{epochs} epochs each, SGD + momentum 0.9, lr 0.05\n")
        print(f"  {'model':34s} {'params':>8s} {'train acc':>10s} {'val acc':>9s} {'time':>7s}")
        names = {"MLP": f"MLP 256 → {hidden} → {len(CLASSES)}",
                 "CNN": "CNN conv8 → conv16 → dense"}
        for k, (n, tr, va, t) in res.items():
            print(f"  {names[k]:34s} {n:8,d} {tr:10.1%} {va:9.1%} {t:6.1f}s")
        ratio = res["MLP"][0] / res["CNN"][0]
        err = (1 - res["MLP"][2]) / max(1 - res["CNN"][2], 1e-9)
        print(f"\n  The CNN has {ratio:.0f}x FEWER parameters and makes {err:.0f}x fewer val "
              "mistakes.\n  Both fit the training set perfectly; the MLP just generalizes worse:"
              " it must learn\n  'vertical bar at column 3' and 'vertical bar at column 9' as"
              " unrelated things.\n")
    return res


def part2(epochs, hidden, verbose=True):
    Xtr, ytr = make_shapes(3000, SIZE, seed=0, cols=(0, HALF))
    Xl, yl = make_shapes(600, SIZE, seed=1, cols=(0, HALF))       # same half as training
    Xr, yr = make_shapes(600, SIZE, seed=2, cols=(HALF, SIZE))    # never-seen positions
    if verbose:
        print(f"PART 2: train ONLY on shapes in the LEFT half ({len(Xtr)} images, {epochs} epochs), "
              "test on the RIGHT half.\n")
        side_by_side([ascii_image(Xtr[0, 0]), ascii_image(Xr[0, 0])],
                     [f"train (left): {CLASSES[ytr[0]]}", f"test (right): {CLASSES[yr[0]]}"])
        print()

    res = {}
    mlp = MLP([SIZE * SIZE, hidden, len(CLASSES)], "relu", seed=0)
    W1_init = mlp.W[0].copy()
    train_mlp(mlp, Xtr, ytr, epochs)
    res["MLP"] = (n_params_mlp(mlp), mlp_acc(mlp, Xl, yl), mlp_acc(mlp, Xr, yr))
    # How much did the first-layer weights of left-half vs right-half pixels move?
    moved = np.abs(mlp.W[0] - W1_init).sum(axis=1).reshape(SIZE, SIZE)  # one number per pixel
    move_left, move_right = moved[:, :HALF].mean(), moved[:, HALF:].mean()

    for head in ("flatten", "gap"):
        cnn = CNN(head=head)
        train(cnn, Xtr, ytr, Xl, yl, epochs, verbose=False)
        res[f"CNN ({head})"] = (cnn.n_params(), cnn_acc(cnn, Xl, yl), cnn_acc(cnn, Xr, yr))
        if head == "flatten":
            cnn_flat = cnn
        else:
            cnn_gap = cnn

    if verbose:
        print(f"  {'model':26s} {'params':>8s} {'left (seen)':>12s} {'right (unseen)':>15s}")
        for k, (n, left, right) in res.items():
            print(f"  {k:26s} {n:8,d} {left:12.1%} {right:15.1%}")
        print(f"  (chance = 1/{len(CLASSES)} = {1 / len(CLASSES):.1%})\n")
        print("  Why the MLP fails: during training, right-half pixels held only noise and dust,")
        print(f"  so their first-layer weights barely moved: on average {move_right:.2f} "
              f"vs {move_left:.2f} for left-half pixels.")
        print("  Why CNN (flatten) fails: conv + pool still DETECT the shape on the right (the")
        print("  feature map just shifts right: translation EQUIVARIANCE), but the dense layer")
        print("  has separate weights for every position and only ever saw activity on the left.")
        print("  Why CNN (gap) works: global average pooling averages each feature map over ALL")
        print("  positions, so 'a diagonal somewhere' looks the same wherever it is (INVARIANCE).")
    return res, (move_left, move_right), (cnn_flat, cnn_gap, Xr, yr)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--hidden", type=int, default=128, help="MLP hidden layer size")
    args = ap.parse_args()
    t0 = time.time()
    part1(args.epochs, args.hidden)
    print("=" * 78 + "\n")
    part2(args.epochs, args.hidden)
    print(f"\n(total {time.time() - t0:.1f} s)")


if __name__ == "__main__":
    main()
