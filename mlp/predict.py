"""
Ask the trained MLP about any point you like.

Type two numbers (x and y, roughly between -1 and 1) and the model tells you
which spiral (class 0, 1 or 2) it thinks the point belongs to, and how sure it is.
Compare with the picture printed by mlp_numpy.py.

Run (after training with mlp_numpy.py):
    python mlp/predict.py
"""
import os

import numpy as np

from mlp_numpy import load_model


def main():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.npz")
    if not os.path.exists(path):
        raise SystemExit(f"No model at {path}. Train first: python mlp/mlp_numpy.py")
    model = load_model(path)

    print("Enter a point as two numbers, e.g.  0.2 -0.5   (empty line or Ctrl+C to quit)\n")
    while True:
        try:
            line = input("x y> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            break
        try:
            x, y = (float(v) for v in line.replace(",", " ").split())
        except ValueError:
            print("  please type exactly two numbers, like: 0.2 -0.5")
            continue
        probs = model.predict_proba(np.array([[x, y]]))[0]
        for c, p in enumerate(probs):
            bar = "#" * int(round(p * 30))
            print(f"  class {c}: {p:6.1%} {bar}")
        print(f"  -> prediction: class {probs.argmax()}\n")


if __name__ == "__main__":
    main()
