"""
Ask the trained CNN about any image you like.

Pick a shape and (optionally) where to put it, or draw your own 16x16 image with
'#' and '.', and the model shows its class probabilities. It also shows a
SALIENCY map: |d score / d pixel|, computed with the same backward pass we use for
training (the dX that gradient_check.py verifies). Bright = "changing this pixel
would change my answer the most" = where the network is looking.

Uses cnn/model.npz if it exists (train it with `python cnn/cnn_numpy.py`),
otherwise trains a quick model first (~5 s).

Run:
    python cnn/predict.py                          (interactive)
    python cnn/predict.py --shape plus --pos 2 9   (one-shot: a plus with top-left at row 2, col 9)
    python cnn/predict.py --shape square --size 7 --thick 2
"""
import argparse

import numpy as np

from cnn_numpy import CLASSES, ascii_image, draw_shape, get_model, side_by_side, softmax

ALIASES = {"-": 0, "h": 0, "horizontal": 0, "|": 1, "v": 1, "vertical": 1,
           "\\": 2, "diag\\": 2, "diag \\": 2, "/": 3, "diag/": 3, "diag /": 3,
           "+": 4, "plus": 4, "square": 5, "[]": 5, "o": 5}
SIZE = 16


def make_image(cls, pos=None, s=6, t=1, seed=None):
    """A 16x16 image with one shape (top-left corner at pos, random if None) + light noise."""
    rng = np.random.default_rng(seed)
    if pos is None:
        pos = rng.integers(0, SIZE - s + 1, 2)
    r, c = int(pos[0]), int(pos[1])
    if not (0 <= r <= SIZE - s and 0 <= c <= SIZE - s):
        raise ValueError(f"position must be between 0 and {SIZE - s} for size {s}")
    img = draw_shape(np.zeros((SIZE, SIZE)), cls, r, c, s, t, 1.0)
    return np.clip(img + rng.normal(0, 0.05, img.shape), 0, 1)


def parse_drawing(lines):
    """Text rows → image. '.' or ' ' = dark, anything else = bright. Missing cells = dark."""
    img = np.zeros((SIZE, SIZE))
    for i, line in enumerate(lines[:SIZE]):
        for j, ch in enumerate(line[:SIZE]):
            img[i, j] = 0.0 if ch in ". " else 1.0
    return img


def bar(p, width=30):
    k = int(round(p * width))
    return "█" * k + "·" * (width - k)


def explain(model, img):
    x = img[None, None]
    logits = model.forward(x)
    p = softmax(logits)[0]
    k = int(p.argmax())
    # Saliency: backprop d(logit_k)/dx, i.e. start the backward pass with a one-hot.
    d = np.zeros_like(logits)
    d[0, k] = 1.0
    sal = np.abs(model.backward(d)[0, 0])
    side_by_side([ascii_image(img), ascii_image(sal, 0, sal.max())],
                 ["your image", f"saliency for '{CLASSES[k]}'"])
    print()
    for i, name in enumerate(CLASSES):
        line = f"  {name:>10s}  {bar(p[i])}  {p[i]:6.1%}"
        print(line + ("   ← prediction" if i == k else ""))
    print()


def read_drawing():
    print(f"  Type up to {SIZE} rows of up to {SIZE} characters ('#' = bright, '.' = dark).")
    print("  An empty line finishes the drawing.")
    lines = []
    while len(lines) < SIZE:
        line = input(f"  row {len(lines):2d}> ")
        if not line.strip():
            break
        lines.append(line)
    return parse_drawing(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shape", help=f"one of: {', '.join(CLASSES)} (or -, |, \\, /, +, [])")
    ap.add_argument("--pos", type=int, nargs=2, metavar=("ROW", "COL"),
                    help="top-left corner of the shape's box (default: random)")
    ap.add_argument("--size", type=int, default=6, help="shape size in pixels (5-7 were trained)")
    ap.add_argument("--thick", type=int, default=1, choices=[1, 2])
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    model = get_model()
    if args.shape:
        if args.shape.lower() not in ALIASES:
            raise SystemExit(f"unknown shape {args.shape!r}; try one of {CLASSES}")
        explain(model, make_image(ALIASES[args.shape.lower()], args.pos, args.size, args.thick,
                                  args.seed))
        return

    print("Commands:")
    print("  plus | square | horizontal | vertical | / | \\   a shape at a random position")
    print("  plus 3 9                                        a shape with its top-left at row 3, col 9")
    print("  draw                                            draw your own image")
    print("  q, empty line or Ctrl+C                         quit\n")
    while True:
        try:
            cmd = input("shape> ").strip()
            if cmd in ("", "q", "quit"):
                break
            if cmd == "draw":
                img = read_drawing()
            else:
                parts = cmd.split()
                pos = None
                if len(parts) >= 3 and parts[-1].isdigit() and parts[-2].isdigit():
                    pos, parts = (int(parts[-2]), int(parts[-1])), parts[:-2]
                name = " ".join(parts).lower()
                if name not in ALIASES:
                    print(f"  unknown shape {name!r}; try: {', '.join(CLASSES)}\n")
                    continue
                img = make_image(ALIASES[name], pos, args.size, args.thick)
            explain(model, img)
        except ValueError as e:
            print(f"  ({e})\n")
        except (EOFError, KeyboardInterrupt):
            print()
            break


if __name__ == "__main__":
    main()
