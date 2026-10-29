"""
The same CNN in PyTorch.

1. Copies the NumPy model's weights into nn.Conv2d / nn.MaxPool2d / nn.Linear and
   checks that PyTorch gives the SAME logits, the SAME parameter gradients and the
   SAME input gradient dX as our hand-written backward() (both heads: flatten and gap).
   Also checks that F.conv2d equals our im2col convolution for stride 2 and padding 1.
   Note: nn.Conv2d stores its weight as (C_out, C_in, kH, kW), exactly like ours, so
   it is copied as-is. nn.Linear stores (out, in), the transpose of our Dense.W.
2. Trains the idiomatic PyTorch version (Dataset/DataLoader, nn.Module, optimizer)
   on the same shape images, by default with BatchNorm after every conv:

       BN(z) = γ · (z - mean) / √(var + ε) + β     (mean, var per channel over N, H, W)

   BatchNorm keeps every channel's activations at a steady scale while the earlier
   layers change, which allows larger learning rates and makes deep CNNs (ResNet)
   trainable. In train() mode it uses the batch statistics; in eval() mode it uses
   running averages collected during training. Forgetting model.eval() is a classic bug.

Run:
    python cnn/cnn_pytorch.py
    python cnn/cnn_pytorch.py --no-bn
    python cnn/cnn_pytorch.py --optim adam --lr 1e-3
"""
import argparse
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from cnn_numpy import CLASSES, CNN, Conv2D, default_data, make_shapes, softmax_cross_entropy


def torch_copy_of(np_model):
    """Build the PyTorch twin of a cnn_numpy.CNN and copy every weight into it."""
    c = np_model.config
    layers = [
        nn.Conv2d(1, c["c1"], 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(c["c1"], c["c2"], 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    ]
    if c["head"] == "flatten":
        layers += [nn.Flatten(), nn.Linear(c["c2"] * (c["size"] // 4) ** 2, c["n_classes"])]
    else:
        layers += [nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(c["c2"], c["n_classes"])]
    tm = nn.Sequential(*layers).double()
    convs = [m for m in tm if isinstance(m, nn.Conv2d)]
    p = dict((name, v) for name, v, _ in np_model.named_params())
    with torch.no_grad():
        for i, conv in enumerate(convs, start=1):
            conv.weight.copy_(torch.from_numpy(p[f"conv{i}.W"]))  # same layout, no transpose
            conv.bias.copy_(torch.from_numpy(p[f"conv{i}.b"]))
        tm[-1].weight.copy_(torch.from_numpy(p["fc.W"].T))  # Linear is (out, in)
        tm[-1].bias.copy_(torch.from_numpy(p["fc.b"]))
    return tm, convs


def check_against_numpy(head):
    rng = np.random.default_rng(0)
    np_model = CNN(head=head, seed=0)
    for _, v, _ in np_model.named_params():  # non-zero biases so every path is tested
        if v.ndim == 1:
            v += rng.standard_normal(v.shape) * 0.1
    X, y = make_shapes(16, seed=5)  # real shape images, float64

    logits = np_model.forward(X)
    loss, dlogits = softmax_cross_entropy(logits, y)
    dx = np_model.backward(dlogits)

    tm, convs = torch_copy_of(np_model)
    xt = torch.from_numpy(X).requires_grad_(True)
    t_logits = tm(xt)
    t_loss = F.cross_entropy(t_logits, torch.from_numpy(y))
    t_loss.backward()  # autograd runs backprop through conv, pool, relu, linear

    assert np.allclose(t_logits.detach().numpy(), logits), "logits differ"
    assert np.isclose(t_loss.item(), loss), "loss differs"
    g = dict((name, grad) for name, _, grad in np_model.named_params())
    for i, conv in enumerate(convs, start=1):
        assert np.allclose(conv.weight.grad.numpy(), g[f"conv{i}.W"]), f"conv{i}.W grad"
        assert np.allclose(conv.bias.grad.numpy(), g[f"conv{i}.b"]), f"conv{i}.b grad"
    assert np.allclose(tm[-1].weight.grad.numpy().T, g["fc.W"]), "fc.W grad"
    assert np.allclose(tm[-1].bias.grad.numpy(), g["fc.b"]), "fc.b grad"
    assert np.allclose(xt.grad.numpy(), dx), "dX differs"
    diff = np.abs(t_logits.detach().numpy() - logits).max()
    print(f"head={head:7s}: logits, loss, all {len(g)} parameter gradients and dX match "
          f"PyTorch ✔  (max logit diff {diff:.1e})")


def check_conv2d():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((2, 3, 9, 9))
    conv = Conv2D(3, 5, 3, stride=2, pad=1, rng=rng)
    conv.params["b"] = rng.standard_normal(5)
    ours = conv.forward(x)
    theirs = F.conv2d(torch.from_numpy(x), torch.from_numpy(conv.params["W"]),
                      torch.from_numpy(conv.params["b"]), stride=2, padding=1).numpy()
    assert np.allclose(ours, theirs)
    print(f"F.conv2d == our im2col conv (C_in 3 → C_out 5, k 3, stride 2, pad 1): "
          f"(2, 3, 9, 9) → {theirs.shape} ✔")
    print("  (both are CROSS-correlation: the kernel is NOT flipped, unlike math convolution)\n")


class ShapeCNN(nn.Module):
    """conv → [BN] → ReLU → pool, twice, then flatten → linear. Same shape as cnn_numpy.CNN."""

    def __init__(self, c1=8, c2=16, n_classes=len(CLASSES), size=16, bn=True):
        super().__init__()

        def block(c_in, c_out):
            # With BN right after the conv, the conv bias is useless (BN subtracts the mean).
            layers = [nn.Conv2d(c_in, c_out, 3, padding=1, bias=not bn)]
            if bn:
                layers.append(nn.BatchNorm2d(c_out))
            return layers + [nn.ReLU(), nn.MaxPool2d(2)]

        self.features = nn.Sequential(*block(1, c1), *block(c1, c2))
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(c2 * (size // 4) ** 2, n_classes))

    def forward(self, x):  # x: (N, 1, H, W)
        return self.head(self.features(x))


@torch.no_grad()
def evaluate(model, X, y):
    model.eval()  # BN uses its running statistics, not this batch's
    logits = model(X)
    return F.cross_entropy(logits, y).item(), (logits.argmax(1) == y).float().mean().item()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-bn", action="store_true", help="train without BatchNorm")
    ap.add_argument("--optim", choices=["sgd", "adam"], default="sgd")
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    check_conv2d()
    for head in ("flatten", "gap"):
        check_against_numpy(head)
    print()

    torch.manual_seed(0)
    # Tiny tensors: one thread is faster than paying thread start-up costs on every op.
    torch.set_num_threads(1)
    Xtr, ytr, Xval, yval = default_data()
    Xtr, Xval = torch.tensor(Xtr, dtype=torch.float32), torch.tensor(Xval, dtype=torch.float32)
    ytr, yval = torch.tensor(ytr), torch.tensor(yval)
    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=args.batch_size, shuffle=True)

    model = ShapeCNN(bn=not args.no_bn)
    if args.optim == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    n = sum(p.numel() for p in model.parameters())
    print(f"ShapeCNN, BatchNorm {'off' if args.no_bn else 'on'}, {args.optim} lr {args.lr}, "
          f"{n:,} parameters")
    print(model, "\n")

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()  # BN uses batch statistics and updates its running averages
        total, correct = 0.0, 0
        for xb, yb in loader:
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(yb)
            correct += (logits.argmax(1) == yb).sum().item()
        vl, va = evaluate(model, Xval, yval)
        print(f"epoch {epoch:3d} | train loss {total / len(Xtr):.3f} acc {correct / len(Xtr):6.1%}"
              f" | val loss {vl:.3f} acc {va:6.1%}")
    print(f"training took {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
