"""
The same MLP in PyTorch.

1. Copies the NumPy model's weights into PyTorch and checks that autograd
   computes exactly the same gradients as our hand-written backward().
   (Autograd is just automated backprop: the chain rule done for you.)
2. Trains an nn.Sequential MLP on the spirals with torch.optim.SGD.

Run:
    python mlp/mlp_pytorch.py
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mlp_numpy import MLP, make_spiral, train_val_split


def check_against_numpy():
    rng = np.random.default_rng(0)
    np_model = MLP([2, 16, 16, 3], "relu")
    X = rng.standard_normal((10, 2))
    y = rng.integers(0, 3, 10)
    _, dW, db = np_model.loss_and_grads(X, y)

    # nn.Linear stores W as (out, in), i.e. the transpose of ours.
    layers = []
    for i, (W, b) in enumerate(zip(np_model.W, np_model.b)):
        lin = nn.Linear(*W.shape).double()
        with torch.no_grad():
            lin.weight.copy_(torch.from_numpy(W.T))
            lin.bias.copy_(torch.from_numpy(b))
        layers.append(lin)
        if i < len(np_model.W) - 1:
            layers.append(nn.ReLU())
    torch_model = nn.Sequential(*layers)

    loss = F.cross_entropy(torch_model(torch.from_numpy(X)), torch.from_numpy(y))
    loss.backward()  # autograd runs backprop

    linears = [m for m in torch_model if isinstance(m, nn.Linear)]
    for i, lin in enumerate(linears):
        assert np.allclose(lin.weight.grad.numpy().T, dW[i]), f"W{i} mismatch"
        assert np.allclose(lin.bias.grad.numpy(), db[i]), f"b{i} mismatch"
    print("PyTorch autograd gradients == our hand-written backprop ✔\n")


def main():
    check_against_numpy()

    torch.manual_seed(0)
    X, y = make_spiral()
    Xtr, ytr, Xval, yval = train_val_split(X, y)
    Xtr, Xval = torch.tensor(Xtr, dtype=torch.float32), torch.tensor(Xval, dtype=torch.float32)
    ytr, yval = torch.tensor(ytr), torch.tensor(yval)

    model = nn.Sequential(
        nn.Linear(2, 64),
        nn.ReLU(),
        nn.Linear(64, 3),  # logits; F.cross_entropy applies softmax for us
    )
    # weight_decay = L2 regularization, same as --reg in mlp_numpy.py
    opt = torch.optim.SGD(model.parameters(), lr=0.5, weight_decay=1e-3)

    for epoch in range(1, 301):
        model.train()
        perm = torch.randperm(len(Xtr))
        for start in range(0, len(Xtr), 32):
            idx = perm[start:start + 32]
            loss = F.cross_entropy(model(Xtr[idx]), ytr[idx])
            opt.zero_grad()   # gradients add up by default, so clear them
            loss.backward()   # backprop
            opt.step()        # gradient descent step

        if epoch == 1 or epoch % 50 == 0:
            model.eval()
            with torch.no_grad():
                tr_acc = (model(Xtr).argmax(1) == ytr).float().mean()
                va_logits = model(Xval)
                va_loss = F.cross_entropy(va_logits, yval)
                va_acc = (va_logits.argmax(1) == yval).float().mean()
            print(f"epoch {epoch:4d} | train acc {tr_acc:.1%} | val loss {va_loss:.3f} acc {va_acc:.1%}")


if __name__ == "__main__":
    main()
