"""
Convolutional Neural Network (CNN) from scratch in NumPy.

Task: classify small 16x16 grayscale images of SHAPES (horizontal bar, vertical
bar, diagonal \\, diagonal /, plus, square outline). Each shape has a random
size, thickness, brightness and, most importantly, a RANDOM POSITION. A CNN
handles "same pattern, different place" naturally because it slides the same
small filters over every position (weight sharing).

Layout: NCHW, like PyTorch. A batch of images has shape (N, C, H, W):
    N = images in the batch, C = channels (1 for grayscale), H, W = height, width.
A conv weight has shape (C_out, C_in, kH, kW): one (C_in, kH, kW) filter per output channel.

Layers (see README sections 2-6):
    conv:   out[n, f, i, j] = Σ_c Σ_a Σ_b  x_pad[n, c, i·s + a, j·s + b] · W[f, c, a, b]  + b[f]
            out size = (n + 2p - k) / s + 1   (floored)
    relu:   max(0, z)
    pool:   max over each non-overlapping k x k window
    dense:  Z = A @ W + b   (same as ../mlp/)
    loss:   softmax + cross-entropy

Backward (README section 7):
    conv:   dW = correlate(x_pad, dout),  db = Σ dout,
            dX = full convolution of dout with the flipped kernel   (stride 1)
    pool:   the gradient goes only to the input that was the max (the argmax)

Default network (≈2.8k parameters):
    (N,1,16,16) → conv 3x3, 1→8, pad 1 → ReLU → maxpool 2 → (N,8,8,8)
                → conv 3x3, 8→16, pad 1 → ReLU → maxpool 2 → (N,16,4,4)
                → flatten (N,256) → dense → (N,6) logits

Run:
    python cnn/cnn_numpy.py
    python cnn/cnn_numpy.py --epochs 20 --c1 16 --c2 32
    python cnn/cnn_numpy.py --head gap          (global average pooling instead of flatten)
"""
import argparse
import os
import time
import warnings

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

# NumPy 2.0 + Apple Accelerate raises spurious "divide by zero / overflow /
# invalid value encountered in matmul" warnings even when results are correct.
# Only silence those; real numerical warnings (exp, log, ...) still show.
warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)

HERE = os.path.dirname(os.path.abspath(__file__))
CLASSES = ["horizontal", "vertical", "diag \\", "diag /", "plus", "square"]
SHADES = " .:-=+*#%@"  # dark → bright, for drawing images in the terminal


# ----------------------------------------------------------------------------
# Data: shapes at random positions
# ----------------------------------------------------------------------------
def draw_shape(img, cls, r, c, s, t=1, v=1.0):
    """Draw shape `cls` inside the s x s box whose top-left corner is (r, c).
    t = line thickness (1 or 2), v = brightness."""
    mid = (s - 1) // 2
    for o in range(t):
        if cls in (0, 4):  # horizontal bar (also half of the plus)
            img[r + mid + o, c:c + s] = v
        if cls in (1, 4):  # vertical bar
            img[r:r + s, c + mid + o] = v
        if cls == 2:  # diagonal "\": top-left to bottom-right
            for i in range(s):
                img[r + i, min(c + i + o, c + s - 1)] = v
        if cls == 3:  # diagonal "/": top-right to bottom-left
            for i in range(s):
                img[r + i, max(c + s - 1 - i - o, c)] = v
        if cls == 5:  # square outline (t=2 adds an inner ring)
            a, b = r + o, c + o
            e = s - 1 - 2 * o
            img[a, b:b + e + 1] = v
            img[a + e, b:b + e + 1] = v
            img[a:a + e + 1, b] = v
            img[a:a + e + 1, b + e] = v
    return img


def make_shapes(n, size=16, seed=0, cols=None, noise=0.1, specks=3, classes=None):
    """n images (n, 1, size, size) with values in [0, 1], and labels (n,).

    Every image: one shape with a random size (5-7 px), thickness (1-2 px),
    brightness (0.6-1.0) and position, plus up to `specks` random "dust" pixels
    and Gaussian pixel noise (std `noise`), clipped to [0, 1].
    cols=(lo, hi) restricts where the shape may sit horizontally: its whole box
    must lie in columns lo..hi-1 (used by cnn_vs_mlp.py: train left, test right).
    """
    rng = np.random.default_rng(seed)
    C = len(CLASSES)
    y = rng.permutation(np.arange(n) % C) if classes is None else np.asarray(classes)
    X = np.zeros((n, 1, size, size))
    lo, hi = (0, size) if cols is None else cols
    for k in range(n):
        s = int(rng.integers(5, 8))
        t = 1 if rng.random() < 0.7 else 2
        v = rng.uniform(0.6, 1.0)
        r = int(rng.integers(0, size - s + 1))
        c = int(rng.integers(lo, hi - s + 1))
        draw_shape(X[k, 0], y[k], r, c, s, t, v)
        m = int(rng.integers(0, specks + 1))  # a few random bright "dust" pixels
        X[k, 0, rng.integers(0, size, m), rng.integers(0, size, m)] = rng.uniform(0.3, 0.9, m)
    X += rng.normal(0.0, noise, X.shape)
    return np.clip(X, 0.0, 1.0), y


def ascii_image(img, vmin=0.0, vmax=1.0):
    """One 2-D image → list of text lines. Each pixel is 2 characters wide so the
    picture keeps its aspect ratio. Dark = ' ', bright = '@'."""
    z = (np.asarray(img, dtype=float) - vmin) / max(vmax - vmin, 1e-12)
    idx = np.clip((z * len(SHADES)).astype(int), 0, len(SHADES) - 1)
    return ["".join(SHADES[i] * 2 for i in row) for row in idx]


def side_by_side(blocks, captions=None, gap=3, per_row=3):
    """Print several ascii_image() blocks next to each other, each in a box,
    at most per_row per line (so it fits in a normal terminal)."""
    if len(blocks) > per_row:
        for i in range(0, len(blocks), per_row):
            side_by_side(blocks[i:i + per_row], captions and captions[i:i + per_row], gap, per_row)
        return
    h = max(len(b) for b in blocks)
    w = [max(len(line) for line in b) for b in blocks]
    sp = " " * gap
    if captions:
        print(sp.join(f" {cap[:wi]:<{wi}} " for cap, wi in zip(captions, w)))
    print(sp.join("┌" + "─" * wi + "┐" for wi in w))
    for i in range(h):
        print(sp.join("│" + (b[i] if i < len(b) else "").ljust(wi) + "│"
                      for b, wi in zip(blocks, w)))
    print(sp.join("└" + "─" * wi + "┘" for wi in w))


# ----------------------------------------------------------------------------
# Convolution: a slow, obvious reference and the fast im2col version
# ----------------------------------------------------------------------------
def out_size(n, k, s=1, p=0):
    """The output-size formula: out = (n + 2p - k) / s + 1, rounded down."""
    return (n + 2 * p - k) // s + 1


def pad_hw(x, p):
    """Zero-pad the last two axes (H and W) of an NCHW array by p on every side."""
    return np.pad(x, ((0, 0), (0, 0), (p, p), (p, p))) if p else x


def conv2d_naive(x, W, b, stride=1, pad=0):
    """Reference convolution with explicit loops: exactly the formula, slow.
    x: (N, C_in, H, W)   W: (C_out, C_in, k, k)   b: (C_out,)  →  (N, C_out, H_out, W_out)"""
    N, C, H, Wd = x.shape
    F, _, kh, kw = W.shape
    Ho, Wo = out_size(H, kh, stride, pad), out_size(Wd, kw, stride, pad)
    xp = pad_hw(x, pad)
    out = np.zeros((N, F, Ho, Wo))
    for n in range(N):
        for f in range(F):
            for i in range(Ho):
                for j in range(Wo):
                    patch = xp[n, :, i * stride:i * stride + kh, j * stride:j * stride + kw]
                    out[n, f, i, j] = np.sum(patch * W[f]) + b[f]  # dot product + bias
    return out


def conv2d_backward_naive(x, W, dout, pad=0):
    """The README backward formulas written literally (stride 1 only):
        dW[f, c] = cross-correlate( x_pad[:, c], dout[:, f] )   (dout plays the kernel)
        dX       = FULL convolution of dout with the 180°-flipped kernel, un-padded
        db       = Σ dout over batch and positions
    Used by self_test() to show that the fast im2col backward computes the same thing."""
    N, C, H, Wd = x.shape
    F, _, k, _ = W.shape
    xp = pad_hw(x, pad)
    Ho, Wo = dout.shape[2:]
    dW = np.zeros_like(W)
    for a in range(k):
        for b_ in range(k):
            # every output position (i, j) used input pixel (i + a, j + b_) with weight W[:, :, a, b_]
            dW[:, :, a, b_] = np.einsum("nfij,ncij->fc", dout, xp[:, :, a:a + Ho, b_:b_ + Wo])
    dout_p = pad_hw(dout, k - 1)  # "full": let the kernel hang off every edge
    Wflip = W[:, :, ::-1, ::-1]
    Hp, Wp = H + 2 * pad, Wd + 2 * pad
    dxp = np.zeros((N, C, Hp, Wp))
    for a in range(k):
        for b_ in range(k):
            dxp += np.einsum("nfij,fc->ncij", dout_p[:, :, a:a + Hp, b_:b_ + Wp], Wflip[:, :, a, b_])
    dx = dxp[:, :, pad:pad + H, pad:pad + Wd]
    return dW, dout.sum(axis=(0, 2, 3)), dx


def im2col(x, k, stride=1, pad=0):
    """Cut every k x k patch (over all input channels) out of x and lay it out as a ROW.
    x: (N, C, H, W)  →  cols: (N·H_out·W_out, C·k·k)
    After this, convolution is ONE matrix multiply: cols @ W.reshape(C_out, -1).T"""
    N, C, H, Wd = x.shape
    Ho, Wo = out_size(H, k, stride, pad), out_size(Wd, k, stride, pad)
    win = sliding_window_view(pad_hw(x, pad), (k, k), axis=(2, 3))  # (N, C, H', W', k, k) view
    win = win[:, :, ::stride, ::stride][:, :, :Ho, :Wo]             # (N, C, Ho, Wo, k, k)
    cols = win.transpose(0, 2, 3, 1, 4, 5).reshape(N * Ho * Wo, C * k * k)  # copies
    return cols, Ho, Wo


def col2im(dcols, x_shape, k, stride=1, pad=0):
    """The backward of im2col: add every patch-row's gradient back to the pixels it
    came from. Pixels shared by overlapping patches collect the SUM of gradients."""
    N, C, H, Wd = x_shape
    Ho, Wo = out_size(H, k, stride, pad), out_size(Wd, k, stride, pad)
    d = dcols.reshape(N, Ho, Wo, C, k, k).transpose(0, 3, 4, 5, 1, 2)  # (N, C, k, k, Ho, Wo)
    dxp = np.zeros((N, C, H + 2 * pad, Wd + 2 * pad))
    for a in range(k):  # only k·k slice-adds, each over all positions at once
        for b in range(k):
            dxp[:, :, a:a + stride * Ho:stride, b:b + stride * Wo:stride] += d[:, :, a, b]
    return dxp[:, :, pad:pad + H, pad:pad + Wd] if pad else dxp


# ----------------------------------------------------------------------------
# Layers. Each has forward(x), backward(dout) -> dx, and dicts params / grads.
# ----------------------------------------------------------------------------
class Conv2D:
    def __init__(self, c_in, c_out, k=3, stride=1, pad=0, rng=None):
        rng = rng if rng is not None else np.random.default_rng(0)
        self.k, self.stride, self.pad = k, stride, pad
        fan_in = c_in * k * k  # each output number looks at C_in·k·k inputs
        self.params = {
            "W": rng.standard_normal((c_out, c_in, k, k)) * np.sqrt(2.0 / fan_in),  # He init
            "b": np.zeros(c_out),
        }
        self.grads = {}

    def forward(self, x):
        W, b = self.params["W"], self.params["b"]
        F = W.shape[0]
        cols, Ho, Wo = im2col(x, self.k, self.stride, self.pad)  # (N·Ho·Wo, C·k·k)
        out = cols @ W.reshape(F, -1).T + b                      # (N·Ho·Wo, F)
        self.cache = (x.shape, cols)
        return out.reshape(x.shape[0], Ho, Wo, F).transpose(0, 3, 1, 2)  # back to NCHW

    def backward(self, dout):
        x_shape, cols = self.cache
        W = self.params["W"]
        F = W.shape[0]
        dout_m = dout.transpose(0, 2, 3, 1).reshape(-1, F)  # (N·Ho·Wo, F), rows match cols
        self.grads["W"] = (dout_m.T @ cols).reshape(W.shape)  # Σ over positions: shared weights
        self.grads["b"] = dout_m.sum(axis=0)
        dcols = dout_m @ W.reshape(F, -1)                      # (N·Ho·Wo, C·k·k)
        return col2im(dcols, x_shape, self.k, self.stride, self.pad)


class ReLU:
    params, grads = {}, {}

    def forward(self, x):
        self.mask = x > 0
        return x * self.mask

    def backward(self, dout):
        return dout * self.mask


class MaxPool2D:
    """Non-overlapping k x k max pooling (stride = k). H and W must be divisible by k."""
    params, grads = {}, {}

    def __init__(self, k=2):
        self.k = k

    def forward(self, x):
        N, C, H, W = x.shape
        k = self.k
        assert H % k == 0 and W % k == 0, "MaxPool2D needs H, W divisible by k"
        win = x.reshape(N, C, H // k, k, W // k, k).transpose(0, 1, 2, 4, 3, 5)
        win = win.reshape(N, C, H // k, W // k, k * k)  # each pooling window as a row of k·k
        self.arg = win.argmax(axis=-1)  # which of the k·k won (first one on ties, like PyTorch)
        self.shape = x.shape
        return np.take_along_axis(win, self.arg[..., None], axis=-1)[..., 0]

    def backward(self, dout):
        N, C, H, W = self.shape
        k = self.k
        dwin = np.zeros((N, C, H // k, W // k, k * k))
        np.put_along_axis(dwin, self.arg[..., None], dout[..., None], axis=-1)  # route to argmax
        dwin = dwin.reshape(N, C, H // k, W // k, k, k).transpose(0, 1, 2, 4, 3, 5)
        return dwin.reshape(N, C, H, W)


class GlobalAvgPool:
    """(N, C, H, W) → (N, C): one number per channel = its average over all positions.
    This makes the network's output (almost) independent of WHERE a pattern is."""
    params, grads = {}, {}

    def forward(self, x):
        self.shape = x.shape
        return x.mean(axis=(2, 3))

    def backward(self, dout):
        N, C, H, W = self.shape
        return np.broadcast_to(dout[:, :, None, None] / (H * W), self.shape).copy()


class Flatten:
    params, grads = {}, {}

    def forward(self, x):
        self.shape = x.shape
        return x.reshape(x.shape[0], -1)  # (N, C·H·W), same order as torch.flatten

    def backward(self, dout):
        return dout.reshape(self.shape)


class Dense:
    """Z = X @ W + b with W: (D_in, D_out). Exactly the MLP layer from ../mlp/."""

    def __init__(self, d_in, d_out, rng=None, gain=1.0):
        rng = rng if rng is not None else np.random.default_rng(0)
        self.params = {"W": rng.standard_normal((d_in, d_out)) * np.sqrt(gain / d_in),
                       "b": np.zeros(d_out)}
        self.grads = {}

    def forward(self, x):
        self.x = x
        return x @ self.params["W"] + self.params["b"]

    def backward(self, dout):
        self.grads["W"] = self.x.T @ dout
        self.grads["b"] = dout.sum(axis=0)
        return dout @ self.params["W"].T


def softmax(logits):
    e = np.exp(logits - logits.max(axis=1, keepdims=True))  # subtract max for stability
    return e / e.sum(axis=1, keepdims=True)


def softmax_cross_entropy(logits, y):
    """Mean cross-entropy loss and its gradient w.r.t. the logits (p - onehot) / N."""
    N = len(y)
    p = softmax(logits)
    loss = -np.log(p[np.arange(N), y] + 1e-12).mean()
    d = p.copy()
    d[np.arange(N), y] -= 1.0
    return loss, d / N


# ----------------------------------------------------------------------------
# The network
# ----------------------------------------------------------------------------
class CNN:
    def __init__(self, n_classes=len(CLASSES), size=16, c1=8, c2=16, head="flatten", seed=0):
        rng = np.random.default_rng(seed)
        self.config = dict(n_classes=n_classes, size=size, c1=c1, c2=c2, head=head)
        self.layers = [
            ("conv1", Conv2D(1, c1, 3, pad=1, rng=rng)),   # (N,1,16,16) → (N,c1,16,16)
            ("relu1", ReLU()),
            ("pool1", MaxPool2D(2)),                        # → (N,c1,8,8)
            ("conv2", Conv2D(c1, c2, 3, pad=1, rng=rng)),  # → (N,c2,8,8)
            ("relu2", ReLU()),
            ("pool2", MaxPool2D(2)),                        # → (N,c2,4,4)
        ]
        if head == "flatten":
            self.layers += [("flatten", Flatten()),                       # → (N, c2·4·4)
                            ("fc", Dense(c2 * (size // 4) ** 2, n_classes, rng))]
        elif head == "gap":
            self.layers += [("gap", GlobalAvgPool()),                     # → (N, c2)
                            ("fc", Dense(c2, n_classes, rng))]
        else:
            raise ValueError(f"unknown head {head!r}")

    def forward(self, x, keep=False):
        """x: (N, 1, H, W) → logits (N, n_classes). keep=True also returns every
        layer's output (for looking inside the network)."""
        acts = {}
        for name, layer in self.layers:
            x = layer.forward(x)
            if keep:
                acts[name] = x
        return (x, acts) if keep else x

    def backward(self, dlogits):
        """Backprop through every layer, last to first. Fills layer.grads; returns dx."""
        d = dlogits
        for _, layer in reversed(self.layers):
            d = layer.backward(d)
        return d

    def loss_and_grads(self, x, y):
        loss, dlogits = softmax_cross_entropy(self.forward(x), y)
        self.backward(dlogits)
        return loss

    def named_params(self):
        """(name, value, gradient) for every parameter, e.g. ('conv1.W', W, dW)."""
        for lname, layer in self.layers:
            for pname, v in layer.params.items():
                yield f"{lname}.{pname}", v, layer.grads.get(pname)

    def n_params(self):
        return sum(v.size for _, v, _ in self.named_params())

    def predict_proba(self, x, batch=500):
        return np.concatenate([softmax(self.forward(x[i:i + batch]))
                               for i in range(0, len(x), batch)])

    def evaluate(self, x, y):
        p = self.predict_proba(x)
        return -np.log(p[np.arange(len(y)), y] + 1e-12).mean(), (p.argmax(1) == y).mean()


class SGDMomentum:
    """v = μ·v - lr·g ;  w += v.
    The velocity v is a running sum of past gradients: steps that agree build up speed,
    zig-zags cancel out. SGD + momentum 0.9 is the classic optimizer for CNNs
    (AlexNet, VGG, ResNet). Adam would also work; see cnn_pytorch.py."""

    def __init__(self, model, lr=0.05, momentum=0.9):
        self.model, self.lr, self.mu = model, lr, momentum
        self.v = {name: np.zeros_like(p) for name, p, _ in model.named_params()}

    def step(self):
        for name, p, g in self.model.named_params():
            self.v[name] = self.mu * self.v[name] - self.lr * g
            p += self.v[name]  # in place, so the layer sees the new value


def train(model, Xtr, ytr, Xval, yval, epochs=15, lr=0.05, momentum=0.9, batch_size=32,
          seed=0, verbose=True):
    """Mini-batch SGD with momentum. Returns a per-epoch history list."""
    rng = np.random.default_rng(seed)
    opt = SGDMomentum(model, lr, momentum)
    history = []
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        perm = rng.permutation(len(Xtr))  # shuffle every epoch
        losses, correct = [], 0
        for start in range(0, len(Xtr), batch_size):
            idx = perm[start:start + batch_size]
            logits = model.forward(Xtr[idx])
            loss, dlogits = softmax_cross_entropy(logits, ytr[idx])
            model.backward(dlogits)
            opt.step()
            losses.append(loss * len(idx))
            correct += (logits.argmax(1) == ytr[idx]).sum()
        # train numbers are averaged over the epoch (the weights moved during it)
        h = {"epoch": epoch, "train_loss": sum(losses) / len(Xtr), "train_acc": correct / len(Xtr)}
        h["val_loss"], h["val_acc"] = model.evaluate(Xval, yval)
        history.append(h)
        if verbose:
            print(f"epoch {epoch:3d} | train loss {h['train_loss']:.3f} acc {h['train_acc']:6.1%}"
                  f" | val loss {h['val_loss']:.3f} acc {h['val_acc']:6.1%}"
                  f" | {time.time() - t0:.1f} s")
    return history


# ----------------------------------------------------------------------------
# Saving, loading, self-test, display
# ----------------------------------------------------------------------------
def save_model(path, model):
    arrays = {name: v for name, v, _ in model.named_params()}
    np.savez(path, classes=np.array(CLASSES), **{f"cfg_{k}": np.array(v)
                                                  for k, v in model.config.items()}, **arrays)


def load_model(path):
    d = np.load(path)
    cfg = {k[4:]: d[k].item() for k in d.files if k.startswith("cfg_")}
    model = CNN(**cfg)
    for lname, layer in model.layers:
        for pname in layer.params:
            layer.params[pname] = d[f"{lname}.{pname}"].astype(float)
    return model


def default_data(n_train=3000, n_val=600, size=16):
    """The standard train / val sets used by every script in this folder."""
    Xtr, ytr = make_shapes(n_train, size, seed=0)
    Xval, yval = make_shapes(n_val, size, seed=1)
    return Xtr, ytr, Xval, yval


def get_model(verbose=True):
    """Load cnn/model.npz if it exists; otherwise quietly train the default model (~5 s)."""
    path = os.path.join(HERE, "model.npz")
    if os.path.exists(path):
        if verbose:
            print(f"(loaded the trained model from {os.path.relpath(path)})\n")
        return load_model(path)
    if verbose:
        print("No cnn/model.npz yet, so training the default model first (~5 s).")
        print("To save it for next time run: python cnn/cnn_numpy.py\n")
    Xtr, ytr, Xval, yval = default_data()
    model = CNN()
    train(model, Xtr, ytr, Xval, yval, verbose=False)
    return model


def self_test():
    """The fast im2col code must agree with the slow, obvious loops."""
    rng = np.random.default_rng(42)
    x = rng.standard_normal((2, 3, 7, 7))
    conv = Conv2D(3, 4, 3, stride=2, pad=1, rng=rng)
    conv.params["b"] = rng.standard_normal(4)
    fast = conv.forward(x)
    slow = conv2d_naive(x, conv.params["W"], conv.params["b"], stride=2, pad=1)
    assert np.allclose(fast, slow), "im2col forward != naive loops"
    print(f"self-test: im2col conv == naive loop conv (stride 2, pad 1, out {fast.shape}) ✔")

    conv = Conv2D(3, 4, 3, stride=1, pad=1, rng=rng)
    out = conv.forward(x)
    dout = rng.standard_normal(out.shape)
    dx = conv.backward(dout)
    dW, db, dx_ref = conv2d_backward_naive(x, conv.params["W"], dout, pad=1)
    assert np.allclose(conv.grads["W"], dW) and np.allclose(conv.grads["b"], db)
    assert np.allclose(dx, dx_ref)
    print("self-test: im2col backward == 'dW = correlate(x, dout)', "
          "'dX = full conv(dout, flipped W)' ✔\n")


def show_predictions(model, X, y, n=6):
    probs = model.predict_proba(X[:n])
    blocks = [ascii_image(img[0]) for img in X[:n]]
    caps = []
    for p, t in zip(probs, y[:n]):
        k = p.argmax()
        mark = "✔" if k == t else "✘"
        caps.append(f"{mark} {CLASSES[k]} {p[k]:.0%}")
    side_by_side(blocks, caps)
    print("   caption = model's guess and its probability;  true labels: "
          + ", ".join(CLASSES[t] for t in y[:n]))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--momentum", type=float, default=0.9, help="0 = plain SGD")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--c1", type=int, default=8, help="channels of conv1")
    ap.add_argument("--c2", type=int, default=16, help="channels of conv2")
    ap.add_argument("--head", choices=["flatten", "gap"], default="flatten",
                    help="flatten + dense (default) or global average pooling + dense")
    ap.add_argument("--n-train", type=int, default=3000)
    ap.add_argument("--n-val", type=int, default=600)
    ap.add_argument("--save", default=os.path.join(HERE, "model.npz"))
    args = ap.parse_args()

    self_test()
    Xtr, ytr, Xval, yval = default_data(args.n_train, args.n_val)
    print(f"data: {len(Xtr)} training and {len(Xval)} validation images of shape "
          f"{Xtr.shape[1:]} (C, H, W), {len(CLASSES)} classes: {', '.join(CLASSES)}")
    print("a few training images (the shape can be anywhere):")
    side_by_side([ascii_image(img[0]) for img in Xtr[:6]], [CLASSES[t] for t in ytr[:6]])

    model = CNN(c1=args.c1, c2=args.c2, head=args.head)
    print("\nlayer      output shape      params   multiply-adds per image")
    x = Xtr[:1]
    print(f"{'input':8s}  {str(x.shape[1:]):16s}  {0:6d}")
    total_macs = 0
    for name, layer in model.layers:
        x = layer.forward(x)
        n = sum(v.size for v in layer.params.values())
        W = layer.params.get("W")
        # conv: every output number is a dot product of length C_in·k·k; dense: D_in per output
        macs = 0 if W is None else x.size * (W[0].size if W.ndim == 4 else W.shape[0])
        total_macs += macs
        print(f"{name:8s}  {str(x.shape[1:]):16s}  {n:6d}   {macs:9,d}" if macs else
              f"{name:8s}  {str(x.shape[1:]):16s}  {n:6d}")
    print(f"total parameters: {model.n_params():,}   total multiply-adds: {total_macs:,}")
    print(f"random guessing loss = ln({len(CLASSES)}) = {np.log(len(CLASSES)):.3f}\n")

    t0 = time.time()
    train(model, Xtr, ytr, Xval, yval, args.epochs, args.lr, args.momentum, args.batch_size)
    print(f"training took {time.time() - t0:.1f} s\n")

    print("some validation images and what the model says:")
    show_predictions(model, Xval, yval)
    save_model(args.save, model)
    print(f"\nSaved model to {os.path.relpath(args.save)}. Try: python cnn/predict.py")


if __name__ == "__main__":
    main()
