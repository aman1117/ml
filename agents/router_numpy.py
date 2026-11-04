"""
The tool router: a tiny learned classifier, from scratch in NumPy.

    request text --hashed bag of words--> x (D,) --Linear--> ReLU --Linear--> softmax over
        {calculator, search_docs, read_file, unit_convert, kv_memory, no_tool}

Why a router? Every agent turn starts with "does this need a tool, and which
one?". A frontier LLM answers that implicitly while generating; production systems
often ALSO put a small, fast, cheap classifier in front (to pick a tool subset,
a model size, or a workflow). Here it is the one LEARNED piece of the agent:
ScriptedPolicy (policies.py) asks it which tool to use for every clause of a
request, and the "grounding check" guardrail (agent.py) asks it whether an answer
given without any tool call should have used one.

Features (the "hashing trick", Weinberger et al. 2009): lowercase words, numbers
-> <num>, file paths -> <path> + <ext:md>, plus word bigrams; each feature is
hashed with crc32 into one of D buckets (stable across runs, unlike Python's
salted hash()). No vocabulary to store, unseen words still land somewhere.

Every layer has a hand-written backward(); gradient_check.py verifies them.

Run (~2 s):
    python agents/router_numpy.py
    python agents/router_numpy.py --hidden 32 --epochs 10
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
import warnings
import zlib

import numpy as np

# NumPy 2.0 + Apple Accelerate raises spurious "... encountered in matmul"
# warnings even when results are correct. Only silence those.
warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)

HERE = os.path.dirname(os.path.abspath(__file__))
LABELS = ["calculator", "search_docs", "read_file", "unit_convert", "kv_memory", "no_tool"]
D_HASH = 2048
MODEL_PATH = os.path.join(HERE, "router.npz")
DEFAULTS = dict(hidden=64, epochs=25, lr=3e-3, batch=64, weight_decay=1e-4, seed=0)


# ----------------------------------------------------------------------------
# Features
# ----------------------------------------------------------------------------
def router_tokens(text: str):
    toks = []
    for t in re.findall(r"[A-Za-z0-9_./\-]+|[+*/^%=?$()-]", text.lower()):
        if re.fullmatch(r"[\w\-]+(/[\w.\-]+)*\.(md|py|txt|json|jsonl)", t):
            toks += ["<path>", "<ext:" + t.rsplit(".", 1)[1] + ">"]
        elif re.fullmatch(r"-?\d[\d,]*(\.\d+)?(e-?\d+)?", t):
            toks.append("<num>")
        else:
            toks.append(t.strip("./-") or t)
    return toks


def feature_ids(text: str, D: int = D_HASH):
    toks = router_tokens(text)
    feats = toks + [a + "_" + b for a, b in zip(toks, toks[1:])] + ["<bias>"]
    return sorted({zlib.crc32(f.encode()) % D for f in feats})


def featurize(texts, D: int = D_HASH, dtype=np.float32):
    """Binary bag of hashed features, (N, D). Row-normalized so long and short
    requests have comparable scale."""
    X = np.zeros((len(texts), D), dtype=dtype)
    for i, t in enumerate(texts):
        ids = feature_ids(t, D)
        X[i, ids] = 1.0 / np.sqrt(len(ids))
    return X


# ----------------------------------------------------------------------------
# Layers (forward caches what backward needs)
# ----------------------------------------------------------------------------
class Linear:
    """y = x W + b.   x: (N, D_in)  W: (D_in, D_out)  b: (D_out,)"""

    def __init__(self, d_in, d_out, rng, dtype=np.float32):
        self.params = {"W": (rng.standard_normal((d_in, d_out)) * np.sqrt(2.0 / d_in)).astype(dtype),
                       "b": np.zeros(d_out, dtype=dtype)}
        self.grads = {k: np.zeros_like(v) for k, v in self.params.items()}

    def forward(self, x):
        self.x = x
        return x @ self.params["W"] + self.params["b"]

    def backward(self, dy):
        self.grads["W"] = self.x.T @ dy          # (D_in, N) @ (N, D_out)
        self.grads["b"] = dy.sum(axis=0)
        return dy @ self.params["W"].T           # (N, D_out) @ (D_out, D_in)


class ReLU:
    params, grads = {}, {}

    def forward(self, x):
        self.mask = x > 0
        return x * self.mask

    def backward(self, dy):
        return dy * self.mask


def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def cross_entropy(logits, y):
    """Mean softmax cross-entropy and dL/dlogits = (p - onehot(y)) / N."""
    N = len(y)
    p = softmax(logits)
    loss = float(-np.log(p[np.arange(N), y] + 1e-12).mean())
    d = p.copy()
    d[np.arange(N), y] -= 1.0
    return loss, d / N


class Router:
    def __init__(self, D=D_HASH, hidden=64, n_out=len(LABELS), seed=0, dtype=np.float32):
        rng = np.random.default_rng(seed)
        self.config = dict(D=D, hidden=hidden, n_out=n_out)
        self.fc1, self.act, self.fc2 = Linear(D, hidden, rng, dtype), ReLU(), Linear(hidden, n_out, rng, dtype)
        self.layers = [self.fc1, self.act, self.fc2]
        self.dtype = dtype

    def params(self):
        return {"fc1.W": self.fc1.params["W"], "fc1.b": self.fc1.params["b"],
                "fc2.W": self.fc2.params["W"], "fc2.b": self.fc2.params["b"]}

    def grads(self):
        return {"fc1.W": self.fc1.grads["W"], "fc1.b": self.fc1.grads["b"],
                "fc2.W": self.fc2.grads["W"], "fc2.b": self.fc2.grads["b"]}

    def forward(self, X):
        h = X
        for layer in self.layers:
            h = layer.forward(h)
        return h  # logits (N, n_out)

    def loss(self, X, y):
        self.logits = self.forward(X)
        loss, self.dlogits = cross_entropy(self.logits, y)
        return loss

    def backward(self):
        d = self.dlogits
        for layer in reversed(self.layers):
            d = layer.backward(d)
        return self.grads()

    def predict_proba(self, texts):
        return softmax(self.forward(featurize(texts, self.config["D"], self.dtype)))

    def route(self, text):
        """-> (label, probability, all probabilities as a dict)."""
        p = self.predict_proba([text])[0]
        i = int(p.argmax())
        return LABELS[i], float(p[i]), {l: float(q) for l, q in zip(LABELS, p)}


# ----------------------------------------------------------------------------
# Data, training, evaluation
# ----------------------------------------------------------------------------
def load_jsonl(name):
    with open(os.path.join(HERE, name)) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return [r["text"] for r in rows], np.array([LABELS.index(r["label"]) for r in rows])


def confusion(y_true, y_pred, n=len(LABELS)):
    C = np.zeros((n, n), dtype=int)
    np.add.at(C, (y_true, y_pred), 1)
    return C


def train(hidden=64, epochs=25, lr=3e-3, batch=64, weight_decay=1e-4, seed=0, verbose=True):
    """Mini-batch Adam with decoupled weight decay on W (not b)."""
    tr_text, ytr = load_jsonl("router_train.jsonl")
    te_text, yte = load_jsonl("router_test.jsonl")
    Xtr, Xte = featurize(tr_text), featurize(te_text)
    model = Router(hidden=hidden, seed=seed)
    params = model.params()
    m = {k: np.zeros_like(v) for k, v in params.items()}
    v = {k: np.zeros_like(p) for k, p in params.items()}
    b1, b2, eps = 0.9, 0.999, 1e-8
    rng = np.random.default_rng(seed)
    hist = {"step": [], "loss": [], "epoch": [], "train_acc": [], "test_acc": [], "test_loss": []}
    t = 0
    t0 = time.time()
    for ep in range(epochs):
        perm = rng.permutation(len(ytr))
        for i in range(0, len(perm), batch):
            idx = perm[i:i + batch]
            loss = model.loss(Xtr[idx], ytr[idx])
            grads = model.backward()
            t += 1
            for k, p in params.items():
                g = grads[k]
                m[k] = b1 * m[k] + (1 - b1) * g
                v[k] = b2 * v[k] + (1 - b2) * g * g
                mh, vh = m[k] / (1 - b1 ** t), v[k] / (1 - b2 ** t)
                if k.endswith("W"):
                    p -= lr * weight_decay * p
                p -= lr * mh / (np.sqrt(vh) + eps)
            hist["step"].append(t)
            hist["loss"].append(loss)
        tr_acc = float((model.forward(Xtr).argmax(1) == ytr).mean())
        te_loss = cross_entropy(model.forward(Xte), yte)[0]
        te_acc = float((model.forward(Xte).argmax(1) == yte).mean())
        for k, val in (("epoch", ep + 1), ("train_acc", tr_acc), ("test_acc", te_acc), ("test_loss", te_loss)):
            hist[k].append(val)
        if verbose and (ep < 3 or (ep + 1) % 5 == 0):
            print(f"epoch {ep + 1:3d} | train loss {np.mean(hist['loss'][-len(range(0, len(perm), batch)):]):.4f} "
                  f"| train acc {tr_acc:.3f} | test loss {te_loss:.4f} | test acc {te_acc:.3f}")
    hist["seconds"] = time.time() - t0
    return model, {k: np.array(val) for k, val in hist.items()}


def save(model, hist, path=MODEL_PATH):
    np.savez(path, **{f"param.{k}": v for k, v in model.params().items()},
             **{f"hist.{k}": v for k, v in hist.items()},
             config=json.dumps(model.config))


_CACHED = None


def load_router(path=MODEL_PATH, verbose=False):
    """The trained router. Trains (and saves) one with the default settings if
    router.npz doesn't exist yet (~2 s), so every script works from a fresh clone."""
    global _CACHED
    if _CACHED is not None:
        return _CACHED
    if os.path.exists(path):
        z = np.load(path)
        cfg = json.loads(str(z["config"]))
        model = Router(D=cfg["D"], hidden=cfg["hidden"], n_out=cfg["n_out"])
        for k, p in model.params().items():
            p[...] = z[f"param.{k}"]
        hist = {k[5:]: z[k] for k in z.files if k.startswith("hist.")}
    else:
        if verbose:
            print("(no agents/router.npz yet: training the router, ~2 s)")
        model, hist = train(**DEFAULTS, verbose=False)
        save(model, hist, path)
    _CACHED = (model, hist)
    return _CACHED


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k, val in DEFAULTS.items():
        ap.add_argument(f"--{k.replace('_', '-')}", type=type(val), default=val)
    args = ap.parse_args()
    cfg = {k: getattr(args, k) for k in DEFAULTS}
    tr_text, ytr = load_jsonl("router_train.jsonl")
    te_text, yte = load_jsonl("router_test.jsonl")
    model = Router(hidden=cfg["hidden"])
    n_params = sum(p.size for p in model.params().values())
    print(f"data: {len(ytr)} train / {len(yte)} test requests, {len(LABELS)} labels "
          f"(test uses held-out phrasings)")
    print(f"model: hashed BoW (D={D_HASH}) -> Linear({D_HASH},{cfg['hidden']}) -> ReLU -> "
          f"Linear({cfg['hidden']},{len(LABELS)}) -> softmax  ({n_params:,} parameters)")
    print(f"first loss should be ≈ ln({len(LABELS)}) = {np.log(len(LABELS)):.3f}: "
          f"{model.loss(featurize(tr_text[:256]), ytr[:256]):.3f}\n")
    model, hist = train(**cfg)
    print(f"\ntrained in {hist['seconds']:.1f} s")
    pred = model.forward(featurize(te_text)).argmax(1)
    C = confusion(yte, pred)
    print(f"\ntest accuracy {(pred == yte).mean():.3f}   (always predicting the most common "
          f"label: {np.bincount(yte).max() / len(yte):.3f})\n")
    w = max(len(l) for l in LABELS)
    print(" " * (w + 2) + "predicted →")
    print(" " * (w + 2) + " ".join(f"{l[:6]:>6s}" for l in LABELS) + "  recall")
    for i, l in enumerate(LABELS):
        print(f"{l:>{w}s}  " + " ".join(f"{c:6d}" for c in C[i]) + f"  {C[i, i] / C[i].sum():.2f}")
    print(f"{'precision':>{w}s}  " + " ".join(f"{C[i, i] / max(1, C[:, i].sum()):6.2f}" for i in range(len(LABELS))))
    wrong = np.where(pred != yte)[0]
    if len(wrong):
        print(f"\nsome mistakes ({len(wrong)} total):")
        for i in wrong[:8]:
            print(f"  true {LABELS[yte[i]]:12s} pred {LABELS[pred[i]]:12s} {te_text[i]!r}")
    print("\nrouting a few new requests:")
    for q in ["Find the parameter count of the mini-GPT in the docs, then multiply it by 4",
              "multiply it by 4", "convert 3 GiB to MB", "What is my favorite optimizer?",
              "How many lines does rnn/input.txt have?", "thanks a lot!"]:
        lab, p, _ = model.route(q)
        print(f"  {lab:12s} {p:5.2f}  {q}")
    save(model, hist)
    print(f"\nsaved {os.path.relpath(MODEL_PATH)}")


if __name__ == "__main__":
    main()
