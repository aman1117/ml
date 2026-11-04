"""
Shared plumbing for serving/: the model we serve, its tokenizer, and the small
statistics helpers that the server, the load tester and the monitor all use.

  - get_model(): the trained mini-GPT from ../transformer/ (transformer/model.npz).
    If it does not exist, train one with the default settings (~10 s, the same
    seed as `python transformer/transformer_numpy.py`) and cache it as
    serving/model_cache.npz (gitignored), so we never write outside this folder.
  - model_version(): a content hash of the weights. Every log line carries it, so
    a quality regression can be traced to the exact weights that produced it.
  - CharTokenizer: text <-> ids for the char-level model, reporting characters the
    model has never seen (a production failure mode we log and monitor).
  - percentile(): the latency-percentile function, written out (numpy's default
    "linear" method), checked against np.percentile in serving_check.py.
  - repetition_ratio() / loop_period(): cheap output-quality signals for the logs.
  - tiny_model(): a small random-weight GPT for fast exact checks.

Not a script: imported by the other files.
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import math
import os
import sys
import warnings

import numpy as np

# NumPy 2.0 + Apple Accelerate raises spurious "divide by zero / overflow /
# invalid value encountered in matmul" warnings even when results are correct.
warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)

HERE = os.path.dirname(os.path.abspath(__file__))
TRANSFORMER_DIR = os.path.join(HERE, "..", "transformer")
sys.path.insert(0, TRANSFORMER_DIR)

from generate import KVCache, decode_step, generate_ids, prefill  # noqa: E402,F401
from transformer_numpy import (DEFAULT_TEXT, GPT, evaluate, load_model,  # noqa: E402,F401
                               load_text, sample_next, save_model, softmax, train)

MODEL_PATH = os.path.join(TRANSFORMER_DIR, "model.npz")
CACHE_PATH = os.path.join(HERE, "model_cache.npz")
HELDOUT_PATH = os.path.join(HERE, "heldout.txt")
LOG_DIR = os.path.join(HERE, "logs")


def get_model(dtype=np.float32, verbose=True):
    """(model, chars, version) of the GPT we serve. Trains it if needed (~10 s)."""
    for path in (MODEL_PATH, CACHE_PATH):
        if os.path.exists(path):
            model, chars, _ = load_model(path, dtype=dtype)
            if verbose:
                print(f"(serving {os.path.relpath(path)}: {model.n_params():,} params, "
                      f"version {model_version(model)})")
            return model, chars, model_version(model)
    if verbose:
        print("No transformer/model.npz yet, so training the default mini-GPT first (~10 s).")
        print("To share it with ../transformer/, run: python transformer/transformer_numpy.py")
    _, chars, data = load_text(DEFAULT_TEXT)
    with contextlib.redirect_stdout(io.StringIO()):
        model32, hist = train(data, len(chars), verbose=False)
    save_model(CACHE_PATH, model32, chars, hist)
    model, chars, _ = load_model(CACHE_PATH, dtype=dtype)
    return model, chars, model_version(model)


def model_version(model):
    """'gpt-<8 hex>': hash of the config and every weight (independent of dtype)."""
    h = hashlib.sha256(json.dumps(model.config, sort_keys=True).encode())
    for name, p in sorted(model.params().items()):
        h.update(name.encode())
        h.update(np.ascontiguousarray(p, dtype=np.float32).tobytes())
    return "gpt-" + h.hexdigest()[:8]


class CharTokenizer:
    def __init__(self, chars):
        self.chars = list(chars)
        self.stoi = {c: i for i, c in enumerate(self.chars)}

    def __len__(self):
        return len(self.chars)

    def encode(self, text):
        """(ids, unknown_chars). Unknown characters are dropped, not guessed."""
        ids, unknown = [], []
        for c in text:
            (ids.append(self.stoi[c]) if c in self.stoi else unknown.append(c))
        return ids, unknown

    def decode(self, ids):
        return "".join(self.chars[i] for i in ids)


def text_ids(tok, path):
    with open(path, encoding="utf-8") as f:
        ids, unknown = tok.encode(f.read())
    assert not unknown, f"{path} has characters outside the vocabulary: {set(unknown)}"
    return np.array(ids, dtype=np.int64)


def heldout_ids(tok):
    """serving/heldout.txt: new sentences in the training text's style, never trained on."""
    return text_ids(tok, HELDOUT_PATH)


def train_ids(tok):
    return text_ids(tok, DEFAULT_TEXT)


def percentile(values, q):
    """q-th percentile (0..100) with linear interpolation between closest ranks:
    the sorted values sit at positions 0..n-1, the percentile at (n-1)·q/100."""
    xs = sorted(values)
    if not xs:
        return float("nan")
    pos = (len(xs) - 1) * q / 100.0
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(xs) - 1)
    return float(xs[lo] + (xs[hi] - xs[lo]) * (pos - lo))


def repetition_ratio(text, n=4):
    """Fraction of n-grams in text that already appeared earlier in it.
    ~0 for varied text, -> 1 for a degenerate loop like 'the the the the'."""
    grams = [text[i:i + n] for i in range(len(text) - n + 1)]
    if not grams:
        return 0.0
    return 1.0 - len(set(grams)) / len(grams)


def loop_period(text, min_repeats=3, max_period=12):
    """Smallest p such that the text ENDS with the same p characters repeated
    min_repeats times in a row ('...and the and the and the' -> 8), else 0."""
    for p in range(1, max_period + 1):
        if len(text) < p * min_repeats:
            break
        unit = text[-p:]
        if unit.strip() and text.endswith(unit * min_repeats):
            return p
    return 0


def tiny_model(V=12, n_layer=2, n_head=2, d_model=16, block_size=16, seed=0, std=0.3,
               dtype=np.float64):
    """Random-weight GPT with peaky (non-uniform) outputs, for exact equivalence checks."""
    rng = np.random.default_rng(seed)
    model = GPT(V, n_layer, n_head, d_model, block_size, seed=seed, dtype=dtype)
    for name, p in model.params().items():
        p[...] = (1.0 + 0.1 * rng.standard_normal(p.shape) if name.endswith("gamma")
                  else std * rng.standard_normal(p.shape))
    return model


def lm_logits(model, x):
    """Final LayerNorm'd hidden states (..., C) -> logits (..., V)."""
    if model.lm_head is None:
        return x @ model.wte.params["weight"].T
    return model.lm_head.forward(x)
