"""
The models under evaluation: this repo's own character-level language models,
trained on demand and wrapped behind ONE small interface, so the harness never
cares what is inside (exactly how a production harness treats an API model).

    gpt         the mini-GPT of ../transformer/transformer_numpy.py, trained on the
                TRAIN split of the story (paragraphs 0-16), default settings
    gpt-full    the same model trained on the WHOLE story, i.e. on the eval set too:
                identical to ../transformer/model.npz, our "contaminated" model
    rnn         the char-RNN of ../rnn/rnn_numpy.py (same loop and settings), train split
    bigram      a smoothed character-bigram counter: the "is it better than trivial?" baseline

Options go after an @, e.g. "gpt@steps=300" (an under-trained model, our fake
regression) or "gpt@seed=1" (same recipe, different seed). Trained weights are
cached in evals/runs/models/ (gitignored); delete them to retrain.

The interface (the `Model` protocol in harness.py):
    model.generate(prompt, n, temperature, max_chars, stop, seed) -> n completions
    model.logprobs(prompt, continuation) -> log p of every continuation character
    model.next_char_logits(prompts) -> (B, V) logits for the next character

Run (trains/loads every model and prints what it says after a prompt):
    python evals/models.py
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import os
import re
import sys
import time
import warnings
from typing import Dict, List, Sequence

import numpy as np

warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "rnn"))
sys.path.insert(0, os.path.join(ROOT, "transformer"))
from transformer_numpy import GPT, load_model, save_model, train as train_gpt  # noqa: E402
from rnn_numpy import RNN  # noqa: E402

STORY_PATH = os.path.join(ROOT, "rnn", "input.txt")
HELDOUT_PARA = 17  # paragraphs 17..23 (the last 23% of the story) are never trained on
MODEL_DIR = os.path.join(HERE, "runs", "models")

BASE_SPECS = {
    "gpt": dict(kind="gpt", data="train", steps=2000, seed=0),
    "gpt-full": dict(kind="gpt", data="full", steps=2000, seed=0),
    "rnn": dict(kind="rnn", data="train", iters=10000, seed=0),
    "bigram": dict(kind="bigram", data="train", alpha=0.1),
}


# ----------------------------------------------------------------------------
# The story and its train / held-out split
# ----------------------------------------------------------------------------
def load_story() -> str:
    with open(STORY_PATH, encoding="utf-8") as f:
        return f.read()


def paragraph_spans(text: str):
    """(start, end) of every paragraph (one line each, separated by blank lines)."""
    return [(m.start(), m.end()) for m in re.finditer(r"[^\n]+", text)]


def split_offset(text: str) -> int:
    """Character offset where the held-out part begins."""
    return paragraph_spans(text)[HELDOUT_PARA][0]


def paragraph_of(text: str, pos: int) -> int:
    spans = paragraph_spans(text)
    for i, (s, e) in enumerate(spans):
        if pos < e + 1:
            return i
    return len(spans) - 1


def story_vocab() -> List[str]:
    """Every model uses the FULL story's 44 characters, so held-out text is
    always encodable (characters seen only in held-out text just get low
    probability, like rare tokens in a real tokenizer)."""
    return sorted(set(load_story()))


# ----------------------------------------------------------------------------
# The common interface
# ----------------------------------------------------------------------------
class CharLM:
    """Shared logic. Subclasses implement start / step / logits on a batch state."""

    kind = "base"

    def __init__(self, name: str, chars: Sequence[str]):
        self.name = name
        self.chars = list(chars)
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.fingerprint = "untrained"

    def encode(self, s: str) -> List[int]:
        try:
            return [self.stoi[c] for c in s]
        except KeyError as e:
            raise ValueError(f"character {e} is not in the model's vocabulary") from None

    # stateful batched decoding: state holds n parallel sequences
    def start(self, prompt_ids: List[int], n: int):
        raise NotImplementedError

    def step(self, state, ids: np.ndarray):
        raise NotImplementedError

    def logits(self, state) -> np.ndarray:  # (n, V)
        raise NotImplementedError

    def next_char_logits(self, prompts: Sequence[str]) -> np.ndarray:
        return np.concatenate([self.logits(self.start(self.encode(p), 1)) for p in prompts])

    def logprobs(self, prompt: str, continuation: str) -> List[float]:
        """log p(continuation[i] | prompt + continuation[:i]) for every i, in nats."""
        ids = self.encode(continuation)
        state = self.start(self.encode(prompt), 1)
        out = []
        for t in ids:
            z = self.logits(state)[0]
            out.append(float(z[t] - logsumexp(z)))
            state = self.step(state, np.array([t]))
        return out

    def generate(self, prompt: str, n: int = 1, temperature: float = 0.0, max_chars: int = 20,
                 stop: str = "", seed: int = 0) -> List[str]:
        """n completions in parallel. temperature 0 = greedy (argmax). Each sample
        ends at the first character in `stop` (not included) or after max_chars."""
        rng = np.random.default_rng(seed)
        state = self.start(self.encode(prompt), n)
        out = [[] for _ in range(n)]
        done = np.zeros(n, dtype=bool)
        stop_ids = {self.stoi[c] for c in stop if c in self.stoi}
        for _ in range(max_chars):
            z = self.logits(state)
            if temperature <= 0:
                nxt = z.argmax(axis=1)
            else:
                # Inverse-CDF sampling, all rows at once: first index whose
                # cumulative probability exceeds a uniform draw.
                cdf = np.cumsum(softmax(z / temperature), axis=1)
                u = rng.random(n)[:, None]
                nxt = np.minimum((cdf < u).sum(axis=1), z.shape[1] - 1)
            for i, t in enumerate(nxt):
                if not done[i]:
                    if int(t) in stop_ids:
                        done[i] = True
                    else:
                        out[i].append(self.chars[int(t)])
            if done.all():
                break
            state = self.step(state, nxt)
        return ["".join(o) for o in out]


def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def logsumexp(z):
    m = z.max()
    return float(m + np.log(np.exp(z - m).sum()))


def fingerprint(arrays: Dict[str, np.ndarray]) -> str:
    """A hash of the weights: the response cache is keyed on it, so retraining a
    model can never serve stale cached answers."""
    h = hashlib.sha1()
    for k in sorted(arrays):
        h.update(k.encode())
        h.update(np.ascontiguousarray(arrays[k], dtype=np.float64).tobytes())
    return h.hexdigest()[:12]


class GPTModel(CharLM):
    """The NumPy mini-GPT. State = the n token sequences; every step re-runs the
    last block_size tokens (no KV cache needed at this size)."""

    kind = "gpt"

    def __init__(self, name, chars, gpt: GPT):
        super().__init__(name, chars)
        self.gpt = gpt
        self.T = gpt.config["block_size"]
        self.fingerprint = fingerprint(gpt.params())

    def start(self, prompt_ids, n):
        return np.tile(np.array(prompt_ids, dtype=np.int64)[None], (n, 1))

    def step(self, state, ids):
        return np.concatenate([state, np.asarray(ids, dtype=np.int64)[:, None]], axis=1)

    def logits(self, state):
        return self.gpt.forward(state[:, -self.T:])[:, -1].astype(np.float64)

    def window_logits(self, windows: np.ndarray, chunk: int = 512) -> np.ndarray:
        """Last-position logits for a (N, L) batch of equal-length windows."""
        return np.concatenate([self.gpt.forward(windows[i:i + chunk])[:, -1]
                               for i in range(0, len(windows), chunk)]).astype(np.float64)

    def logprobs(self, prompt, continuation):
        """Sliding window: every continuation character is predicted from the
        block_size characters right before it, all windows in one batch.
        (Scoring non-overlapping chunks instead gives worse numbers: README section 7.)"""
        ids = np.array(self.encode(prompt + continuation), dtype=np.int64)
        targets = list(range(len(prompt), len(ids)))
        out = np.zeros(len(targets))
        by_len: Dict[int, List[int]] = {}
        for j, i in enumerate(targets):
            if i == 0:
                raise ValueError("need at least one character of prompt")
            by_len.setdefault(min(i, self.T), []).append(j)
        for L, js in by_len.items():
            windows = np.stack([ids[targets[j] - L:targets[j]] for j in js])
            z = self.window_logits(windows)
            z = z - z.max(axis=1, keepdims=True)
            lp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
            out[js] = lp[np.arange(len(js)), ids[[targets[j] for j in js]]]
        return out.tolist()

    def next_char_logits(self, prompts):
        by_len: Dict[int, List[int]] = {}
        enc = [self.encode(p)[-self.T:] for p in prompts]
        for j, e in enumerate(enc):
            by_len.setdefault(len(e), []).append(j)
        out = np.zeros((len(prompts), len(self.chars)))
        for _, js in by_len.items():
            out[js] = self.window_logits(np.array([enc[j] for j in js], dtype=np.int64))
        return out


class RNNModel(CharLM):
    """The NumPy char-RNN. State = hidden vectors of the n sequences, (H, n)."""

    kind = "rnn"

    def __init__(self, name, chars, rnn: RNN):
        super().__init__(name, chars)
        self.P = rnn.params
        self.H = rnn.H
        self.fingerprint = fingerprint(rnn.params)

    def _advance(self, h, ids):
        P = self.P
        # W_xh @ onehot(id) is just column id of W_xh, for every sequence at once.
        return np.tanh(P["W_xh"][:, ids] + P["W_hh"] @ h + P["b_h"])

    def start(self, prompt_ids, n):
        h = np.zeros((self.H, n))
        for t in prompt_ids[:-1]:
            h = self._advance(h, np.full(n, t))
        # The last prompt char is the input that produces the first prediction.
        h = self._advance(h, np.full(n, prompt_ids[-1]))
        return h

    def step(self, h, ids):
        return self._advance(h, np.asarray(ids))

    def logits(self, h):
        return (self.P["W_hy"] @ h + self.P["b_y"]).T


class BigramModel(CharLM):
    """p(next | previous char) from smoothed counts on the training text."""

    kind = "bigram"

    def __init__(self, name, chars, text, alpha):
        super().__init__(name, chars)
        V = len(chars)
        counts = np.full((V, V), alpha)
        ids = self.encode(text)
        np.add.at(counts, (ids[:-1], ids[1:]), 1.0)
        self.logp = np.log(counts / counts.sum(axis=1, keepdims=True))
        self.fingerprint = fingerprint({"logp": self.logp})

    def start(self, prompt_ids, n):
        return np.full(n, prompt_ids[-1])

    def step(self, state, ids):
        return np.asarray(ids)

    def logits(self, state):
        return self.logp[state]


# ----------------------------------------------------------------------------
# Training on demand (cached in evals/runs/models/)
# ----------------------------------------------------------------------------
def parse_spec(spec: str) -> Dict:
    """'gpt@steps=300,seed=1' -> the base config with those values replaced."""
    name, _, opts = spec.partition("@")
    if name not in BASE_SPECS:
        raise SystemExit(f"unknown model {name!r}; choose from {sorted(BASE_SPECS)} "
                         "(options after @, e.g. gpt@steps=300,seed=1)")
    cfg = dict(BASE_SPECS[name])
    for kv in filter(None, opts.split(",")):
        k, _, v = kv.partition("=")
        if k not in cfg or k == "kind":
            raise SystemExit(f"unknown option {k!r} for {name}; options: "
                             f"{[c for c in cfg if c != 'kind']}")
        cfg[k] = type(cfg[k])(v)
    return cfg


def training_text(cfg) -> str:
    text = load_story()
    return text if cfg["data"] == "full" else text[:split_offset(text)]


def train_rnn(text, chars, hidden=100, seq_len=25, lr=0.1, iters=10000, clip=5.0, seed=0):
    """The loop of ../rnn/rnn_numpy.train (Adagrad, truncated BPTT, same defaults),
    without the printing and with the full-story vocabulary."""
    stoi = {c: i for i, c in enumerate(chars)}
    data = [stoi[c] for c in text]
    model = RNN(len(chars), hidden, seed)
    mem = {k: np.zeros_like(v) for k, v in model.params.items()}
    p, h = 0, np.zeros((hidden, 1))
    for _ in range(iters + 1):
        if p + seq_len + 1 > len(data):
            p, h = 0, np.zeros((hidden, 1))
        _, h, cache = model.forward(data[p:p + seq_len], data[p + 1:p + seq_len + 1], h)
        grads = model.backward(cache)
        for k, g in grads.items():
            np.clip(g, -clip, clip, out=g)
            mem[k] += g * g
            model.params[k] -= lr * g / np.sqrt(mem[k] + 1e-8)
        p += seq_len
    return model


def cache_path(cfg) -> str:
    key = "_".join(f"{k}{v}" for k, v in cfg.items() if k not in ("kind",))
    return os.path.join(MODEL_DIR, f"{cfg['kind']}_{key}.npz")


def get_model(spec: str, verbose: bool = True) -> CharLM:
    """Build (train or load) the model named by spec."""
    cfg = parse_spec(spec)
    chars = story_vocab()
    text = training_text(cfg)
    if cfg["kind"] == "bigram":
        return BigramModel(spec, chars, text, cfg["alpha"])
    path = cache_path(cfg)
    os.makedirs(MODEL_DIR, exist_ok=True)
    if cfg["kind"] == "gpt":
        if not os.path.exists(path):
            if verbose:
                print(f"[models] training {spec} on {cfg['data']} text ({len(text)} chars, "
                      f"{cfg['steps']} steps, ~{cfg['steps'] / 230:.0f} s)...", flush=True)
            t0 = time.time()
            data = np.array([chars.index(c) for c in text], dtype=np.int64)
            with contextlib.redirect_stdout(io.StringIO()):
                m32, hist = train_gpt(data, len(chars), steps=cfg["steps"], seed=cfg["seed"],
                                      verbose=False)
            save_model(path, m32, chars, hist)
            if verbose:
                print(f"[models]   done in {time.time() - t0:.1f} s, cached in "
                      f"{os.path.relpath(path)}", flush=True)
        gpt, _, _ = load_model(path, dtype=np.float64)  # evaluate in float64: deterministic
        return GPTModel(spec, chars, gpt)
    if not os.path.exists(path):
        if verbose:
            print(f"[models] training {spec} on {cfg['data']} text ({len(text)} chars, "
                  f"{cfg['iters']} iterations, ~{cfg['iters'] / 900:.0f} s)...", flush=True)
        t0 = time.time()
        rnn = train_rnn(text, chars, iters=cfg["iters"], seed=cfg["seed"])
        np.savez(path, **rnn.params)
        if verbose:
            print(f"[models]   done in {time.time() - t0:.1f} s, cached in "
                  f"{os.path.relpath(path)}", flush=True)
    d = np.load(path)
    rnn = RNN(len(chars), d["W_hh"].shape[0])
    for k in rnn.params:
        rnn.params[k] = d[k].astype(np.float64)
    return RNNModel(spec, chars, rnn)


def main():
    text = load_story()
    off = split_offset(text)
    print(f"story: {len(text)} chars. Train split = paragraphs 0-{HELDOUT_PARA - 1} "
          f"({off} chars); held out = paragraphs {HELDOUT_PARA}-"
          f"{len(paragraph_spans(text)) - 1} ({len(text) - off} chars)\n")
    prompt = "The girl saw the storm from her window in the "
    for spec in ["bigram", "rnn", "gpt", "gpt-full"]:
        m = get_model(spec)
        out = m.generate(prompt, max_chars=40, stop="\n")[0]
        lp = m.logprobs(text[off - 40:off], text[off:off + 200])
        print(f"  {spec:9s} fingerprint {m.fingerprint}  greedy: ...in the {out!r}")
        print(f"  {'':9s} held-out text, first 200 chars: {-np.mean(lp) / np.log(2):.3f} bits/char")


if __name__ == "__main__":
    main()
