"""
Byte-level BPE (byte-pair encoding) from scratch: the tokenizer family of GPT-2/3/4,
LLaMA-3 and most modern LLMs, in pure Python.

What it teaches, top to bottom:
  1. Pre-tokenization: a regex splits text into chunks (words with their leading
     space, numbers, punctuation runs, whitespace). Merges never cross a chunk
     boundary, so "dog." and "dog!" share the token " dog" instead of learning
     "dog." and "dog!" separately. GPT2_PATTERN / GPT4_PATTERN below are stdlib-`re`
     approximations of the real patterns (which use `\\p{L}`, not available in `re`).
  2. Training: start from the 256 byte values; count every adjacent pair in the
     corpus; merge the most frequent pair into a new token; repeat until the
     vocabulary is big enough. `train_bpe` updates the pair counts incrementally
     (only the chunks that contain the merged pair are touched), with a heap to
     find the next best pair, so it learns 3,840 merges from ~178 KB in seconds.
  3. Encoding new text: inside each chunk, repeatedly apply the learned merge with
     the lowest rank (= learned earliest), like tiktoken and minbpe. `_encode_chunk`
     does that with a heap over a linked list, plus a cache of seen chunks.
  4. Decoding: concatenate the bytes of the tokens, then UTF-8 decode with
     errors="replace" (a token sequence can end in the middle of a character).
  5. Special tokens (<|endoftext|>, <|im_start|>, <|im_end|>): added after training,
     matched BEFORE the regex, never split, and only allowed when the caller says so.
  6. Save / load as JSON.

Corpus: ../rnn/input.txt plus the READMEs of the six earlier modules (read at run
time, ~197 KB of English Markdown with equations and code). The first 90% is the
training split; the last 10% (the end of transformer/README.md) is held out.

Run:
    python tokenizer/bpe.py                         (trains V = 4096, saves tokenizer/tokenizer.json)
    python tokenizer/bpe.py --vocab-size 1024 --pattern gpt2
    python tokenizer/bpe.py --encode "Tokenization is the root of all weirdness."
"""
from __future__ import annotations

import argparse
import hashlib
import heapq
import itertools
import json
import os
import re
import time
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
CORPUS_FILES = ["rnn/input.txt", "mlp/README.md", "cnn/README.md", "rnn/README.md",
                "lstm/README.md", "attention/README.md", "transformer/README.md"]
DEFAULT_PATH = os.path.join(HERE, "tokenizer.json")
DEFAULT_VOCAB = 4096
VAL_FRAC = 0.1

# ----------------------------------------------------------------------------
# 1. Pre-tokenization patterns
# ----------------------------------------------------------------------------
# The real GPT-2 pattern (needs the third-party `regex` module for \p{L}, \p{N}):
GPT2_PATTERN_ORIGINAL = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
# The real GPT-4 (cl100k_base) pattern, also used by LLaMA-3:
GPT4_PATTERN_ORIGINAL = (r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}|"""
                         r""" ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+""")

# stdlib `re` has no \p{...}. Python's \w = letters + digits + underscore (Unicode-aware),
# so "a letter" = a word char that is not a digit and not "_". Differences from the real
# classes only on rare characters (e.g. '²' or 'Ⅻ' count as letters here, numbers there).
_L = r"[^\W\d_]"            # ≈ \p{L}
_N = r"\d"                  # ≈ \p{N}  (Unicode decimal digits)
_O = r"(?:[^\s\w]|_)"       # ≈ [^\s\p{L}\p{N}]: not space, not letter, not digit
GPT2_PATTERN = rf"""'(?:[sdmt]|ll|ve|re)| ?{_L}+| ?{_N}+| ?{_O}+|\s+(?!\S)|\s+"""
# Possessive `?+` / `++` (Python 3.11+) don't change what these alternatives match.
GPT4_PATTERN = (rf"""'(?i:[sdmt]|ll|ve|re)|(?:[^\r\n\w]|_)?{_L}+|{_N}{{1,3}}|"""
                rf""" ?{_O}+[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+""")
PATTERNS = {"gpt2": GPT2_PATTERN, "gpt4": GPT4_PATTERN}
SPECIAL_TOKENS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]


def split_text(text: str, pattern: str = GPT4_PATTERN) -> List[str]:
    """Pre-tokenize. Every character lands in exactly one chunk, so ''.join(chunks) == text."""
    return re.findall(pattern, text)


# ----------------------------------------------------------------------------
# 2. The two building blocks: count pairs, merge a pair
# ----------------------------------------------------------------------------
def get_stats(seqs: Iterable[List[int]], weights: Optional[Iterable[int]] = None) -> Dict:
    """How often each adjacent pair occurs (each sequence counted `weight` times)."""
    stats: Dict[Tuple[int, int], int] = defaultdict(int)
    for ids, w in zip(seqs, weights if weights is not None else itertools.repeat(1)):
        for pair in zip(ids, ids[1:]):
            stats[pair] += w
    return stats


def merge(ids: List[int], pair: Tuple[int, int], idx: int) -> List[int]:
    """Replace every occurrence of `pair` in ids by the new token idx, left to right
    (so "a a a" with pair (a, a) becomes "aa a", never "a aa")."""
    a, b = pair
    out, i, n = [], 0, len(ids)
    while i < n:
        if i < n - 1 and ids[i] == a and ids[i + 1] == b:
            out.append(idx)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


def best_pair(stats: Dict) -> Tuple[int, int]:
    """Most frequent pair; ties go to the smallest (a, b), so training is deterministic."""
    return min(stats, key=lambda p: (-stats[p], p))


# ----------------------------------------------------------------------------
# 3. Training with incremental pair counts
# ----------------------------------------------------------------------------
def train_bpe(text: str, vocab_size: int, pattern: str = GPT4_PATTERN,
              verbose: bool = False) -> List[Tuple[int, int]]:
    """Learn vocab_size - 256 merges. Returns them in order; merge k creates token 256 + k.

    Same result as the naive loop "recount all pairs, merge the best, repeat"
    (bpe_check.py verifies it), but each step only touches the chunks that contain
    the merged pair:
      - identical chunks are counted once, with a frequency (" the" is one entry),
      - stats[pair] = Σ freq over all occurrences, where[pair] = chunks containing it,
      - after a merge, only those chunks are re-counted, and changed counts are pushed on
        a heap (stale heap entries are skipped when popped).
    """
    assert vocab_size >= 256
    chunk_freq = Counter(split_text(text, pattern))
    words = [list(c.encode("utf-8")) for c in chunk_freq]
    freqs = list(chunk_freq.values())
    stats: Dict[Tuple[int, int], int] = defaultdict(int)
    where: Dict[Tuple[int, int], set] = defaultdict(set)
    for wi, (w, f) in enumerate(zip(words, freqs)):
        for p in zip(w, w[1:]):
            stats[p] += f
            where[p].add(wi)
    heap = [(-c, p) for p, c in stats.items()]
    heapq.heapify(heap)  # smallest (-count, pair) first = most frequent, then smallest pair

    merges: List[Tuple[int, int]] = []
    vocab = {i: bytes([i]) for i in range(256)}
    t0 = time.time()
    for k in range(vocab_size - 256):
        while heap and stats.get(heap[0][1], 0) != -heap[0][0]:
            heapq.heappop(heap)  # stale: this pair's count changed after it was pushed
        if not heap:
            if verbose:
                print(f"  stopped early: no pair left to merge after {k} merges")
            break
        negc, pair = heapq.heappop(heap)
        idx = 256 + k
        merges.append(pair)
        vocab[idx] = vocab[pair[0]] + vocab[pair[1]]

        delta: Dict[Tuple[int, int], int] = defaultdict(int)
        for wi in where.pop(pair):
            w, f = words[wi], freqs[wi]
            new = merge(w, pair, idx)
            for p in zip(w, w[1:]):
                delta[p] -= f
            for p in zip(new, new[1:]):
                delta[p] += f
            new_pairs = set(zip(new, new[1:]))
            for p in set(zip(w, w[1:])) - new_pairs:
                if p in where:
                    where[p].discard(wi)
            for p in new_pairs:
                where[p].add(wi)
            words[wi] = new
        for p, d in delta.items():
            if d == 0:
                continue
            c = stats.get(p, 0) + d
            if c > 0:
                stats[p] = c
                heapq.heappush(heap, (-c, p))
            else:
                stats.pop(p, None)
                where.pop(p, None)

        if verbose and (k < 8 or (k + 1) % 500 == 0):
            print(f"  merge {k + 1:5d}/{vocab_size - 256}: {str(pair):12s} -> {idx:5d} "
                  f"{show(vocab[idx])!r:20s} count {-negc:6d}  ({time.time() - t0:.1f} s)")
    return merges


# ----------------------------------------------------------------------------
# 4. The tokenizer: encode, decode, special tokens, save / load
# ----------------------------------------------------------------------------
class Tokenizer:
    def __init__(self, merges: List[Tuple[int, int]], pattern: str = GPT4_PATTERN,
                 special_tokens: Iterable[str] = SPECIAL_TOKENS, meta: Optional[dict] = None):
        self.pattern = pattern
        self.meta = dict(meta or {})
        # merges[(a, b)] = new id. Ids grow in training order, so the id IS the rank.
        self.merges = {tuple(p): 256 + i for i, p in enumerate(merges)}
        self.vocab = {i: bytes([i]) for i in range(256)}
        for (a, b), idx in self.merges.items():
            self.vocab[idx] = self.vocab[a] + self.vocab[b]
        n = 256 + len(self.merges)
        self.special_tokens = {s: n + i for i, s in enumerate(special_tokens)}
        self.inverse_special = {i: s for s, i in self.special_tokens.items()}
        for s, i in self.special_tokens.items():
            self.vocab[i] = s.encode("utf-8")
        self._cache: Dict[bytes, List[int]] = {}

    @property
    def n_vocab(self) -> int:
        """Total ids: 256 bytes + merges + special tokens (GPT-2: 256 + 50,000 + 1 = 50,257)."""
        return len(self.vocab)

    def merge_list(self) -> List[Tuple[int, int]]:
        return list(self.merges)

    def truncated(self, vocab_size: int) -> "Tokenizer":
        """The tokenizer with only the first vocab_size - 256 merges. Training is greedy,
        so this is EXACTLY what training with the smaller vocab_size would have produced."""
        return Tokenizer(self.merge_list()[:vocab_size - 256], self.pattern,
                         list(self.special_tokens), self.meta)

    # --- encoding ---------------------------------------------------------
    def _encode_chunk(self, chunk: bytes) -> List[int]:
        """Apply merges by rank inside one chunk: always merge the adjacent pair whose
        merge was learned earliest (leftmost first among equal ranks).

        Heap of (rank, left position) over a doubly linked list: O(n log n) per chunk
        instead of the naive O(n²) "recount the pairs after every merge" loop.
        """
        cached = self._cache.get(chunk)
        if cached is not None:
            return cached
        ids: List[Optional[int]] = list(chunk)
        n = len(ids)
        nxt = list(range(1, n + 1))
        prv = list(range(-1, n - 1))
        merges = self.merges
        heap = [(merges[(ids[i], ids[i + 1])], i) for i in range(n - 1)
                if (ids[i], ids[i + 1]) in merges]
        heapq.heapify(heap)
        while heap:
            rank, i = heapq.heappop(heap)
            j = nxt[i]
            if ids[i] is None or j >= n or merges.get((ids[i], ids[j])) != rank:
                continue  # stale entry: one of the two tokens was merged away already
            ids[i], ids[j] = rank, None  # the left node becomes the new token
            nxt[i] = nxt[j]
            if nxt[j] < n:
                prv[nxt[j]] = i
            for left in (prv[i], i):  # the two new neighbouring pairs
                if left >= 0 and nxt[left] < n:
                    r = merges.get((ids[left], ids[nxt[left]]))
                    if r is not None:
                        heapq.heappush(heap, (r, left))
        out = [t for t in ids if t is not None]
        if len(self._cache) > 200_000:
            self._cache.clear()
        self._cache[chunk] = out
        return out

    def encode_ordinary(self, text: str) -> List[int]:
        """Encode, treating special-token strings like any other text."""
        ids: List[int] = []
        for chunk in split_text(text, self.pattern):
            ids.extend(self._encode_chunk(chunk.encode("utf-8")))
        return ids

    def encode(self, text: str, allowed_special="none_raise") -> List[int]:
        """allowed_special: "all", "none" (encode them as plain text), "none_raise"
        (default, like tiktoken: refuse text that contains a special-token string, so
        user input can't smuggle in an <|im_end|>), or a set of allowed strings."""
        if allowed_special == "all":
            special = self.special_tokens
        elif allowed_special == "none":
            special = {}
        elif allowed_special == "none_raise":
            special = {}
            found = [s for s in self.special_tokens if s in text]
            if found:
                raise ValueError(f"text contains special token(s) {found}; pass "
                                 "allowed_special='all' or 'none' to choose explicitly")
        else:
            special = {s: self.special_tokens[s] for s in allowed_special}
        if not special:
            return self.encode_ordinary(text)
        # Split on the special strings FIRST (longest first), so they are never broken up.
        pat = "(" + "|".join(re.escape(s) for s in sorted(special, key=len, reverse=True)) + ")"
        ids: List[int] = []
        for part in re.split(pat, text):
            if part in special:
                ids.append(special[part])
            elif part:
                ids.extend(self.encode_ordinary(part))
        return ids

    # --- decoding ---------------------------------------------------------
    def decode_bytes(self, ids: Iterable[int]) -> bytes:
        return b"".join(self.vocab[i] for i in ids)

    def decode(self, ids: Iterable[int]) -> str:
        # A token can hold part of a multi-byte character (e.g. 2 of the 3 bytes of '世'),
        # and a model can sample an invalid sequence: replace those bytes with U+FFFD.
        return self.decode_bytes(ids).decode("utf-8", errors="replace")

    def token_str(self, i: int) -> str:
        return self.inverse_special.get(i) or show(self.vocab[i])

    # --- save / load ------------------------------------------------------
    def save(self, path: str = DEFAULT_PATH) -> None:
        pattern_name = next((k for k, v in PATTERNS.items() if v == self.pattern), "custom")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"type": "byte-level BPE", "pattern_name": pattern_name,
                       "pattern": self.pattern, "meta": self.meta,
                       "special_tokens": list(self.special_tokens),
                       "merges": [list(p) for p in self.merges]}, f)

    @classmethod
    def load(cls, path: str = DEFAULT_PATH) -> "Tokenizer":
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        return cls([tuple(p) for p in d["merges"]], d["pattern"], d["special_tokens"], d["meta"])


def show(b: bytes) -> str:
    """Readable form of a token's bytes: '␣' for space, '↵' for newline, \\xNN for bytes
    that are not a complete UTF-8 character on their own."""
    s = b.decode("utf-8", errors="backslashreplace")
    return s.replace(" ", "␣").replace("\n", "↵").replace("\t", "⇥")


def render_chat(messages: List[dict]) -> str:
    """ChatML, the chat template of GPT-3.5/4 and many open models (Qwen, ...). The model only
    ever sees ONE token stream; roles and turn ends are marked with special tokens."""
    out = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
    return out + "<|im_start|>assistant\n"  # the model continues from here


# ----------------------------------------------------------------------------
# 5. The corpus and a cached default tokenizer
# ----------------------------------------------------------------------------
def load_corpus(split: str = "train") -> str:
    """'all', 'train' (first 90%) or 'val' (last 10%) of the concatenated corpus."""
    docs = []
    for rel in CORPUS_FILES:
        with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
            docs.append(f.read())
    text = "\n\n".join(docs)
    n_train = int(len(text) * (1 - VAL_FRAC))
    return {"all": text, "train": text[:n_train], "val": text[n_train:]}[split]


def get_tokenizer(vocab_size: int = DEFAULT_VOCAB, pattern_name: str = "gpt4",
                  path: str = DEFAULT_PATH, verbose: bool = True) -> Tokenizer:
    """Load tokenizer.json if it was trained on the current corpus with these settings and
    at least vocab_size tokens (then truncate); otherwise train it and save it."""
    text = load_corpus("train")
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
    if os.path.exists(path):
        tok = Tokenizer.load(path)
        if (tok.meta.get("corpus_sha1") == digest and tok.meta.get("pattern_name") == pattern_name
                and 256 + len(tok.merges) >= vocab_size):
            return tok.truncated(vocab_size) if 256 + len(tok.merges) > vocab_size else tok
    train_size = max(vocab_size, DEFAULT_VOCAB)
    if verbose:
        print(f"(training the BPE tokenizer, V = {train_size}, on {len(text):,} chars; "
              f"saving {os.path.relpath(path)})")
    t0 = time.time()
    merges = train_bpe(text, train_size, PATTERNS[pattern_name])
    meta = {"corpus_sha1": digest, "pattern_name": pattern_name, "train_chars": len(text),
            "train_seconds": round(time.time() - t0, 2)}
    tok = Tokenizer(merges, PATTERNS[pattern_name], meta=meta)
    if pattern_name == "gpt4":  # only the default tokenizer is cached on disk
        tok.save(path)
    return tok.truncated(vocab_size) if train_size > vocab_size else tok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB,
                    help="256 bytes + merges (special tokens come on top)")
    ap.add_argument("--pattern", choices=list(PATTERNS), default="gpt4")
    ap.add_argument("--out", default=DEFAULT_PATH)
    ap.add_argument("--encode", default="The keeper lit the lamp. Tokenization isn't magic: "
                                        "12345 + 678 = 13023 🙂")
    args = ap.parse_args()

    train, val = load_corpus("train"), load_corpus("val")
    print(f"corpus: {len(CORPUS_FILES)} files, train {len(train):,} chars "
          f"({len(train.encode()):,} bytes), val {len(val):,} chars ({len(val.encode()):,} bytes)")
    chunks = split_text(train, PATTERNS[args.pattern])
    print(f"pre-tokenization ({args.pattern} pattern): {len(chunks):,} chunks, "
          f"{len(set(chunks)):,} unique. First ones: {chunks[:10]}")

    print(f"\ntraining {args.vocab_size - 256} merges:")
    t0 = time.time()
    merges = train_bpe(train, args.vocab_size, PATTERNS[args.pattern], verbose=True)
    secs = time.time() - t0
    digest = hashlib.sha1(train.encode("utf-8")).hexdigest()[:12]
    tok = Tokenizer(merges, PATTERNS[args.pattern],
                    meta={"corpus_sha1": digest, "pattern_name": args.pattern,
                          "train_chars": len(train), "train_seconds": round(secs, 2)})
    print(f"done in {secs:.1f} s: {len(merges)} merges, n_vocab = 256 + {len(merges)} + "
          f"{len(tok.special_tokens)} special = {tok.n_vocab}")

    t0 = time.time()
    tr_ids = tok.encode_ordinary(train)
    enc_secs = time.time() - t0
    va_ids = tok.encode_ordinary(val)
    assert tok.decode(tr_ids) == train and tok.decode(va_ids) == val
    print(f"encode train: {len(tr_ids):,} tokens in {enc_secs:.2f} s; "
          f"{len(train.encode()) / len(tr_ids):.2f} bytes/token on train, "
          f"{len(val.encode()) / len(va_ids):.2f} on held-out val (decode round-trips ✔)")

    longest = sorted(range(256, 256 + len(merges)), key=lambda i: (-len(tok.vocab[i]), i))[:8]
    print("longest tokens:", ", ".join(repr(tok.token_str(i)) for i in longest))
    print("last merges:   ", ", ".join(repr(tok.token_str(255 + len(merges) - k))
                                        for k in range(8)))

    ids = tok.encode(args.encode, allowed_special="all")
    print(f"\n{args.encode!r}\n  -> {len(ids)} tokens: {ids}")
    print("  -> " + "|".join(tok.token_str(i) for i in ids))
    print(f"  -> decode: {tok.decode(ids)!r}")

    chat = render_chat([{"role": "system", "content": "You are terse."},
                        {"role": "user", "content": "Hi!"}])
    ids = tok.encode(chat, allowed_special="all")
    print(f"\nChatML prompt -> {len(ids)} tokens: "
          + " ".join(f"[{tok.token_str(i)}]" if i in tok.inverse_special else tok.token_str(i)
                     for i in ids))
    try:
        tok.encode("user text with <|im_end|> inside")
    except ValueError as e:
        print(f"encode() of untrusted text containing a special string -> ValueError: {e}")

    tok.save(args.out)
    print(f"\nsaved {os.path.relpath(args.out)} ({os.path.getsize(args.out) // 1024} KB)")


if __name__ == "__main__":
    main()
