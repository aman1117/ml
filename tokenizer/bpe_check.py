"""
BPE check: verify the tokenizer in bpe.py against slow, obviously-correct references.

This is the tokenizer's analog of gradient_check.py. Every line is one property:
  - pre-tokenization is lossless and splits like GPT-2 / GPT-4 on known examples,
  - decode(encode(s)) == s for random Unicode strings (emoji, ZWJ sequences, flags,
    CJK, accents, combining marks, whitespace runs, control characters, random code
    points) and decode never crashes on arbitrary token ids (invalid UTF-8),
  - the fast heap encoder == the minbpe-style naive encoder (recount the pairs after
    every merge) == replaying all merges in training order,
  - the fast incremental trainer learns exactly the same merges as the naive trainer
    (recount every pair in the whole corpus after every merge), including ties and
    overlapping pairs ("aaaa"), and is deterministic,
  - truncating a trained tokenizer == training with a smaller vocabulary,
  - special tokens are never split, never produced from plain text unless allowed,
  - vocabulary bookkeeping (sizes, ids, bytes) and JSON save/load,
  - optional: if `tiktoken` is installed (it is NOT a requirement), our encoder with
    the real GPT-2 merges reproduces tiktoken's GPT-2 ids. Otherwise: SKIP.

Prints one line per check and ends with "BPE check PASSED" (exit 0) or FAILED (exit 1).

Run (~1.5 s):
    python tokenizer/bpe_check.py
"""
import os
import random
import time

from bpe import (GPT2_PATTERN, GPT4_PATTERN, HERE, PATTERNS, Tokenizer, best_pair, get_stats,
                 load_corpus, merge, split_text, train_bpe)

RESULTS = []


def report(name, ok, detail=""):
    RESULTS.append(bool(ok))
    print(f"  {name:68s} {'OK' if ok else 'FAIL'}  {detail}")
    return ok


# ----------------------------------------------------------------------------
# Reference implementations: slow, simple, obviously correct
# ----------------------------------------------------------------------------
def train_naive(text, vocab_size, pattern):
    """Recount every pair in the whole corpus after every merge (no dedup, no heap)."""
    chunks = [list(c.encode("utf-8")) for c in split_text(text, pattern)]
    merges = []
    for k in range(vocab_size - 256):
        stats = get_stats(chunks)
        if not stats:
            break
        pair = best_pair(stats)
        chunks = [merge(c, pair, 256 + k) for c in chunks]
        merges.append(pair)
    return merges


def encode_naive(tok, text):
    """minbpe: per chunk, recount the pairs, merge the one with the lowest rank, repeat."""
    out = []
    for chunk in split_text(text, tok.pattern):
        ids = list(chunk.encode("utf-8"))
        while len(ids) >= 2:
            stats = get_stats([ids])
            pair = min(stats, key=lambda p: tok.merges.get(p, float("inf")))
            if pair not in tok.merges:
                break
            ids = merge(ids, pair, tok.merges[pair])
        out.extend(ids)
    return out


def encode_replay(tok, text):
    """The definition: apply every merge, in the order it was learned, to every chunk."""
    chunks = [list(c.encode("utf-8")) for c in split_text(text, tok.pattern)]
    for pair, idx in tok.merges.items():
        a = pair[0]
        chunks = [merge(c, pair, idx) if a in c else c for c in chunks]
    return [i for c in chunks for i in c]


# ----------------------------------------------------------------------------
# Random test strings
# ----------------------------------------------------------------------------
POOLS = [
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ",
    "0123456789", ".,;:!?'\"()[]{}<>|/\\-_=+*&^%$#@~`",
    [" ", "  ", "   ", "\t", "\n", "\n\n", "\r\n", "\u00a0", "\u3000", " \n "],
    "éèêëàâäçñüöïîôûœæßøåÉÀÇ", ["e\u0301", "a\u0308", "n\u0303"],      # precomposed + combining
    "的一是不了人我在有他这中大来上国个到说们为子和你地出道也时年得就那要下以生会自着去之过家学对可她里后小么心多天而能好都然没日于起还发成事只作当想看文",
    "あいうえおカタカナ漢字한국어Привет мирαβγδ שלום مرحبا नमस्ते",
    ["🙂", "👍🏽", "👩‍💻", "🇺🇸", "🏳️‍🌈", "❤️", "🤖", "🐍"],
    ["'s", "'t", "'re", "'ve", "'m", "'ll", "'d", "'S", "'LL"],
    [" the", " of", " and", " token", "ization", " model", " 1234", " 3.14"],
]


def random_string(rng, max_len=40):
    parts = []
    for _ in range(rng.randint(0, max_len)):
        r = rng.random()
        if r < 0.08:  # any Unicode scalar value (no surrogates: not encodable as UTF-8)
            cp = rng.randint(0, 0x10FFFF)
            parts.append(chr(cp) if not 0xD800 <= cp <= 0xDFFF else "?")
        else:
            parts.append(rng.choice(rng.choice(POOLS)))
    return "".join(parts)


# ----------------------------------------------------------------------------
# The checks
# ----------------------------------------------------------------------------
def check_pretokenization(strings):
    print("Pre-tokenization")
    ok = all("".join(split_text(s, p)) == s for s in strings for p in (GPT2_PATTERN, GPT4_PATTERN))
    report("chunks concatenate back to the text (both patterns)", ok, f"{len(strings)} strings")
    cases = [  # what the real GPT-2 / GPT-4 (cl100k_base) regexes produce
        (GPT2_PATTERN, "Hello world's  123!!", ["Hello", " world", "'s", " ", " 123", "!!"]),
        (GPT4_PATTERN, "Hello world's  123!!", ["Hello", " world", "'s", " ", " ", "123", "!!"]),
        (GPT2_PATTERN, "12345678", ["12345678"]),
        (GPT4_PATTERN, "12345678", ["123", "456", "78"]),
        (GPT2_PATTERN, "WE'LL", ["WE", "'", "LL"]),
        (GPT4_PATTERN, "WE'LL", ["WE", "'LL"]),
        (GPT4_PATTERN, "foo_bar(1)\n    x", ["foo", "_bar", "(", "1", ")\n", "   ", " x"]),
        (GPT2_PATTERN, "héllo 世界🙂", ["héllo", " 世界", "🙂"]),
    ]
    ok = all(split_text(s, p) == want for p, s, want in cases)
    report("known GPT-2 / GPT-4 splits (numbers, contractions, code, Unicode)", ok,
           f"{len(cases)} cases")


def check_roundtrip(toks, strings, rng):
    print("Encode / decode")
    ok, n_tok = True, 0
    for tok in toks:
        for s in strings:
            ids = tok.encode_ordinary(s)
            n_tok += len(ids)
            ok &= tok.decode(ids) == s and tok.decode_bytes(ids) == s.encode("utf-8")
    report("decode(encode(s)) == s on random Unicode strings", ok,
           f"{len(strings)} strings x {len(toks)} tokenizers, {n_tok:,} tokens")
    tok = toks[0]
    ok = True
    for _ in range(300):  # arbitrary ids: partial UTF-8 must decode to U+FFFD, not crash
        ids = [rng.randrange(tok.n_vocab) for _ in range(rng.randint(0, 12))]
        out = tok.decode(ids)
        ok &= isinstance(out, str) and out.encode("utf-8", "surrogatepass") is not None
    lone = tok.decode([0xE4])  # first byte of a 3-byte character only
    report("decode of random id sequences never crashes (invalid UTF-8 -> U+FFFD)",
           ok and lone == "\ufffd", f"decode([0xE4]) = {lone!r}")


def check_encoders(toks, strings, corpus_sample):
    print("Fast encoder vs references")
    ok_naive, ok_replay = True, True
    for tok in toks:
        for s in strings + [corpus_sample]:
            fast = tok.encode_ordinary(s)
            ok_naive &= fast == encode_naive(tok, s)
        for s in strings[:60] + [corpus_sample[:3000]]:
            ok_replay &= tok.encode_ordinary(s) == encode_replay(tok, s)
    report("heap encoder == naive recount-the-pairs encoder (minbpe)", ok_naive,
           f"{len(strings) + 1} texts x {len(toks)} tokenizers")
    report("heap encoder == replaying every merge in training order", ok_replay,
           f"{61} texts x {len(toks)} tokenizers")
    tok = toks[-1]
    tok._cache.clear()
    a = tok.encode_ordinary(corpus_sample)
    b = tok.encode_ordinary(corpus_sample)  # now served from the chunk cache
    report("encoding is deterministic (cold cache == warm cache)", a == b, f"{len(a):,} tokens")


def check_training(small_text, rng):
    print("Fast incremental trainer vs naive trainer")
    corpora = [
        ("rnn/input.txt + 6 KB of README", small_text, 256 + 300, GPT4_PATTERN),
        ("same, GPT-2 pattern", small_text, 256 + 300, GPT2_PATTERN),
        ("ties + overlapping pairs ('aaaa', 'abab')",
         "aaaa aaa abab ab ab ba ba bb aaaaaaa abababab xyz xyz", 256 + 20, GPT4_PATTERN),
        ("random Unicode strings", " ".join(random_string(rng, 30) for _ in range(150)),
         256 + 150, GPT4_PATTERN),
    ]
    t_fast = t_naive = 0.0
    for name, text, V, pat in corpora:
        t0 = time.time()
        fast = train_bpe(text, V, pat)
        t1 = time.time()
        naive = train_naive(text, V, pat)
        t_fast, t_naive = t_fast + t1 - t0, t_naive + time.time() - t1
        report(f"same merges: {name}", fast == naive, f"{len(fast)} merges")
    print(f"  (time: incremental {t_fast:.2f} s, naive {t_naive:.2f} s)")

    a, b = train_bpe(small_text, 600, GPT4_PATTERN), train_bpe(small_text, 600, GPT4_PATTERN)
    report("training is deterministic (same text twice -> same merges)", a == b,
           f"{len(a)} merges")


def check_truncation(text):
    big = Tokenizer(train_bpe(text, 256 + 400, GPT4_PATTERN))
    small = Tokenizer(train_bpe(text, 256 + 150, GPT4_PATTERN))
    t = big.truncated(256 + 150)
    report("first k merges of a big tokenizer == training with V = 256 + k",
           t.merges == small.merges and t.encode_ordinary(text) == small.encode_ordinary(text),
           "k = 150 of 400")


def check_special(tok, strings, rng):
    print("Special tokens")
    specials = list(tok.special_tokens)
    ok, n_special = True, 0
    for s in strings:
        pieces, want = [], []
        for part in [s[:len(s) // 2], s[len(s) // 2:], random_string(rng, 8)]:
            sp = rng.choice(specials)
            pieces += [part, sp]
            want += [part, sp]
        text = "".join(pieces)
        ids = tok.encode(text, allowed_special="all")
        ref = []
        for w in want:
            ref += [tok.special_tokens[w]] if w in tok.special_tokens else tok.encode_ordinary(w)
        n_special += sum(i in tok.inverse_special for i in ids)
        ok &= ids == ref and tok.decode(ids) == text
    report("allowed specials are one token each, text around them unchanged", ok,
           f"{n_special} special tokens")
    text = "a<|endoftext|>b <|im_start|>user"
    plain = tok.encode(text, allowed_special="none")
    ok = (not any(i in tok.inverse_special for i in plain) and tok.decode(plain) == text
          and len(plain) > 2)
    report("allowed_special='none': specials encoded as ordinary text", ok,
           f"'<|endoftext|>' -> {len(tok.encode_ordinary('<|endoftext|>'))} tokens")
    try:
        tok.encode(text)
        raised = False
    except ValueError:
        raised = True
    only_eot = tok.encode(text, allowed_special={"<|endoftext|>"})
    ok = raised and only_eot.count(tok.special_tokens["<|endoftext|>"]) == 1 and \
        tok.special_tokens["<|im_start|>"] not in only_eot
    report("default refuses specials in text; a subset can be allowed", ok, "")
    near = "<|endoftext| <|endoftext <|im_end>"
    ok = not any(i in tok.inverse_special for i in tok.encode(near, allowed_special="all"))
    report("near-misses ('<|endoftext|' ...) never become special tokens", ok, "")


def check_vocab(tok, text):
    print("Vocabulary bookkeeping and save / load")
    n_merges = len(tok.merges)
    ok = (tok.n_vocab == 256 + n_merges + len(tok.special_tokens)
          and sorted(tok.merges.values()) == list(range(256, 256 + n_merges))
          and sorted(tok.special_tokens.values()) == list(range(256 + n_merges, tok.n_vocab))
          and all(tok.vocab[i] == tok.vocab[a] + tok.vocab[b] for (a, b), i in tok.merges.items())
          and all(a < i and b < i for (a, b), i in tok.merges.items()))
    report("n_vocab = 256 + merges + specials; ids contiguous; parts before wholes", ok,
           f"{tok.n_vocab} = 256 + {n_merges} + {len(tok.special_tokens)}")
    ids = tok.encode_ordinary(text)
    report("every encoded id is a valid ordinary token", all(0 <= i < 256 + n_merges for i in ids),
           f"max id {max(ids)}")
    path = os.path.join(HERE, "_check_tmp.json")
    try:
        tok.save(path)
        back = Tokenizer.load(path)
    finally:
        if os.path.exists(path):
            os.remove(path)
    ok = (back.merges == tok.merges and back.special_tokens == tok.special_tokens
          and back.pattern == tok.pattern and back.encode_ordinary(text) == ids)
    report("JSON save -> load gives the same tokenizer", ok, "")


def check_tiktoken(ascii_texts):
    print("Optional: compare with tiktoken's GPT-2")
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")  # downloads the merges on first use
        ranks = enc._mergeable_ranks
    except Exception as e:  # not installed, offline, or API changed: this check is optional
        print(f"  tiktoken GPT-2 comparison: SKIP ({type(e).__name__}: {str(e)[:60]})")
        return
    byte_id = [ranks[bytes([b])] for b in range(256)]  # GPT-2 permutes the 256 byte ids
    merges = []
    for token, rank in sorted(ranks.items(), key=lambda kv: kv[1]):
        if len(token) == 1:
            continue
        parts = [bytes([b]) for b in token]  # recover (left, right): re-run BPE below `rank`
        while len(parts) > 2:
            best = min((ranks.get(parts[i] + parts[i + 1], rank), i) for i in range(len(parts) - 1))
            if best[0] >= rank:
                break
            i = best[1]
            parts[i:i + 2] = [parts[i] + parts[i + 1]]
        merges.append((ranks[parts[0]], ranks[parts[1]]))
    ours = Tokenizer(merges, GPT2_PATTERN, special_tokens=[])
    ok = True
    for text in ascii_texts:  # ASCII: our stdlib regex == GPT-2's \p{L} regex here
        mine = []
        for chunk in split_text(text, GPT2_PATTERN):
            mine += ours._encode_chunk(tuple(byte_id[b] for b in chunk.encode("utf-8")))
        ok &= mine == enc.encode_ordinary(text)
    report("our encoder + GPT-2 merges == tiktoken gpt2", ok, f"{len(ascii_texts)} texts")


def main():
    t0 = time.time()
    rng = random.Random(0)
    strings = [random_string(rng) for _ in range(250)]
    train = load_corpus("train")
    val = load_corpus("val")
    toks = [Tokenizer(train_bpe(train, 1024, PATTERNS[p])) for p in ("gpt4", "gpt2")]
    toks.append(Tokenizer(train_bpe(train, 4096, GPT4_PATTERN)))
    with open(os.path.join(HERE, "..", "rnn", "input.txt"), encoding="utf-8") as f:
        story = f.read()
    small_text = story + train[20000:26000]

    check_pretokenization(strings)
    check_roundtrip(toks, strings, rng)
    check_encoders(toks[:2], strings, val[:6000])
    check_training(small_text, rng)
    check_truncation(small_text)
    check_special(toks[2], strings[:100], rng)
    check_vocab(toks[2], val)
    ascii_texts = [story] + ["".join(c for c in s if ord(c) < 128) for s in strings[:100]]
    check_tiktoken(ascii_texts)

    ok = all(RESULTS)
    print(f"\n{sum(RESULTS)}/{len(RESULTS)} checks OK in {time.time() - t0:.1f} s")
    print("BPE check", "PASSED" if ok else "FAILED")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
