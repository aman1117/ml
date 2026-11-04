"""
Watch byte-level BPE learn and apply merges, step by step, in your terminal.

You will see:
  - text -> Unicode code points -> UTF-8 bytes (1 to 4 bytes per character),
  - the pre-tokenization regex cutting the text into chunks (merges never cross them),
  - the training loop on a tiny corpus, one merge at a time: the table of pair
    counts, the winning pair, the new token, and the sequence shrinking,
  - the learned merge table, and how a NEW word is encoded with it (always apply the
    earliest-learned merge that is present),
  - the real tokenizer (V = 4096, trained on ~170 KB by bpe.py) on a few sentences,
    with coloured token boundaries: English, numbers, code, other languages, emoji.

Uses tokenizer/tokenizer.json if it matches the corpus, otherwise trains it first (<1 s).

Run:
    python tokenizer/bpe_visual.py                 (press Enter for the next step)
    python tokenizer/bpe_visual.py --word " unhatted" --merges 12
    python tokenizer/bpe_visual.py --no-pause      (or with stdin not a terminal)
"""
import argparse
import sys
from collections import Counter

from bpe import GPT4_PATTERN, best_pair, get_stats, get_tokenizer, merge, show, split_text

TTY = sys.stdout.isatty()
PAUSE = True
TOY = "the cat sat on the mat. the hat! that cat ate the rat; the rats sat. cats hate hats."
BG = ["48;5;153", "48;5;223", "48;5;194", "48;5;225", "48;5;230", "48;5;189"]  # pastel cycle


def color(code, s):
    return f"\033[{code}m{s}\033[0m" if TTY else s


def cyan(s): return color("36;1", s)      # bytes / ids
def yellow(s): return color("33;1", s)    # the pair being merged
def green(s): return color("32;1", s)     # new tokens / results
def dim(s): return color("2", s)


def pause():
    if PAUSE and sys.stdin.isatty():
        try:
            input(dim("\n  [Enter] next step "))
        except EOFError:
            print()
    print("\n" + "=" * 78 + "\n")


def boxes(pieces, highlight=None):
    """Tokens with visible boundaries: alternating background colours on a terminal,
    '|' separators otherwise. `highlight` = set of positions to colour as new tokens."""
    if not TTY:
        return "|".join(f"[{p}]" if highlight and k in highlight else p
                        for k, p in enumerate(pieces))
    out = []
    for k, p in enumerate(pieces):
        code = "30;48;5;120;1" if highlight and k in highlight else "30;" + BG[k % len(BG)]
        out.append(f"\033[{code}m{p}\033[0m")
    return "".join(out)


def bar(n, top, width=20):
    k = max(1, round(n / top * width))
    return "█" * k + "·" * (width - k)


def main():
    global PAUSE
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--text", default=TOY, help="the tiny training corpus for steps 2-4")
    ap.add_argument("--merges", type=int, default=10, help="how many merges to learn")
    ap.add_argument("--word", default=" thats", help="a new word to encode in step 5")
    ap.add_argument("--no-pause", action="store_true")
    args = ap.parse_args()
    PAUSE = not args.no_pause

    # ------------------------------------------------------------------ STEP 1
    print("STEP 1: text is Unicode characters; computers store it as UTF-8 BYTES\n")
    print("   Every character has a code point (an integer up to 0x10FFFF). UTF-8 writes it")
    print("   with 1 byte (ASCII), 2 (accents, Greek, Cyrillic), 3 (most CJK) or 4 (emoji).\n")
    for c in "aé世🙂":
        b = c.encode("utf-8")
        print(f"   {c!r:5s} U+{ord(c):05X}  ->  {cyan(' '.join(f'{x:02X}' for x in b)):s}"
              f"   ({len(b)} byte{'s' if len(b) > 1 else ''}: ids {list(b)})")
    print(dim("\n   Byte-level BPE starts from these 256 possible byte values: every string in"))
    print(dim("   every language (and any binary junk) can be encoded, with no <unk> token."))
    pause()

    # ------------------------------------------------------------------ STEP 2
    text = args.text
    chunks = split_text(text, GPT4_PATTERN)
    freq = Counter(chunks)
    print("STEP 2: pre-tokenize with the GPT-4 regex: words (with their leading space),")
    print("        numbers (1-3 digits), punctuation runs, whitespace\n")
    print(f"   {text!r}\n")
    print("   " + boxes([show(c.encode()) for c in chunks]))
    print(f"\n   {len(chunks)} chunks, {len(freq)} unique. Count each unique chunk once:\n")
    print("   " + "  ".join(f"{show(c.encode())!r}×{n}" for c, n in freq.most_common()))
    print(dim("\n   Merges happen INSIDE chunks only, so 'cat.' can never become one token:"))
    print(dim("   ' cat' is learned once and reused before '.', '!', ' sat', ..."))
    pause()

    # ------------------------------------------------------------------ STEP 3
    words = [list(c.encode("utf-8")) for c in freq]
    weights = list(freq.values())
    vocab = {i: bytes([i]) for i in range(256)}
    n_tokens = sum(len(w) * f for w, f in zip(words, weights))
    print("STEP 3: start from bytes. The whole text is a sequence of byte ids:\n")
    seq = [show(vocab[i]) for c in chunks for i in c.encode()]
    print("   " + boxes(seq))
    print(f"\n   {cyan(str(n_tokens))} tokens (= {len(text.encode())} bytes). "
          "Vocabulary: 256 byte values. Now repeat:")
    print("     1. count every adjacent pair (inside chunks, × chunk frequency)")
    print("     2. merge the most frequent pair into a NEW token id 256, 257, ...")
    pause()

    merges = []
    for k in range(args.merges):
        stats = get_stats(words, weights)
        if not stats:
            print("   (no pairs left: every chunk is a single token)")
            break
        pair = best_pair(stats)
        idx = 256 + k
        top = sorted(stats.items(), key=lambda kv: (-kv[1], kv[0]))[:6]
        print(f"MERGE {k + 1}: pair counts (top {len(top)} of {len(stats)} distinct pairs)\n")
        for p, n in top:
            label = f"{show(vocab[p[0]])!r} + {show(vocab[p[1]])!r}"
            line = f"   {label:16s} ({p[0]:3d},{p[1]:3d})  {bar(n, top[0][1])} {n:3d}"
            print(yellow(line) if p == pair else line)
        vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
        merges.append(pair)
        words = [merge(w, pair, idx) for w in words]
        new_tokens = [vocab[i] for c in chunks for i in words[list(freq).index(c)]]
        new_len = len(new_tokens)
        hl = {j for j, t in enumerate(new_tokens) if t == vocab[idx]}
        print(f"\n   new token {green(str(idx))} = {show(vocab[pair[0]])!r} + "
              f"{show(vocab[pair[1]])!r} = {green(repr(show(vocab[idx])))}\n")
        print("   " + boxes([show(t) for t in new_tokens], hl))
        print(f"\n   tokens: {n_tokens} -> {green(str(new_len))}   (-{n_tokens - new_len}, "
              f"the count of the pair; vocabulary: {257 + k})")
        n_tokens = new_len
        if k < 4 or k == args.merges - 1:
            pause()
        else:
            print()

    # ------------------------------------------------------------------ STEP 4
    print("STEP 4: the result of training is just this ordered list of merges\n")
    print("   rank  new id   merge                   token")
    for k, (a, b) in enumerate(merges):
        print(f"   {k:4d}  {256 + k:6d}   ({a:3d},{b:3d}) {show(vocab[a])!r:>7s} + "
              f"{show(vocab[b])!r:<7s}  {green(repr(show(vocab[256 + k])))}")
    total = len(text.encode())
    print(f"\n   {total} bytes -> {n_tokens} tokens: {total / n_tokens:.2f} bytes per token. "
          "Real tokenizers learn 50k-200k merges")
    print("   on terabytes of text.")
    print(dim("\n   Note 'the' at the very start is still th|e: without a leading space it is a"))
    print(dim("   different chunk from ' the', so it needs its own merges (and its own token)."))
    pause()

    # ------------------------------------------------------------------ STEP 5
    word = args.word
    rank = {p: 256 + k for k, p in enumerate(merges)}
    print(f"STEP 5: encode a NEW chunk, {word!r}, with the learned merges\n")
    print("   Rule (tiktoken, minbpe): among the adjacent pairs present, apply the merge with")
    print("   the LOWEST rank (learned first), because that's the order training applied them.\n")
    ids = list(word.encode("utf-8"))
    print(f"   bytes:  {boxes([show(vocab[i]) for i in ids])}   {cyan(str(ids))}")
    while len(ids) >= 2:
        present = [(rank[p], p) for p in zip(ids, ids[1:]) if p in rank]
        if not present:
            break
        r, p = min(present)
        options = ", ".join(f"{show(vocab[q[0]] + vocab[q[1]])!r}:{rq - 256}"
                            for rq, q in sorted(set(present)))
        ids = merge(ids, p, r)
        hl = {j for j, i in enumerate(ids) if i == r}
        print(f"   rank {r - 256:2d}: {boxes([show(vocab[i]) for i in ids], hl)}   "
              f"{dim('(present merges, by rank: ' + options + ')')}")
    print(f"\n   {word!r} -> {len(ids)} tokens {green(str(ids))} = "
          f"{[show(vocab[i]) for i in ids]}")
    print(dim("   It never appeared in the training text, yet it is encoded from known pieces."))
    print(dim("   Worst case it falls back to single bytes, so nothing is ever 'unknown'."))
    pause()

    # ------------------------------------------------------------------ STEP 6
    tok = get_tokenizer(4096)
    print(f"STEP 6: the real tokenizer from bpe.py: {len(tok.merges)} merges learned from "
          f"{tok.meta.get('train_chars', 0):,} characters of")
    print("        ../rnn/input.txt and the earlier modules' READMEs (English Markdown)\n")
    samples = ["The keeper lit the lamp and watched the ships.",
               "Backpropagation computes the gradient of the loss.",
               "Tokenization? Untokenizable!",
               "12345 + 678 = 13023",
               "def forward(self, x):\n    return x @ self.W",
               "Le gardien alluma la lampe.",
               "灯塔看守人点亮了灯。",
               "🙂👍🏽"]
    for s in samples:
        ids = tok.encode(s)
        pieces = [tok.token_str(i) for i in ids]
        print(f"   {len(ids):3d} tokens / {len(s.encode()):3d} bytes  {boxes(pieces)}")
    print(dim("\n   Frequent English words are one token (with their leading space). Rare words,"))
    print(dim("   numbers and other languages split into pieces, down to raw bytes (\\xNN = one"))
    print(dim("   byte of a multi-byte character): the same meaning costs more tokens."))
    ids = tok.encode(samples[0])
    print(f"\n   ids {cyan(str(ids))}\n   decode -> {tok.decode(ids)!r}")
    pause()

    print("RECAP\n")
    print("  * Tokens = byte sequences. Start with the 256 bytes, so any text can be encoded.")
    print("  * Training: count adjacent pairs, merge the most frequent, repeat V - 256 times.")
    print("  * The regex splits text first: no token spans a word + punctuation boundary.")
    print("  * Encoding: apply the learned merges in rank order. Decoding: concatenate bytes,")
    print("    then UTF-8 decode (errors='replace' for partial characters).")
    print("  * The model only ever sees the ids: it never sees letters, digits or spaces.")
    print("\nNext: README sections 3-5, then python tokenizer/bpe_check.py and "
          "python tokenizer/why_tokenization.py")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n(stopped)")
