"""
Why tokenization matters: the trade-offs and failure modes of BPE, measured.

1. UNITS. Words vs characters vs raw bytes vs BPE at several vocabulary sizes, on the
   same corpus (train = first 90%, held-out val = last 10%):
     - compression: bytes per token (more = shorter sequences = cheaper attention,
       more text in a fixed context window),
     - open vocabulary: how much held-out text a closed vocabulary can't represent
       (words and characters need <unk>; bytes and byte-level BPE never do),
     - cost: the embedding table + output head grow like 2·V·d parameters,
     - data per token: how many training occurrences each vocabulary entry gets,
     - and what the pre-tokenization regex buys (the same BPE trained on whole lines).
2. ZIPF. Token frequencies follow a power law: a few tokens are everywhere, most are
   rare, and some never appear as themselves at all (they only exist as stepping
   stones to longer merges). Their embeddings barely get trained: "glitch tokens".
3. FAILURE MODES, shown with our own tokenizer:
     - counting letters (the model sees ids, not characters),
     - numbers split inconsistently (GPT-2 regex) vs 3-digit groups (GPT-4 regex),
     - " the" vs "the" vs " The": different tokens for the same word,
     - other languages and emoji cost several times more tokens than English,
     - a prompt with a trailing space ends in a token that almost never precedes a word.

make_figures.py imports these functions to draw compression.png, zipf.png and
failure_modes.png from the same numbers.

Run (~4 s):
    python tokenizer/why_tokenization.py
"""
import re
import warnings
from collections import Counter

import numpy as np

from bpe import GPT2_PATTERN, Tokenizer, get_tokenizer, load_corpus, show, split_text, train_bpe

warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)

VOCABS = [256, 512, 1024, 2048, 4096]
SENTENCES = [  # the same sentence; translations kept simple and literal
    ("English", "The weather is nice today, so I am going for a walk in the park."),
    ("French", "Il fait beau aujourd'hui, alors je vais me promener dans le parc."),
    ("Spanish", "Hoy hace buen tiempo, así que voy a dar un paseo por el parque."),
    ("German", "Das Wetter ist heute schön, also gehe ich im Park spazieren."),
    ("Russian", "Сегодня хорошая погода, поэтому я иду гулять в парк."),
    ("Arabic", "الطقس جميل اليوم، لذلك سأذهب للتنزه في الحديقة."),
    ("Hindi", "आज मौसम अच्छा है, इसलिए मैं पार्क में टहलने जा रहा हूँ।"),
    ("Chinese", "今天天气很好，所以我要去公园散步。"),
    ("Japanese", "今日は天気がいいので、公園に散歩に行きます。"),
    ("Korean", "오늘 날씨가 좋아서 공원에 산책하러 갑니다."),
    ("Python *", "if weather.is_nice(today):\n    me.walk(park)"),
    ("Emoji *", "☀️😎 → 🚶‍♂️🌳"),
]


def compression_table(tok_full, train, val, d_small=64, d_gpt2=768):
    """One row per tokenization scheme, with everything section 1 of the README quotes."""
    btr, bva = len(train.encode("utf-8")), len(val.encode("utf-8"))
    rows = []
    # words: the 4,096 most frequent pre-tokenizer chunks of the training text + <unk>
    chunks_tr, chunks_va = split_text(train), split_text(val)
    top = {w for w, _ in Counter(chunks_tr).most_common(4095)}
    unk = sum(len(w.encode()) for w in chunks_va if w not in top)
    rows.append(dict(name="words (4095 + <unk>)", V=4096, tr=len(chunks_tr),
                     va=len(chunks_va), unk_bytes=unk))
    chars = set(train)
    unk = sum(len(c.encode()) for c in val if c not in chars)
    rows.append(dict(name="characters (+ <unk>)", V=len(chars) + 1, tr=len(train), va=len(val),
                     unk_bytes=unk))
    for V in VOCABS + [256 + len(tok_full.merges)]:
        t = tok_full.truncated(V)
        name = "bytes (BPE, 0 merges)" if V == 256 else f"BPE {V}"
        if V == 256 + len(tok_full.merges) and V not in VOCABS:
            name = f"BPE {V} (every chunk merged)"
        rows.append(dict(name=name, V=V, tr=len(t.encode_ordinary(train)),
                         va=len(t.encode_ordinary(val)), unk_bytes=0))
    for r in rows:
        r["bpt_tr"], r["bpt_va"] = btr / r["tr"], bva / r["va"]
        r["unk_frac"] = r["unk_bytes"] / bva
        r["ctx256"] = 256 * r["bpt_va"]
        r["emb_small"], r["emb_gpt2"] = 2 * r["V"] * d_small, 2 * r["V"] * d_gpt2
        r["per_entry"] = r["tr"] / r["V"]
    return rows


LINES_PATTERN = r"[^\n]*\n|[^\n]+"  # "no pre-tokenization": only cut at newlines


def pretokenization_effect(tok, train, val):
    """Train the same V=4096 BPE on whole lines instead of regex chunks."""
    lines = Tokenizer(train_bpe(train, 4096, LINES_PATTERN), LINES_PATTERN)
    spans = lambda t: [i for i in range(256, 4096)  # noqa: E731
                       if re.search(r"\S ", t.vocab[i].decode("utf-8", "replace"))]
    return dict(lines=lines, cross_lines=spans(lines), cross_regex=spans(tok),
                bpt=[(len(s.encode()) / len(t.encode_ordinary(s))) for t in (tok, lines)
                     for s in (train, val)])


def zipf_stats(tok, train):
    """Count how often every token id occurs in the encoded training text."""
    ids = tok.encode_ordinary(train)
    counts = np.bincount(ids, minlength=256 + len(tok.merges))
    learned = counts[256:]
    nz = np.sort(counts[counts > 0])[::-1]
    ranks = np.arange(1, len(nz) + 1)
    sel = (ranks >= 10) & (ranks <= 1000)
    slope = np.polyfit(np.log(ranks[sel]), np.log(nz[sel]), 1)[0]
    zero = [256 + i for i in np.flatnonzero(learned == 0)]
    return dict(ids=ids, counts=counts, sorted=nz, slope=slope, zero=zero,
                le5=int((learned <= 5).sum()), n_learned=len(learned),
                top10=[(int(i), int(counts[i])) for i in np.argsort(-counts)[:10]])


def digit_splits(toks):
    """Encode every 4-digit number with each tokenizer; return the split shapes."""
    out = {}
    for name, tok in toks.items():
        shapes = Counter()
        examples = {}
        for n in range(1000, 10000):
            s = str(n)
            parts = [tok.token_str(i) for i in tok.encode_ordinary(s)]
            shape = "+".join(str(len(p)) for p in parts)
            shapes[shape] += 1
            examples.setdefault(shape, "|".join(parts))
        out[name] = (shapes, examples)
    return out


def language_costs(tok):
    base = None
    rows = []
    for lang, s in SENTENCES:
        n = len(tok.encode_ordinary(s))
        base = base or n
        rows.append(dict(lang=lang, text=s, chars=len(s), bytes=len(s.encode()), tokens=n,
                         ratio=n / base))
    return rows


def trailing_space(tok, train_ids):
    """What follows a lone ' ' token in the training data?"""
    sp = tok.encode_ordinary(" ")[0]
    after = Counter()
    for a, b in zip(train_ids, train_ids[1:]):
        if a == sp:
            first = tok.vocab[b][:1].decode("utf-8", "replace")
            after["letter" if first.isalpha() else "digit" if first.isdigit()
                  else "space/newline" if first.isspace() else "other"] += 1
    return sp, after


def fmt_params(n):
    return f"{n / 1e6:.1f}M" if n >= 1e6 else f"{n / 1e3:.0f}k"


def main():
    train, val = load_corpus("train"), load_corpus("val")
    tok_full = Tokenizer(train_bpe(train, 100_000))  # merge until every chunk is one token
    tok = get_tokenizer(4096)
    assert tok.merges == tok_full.truncated(4096).merges
    print(f"corpus: train {len(train.encode()):,} bytes, held-out val {len(val.encode()):,} bytes "
          f"(GPT-4 regex; merging ran out of pairs after {len(tok_full.merges):,} merges)\n")

    print("1. UNITS: compression, open vocabulary, and the cost of a big vocabulary\n")
    rows = compression_table(tok_full, train, val)
    print(f"  {'scheme':30s}{'V':>6s}{'train tok':>10s}{'bytes/tok':>10s}{'val b/tok':>10s}"
          f"{'val <unk>':>10s}{'256 tok =':>10s}{'2·V·d d=64':>11s}{'d=768':>8s}{'tok/entry':>10s}")
    for r in rows:
        print(f"  {r['name']:30s}{r['V']:6d}{r['tr']:10,d}{r['bpt_tr']:10.2f}{r['bpt_va']:10.2f}"
              f"{r['unk_frac']:10.1%}{r['ctx256']:8.0f} B{fmt_params(r['emb_small']):>11s}"
              f"{fmt_params(r['emb_gpt2']):>8s}{r['per_entry']:10.0f}")
    print("\n  bytes/tok: compression (higher = shorter sequences). val <unk>: share of held-out")
    print("  bytes a closed vocabulary must replace by <unk>. 256 tok =: how many bytes of text")
    print("  a 256-token context window holds. 2·V·d: embedding + output-head parameters")
    print("  (our mini-GPT's 2 blocks at d=64 have 99k; GPT-2 small's 12 blocks have 85M).")
    print("  tok/entry: training tokens per vocabulary entry (how much data each embedding gets).")
    last = rows[-1]
    print(f"  Held-out compression grows slower than train compression ({rows[-2]['bpt_tr']:.2f} vs "
          f"{rows[-2]['bpt_va']:.2f} at V=4096,")
    print(f"  {last['bpt_tr']:.2f} vs {last['bpt_va']:.2f} at V={last['V']}): "
          "late merges memorize rare strings of the training text.")

    print("\n  Why pre-tokenize? The same V=4096 BPE trained on whole lines (no regex):")
    p = pretokenization_effect(tok, train, val)
    print(f"     bytes/token train {p['bpt'][2]:.2f}, val {p['bpt'][3]:.2f}   (GPT-4 regex: "
          f"train {p['bpt'][0]:.2f}, val {p['bpt'][1]:.2f})")
    print(f"     {len(p['cross_lines']):,} of 3,840 learned tokens glue a word to the next one "
          f"(GPT-4 regex: {len(p['cross_regex'])}), e.g. "
          + ", ".join(repr(show(p["lines"].vocab[i])) for i in p["cross_lines"][:8]))
    for w in [" keeper.", " keeper's"]:
        print(f"     {w!r:12s} lines: {'|'.join(p['lines'].token_str(i) for i in p['lines'].encode_ordinary(w)):18s}"
              f" regex: {'|'.join(tok.token_str(i) for i in tok.encode_ordinary(w))}")
    print("     Without the regex, BPE compresses this text better but spends half its vocabulary on")
    print("     word+space+punctuation combinations: 'the␣', '.␣', ',␣' instead of reusable words.")

    print("\n2. ZIPF: a few tokens are everywhere, most are rare\n")
    z = zipf_stats(tok, train)
    print(f"  V = 4096 on the training text: {len(z['ids']):,} tokens. Most frequent: "
          + ", ".join(f"{tok.token_str(i)!r} {c}" for i, c in z["top10"][:8]))
    print(f"  log-log slope of count vs rank (ranks 10-1000): {z['slope']:.2f} "
          "(Zipf's law: about -1)")
    print(f"  {z['le5']:,} of the {z['n_learned']:,} learned tokens occur <= 5 times; "
          f"{len(z['zero'])} occur ZERO times as themselves, e.g. "
          + ", ".join(repr(tok.token_str(i)) for i in z["zero"][:6]))
    print("  A zero-count token was a stepping stone: it existed only to build a longer token")
    print("  that always swallows it. Its input embedding row never gets a gradient (only weight")
    print("  decay), so if a user ever sends it, the model sees an untrained vector ('glitch token').")

    print("\n3. FAILURE MODES\n")
    print("  a) Counting letters: the model sees ids, not characters")
    for w in [" strawberry", " lighthouse", " backpropagation", " Mississippi"]:
        ids = tok.encode_ordinary(w)
        print(f"     {w!r:20s} -> {'|'.join(tok.token_str(i) for i in ids):28s} {ids}")
    print("     'How many r in strawberry?' needs the spelling of each token, which is stored")
    print("     (if at all) somewhere in its embedding, not in the input.")

    print("\n  b) Numbers (all 4-digit numbers 1000-9999, our V=4096 tokenizers)")
    tok2 = Tokenizer(train_bpe(train, 4096, GPT2_PATTERN), GPT2_PATTERN)
    splits = digit_splits({"GPT-2 regex": tok2, "GPT-4 regex": tok})
    for name, (shapes, ex) in splits.items():
        top = ", ".join(f"{k} digits {v}x (e.g. {ex[k]})" for k, v in shapes.most_common(4))
        print(f"     {name}: {len(shapes)} different split shapes. Most common: {top}")
    for n in ["1999", "2023", "2024", "3.14159", "1000000"]:
        print(f"     {n:>8s}: GPT-2 regex {'|'.join(tok2.token_str(i) for i in tok2.encode_ordinary(n)):14s}"
              f" GPT-4 regex {'|'.join(tok.token_str(i) for i in tok.encode_ordinary(n))}")
    print("     Column addition needs aligned digits; with inconsistent chunks the same digit")
    print("     position lands in a different token every time. GPT-4 caps chunks at 3 digits;")
    print("     LLaMA-1/2 split every digit; some models even split right-to-left.")

    print("\n  c) Same word, different tokens")
    variants = ["the", " the", "The", " The", " THE", "the."]
    print("     " + "   ".join(f"{v!r} -> {[tok.token_str(i) for i in tok.encode_ordinary(v)]}"
                              for v in variants))
    words = {tok.vocab[i].decode("utf-8", "replace") for i in range(256, 4096)}
    alpha = {w for w in words if w.isalpha()}
    both = sorted(w for w in alpha if " " + w in words)
    print(f"     {len(both)} words exist as both 'w' and ' w' tokens (e.g. "
          + ", ".join(repr(w) for w in both[:6]) + "): unrelated ids, so the")
    print("     model must learn separately that they mean the same thing.")

    print("\n  d) The same sentence in other languages (English-trained V=4096 tokenizer)")
    print(f"     {'':10s}{'chars':>6s}{'bytes':>6s}{'tokens':>7s}{'vs English':>11s}")
    for r in language_costs(tok):
        print(f"     {r['lang']:10s}{r['chars']:6d}{r['bytes']:6d}{r['tokens']:7d}"
              f"{r['ratio']:10.1f}x")
    print("     (* not translations: code and emoji for comparison.)")
    print("     Same meaning, up to several times the tokens: more cost per request, less text per")
    print("     context window, and fewer training tokens per idea (Petrov et al. 2023).")

    print("\n  e) Trailing whitespace")
    for p in ["The answer is", "The answer is "]:
        print(f"     {p!r:18s} -> {'|'.join(tok.token_str(i) for i in tok.encode_ordinary(p))}")
    sp, after = trailing_space(tok, z["ids"])
    total = sum(after.values())
    print(f"     In training, the lone ' ' token (id {sp}) appears {total:,} times. Next token starts with: "
          + ", ".join(f"{k} {v:,}" for k, v in after.most_common()))
    print("     A letter almost never follows it (the regex glues the space to the word: ' answer'),")
    print("     so a prompt ending in a space asks the model to continue from a state it has")
    print("     barely seen. Strip trailing spaces from prompts (or use 'token healing').")


if __name__ == "__main__":
    main()
