"""
The two toy seq2seq tasks used in this folder. All data is generated in code.

1. DATES (character level): many human formats -> one machine format.
       "march 3, 2021"           -> "2021-03-03"
       "3 mar 21"                -> "2021-03-03"
       "03/03/2021"              -> "2021-03-03"   (US order: month/day/year)
       "wednesday 3 march 2021"  -> "2021-03-03"
       "3rd of march 2021"       -> "2021-03-03"
       "wed, mar 3, 21"          -> "2021-03-03"
       "03-mar-2021"             -> "2021-03-03"
   Years are 1950..2049, so a 2-digit year is unambiguous: 50..99 -> 19xx,
   00..49 -> 20xx. Every 10th day is held out of training (test set only),
   so test accuracy measures generalization to dates never seen.

2. REVERSE (the bottleneck experiment): a digit string -> the same digits
   reversed, e.g. "48213" -> "31284". Lengths vary (5..40), which is what
   makes the fixed-size bottleneck vector hurt.

Both tasks share one tiny vocabulary class with three special tokens:
    <pad> = 0 (fills short sequences up to the batch length)
    <sos> = 1 (start of sequence: the decoder's first input)
    <eos> = 2 (end of sequence: the decoder says "I'm done")

Run (prints some examples):
    python attention/tasks.py
"""
import datetime as dt

import numpy as np

PAD, SOS, EOS = 0, 1, 2
SPECIALS = ["<pad>", "<sos>", "<eos>"]

MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august",
          "september", "october", "november", "december"]
WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
FIRST = dt.date(1950, 1, 1)
LAST = dt.date(2049, 12, 31)
N_DAYS = (LAST - FIRST).days + 1


class Vocab:
    """Characters <-> integer ids. Ids 0, 1, 2 are <pad>, <sos>, <eos>."""

    def __init__(self, chars):
        self.itos = SPECIALS + list(chars)
        self.stoi = {c: i for i, c in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, ids):
        """ids -> string, stopping at <eos> and skipping <pad>/<sos>."""
        out = []
        for i in ids:
            i = int(i)
            if i == EOS:
                break
            if i > EOS:
                out.append(self.itos[i])
        return "".join(out)


# ---------------------------------------------------------------- dates

DATE_VOCAB = Vocab(sorted(set("abcdefghijklmnopqrstuvwxyz0123456789 ,/-")))


def ordinal(d):
    if d in (1, 21, 31):
        return f"{d}st"
    if d in (2, 22):
        return f"{d}nd"
    if d in (3, 23):
        return f"{d}rd"
    return f"{d}th"


DATE_FORMATS = [
    lambda x: f"{MONTHS[x.month - 1]} {x.day}, {x.year}",
    lambda x: f"{x.day} {MONTHS[x.month - 1][:3]} {x.year % 100:02d}",
    lambda x: f"{x.month:02d}/{x.day:02d}/{x.year}",
    lambda x: f"{WEEKDAYS[x.weekday()]} {x.day} {MONTHS[x.month - 1]} {x.year}",
    lambda x: f"{ordinal(x.day)} of {MONTHS[x.month - 1]} {x.year}",
    lambda x: f"{WEEKDAYS[x.weekday()][:3]}, {MONTHS[x.month - 1][:3]} {x.day}, {x.year % 100:02d}",
    lambda x: f"{x.day:02d}-{MONTHS[x.month - 1][:3]}-{x.year}",
]


def format_date(date, fmt):
    return DATE_FORMATS[fmt](date)


def target_date(date):
    return date.isoformat()  # "YYYY-MM-DD"


def is_test_day(k):
    """Every 10th day (counted from 1950-01-01) is only ever used for testing."""
    return k % 10 == 0


def sample_dates(n, rng, split="train"):
    """n random (input string, target string) pairs from the given split."""
    pairs = []
    while len(pairs) < n:
        k = int(rng.integers(N_DAYS))
        if is_test_day(k) != (split == "test"):
            continue
        date = FIRST + dt.timedelta(days=k)
        fmt = int(rng.integers(len(DATE_FORMATS)))
        pairs.append((format_date(date, fmt), target_date(date)))
    return pairs


def clean_date_text(s):
    """What translate.py does to user input: lowercase, single spaces, known chars only."""
    s = " ".join(s.lower().split())
    return "".join(c for c in s if c in DATE_VOCAB.stoi)


# ---------------------------------------------------------------- reverse

DIGIT_VOCAB = Vocab("0123456789")


def sample_reverse(n, rng, min_len=5, max_len=40, length=None):
    """n random digit strings (length uniform in [min_len, max_len], or fixed) and their reverse."""
    pairs = []
    for _ in range(n):
        L = length if length is not None else int(rng.integers(min_len, max_len + 1))
        s = "".join(str(d) for d in rng.integers(0, 10, L))
        pairs.append((s, s[::-1]))
    return pairs


# ---------------------------------------------------------------- batching

def to_arrays(pairs, vocab):
    """
    Pad a list of (src, tgt) strings into integer arrays.
        src     (B, S)     input ids, padded with <pad>
        src_len (B,)       real input lengths (for the padding mask)
        dec_in  (B, T)     <sos> y_1 ... y_{T-1}   (teacher forcing: the TRUE previous char)
        dec_out (B, T)     y_1 ... y_{T-1} <eos>   (what the decoder must predict)
    """
    B = len(pairs)
    S = max(len(s) for s, _ in pairs)
    T = max(len(t) for _, t in pairs) + 1
    src = np.full((B, S), PAD, dtype=np.int64)
    dec_in = np.full((B, T), PAD, dtype=np.int64)
    dec_out = np.full((B, T), PAD, dtype=np.int64)
    src_len = np.zeros(B, dtype=np.int64)
    for i, (s, t) in enumerate(pairs):
        src[i, :len(s)] = vocab.encode(s)
        src_len[i] = len(s)
        ids = vocab.encode(t)
        dec_in[i, :len(ids) + 1] = [SOS] + ids
        dec_out[i, :len(ids) + 1] = ids + [EOS]
    return src, src_len, dec_in, dec_out


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    print(f"date vocab ({len(DATE_VOCAB)}): {''.join(DATE_VOCAB.itos[3:])!r}\n")
    for s, t in sample_dates(10, rng):
        print(f"  {s:<28} -> {t}")
    print(f"\ndigit vocab ({len(DIGIT_VOCAB)})\n")
    for s, t in sample_reverse(4, rng):
        print(f"  {s:<40} -> {t}")
