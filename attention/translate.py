"""
Type a date in (almost) any format, get it back as YYYY-MM-DD, and SEE where the
model looked: an attention heatmap with one row per output char and one column
per input char. Darker = more attention weight (each row sums to 1).

Supported input formats (lowercase is added for you):
    march 3, 2021        3 mar 21           03/03/2021 (month/day/year)
    wednesday 3 march 2021                  3rd of march 2021
    wed, mar 3, 21       03-mar-2021
Years 1950..2049 (2-digit years: 50..99 -> 19xx, 00..49 -> 20xx).
Formats it never saw (e.g. "2021/03/03") show you what a model does off its
training distribution: it still answers, often wrongly, and the heatmap shows why.

Loads attention/date_model.pt (train it with `python attention/seq2seq_pytorch.py`);
if it doesn't exist yet, trains and saves one first (~15 s).

Run:
    python attention/translate.py                                (interactive)
    python attention/translate.py --date "wednesday 3 march 2021"
"""
import argparse
import sys

import numpy as np

from seq2seq_pytorch import load_or_train_date_model, translate_date
from tasks import clean_date_text

SHADES = " ·░▒▓█"  # weight 0 ... 1
TTY = sys.stdout.isatty()


def color(code, s):
    return f"\033[{code}m{s}\033[0m" if TTY else s


def shade(w):
    return SHADES[min(len(SHADES) - 1, int(np.ceil(w * (len(SHADES) - 1) - 1e-9)))]


def heatmap_lines(src, out, alpha, top=True):
    """ASCII heatmap: rows = output chars (+ <eos>), columns = input chars."""
    lines = ["          " + " ".join(c if c != " " else "␣" for c in src)]
    for t, row in enumerate(alpha):
        o = out[t] if t < len(out) else "<eos>"
        cells = " ".join(shade(w) for w in row)
        note = ""
        if top:
            j = int(row.argmax())
            note = f"  max {row[j]:.2f} on {src[j]!r} (#{j})"
        lines.append(f"  {o:>5}   {cells}{note}")
    return lines


def show(model, text):
    clean = clean_date_text(text)
    if clean != text:
        print(f"(read as {clean!r})")
    if not clean:
        print("(nothing left to translate)")
        return
    out, alpha = translate_date(model, clean)
    print(f"\n  {clean}  ->  {color('32;1', out)}\n")
    for line in heatmap_lines(clean, out, alpha):
        print(line)
    print(f"\n  shading: '{SHADES[1]}' < 0.2 < '{SHADES[2]}' < 0.4 < '{SHADES[3]}' < 0.6 < "
          f"'{SHADES[4]}' < 0.8 < '{SHADES[5]}'   (each row sums to 1)\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", help="one date to translate; omit for interactive mode")
    args = ap.parse_args()

    model = load_or_train_date_model()
    if args.date is not None:
        show(model, args.date)
        return

    print("Type a date and press Enter (e.g. 'wednesday 3 march 2021', '3 mar 21', "
          "'03/03/2021'). Empty line or Ctrl+C to quit.\n")
    while True:
        try:
            text = input("date> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text.strip():
            break
        show(model, text)


if __name__ == "__main__":
    main()
