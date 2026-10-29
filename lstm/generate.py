"""
Give the trained LSTM (or GRU) some starting text (a prompt) and let it continue.

This is a character-level language model: it does NOT answer questions.
It only predicts "what character comes next" in the style of its training
text (../rnn/input.txt), so the best prompts look like the beginning of a
sentence from it, e.g. "The keeper" or "Good morning".

How it works (same as ../rnn/generate.py, but the memory is (h, c) for an LSTM):
  1. Feed the prompt one character at a time to build up the state
     (the model "reads" your text).
  2. Then repeatedly: predict the next-char probabilities, sample one char,
     and feed it back in as the next input.

Run (after training with lstm_numpy.py):
    python lstm/generate.py                        # interactive: type prompts
    python lstm/generate.py --prompt "The girl"    # one-shot
    python lstm/generate.py --cell gru --prompt "The sea" --temperature 0.5 --length 500
"""
import argparse
import os

import numpy as np

from lstm_numpy import load_model, sample

HERE = os.path.dirname(os.path.abspath(__file__))


def generate(model, chars, prompt, length=300, temperature=0.8):
    stoi = {c: i for i, c in enumerate(chars)}
    unknown = sorted({c for c in prompt if c not in stoi})
    if unknown:
        print(f"(skipping characters never seen in training: {unknown})")
    ixs = [stoi[c] for c in prompt if c in stoi]
    if not ixs:
        raise ValueError("the prompt has no characters the model knows")

    # Read all but the last prompt char to build up memory. sample() then
    # takes the last char as its first input and starts predicting.
    state = model.init_state()
    for ix in ixs[:-1]:
        state, _ = model.step(ix, state)

    out = sample(model, state, ixs[-1], length, temperature)
    return "".join(chars[i] for i in out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cell", choices=["lstm", "gru"], default="lstm",
                    help="which saved model to load (model.npz or model_gru.npz)")
    ap.add_argument("--model", default=None, help="path to a saved model (overrides --cell)")
    ap.add_argument("--prompt", help="starting text; omit for interactive mode")
    ap.add_argument("--length", type=int, default=300, help="characters to generate")
    ap.add_argument("--temperature", type=float, default=0.8,
                    help="<1 = safer/repetitive, >1 = more random")
    ap.add_argument("--seed", type=int, default=None, help="for reproducible output")
    args = ap.parse_args()

    path = args.model or os.path.join(HERE, "model.npz" if args.cell == "lstm"
                                      else "model_gru.npz")
    if not os.path.exists(path):
        flag = "" if args.cell == "lstm" else " --cell gru"
        raise SystemExit(f"No model at {path}. Train first: python lstm/lstm_numpy.py{flag}")
    model, chars = load_model(path)
    model.rng = np.random.default_rng(args.seed)
    print(f"(loaded a {model.cell.upper()} with H = {model.H} from {os.path.relpath(path)})")

    if args.prompt is not None:
        print(args.prompt + generate(model, chars, args.prompt, args.length, args.temperature))
        return

    print("Type the start of some text and press Enter. Empty line or Ctrl+C to quit.\n")
    while True:
        try:
            prompt = input("prompt> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not prompt:
            break
        try:
            print(prompt + generate(model, chars, prompt, args.length, args.temperature))
        except ValueError as e:
            print(f"({e})")
        print()


if __name__ == "__main__":
    main()
