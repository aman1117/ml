"""
Give the trained RNN some starting text (a prompt) and let it continue.

This is a character-level language model: it does NOT answer questions.
It only predicts "what character comes next" in the style of its training
text, so the best prompts look like the beginning of a sentence from it,
e.g. "The keeper" or "Good morning".

How it works:
  1. Feed the prompt one character at a time to build up the hidden state h
     (the model "reads" your text).
  2. Then repeatedly: predict the next-char probabilities, sample one char,
     and feed it back in as the next input.

Run (after training with rnn_numpy.py):
    python rnn/generate.py                       # interactive: type prompts
    python rnn/generate.py --prompt "The girl"   # one-shot
    python rnn/generate.py --prompt "The sea" --temperature 0.5 --length 500
"""
import argparse
import os

import numpy as np

from rnn_numpy import load_model


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
    P = model.params
    h = np.zeros((model.H, 1))
    for ix in ixs[:-1]:
        x = np.zeros((model.V, 1))
        x[ix] = 1.0
        h = np.tanh(P["W_xh"] @ x + P["W_hh"] @ h + P["b_h"])

    out = model.sample(h, ixs[-1], length, temperature)
    return "".join(chars[i] for i in out)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=os.path.join(here, "model.npz"))
    ap.add_argument("--prompt", help="starting text; omit for interactive mode")
    ap.add_argument("--length", type=int, default=300, help="characters to generate")
    ap.add_argument("--temperature", type=float, default=0.8,
                    help="<1 = safer/repetitive, >1 = more random")
    ap.add_argument("--seed", type=int, default=None, help="for reproducible output")
    args = ap.parse_args()

    if not os.path.exists(args.model):
        raise SystemExit(f"No model at {args.model}. Train first: python rnn/rnn_numpy.py")
    model, chars = load_model(args.model)
    model.rng = np.random.default_rng(args.seed)

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
