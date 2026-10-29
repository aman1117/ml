"""
THE key insight of this folder: an LSTM/GRU can remember something for a long
time; a vanilla RNN cannot. Plus a surprise about initialization.

Task ("remember the first token"):
    input : A  x q t q w e ... r  ?        first token = one of K=4 signals (A B C D),
            ^  └── D random distractors ──┘  then D random distractor tokens, then "?"
    target: A                              at "?" the model must output the FIRST token.

Chance accuracy is 1/K = 25%. The only way to beat it is to carry the first
token across D steps, so the learning signal must also travel D steps back:

    vanilla RNN :  dL/dh_k = product of D Jacobians diag(tanh')·W_hhᵀ   -> vanishes
    LSTM        :  dL/dc_k ≈ dL/dc_T ⊙ f_T ⊙ ... ⊙ f_{k+1}               -> survives if f ≈ 1
    GRU         :  dL/dh_k gets a direct term dL/dh_T ⊙ z_T ⊙ ... ⊙ z_{k+1}

We train 4 models for every delay D and print a table of test accuracy:
    RNN               vanilla tanh RNN
    LSTM (b_f=1)      LSTM with the standard forget-gate bias of 1 (f starts ≈ 0.73)
    LSTM (chrono)     LSTM with "chrono" init (Tallec & Ollivier 2018):
                      b_f = log(U(1, T)), b_i = -b_f, so f starts between 0.5 and 1 - 1/T
    GRU (chrono)      GRU with the same idea for its keep gate: b_z = log(U(1, T))
Then we measure how much gradient reaches each earlier step (||dL/dh_k|| for
the RNN, ||dL/dc_k|| for the LSTM) relative to the last step.

PyTorch (CPU) is used for speed. Results are saved to lstm/long_memory.npz
for make_figures.py.

Run (about 45 s):
    python lstm/long_memory.py
    python lstm/long_memory.py --delays 5 10 20 40 80 160 --seeds 0
"""
import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
K, M = 4, 8                 # number of signal symbols, number of distractor symbols
VOCAB = K + M + 1           # + the query token "?"
QUERY = VOCAB - 1
MODELS = ["RNN", "LSTM (b_f=1)", "LSTM (chrono)", "GRU (chrono)"]
torch.set_num_threads(1)    # tiny matrices: threads only add overhead


def make_batch(B, D, gen):
    """B sequences [signal, D distractors, ?] (B, D+2) and their targets (B,)."""
    signal = torch.randint(0, K, (B,), generator=gen)
    distract = torch.randint(K, K + M, (B, D), generator=gen)
    query = torch.full((B, 1), QUERY)
    return torch.cat([signal[:, None], distract, query], dim=1), signal


class Recall(nn.Module):
    """Embedding -> one recurrent layer -> Linear on the LAST hidden state."""

    def __init__(self, kind, T, H=32, E=16):
        super().__init__()
        self.kind = kind
        self.embed = nn.Embedding(VOCAB, E)
        if kind == "RNN":
            self.rnn = nn.RNN(E, H, batch_first=True)
        elif kind.startswith("LSTM"):
            self.rnn = nn.LSTM(E, H, batch_first=True)
        else:
            self.rnn = nn.GRU(E, H, batch_first=True)
        self.head = nn.Linear(H, K)

        # PyTorch adds bias_ih + bias_hh, so we put the whole bias in bias_ih.
        with torch.no_grad():
            b_ih, b_hh = self.rnn.bias_ih_l0, self.rnn.bias_hh_l0
            if kind == "LSTM (b_f=1)":          # gate order i, f, g, o
                b_ih[H:2 * H], b_hh[H:2 * H] = 1.0, 0.0
            elif kind == "LSTM (chrono)":
                b = torch.log(torch.empty(H).uniform_(1, T - 1))
                b_ih[H:2 * H], b_hh[H:2 * H] = b, 0.0      # forget: keep for ~e^b steps
                b_ih[0:H], b_hh[0:H] = -b, 0.0             # input: write little
            elif kind == "GRU (chrono)":         # gate order r, z, n; h = (1-z)n + z h
                b = torch.log(torch.empty(H).uniform_(1, T - 1))
                b_ih[H:2 * H], b_hh[H:2 * H] = b, 0.0      # z ≈ 1: keep the old h

    def forward(self, x):
        out, _ = self.rnn(self.embed(x))   # out: (B, T, H)
        return self.head(out[:, -1])       # logits from the last step: (B, K)


def train_one(kind, D, seed, steps=800, batch=64, lr=1e-2):
    """Train on fresh random batches; stop early once the task is solved."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    model = Recall(kind, T=D + 2)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    recent = []
    for step in range(1, steps + 1):
        x, y = make_batch(batch, D, gen)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        recent = (recent + [(logits.argmax(1) == y).float().mean().item()])[-50:]
        if len(recent) == 50 and min(recent) == 1.0:
            break  # 50 batches in a row all correct: solved
    return model, step


@torch.no_grad()
def accuracy(model, D, n=2000):
    x, y = make_batch(n, D, torch.Generator().manual_seed(12345))  # same test set for all
    return (model(x).argmax(1) == y).float().mean().item()


def gradient_flow(model, D, n=256):
    """
    ||dL/dstate_k|| / ||dL/dstate_last|| for every step k, where L is the loss
    at the final "?" and state = h for the RNN/GRU, c for the LSTM. Computed
    with an explicit loop over PyTorch's own weights so we can grab every state.
    """
    x, y = make_batch(n, D, torch.Generator().manual_seed(777))
    emb = model.embed(x).detach()
    r = model.rnn
    W_ih, W_hh, b = r.weight_ih_l0, r.weight_hh_l0, r.bias_ih_l0 + r.bias_hh_l0
    H = W_hh.shape[1]
    h = torch.zeros(n, H)
    c = torch.zeros(n, H)
    states = []
    for t in range(x.shape[1]):
        if model.kind == "RNN":
            h = torch.tanh(emb[:, t] @ W_ih.T + h @ W_hh.T + b)
            s = h
        elif model.kind.startswith("LSTM"):
            i, f, g, o = (emb[:, t] @ W_ih.T + h @ W_hh.T + b).chunk(4, dim=1)
            c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
            h = torch.sigmoid(o) * torch.tanh(c)
            s = c
        else:
            gx = emb[:, t] @ W_ih.T + r.bias_ih_l0
            gh = h @ W_hh.T + r.bias_hh_l0
            rr, z = torch.sigmoid(gx[:, :2 * H] + gh[:, :2 * H]).chunk(2, dim=1)
            nn_ = torch.tanh(gx[:, 2 * H:] + rr * gh[:, 2 * H:])
            h = (1 - z) * nn_ + z * h
            s = h
        s.retain_grad()
        states.append(s)
    loss = F.cross_entropy(model.head(h), y, reduction="sum")
    loss.backward()
    norms = np.array([s.grad.norm(dim=1).mean().item() for s in states])
    model.zero_grad()
    return norms[::-1] / norms[-1]  # index = distance back from the last step


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--delays", type=int, nargs="+", default=[5, 10, 20, 40, 80])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=800, help="max training steps per run")
    ap.add_argument("--save", default=os.path.join(HERE, "long_memory.npz"))
    args = ap.parse_args()
    t0 = time.time()

    print(f"Task: first token is one of {K} signals, then D random distractors "
          f"(from {M} symbols), then '?'.")
    print(f"Answer the first token. Chance = {1 / K:.0%}. Each model: H=32, Adam, "
          f"batch 64, up to {args.steps} steps (stops early when solved).\n")
    acc = np.zeros((len(MODELS), len(args.delays), len(args.seeds)))
    used = np.zeros_like(acc)
    trained = {}
    header = f"{'delay D':>8} | " + " | ".join(f"{m:>13}" for m in MODELS)
    print("test accuracy (2000 fresh sequences)" +
          ("" if len(args.seeds) == 1 else f", one number per seed {args.seeds}"))
    print(header)
    print("-" * len(header))
    for j, D in enumerate(args.delays):
        cells = []
        for i, kind in enumerate(MODELS):
            for s, seed in enumerate(args.seeds):
                model, steps = train_one(kind, D, seed, steps=args.steps)
                acc[i, j, s], used[i, j, s] = accuracy(model, D), steps
                if s == 0:
                    trained[(kind, D)] = model
            cells.append(" ".join(f"{a:4.0%}" for a in acc[i, j]))
        print(f"{D:>8} | " + " | ".join(f"{c:>13}" for c in cells), flush=True)
    print(f"\ntraining steps used (seed {args.seeds[0]}; {args.steps} = never solved):")
    for i, kind in enumerate(MODELS):
        print(f"  {kind:<14} " + "  ".join(f"D={D}: {int(used[i, j, 0]):>4}"
                                           for j, D in enumerate(args.delays)))

    # Gradient flow at the longest delay: at initialization and after training.
    D = max(args.delays)
    flows = {}
    print(f"\nHow much gradient reaches d steps back (delay D={D}), relative to the last step:")
    print("  RNN: ||dL/dh_k||,  LSTM: ||dL/dc_k||,  GRU: ||dL/dh_k||")
    ds = [d for d in [1, 5, 10, 20, 40, 80] if d <= D + 1]
    print(f"  {'':<26}" + "".join(f"d={d:<8}" for d in ds))
    for kind in MODELS:
        for when in ["init", "trained"]:
            if when == "init":
                torch.manual_seed(args.seeds[0])
                model = Recall(kind, T=D + 2)
            else:
                model = trained[(kind, D)]
            g = gradient_flow(model, D)
            flows[(kind, when)] = g
            print(f"  {kind + ', ' + when:<26}" + "".join(f"{g[d]:<10.1e}" for d in ds))

    print("\nTakeaways:")
    print(" * The vanilla RNN solves short delays but drops to chance once D is large:")
    print("   its gradient shrinks by a constant factor per step (a product of Jacobians).")
    print(" * The LSTM's cell path multiplies the gradient by f, not by W_hh·tanh'.")
    print("   With the usual b_f = 1, f ≈ 0.73 and 0.73^80 ≈ 1e-11: still too small.")
    print("   Chrono init puts some f's near 1, the gradient survives, the task is solved.")
    print(" * Gating makes long memory POSSIBLE; initialization makes it FINDABLE.")

    np.savez(args.save, models=np.array(MODELS), delays=np.array(args.delays),
             seeds=np.array(args.seeds), acc=acc, steps_used=used, flow_delay=D,
             **{f"flow_{i}_{w}": flows[(k, w)] for i, k in enumerate(MODELS)
                for w in ["init", "trained"]})
    print(f"\nSaved results to {os.path.relpath(args.save)}. "
          f"Total time {time.time() - t0:.1f} s.")


if __name__ == "__main__":
    main()
