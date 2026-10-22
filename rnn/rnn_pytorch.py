"""
The same character-level RNN, in PyTorch.

1. Verifies that nn.RNN computes exactly:
       h_t = tanh(x_t W_ih^T + b_ih + h_{t-1} W_hh^T + b_hh)
   by re-implementing it with an explicit loop.
2. Trains a CharRNN (Embedding -> nn.RNN -> Linear) with Adam, using
   truncated BPTT and gradient-norm clipping, then samples text.

Note: nn.Embedding(V, E) is equivalent to multiplying a one-hot vector by a
(V, E) matrix, i.e. the W_xh lookup in rnn_numpy.py.

Run:
    python rnn/rnn_pytorch.py
"""
import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


def manual_rnn(rnn: nn.RNN, x, h0):
    """Explicit-loop version of a single-layer nn.RNN (batch_first=True)."""
    W_ih, W_hh = rnn.weight_ih_l0, rnn.weight_hh_l0
    b_ih, b_hh = rnn.bias_ih_l0, rnn.bias_hh_l0
    h = h0[0]  # (B, H)
    outs = []
    for t in range(x.shape[1]):
        h = torch.tanh(x[:, t] @ W_ih.T + b_ih + h @ W_hh.T + b_hh)
        outs.append(h)
    return torch.stack(outs, dim=1), h.unsqueeze(0)


def check_equivalence():
    torch.manual_seed(0)
    B, T, I, H = 3, 10, 5, 7
    rnn = nn.RNN(I, H, batch_first=True)
    x = torch.randn(B, T, I)
    h0 = torch.randn(1, B, H)
    out_ref, h_ref = rnn(x, h0)
    out_man, h_man = manual_rnn(rnn, x, h0)
    assert torch.allclose(out_ref, out_man, atol=1e-6)
    assert torch.allclose(h_ref, h_man, atol=1e-6)
    print("nn.RNN matches the hand-written equations ✔")
    print(f"  output shape (B, T, H) = {tuple(out_ref.shape)}, "
          f"h_n shape (num_layers, B, H) = {tuple(h_ref.shape)}\n")


class CharRNN(nn.Module):
    def __init__(self, vocab_size, embed_size=32, hidden_size=128, num_layers=1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.RNN(embed_size, hidden_size, num_layers=num_layers,
                          nonlinearity="tanh", batch_first=True)
        self.head = nn.Linear(hidden_size, vocab_size)

    def forward(self, idx, h=None):
        out, h = self.rnn(self.embed(idx), h)  # out: (B, T, H)
        return self.head(out), h  # logits: (B, T, V)

    @torch.no_grad()
    def sample(self, start_ix, n, temperature=1.0):
        idx = torch.tensor([[start_ix]])
        h, out = None, [start_ix]
        for _ in range(n):
            logits, h = self(idx, h)
            probs = F.softmax(logits[:, -1] / temperature, dim=-1)
            idx = torch.multinomial(probs, 1)
            out.append(idx.item())
        return out


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=os.path.join(here, "input.txt"))
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--seq-len", type=int, default=50)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--temperature", type=float, default=0.8)
    args = ap.parse_args()

    check_equivalence()

    torch.manual_seed(0)
    with open(args.file, encoding="utf-8") as f:
        text = f.read()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    data = torch.tensor([stoi[c] for c in text])
    V = len(chars)

    # Split the text into `batch` parallel streams. Each stream is read in
    # consecutive chunks so the hidden state can carry context between them.
    n = (len(data) - 1) // args.batch
    xs = data[: n * args.batch].view(args.batch, n)
    ys = data[1: n * args.batch + 1].view(args.batch, n)

    model = CharRNN(V, hidden_size=args.hidden)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f"vocab {V}, params {sum(p.numel() for p in model.parameters()):,}")

    h, pos = None, 0
    for step in range(args.steps + 1):
        if pos + args.seq_len > n:
            h, pos = None, 0  # end of streams: reset memory
        x = xs[:, pos:pos + args.seq_len]
        y = ys[:, pos:pos + args.seq_len]
        pos += args.seq_len

        if h is not None:
            h = h.detach()  # truncated BPTT: keep the values, drop the graph

        logits, h = model(x, h)
        loss = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1))

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 250 == 0:
            print(f"step {step:5d}  loss/char {loss.item():.3f}")

    out = model.sample(stoi[text[0]], 400, args.temperature)
    print(f"\n===== sample (temperature {args.temperature}) =====")
    print("".join(itos[i] for i in out))


if __name__ == "__main__":
    main()
