"""
The same LSTM / GRU, in PyTorch.

1. Verifies that PyTorch's built-in layers compute exactly our equations:
     nn.LSTMCell / nn.LSTM (gate order i, f, g, o):
         i, f, g, o = chunks of  x W_ihᵀ + b_ih + h W_hhᵀ + b_hh
         c' = σ(f) ⊙ c + σ(i) ⊙ tanh(g),   h' = σ(o) ⊙ tanh(c')
     nn.GRU (gate order r, z, n):
         r, z = σ(x W_irᵀ + b_ir + h W_hrᵀ + b_hr), σ(...)
         n    = tanh(x W_inᵀ + b_in + r ⊙ (h W_hnᵀ + b_hn))    <- r is applied AFTER W_hn
         h'   = (1 - z) ⊙ n + z ⊙ h
   by re-implementing them with explicit loops.
2. Copies the weights of our NumPy LSTM and GRU (lstm_numpy.py) into PyTorch
   and checks that autograd gives the same gradients as our hand-written BPTT,
   including the gradients for the initial state (h0, c0).
3. Trains a CharLSTM (Embedding -> nn.LSTM -> Linear) with Adam, truncated
   BPTT (detach BOTH h and c), clip_grad_norm_, then samples text.

Run:
    python lstm/lstm_pytorch.py
    python lstm/lstm_pytorch.py --cell gru
"""
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lstm_numpy import DEFAULT_TEXT, GRU, LSTM


# ---------------------------------------------------------------- 1. equations

def manual_lstm_cell(cell, x, h, c):
    """One step of nn.LSTMCell, written out. x: (B, I), h, c: (B, H)."""
    gates = x @ cell.weight_ih.T + cell.bias_ih + h @ cell.weight_hh.T + cell.bias_hh  # (B, 4H)
    i, f, g, o = gates.chunk(4, dim=1)  # PyTorch's order: input, forget, cell(g), output
    c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
    h = torch.sigmoid(o) * torch.tanh(c)
    return h, c


def manual_lstm(lstm, x, h0, c0):
    """Explicit-loop version of a single-layer nn.LSTM (batch_first=True)."""
    W_ih, W_hh = lstm.weight_ih_l0, lstm.weight_hh_l0
    b = lstm.bias_ih_l0 + lstm.bias_hh_l0  # the two biases are simply added
    h, c = h0[0], c0[0]  # (B, H): drop the num_layers dimension
    outs = []
    for t in range(x.shape[1]):
        i, f, g, o = (x[:, t] @ W_ih.T + h @ W_hh.T + b).chunk(4, dim=1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h = torch.sigmoid(o) * torch.tanh(c)
        outs.append(h)
    return torch.stack(outs, dim=1), (h.unsqueeze(0), c.unsqueeze(0))


def manual_gru(gru, x, h0):
    """Explicit-loop version of a single-layer nn.GRU (batch_first=True)."""
    W_ih, W_hh = gru.weight_ih_l0, gru.weight_hh_l0
    h = h0[0]
    H = h.shape[1]
    outs = []
    for t in range(x.shape[1]):
        gx = x[:, t] @ W_ih.T + gru.bias_ih_l0  # (B, 3H): r, z, n votes from the input
        gh = h @ W_hh.T + gru.bias_hh_l0        # (B, 3H): ... and from the memory
        r = torch.sigmoid(gx[:, :H] + gh[:, :H])
        z = torch.sigmoid(gx[:, H:2 * H] + gh[:, H:2 * H])
        n = torch.tanh(gx[:, 2 * H:] + r * gh[:, 2 * H:])  # r scales (W_hn h + b_hn)
        h = (1 - z) * n + z * h
        outs.append(h)
    return torch.stack(outs, dim=1), h.unsqueeze(0)


def check_equations():
    torch.manual_seed(0)
    B, T, I, H = 3, 10, 5, 7

    cell = nn.LSTMCell(I, H)
    x, h, c = torch.randn(B, I), torch.randn(B, H), torch.randn(B, H)
    h_ref, c_ref = cell(x, (h, c))
    h_man, c_man = manual_lstm_cell(cell, x, h, c)
    assert torch.allclose(h_ref, h_man, atol=1e-6) and torch.allclose(c_ref, c_man, atol=1e-6)
    print("nn.LSTMCell matches the hand-written equations (gate order i, f, g, o) ✔")

    lstm = nn.LSTM(I, H, batch_first=True)
    x = torch.randn(B, T, I)
    h0, c0 = torch.randn(1, B, H), torch.randn(1, B, H)
    out_ref, (h_ref, c_ref) = lstm(x, (h0, c0))
    out_man, (h_man, c_man) = manual_lstm(lstm, x, h0, c0)
    assert torch.allclose(out_ref, out_man, atol=1e-6)
    assert torch.allclose(h_ref, h_man, atol=1e-6) and torch.allclose(c_ref, c_man, atol=1e-6)
    print("nn.LSTM matches an explicit loop over time ✔")
    print(f"  weight_ih_l0 {tuple(lstm.weight_ih_l0.shape)} = (4H, I), "
          f"weight_hh_l0 {tuple(lstm.weight_hh_l0.shape)} = (4H, H)")
    print(f"  output (B, T, H) = {tuple(out_ref.shape)}, h_n and c_n (num_layers, B, H) = "
          f"{tuple(h_ref.shape)} and {tuple(c_ref.shape)}")

    gru = nn.GRU(I, H, batch_first=True)
    out_ref, h_ref = gru(x, h0)
    out_man, h_man = manual_gru(gru, x, h0)
    assert torch.allclose(out_ref, out_man, atol=1e-6) and torch.allclose(h_ref, h_man, atol=1e-6)
    print("nn.GRU matches an explicit loop (gate order r, z, n; r ⊙ (W_hn h + b_hn)) ✔")
    print(f"  weight_ih_l0 {tuple(gru.weight_ih_l0.shape)} = (3H, I): "
          "3 blocks instead of the LSTM's 4\n")


# ---------------------------------------------------------------- 2. gradients

def check_against_numpy(cls):
    """Copy a NumPy model into PyTorch and compare autograd with our backward()."""
    rng = np.random.default_rng(0)
    V, H, T = 6, 5, 8
    np_model = cls(V, H)
    for k in np_model.params:  # non-trivial weights so every path matters
        np_model.params[k] = rng.standard_normal(np_model.params[k].shape) * 0.5
    inputs = rng.integers(0, V, T).tolist()
    targets = rng.integers(0, V, T).tolist()
    state = tuple(rng.standard_normal((H, 1)) * 0.5 for _ in np_model.init_state())
    _, _, cache = np_model.forward(inputs, targets, state)
    grads, dstate = np_model.backward(cache)

    P = {k: torch.tensor(v) for k, v in np_model.params.items()}  # float64
    rnn = (nn.LSTM if cls is LSTM else nn.GRU)(V, H, batch_first=True).double()
    head = nn.Linear(H, V).double()
    with torch.no_grad():
        rnn.weight_ih_l0.copy_(P["W_xh"])  # our (4H, V) or (3H, V) = PyTorch's layout
        rnn.weight_hh_l0.copy_(P["W_hh"])
        if cls is LSTM:
            rnn.bias_ih_l0.copy_(P["b_h"][:, 0])  # our single b_h = bias_ih + bias_hh
            rnn.bias_hh_l0.zero_()
        else:
            rnn.bias_ih_l0.copy_(P["b_xh"][:, 0])
            rnn.bias_hh_l0.copy_(P["b_hh"][:, 0])
        head.weight.copy_(P["W_hy"])
        head.bias.copy_(P["b_y"][:, 0])

    x = F.one_hot(torch.tensor([inputs]), V).double()  # (1, T, V) one-hot, like ours
    s0 = [torch.tensor(s.T[None], requires_grad=True) for s in state]  # (1, 1, H) each
    out, _ = rnn(x, tuple(s0) if cls is LSTM else s0[0])
    loss = F.cross_entropy(head(out[0]), torch.tensor(targets), reduction="sum")
    loss.backward()

    torch_grads = {"W_xh": rnn.weight_ih_l0.grad, "W_hh": rnn.weight_hh_l0.grad,
                   "W_hy": head.weight.grad, "b_y": head.bias.grad[:, None]}
    if cls is LSTM:
        torch_grads["b_h"] = rnn.bias_ih_l0.grad[:, None]
    else:
        torch_grads["b_xh"] = rnn.bias_ih_l0.grad[:, None]
        torch_grads["b_hh"] = rnn.bias_hh_l0.grad[:, None]
    for k, g in torch_grads.items():
        assert np.allclose(g.numpy(), grads[k], atol=1e-10), f"{cls.__name__} {k} mismatch"
    for name, s, ds in zip(["h0", "c0"], s0, dstate):
        assert np.allclose(s.grad[0].numpy().T, ds, atol=1e-10), f"{name} mismatch"
    names = ", ".join(list(torch_grads) + ["h0", "c0"][:len(s0)])
    print(f"PyTorch autograd == our hand-written {cls.__name__} BPTT ✔  ({names})")


# ---------------------------------------------------------------- 3. training

class CharModel(nn.Module):
    def __init__(self, vocab_size, cell="lstm", embed_size=32, hidden_size=128, num_layers=1):
        super().__init__()
        self.cell = cell
        self.embed = nn.Embedding(vocab_size, embed_size)
        rnn_cls = nn.LSTM if cell == "lstm" else nn.GRU
        self.rnn = rnn_cls(embed_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.head = nn.Linear(hidden_size, vocab_size)
        if cell == "lstm":
            # Forget-gate bias = 1 (PyTorch's default is a small random bias).
            # Gate order is i, f, g, o, so the f block is [H:2H]; the two biases add up.
            H = hidden_size
            with torch.no_grad():
                for name, b in self.rnn.named_parameters():
                    if name.startswith("bias_ih"):
                        b[H:2 * H] = 1.0
                    elif name.startswith("bias_hh"):
                        b[H:2 * H] = 0.0

    def forward(self, idx, state=None):
        out, state = self.rnn(self.embed(idx), state)  # out: (B, T, H)
        return self.head(out), state  # logits: (B, T, V)

    @torch.no_grad()
    def sample(self, start_ix, n, temperature=1.0):
        idx = torch.tensor([[start_ix]])
        state, out = None, [start_ix]
        for _ in range(n):
            logits, state = self(idx, state)
            probs = F.softmax(logits[:, -1] / temperature, dim=-1)
            idx = torch.multinomial(probs, 1)
            out.append(idx.item())
        return out


def detach(state):
    """Truncated BPTT: keep the values, drop the graph. LSTM state is a tuple (h, c)."""
    if isinstance(state, tuple):
        return tuple(s.detach() for s in state)
    return state.detach()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", default=DEFAULT_TEXT)
    ap.add_argument("--cell", choices=["lstm", "gru"], default="lstm")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--seq-len", type=int, default=50)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--temperature", type=float, default=0.8)
    args = ap.parse_args()

    check_equations()
    check_against_numpy(LSTM)
    check_against_numpy(GRU)
    print()

    torch.manual_seed(0)
    with open(args.file, encoding="utf-8") as f:
        text = f.read()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    data = torch.tensor([stoi[c] for c in text])
    V = len(chars)

    # Split the text into `batch` parallel streams, read in consecutive chunks
    # so the state can carry context from one chunk to the next.
    n = (len(data) - 1) // args.batch
    xs = data[: n * args.batch].view(args.batch, n)
    ys = data[1: n * args.batch + 1].view(args.batch, n)

    model = CharModel(V, args.cell, hidden_size=args.hidden)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f"Char{args.cell.upper()}: vocab {V}, params "
          f"{sum(p.numel() for p in model.parameters()):,}")

    state, pos = None, 0
    for step in range(args.steps + 1):
        if pos + args.seq_len > n:
            state, pos = None, 0  # end of the streams: reset memory
        x = xs[:, pos:pos + args.seq_len]
        y = ys[:, pos:pos + args.seq_len]
        pos += args.seq_len

        if state is not None:
            state = detach(state)  # for an LSTM, forgetting to detach c is a classic bug

        logits, state = model(x, state)
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
