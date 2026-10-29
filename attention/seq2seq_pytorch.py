"""
Seq2seq (GRU encoder-decoder) WITH and WITHOUT attention, in PyTorch.

Part 1 proves that PyTorch's built-in attention computes exactly our NumPy equations:
    F.scaled_dot_product_attention  ==  softmax(Q Kᵀ / √d + mask) V   (attention_numpy.py)
    nn.MultiheadAttention (1 head, identity projections)  ==  the same thing
    autograd gradients  ==  our hand-written backward (scaled dot-product AND additive)

Part 2 trains a character-level encoder-decoder on date normalization
("wednesday 3 march 2021" -> "2021-03-03"), then saves it for translate.py.

The model (README section 2 and 4):
    encoder:   h_1..h_S  = BiGRU(embed(x_1..x_S))           (B, S, 2H)  one vector PER input char
    summary:   c         = [h_fwd_last ; h_bwd_last]          (B, 2H)     the whole input in ONE vector
    decoder:   s_0 = tanh(W_b c),  s_t = GRU(embed(y_{t-1}), s_{t-1})     (B, D)
    context:   none:      c_t = c                             the same fixed vector at every step
               attention: c_t = Σ_j α_tj h_j,  α_t = softmax_j(score(s_t, h_j))   (masked at <pad>)
                 dot:      score = (s_t W_q) · (h_j W_k) / √d_a           (Luong "general", scaled)
                 additive: score = vᵀ tanh(s_t W_q + h_j W_k)             (Bahdanau)
    output:    p(y_t) = softmax(W_o tanh(W_c [s_t ; c_t]))

The only difference between "none" and attention is c_t: fixed vs. recomputed for every
output character. Training uses teacher forcing (the decoder reads the TRUE previous char),
so all T decoder steps run in parallel. Inference is greedy: feed back the argmax char.

Run:
    python attention/seq2seq_pytorch.py                       (dot attention, ~15 s, saves date_model.pt)
    python attention/seq2seq_pytorch.py --attention none      (the bottleneck baseline)
    python attention/seq2seq_pytorch.py --attention additive  (Bahdanau scores)
    python attention/seq2seq_pytorch.py --check-only          (just Part 1)
"""
import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from attention_numpy import (additive_attention, additive_attention_backward,
                             scaled_dot_product_attention, scaled_dot_product_attention_backward)
from tasks import DATE_VOCAB, DIGIT_VOCAB, EOS, PAD, SOS, sample_dates, sample_reverse, to_arrays

HERE = os.path.dirname(os.path.abspath(__file__))
torch.set_num_threads(min(4, torch.get_num_threads()))  # small models: more threads only add overhead


# ======================================================================= Part 1: checks

def check_against_numpy():
    rng = np.random.default_rng(0)
    B, n, m, d, dv = 2, 3, 5, 4, 6
    Q, K, V = rng.standard_normal((B, n, d)), rng.standard_normal((B, m, d)), \
        rng.standard_normal((B, m, dv))
    lengths = [3, 5]  # example 0 has 2 padded keys
    mask = (np.arange(m)[None, :] < np.array(lengths)[:, None])[:, None, :]  # (B, 1, m) True = keep
    out_np, A_np, cache = scaled_dot_product_attention(Q, K, V, mask)

    Qt, Kt, Vt = (torch.tensor(x, requires_grad=True) for x in (Q, K, V))
    out_t = F.scaled_dot_product_attention(Qt, Kt, Vt, attn_mask=torch.tensor(mask))
    assert torch.allclose(out_t, torch.tensor(out_np), atol=1e-10)
    print("F.scaled_dot_product_attention == our NumPy softmax(Q Kᵀ/√d + mask) V ✔")
    print(f"  Q {Q.shape}, K {K.shape}, V {V.shape} -> out {out_np.shape}, "
          f"weights {A_np.shape}; padded keys got weight {A_np[0, :, 3:].max():.0f}")

    # nn.MultiheadAttention = projections + the same attention. With 1 head and identity
    # projections, it must give the same output and weights. Note the OPPOSITE mask
    # convention: key_padding_mask=True means "ignore this key".
    mha = nn.MultiheadAttention(d, num_heads=1, batch_first=True, bias=False).double()
    with torch.no_grad():
        mha.in_proj_weight.copy_(torch.cat([torch.eye(d)] * 3))  # W_Q = W_K = W_V = I
        mha.out_proj.weight.copy_(torch.eye(d))
    V4 = rng.standard_normal((B, m, d))  # MHA needs d_v = d
    out_mha, A_mha = mha(Qt.detach(), Kt.detach(), torch.tensor(V4),
                         key_padding_mask=torch.tensor(~mask[:, 0, :]))
    out_np4, _, _ = scaled_dot_product_attention(Q, K, V4, mask)
    assert torch.allclose(out_mha, torch.tensor(out_np4), atol=1e-10)
    assert torch.allclose(A_mha, torch.tensor(A_np), atol=1e-10)
    print("nn.MultiheadAttention (1 head, identity projections) == the same equations ✔")

    R = rng.standard_normal(out_np.shape)  # loss L = Σ out ⊙ R, so dOut = R
    (out_t * torch.tensor(R)).sum().backward()
    for name, g_np, x in zip(["dQ", "dK", "dV"], scaled_dot_product_attention_backward(R, cache),
                             (Qt, Kt, Vt)):
        assert torch.allclose(x.grad, torch.tensor(g_np), atol=1e-10), name
    print("autograd dQ, dK, dV == our hand-written backward (with the padding mask) ✔")

    # Additive (Bahdanau) attention: re-implemented with torch ops, autograd vs our backward.
    da = 7
    arrays = [rng.standard_normal(s) for s in [(B, n, 5), (B, m, d), (B, m, dv),
                                               (5, da), (d, da), (da,)]]
    out_a, _, cache_a = additive_attention(*arrays, mask=mask)
    q, Kk, Vv, Wq, Wk, v = (torch.tensor(a, requires_grad=True) for a in arrays)
    s = torch.tanh((q @ Wq)[:, :, None, :] + (Kk @ Wk)[:, None, :, :]) @ v
    s = s.masked_fill(~torch.tensor(mask), float("-inf"))
    out_ta = torch.softmax(s, dim=-1) @ Vv
    assert torch.allclose(out_ta, torch.tensor(out_a), atol=1e-10)
    R = rng.standard_normal(out_a.shape)
    (out_ta * torch.tensor(R)).sum().backward()
    for name, g_np, x in zip(["dq", "dK", "dV", "dW_q", "dW_k", "dv"],
                             additive_attention_backward(R, cache_a), (q, Kk, Vv, Wq, Wk, v)):
        assert torch.allclose(x.grad, torch.tensor(g_np), atol=1e-10), name
    print("additive attention: forward and all 6 autograd gradients == our NumPy code ✔\n")


# ======================================================================= Part 2: the model

class Seq2Seq(nn.Module):
    def __init__(self, vocab_size, emb=32, hidden=64, attention="dot", attn_dim=64):
        super().__init__()
        H, D = hidden, 2 * hidden  # encoder: H per direction; decoder state D = 2H
        self.attention = attention
        self.embed = nn.Embedding(vocab_size, emb, padding_idx=PAD)  # shared by both sides
        self.encoder = nn.GRU(emb, H, batch_first=True, bidirectional=True)
        self.bridge = nn.Linear(2 * H, D)
        self.decoder = nn.GRU(emb, D, batch_first=True)
        self.W_q = nn.Linear(D, attn_dim, bias=False)  # query = what the decoder needs now
        self.W_k = nn.Linear(2 * H, attn_dim, bias=False)  # key = what each input char offers
        self.v = nn.Linear(attn_dim, 1, bias=False)  # only used by additive attention
        self.combine = nn.Linear(D + 2 * H, D)
        self.out = nn.Linear(D, vocab_size)
        # The no-attention model keeps (but never uses) W_q, W_k, v, so both models have
        # exactly the same parameters and the comparison is only about c_t.

    def encode(self, src, src_len):
        """Returns per-char states h (B, S, 2H), the summary c (B, 2H) and the pad mask (B, S)."""
        packed = nn.utils.rnn.pack_padded_sequence(self.embed(src), src_len.cpu(),
                                                   batch_first=True, enforce_sorted=False)
        out, h_n = self.encoder(packed)  # packing: the backward GRU starts at the real last char
        h, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=src.shape[1])
        c = torch.cat([h_n[0], h_n[1]], dim=-1)  # last forward state ; last backward state
        return h, c, src != PAD

    def attend(self, s, h, mask):
        """s (B, T, D) decoder states = queries; h (B, S, 2H) = keys AND values. -> c_t, α."""
        q, k = self.W_q(s), self.W_k(h)  # (B, T, A), (B, S, A)
        if self.attention == "dot":
            scores = q @ k.transpose(1, 2) / q.shape[-1] ** 0.5  # (B, T, S)
        else:  # additive: v · tanh(q + k) for every (output step, input char) pair
            scores = self.v(torch.tanh(q[:, :, None, :] + k[:, None, :, :])).squeeze(-1)
        scores = scores.masked_fill(~mask[:, None, :], float("-inf"))  # never attend to <pad>
        alpha = torch.softmax(scores, dim=-1)  # (B, T, S), each row sums to 1
        return alpha @ h, alpha  # context (B, T, 2H) = weighted sum of the VALUES h_j

    def decode(self, y_prev, s, h, c, mask):
        """Run the decoder over y_prev (B, T) from state s. Returns logits, new state, α."""
        out, s = self.decoder(self.embed(y_prev), s)  # (B, T, D)
        if self.attention == "none":
            ctx, alpha = c[:, None, :].expand(-1, out.shape[1], -1), None  # the bottleneck
        else:
            ctx, alpha = self.attend(out, h, mask)
        logits = self.out(torch.tanh(self.combine(torch.cat([out, ctx], dim=-1))))
        return logits, s, alpha

    def forward(self, src, src_len, dec_in):
        """Teacher forcing: all decoder inputs are known, so all steps run at once."""
        h, c, mask = self.encode(src, src_len)
        s0 = torch.tanh(self.bridge(c))[None]  # (1, B, D)
        logits, _, alpha = self.decode(dec_in, s0, h, c, mask)
        return logits, alpha

    @torch.no_grad()
    def greedy(self, src, src_len, max_len):
        """Generate one char at a time, feeding back the argmax. Returns ids (B, T), α (B, T, S)."""
        h, c, mask = self.encode(src, src_len)
        s = torch.tanh(self.bridge(c))[None]
        y = torch.full((src.shape[0], 1), SOS, dtype=torch.long)
        ids, alphas = [], []
        for _ in range(max_len):
            logits, s, alpha = self.decode(y, s, h, c, mask)
            y = logits.argmax(-1)  # (B, 1)
            ids.append(y)
            if alpha is not None:
                alphas.append(alpha)
            if (torch.cat(ids, 1) == EOS).any(1).all():
                break
        return torch.cat(ids, 1), (torch.cat(alphas, 1) if alphas else None)


# ======================================================================= training helpers

TASKS = {
    # name: (vocab, sampler(n, rng, split), max output length incl. <eos>)
    "dates": (DATE_VOCAB, lambda n, rng, split="train": sample_dates(n, rng, split), 11),
    "reverse": (DIGIT_VOCAB, lambda n, rng, split="train": sample_reverse(n, rng), 41),
}


def tensors(pairs, vocab):
    return [torch.from_numpy(a) for a in to_arrays(pairs, vocab)]


def exact_match(model, pairs, vocab, max_len, batch=500):
    """Fraction of inputs whose greedy output equals the target exactly, plus the outputs."""
    model.eval()
    preds = []
    for i in range(0, len(pairs), batch):
        src, src_len, _, _ = tensors(pairs[i:i + batch], vocab)
        ids, _ = model.greedy(src, src_len, max_len)
        preds += [vocab.decode(row) for row in ids.tolist()]
    model.train()
    acc = float(np.mean([p == t for p, (_, t) in zip(preds, pairs)]))
    return acc, preds


def train(task="dates", attention="dot", steps=1500, batch=64, lr=3e-3, emb=32, hidden=64,
          seed=0, eval_pairs=None, eval_every=100, verbose=True, sampler=None):
    """
    Trains a Seq2Seq model on fresh random examples every step.
    Returns the model and a history list of (step, train loss, exact-match acc on eval_pairs).
    """
    vocab, default_sampler, max_len = TASKS[task]
    sampler = sampler or default_sampler
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = Seq2Seq(len(vocab), emb, hidden, attention)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    # Cosine learning-rate decay: big steps early, tiny careful steps at the end.
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda i: 0.5 * (1 + np.cos(np.pi * min(i, steps) / steps)))
    history, t0 = [], time.time()
    for step in range(1, steps + 1):
        src, src_len, dec_in, dec_out = tensors(sampler(batch, rng), vocab)
        logits, _ = model(src, src_len, dec_in)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), dec_out.reshape(-1),
                               ignore_index=PAD)  # no loss on padding positions
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        acc = None
        if eval_pairs is not None and (step % eval_every == 0 or step == steps):
            acc, _ = exact_match(model, eval_pairs, vocab, max_len)
            if verbose:
                print(f"step {step:5d} | loss {loss.item():.3f} | test exact match {acc:6.1%} "
                      f"| {time.time() - t0:5.1f} s")
        history.append((step, loss.item(), acc))
    return model, history


def save_model(path, model, config):
    torch.save({"config": config, "state_dict": model.state_dict()}, path)


def load_model(path):
    ckpt = torch.load(path, map_location="cpu")
    cfg = ckpt["config"]
    model = Seq2Seq(len(TASKS[cfg["task"]][0]), cfg["emb"], cfg["hidden"], cfg["attention"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, cfg


DATE_CONFIG = dict(task="dates", attention="dot", steps=800, batch=64, lr=3e-3, emb=32,
                   hidden=64, seed=0)
DATE_MODEL_PATH = os.path.join(HERE, "date_model.pt")


def load_or_train_date_model(verbose=True):
    """The saved date model, or (if there is none yet) a freshly trained and saved one."""
    if os.path.exists(DATE_MODEL_PATH):
        return load_model(DATE_MODEL_PATH)[0]
    if verbose:
        print(f"No {os.path.relpath(DATE_MODEL_PATH)} yet, so training one first "
              "(same settings as seq2seq_pytorch.py, ~15 s)...")
    model, _ = train(verbose=False, **DATE_CONFIG)
    model.eval()
    save_model(DATE_MODEL_PATH, model, DATE_CONFIG)
    if verbose:
        print(f"saved {os.path.relpath(DATE_MODEL_PATH)}\n")
    return model


@torch.no_grad()
def translate_date(model, text):
    """One input string -> (output string, attention weights (T_out incl. <eos>, S) as NumPy)."""
    src, src_len, _, _ = tensors([(text, "")], DATE_VOCAB)
    ids, alpha = model.greedy(src, src_len, 11)
    out = DATE_VOCAB.decode(ids[0].tolist())
    T = min(len(out) + 1, alpha.shape[1])  # keep the step that produced <eos>
    return out, alpha[0, :T].numpy()


def date_test_set(n=1000, seed=1):
    """n DIFFERENT test examples, all on days that never appear in training."""
    pairs = list(dict.fromkeys(sample_dates(2 * n, np.random.default_rng(seed), split="test")))
    return pairs[:n]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--attention", choices=["additive", "dot", "none"],
                    default=DATE_CONFIG["attention"])
    ap.add_argument("--steps", type=int, default=DATE_CONFIG["steps"])
    ap.add_argument("--batch", type=int, default=DATE_CONFIG["batch"])
    ap.add_argument("--lr", type=float, default=DATE_CONFIG["lr"])
    ap.add_argument("--hidden", type=int, default=DATE_CONFIG["hidden"],
                    help="encoder GRU size per direction (decoder is 2x this)")
    ap.add_argument("--seed", type=int, default=DATE_CONFIG["seed"])
    ap.add_argument("--save", default=DATE_MODEL_PATH)
    ap.add_argument("--check-only", action="store_true", help="only run the equivalence checks")
    args = ap.parse_args()

    check_against_numpy()
    if args.check_only:
        return

    cfg = dict(DATE_CONFIG, attention=args.attention, steps=args.steps, batch=args.batch,
               lr=args.lr, hidden=args.hidden, seed=args.seed)
    test = date_test_set()
    n_params = sum(p.numel() for p in Seq2Seq(len(DATE_VOCAB), cfg["emb"], cfg["hidden"],
                                              cfg["attention"]).parameters())
    print(f"Dates: {len(test)} test examples (days never seen in training). "
          f"attention = {cfg['attention']}, {n_params:,} parameters\n")
    t0 = time.time()
    model, _ = train(eval_pairs=test, **cfg)
    acc, preds = exact_match(model, test, DATE_VOCAB, 11)
    print(f"\nfinal test exact match: {acc:.1%}  (trained in {time.time() - t0:.0f} s)\n")
    for (s, t), p in list(zip(test, preds))[:8]:
        print(f"  {s:<28} -> {p}  {'✔' if p == t else '✘ (want ' + t + ')'}")
    wrong = [(s, t, p) for (s, t), p in zip(test, preds) if p != t]
    if wrong:
        print(f"\n{len(wrong)} mistake(s), for example:")
        for s, t, p in wrong[:5]:
            print(f"  {s:<28} -> {p}  (want {t})")

    if cfg["attention"] == "none" and args.save == DATE_MODEL_PATH:
        print("\n(No-attention model not saved, so date_model.pt keeps the attention model "
              "that translate.py needs.)")
        return
    save_model(args.save, model, cfg)
    print(f"\nSaved model to {os.path.relpath(args.save)}. Try: python attention/translate.py")


if __name__ == "__main__":
    main()
