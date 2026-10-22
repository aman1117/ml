# Recurrent Neural Networks (RNNs)

A hands-on guide to vanilla RNNs: the intuition, the math, a from-scratch NumPy
implementation, and the same model in PyTorch.

## Files

| File | What it teaches |
|------|-----------------|
| `rnn_numpy.py` | Full RNN in NumPy: forward pass, backprop through time (BPTT), Adagrad, sampling. Trains a character-level language model and saves it to `model.npz`. |
| `generate.py` | Loads the trained model. You type the start of some text and it continues it. |
| `gradient_check.py` | Verifies the hand-written BPTT gradients against numerical gradients. |
| `vanishing_gradients.py` | Shows numerically why gradients vanish or explode over long sequences. |
| `rnn_pytorch.py` | Same model in PyTorch, plus a check that `nn.RNN` computes exactly the equations below. |
| `input.txt` | Small training text. Replace it with any text you like. |

### Suggested order

1. Read sections 1–4 of this file.
2. Read `rnn_numpy.py` top to bottom, then run `python rnn/gradient_check.py`.
3. Train: `python rnn/rnn_numpy.py`. Watch the samples improve.
   Then talk to it: `python rnn/generate.py` and type a prompt like `The keeper`.
   It's a text *continuer*, not a chatbot: it only predicts the next character
   in the style of `input.txt`.
4. Read section 5, then run `python rnn/vanishing_gradients.py`.
5. Run `python rnn/rnn_pytorch.py` and compare it with the NumPy version.
6. Do the exercises at the end.

Run the scripts from the repo root with the venv active (see the root `README.md`).
On some Macs, NumPy (with Apple Accelerate) prints harmless
`RuntimeWarning: divide by zero encountered in matmul` messages. The results are
correct. Silence them with `python -W ignore::RuntimeWarning ...`.

---

## 1. Why RNNs?

A feed-forward network (MLP/CNN) maps one fixed-size input to one output and has
**no memory**. Many problems are **sequences** where order matters and lengths vary:
text, audio, time series, DNA.

An RNN processes a sequence **one element at a time** and keeps a **hidden state**
`h_t`, a vector that summarizes everything seen so far. At each step it:

1. reads the current input `x_t`,
2. combines it with the previous memory `h_{t-1}`,
3. produces a new memory `h_t` (and optionally an output `y_t`).

The **same weights are reused at every timestep** (weight sharing). This is what lets
an RNN handle sequences of any length with a fixed number of parameters.

## 2. The equations

```
h_t = tanh(W_xh · x_t + W_hh · h_{t-1} + b_h)     # new hidden state
y_t = W_hy · h_t + b_y                            # output logits
p_t = softmax(y_t)                                # probabilities over the vocabulary
```

Shapes (column vectors, as in `rnn_numpy.py`), with `V` = input/vocab size and `H` = hidden size:

| Symbol | Shape | Meaning |
|--------|-------|---------|
| `x_t`  | (V, 1) | input at step t (one-hot character) |
| `h_t`  | (H, 1) | hidden state / memory |
| `W_xh` | (H, V) | input → hidden |
| `W_hh` | (H, H) | hidden → hidden (**the recurrence**) |
| `W_hy` | (V, H) | hidden → output |
| `b_h`, `b_y` | (H, 1), (V, 1) | biases |

`h_{-1}` (the initial state) is usually zeros.

**Why `tanh`?** It keeps `h` in (-1, 1), so repeatedly applying `W_hh` doesn't blow
up the activations, and it's zero-centered. ReLU RNNs exist but are harder to keep stable.

### Unrolling

An RNN is a deep feed-forward network with one layer per timestep, all sharing weights:

```
          y_0         y_1         y_2
           ↑           ↑           ↑
         [W_hy]      [W_hy]      [W_hy]
           ↑           ↑           ↑
h_{-1} → [ h_0 ] ──→ [ h_1 ] ──→ [ h_2 ] ──→ ...
   W_hh    ↑    W_hh   ↑    W_hh   ↑
         [W_xh]      [W_xh]      [W_xh]
           ↑           ↑           ↑
          x_0         x_1         x_2
```

Sequence length = depth. This view explains both how training works (backprop
through the unrolled graph) and why it's hard (very deep networks → vanishing gradients).

## 3. Training: loss and Backpropagation Through Time (BPTT)

For a character language model, the target at step `t` is the next character.
The loss is the summed cross-entropy:

```
L = Σ_t  -log p_t[target_t]
```

**BPTT** is ordinary backprop on the unrolled graph. Because weights are shared,
**each weight's gradient is the sum of its gradients at every timestep**.

Going backwards from the last step, with `dh_next = 0` initially:

```
dy_t   = p_t - onehot(target_t)          # softmax + cross-entropy gradient
dW_hy += dy_t · h_tᵀ
db_y  += dy_t
dh_t   = W_hyᵀ · dy_t + dh_next          # from the output AND from the future
da_t   = (1 - h_t²) ⊙ dh_t               # through tanh  (tanh' = 1 - tanh²)
dW_xh += da_t · x_tᵀ
dW_hh += da_t · h_{t-1}ᵀ
db_h  += da_t
dh_next = W_hhᵀ · da_t                   # pass gradient to step t-1
```

The key line is `dh_t = W_hyᵀ·dy_t + dh_next`. The hidden state affects the loss
**now** (through the output) **and in the future** (through `h_{t+1}`), so both paths
contribute to the gradient.

### Truncated BPTT

Backpropagating through a whole book is impossible, so we split the text into chunks
of `seq_len` (e.g. 25) characters:

- **Forward**: carry the hidden state across chunks, so the model keeps context.
- **Backward**: only backprop within the chunk. Gradients don't flow across chunk boundaries.

In NumPy this happens naturally. In PyTorch you call `h = h.detach()` between chunks.

### Gradient clipping

Gradients can occasionally explode (section 5). Clipping, either element-wise
(`np.clip`) or by global norm (`clip_grad_norm_`), is a standard, cheap fix.

## 4. Inference: sampling text

To generate text: feed a seed character, get `p_t`, **sample** the next character
from `p_t`, feed it back as the next input, and repeat.

**Temperature** controls randomness: `p = softmax(y / T)`.
- `T < 1`: more confident and repetitive.
- `T > 1`: more random and creative, with more errors.

During training we use **teacher forcing**: the true previous character is the
input, not the model's own prediction.

## 5. Vanishing and exploding gradients

How much does the loss at step `t` depend on the hidden state at an earlier step `k`?

```
∂h_t/∂h_k = Π_{i=k+1..t}  diag(1 - h_i²) · W_hh
```

This is a product of `t - k` matrices. Roughly:

- If the largest singular value of `W_hh` (times the tanh slope ≤ 1) is **< 1**, the
  product **shrinks exponentially** → **vanishing gradients**. The model can't
  learn long-range dependencies because the signal from far back is ~0.
- If it is **> 1**, the product can **grow exponentially** → **exploding gradients**,
  which cause unstable training and NaNs.
- Saturated tanh units (`|h| ≈ 1` → slope ≈ 0) make vanishing even worse.

`vanishing_gradients.py` shows this numerically.

**Fixes:**
| Problem | Fix |
|---------|-----|
| Exploding | Gradient clipping |
| Vanishing | **LSTM / GRU** (gated, additive memory paths), careful init (orthogonal / identity `W_hh`), skip connections, and eventually attention / Transformers |

## 6. Architectures and variants

**Input/output patterns**
- **Many-to-one**: sentiment classification (use only the final `h_T`).
- **One-to-many**: image captioning (an image feature initializes `h`).
- **Many-to-many, aligned**: char language model, POS tagging (an output at every step).
- **Many-to-many, seq2seq**: translation (an encoder RNN produces a context, a decoder RNN generates the output).

**Variants**
- **Stacked / deep RNN**: the `h_t` of layer 1 is the input `x_t` of layer 2.
- **Bidirectional RNN**: one RNN reads left→right, another right→left, and their states are concatenated. Useful when the full sequence is available (not for generation).
- **LSTM** (preview, the next folder):
  ```
  f, i, o = σ(...), σ(...), σ(...)        # forget, input, output gates
  g       = tanh(...)                     # candidate memory
  c_t     = f ⊙ c_{t-1} + i ⊙ g           # ADDITIVE cell update → gradients flow
  h_t     = o ⊙ tanh(c_t)
  ```
  The additive `c_t` path lets gradients flow across many steps when `f ≈ 1`.
- **GRU**: a simpler gated variant with 2 gates (update, reset) and no separate cell state.

## 7. Practical tips

- Initialize `W_hh` small (or orthogonal). With large random weights the activations saturate.
- A healthy starting loss for a char model ≈ `seq_len · ln(V)` (uniform guessing).
  `rnn_numpy.py` prints the **summed** loss over `seq_len` chars, while `rnn_pytorch.py`
  prints the **mean per character**. Divide the NumPy number by `seq_len` to compare.
- `input.txt` is tiny (~4 KB), so the PyTorch model memorizes it (loss → ~0.02).
  That's **overfitting**. For real learning, use a bigger text (1 MB+) and hold out a
  validation split to watch for overfitting.
- If the loss is NaN: lower the learning rate and check clipping.
- In PyTorch, mind `batch_first=True`, and that the hidden state shape is `(num_layers, B, H)`.
- RNNs are inherently **sequential** (step t needs step t-1), which makes them slow
  on GPUs. This is a big reason Transformers replaced them.

## 8. Exercises

1. Derive the BPTT equations in section 3 on paper. Then break one line in
   `backward()` and confirm that `gradient_check.py` catches it.
2. Train with `--hidden 32` vs `--hidden 256`, and `--seq-len 5` vs `--seq-len 50`. Compare the final loss and samples.
3. Sample at temperatures 0.5, 1.0 and 1.5.
4. Replace `tanh` with ReLU in `rnn_numpy.py`. What happens without clipping?
5. Initialize `W_hh` as an orthogonal matrix. Does training change?
6. Turn it into a **many-to-one** classifier (e.g. predict whether a binary string has an even number of 1s).
7. Add a second layer (stacked RNN) to the NumPy model.
8. Implement an LSTM cell in NumPy and reuse the gradient checker.

## 9. Resources

- Andrej Karpathy, *The Unreasonable Effectiveness of Recurrent Neural Networks* (blog), and `min-char-rnn.py` (gist), on which `rnn_numpy.py` is modeled.
- Christopher Olah, *Understanding LSTM Networks* (blog).
- Stanford CS231n, Lecture 10: Recurrent Neural Networks.
- Goodfellow, Bengio & Courville, *Deep Learning*, Chapter 10.
- Pascanu, Mikolov & Bengio (2013), *On the difficulty of training recurrent neural networks*.
