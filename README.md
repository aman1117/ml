# ML from Scratch

Personal learning repo for implementing classic deep learning models from scratch.

## Goal

**Land an AI engineer job at OpenAI-like companies** (frontier AI labs).

Everything here is in service of that: deeply understand how models work by
building them from scratch, the kind of fundamentals these interviews and roles expect.

## Planned implementations

Work through these in order. Each builds on the previous one.

- [ ] Multi-Layer Perceptron (MLP): [`mlp/`](mlp/)
- [ ] Convolutional Neural Network (CNN)
- [ ] Recurrent Neural Network (RNN): [`rnn/`](rnn/)
- [ ] LSTM / GRU
- [ ] Transformer

## Structure

Each model lives in its own folder (e.g. `rnn/`, `cnn/`, `transformer/`) with its implementation and notes.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
