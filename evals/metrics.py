"""
Eval metrics from scratch: what number do you compute for ONE example, and how do
you turn many examples into one score? (README sections 4-7.)

  answer matching   exact match, SQuAD-normalized exact match, token-level F1
  sampling          the unbiased pass@k estimator (Chen et al. 2021) and pass^k
  likelihood        perplexity, bits per character, bits per byte from log-probs
  calibration       ECE (equal-width and equal-mass bins), reliability-diagram data,
                    Brier score, temperature scaling with a hand-derived gradient
  agreement         raw agreement and Cohen's kappa (judge vs human, human vs human)
  error bars        the standard error of an accuracy (and the Wilson interval)

Everything is plain Python / NumPy so you can read every line. metrics_check.py
verifies each function against brute force, simulation, or known values.

Run (prints a few worked examples):
    python evals/metrics.py
"""
from __future__ import annotations

import math
import re
import string
import warnings
from collections import Counter
from typing import Dict, List, Optional, Sequence

import numpy as np

# NumPy 2.0 + Apple Accelerate raises spurious "divide by zero / overflow /
# invalid value encountered in matmul" warnings even when results are correct.
warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)

LN2 = math.log(2.0)
PUNCT = set(string.punctuation)


# ----------------------------------------------------------------------------
# Answer matching (SQuAD style)
# ----------------------------------------------------------------------------
def normalize_answer(s: str, remove_articles: bool = True) -> str:
    """The SQuAD v1.1 normalization: lowercase, drop punctuation, drop the
    articles a/an/the, collapse whitespace. Same order as the official script.

    remove_articles=False exists because dropping articles is WRONG when the
    answer itself can be an article: "the" and "a" both normalize to "" and
    would count as a match (README section 4 measures how often that happens)."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in PUNCT)
    if remove_articles:
        s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(pred: str, gold: str) -> float:
    """Strict: the raw strings must be identical (after stripping outer whitespace)."""
    return float(pred.strip() == gold.strip())


def normalized_exact_match(pred: str, gold: str, remove_articles: bool = True) -> float:
    return float(normalize_answer(pred, remove_articles) == normalize_answer(gold, remove_articles))


def token_f1(pred: str, gold: str, remove_articles: bool = True) -> float:
    """Bag-of-tokens F1 between normalized answers (the SQuAD definition).
    Precision = shared / len(pred), recall = shared / len(gold), with token
    multiplicities (Counter intersection). If either side is empty, F1 is 1
    when both are empty and 0 otherwise."""
    p = normalize_answer(pred, remove_articles).split()
    g = normalize_answer(gold, remove_articles).split()
    if not p or not g:
        return float(p == g)
    same = sum((Counter(p) & Counter(g)).values())
    if same == 0:
        return 0.0
    precision, recall = same / len(p), same / len(g)
    return 2 * precision * recall / (precision + recall)


def best_over_golds(metric, pred: str, golds: Sequence[str], **kw) -> float:
    """SQuAD and most QA sets give several acceptable answers: take the max."""
    return max(metric(pred, g, **kw) for g in golds)


# ----------------------------------------------------------------------------
# pass@k (Chen et al. 2021, "Evaluating Large Language Models Trained on Code")
# ----------------------------------------------------------------------------
def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimate of P(at least one of k samples is correct), given that
    c of n >= k samples were correct.

        pass@k = 1 - C(n-c, k) / C(n, k)        (choose k of the n without a correct one)
               = 1 - prod_{i=n-c+1..n} (1 - k / i)

    The product form never builds huge binomial coefficients, so it is
    numerically stable for n in the hundreds (Chen et al., Figure 3)."""
    if k > n:
        raise ValueError(f"need n >= k samples (n={n}, k={k})")
    if n - c < k:
        return 1.0  # every k-subset contains a correct sample
    return float(1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))


def pass_hat_k(n: int, c: int, k: int) -> float:
    """Unbiased estimate of pass^k = P(ALL k samples are correct) = p^k: the
    reliability metric for agents (tau-bench, Yao et al. 2024; see ../agents/).

        pass^k = C(c, k) / C(n, k) = prod_{i=0..k-1} (c - i) / (n - i)"""
    if k > n:
        raise ValueError(f"need n >= k samples (n={n}, k={k})")
    if c < k:
        return 0.0
    i = np.arange(k)
    return float(np.prod((c - i) / (n - i)))


def pass_at_k_naive(per_sample_p: float, k: int) -> float:
    """The textbook formula if you KNEW the per-sample success rate p."""
    return 1.0 - (1.0 - per_sample_p) ** k


# ----------------------------------------------------------------------------
# Likelihood metrics
# ----------------------------------------------------------------------------
def perplexity(mean_nll_nats: float) -> float:
    """exp(mean negative log-likelihood per token, in nats)."""
    return math.exp(mean_nll_nats)


def bits_per_char(total_nll_nats: float, n_chars: int) -> float:
    """Total NLL in nats -> bits (divide by ln 2) -> per character."""
    return total_nll_nats / LN2 / n_chars


def bits_per_byte(total_nll_nats: float, n_bytes: int) -> float:
    """Like bits_per_char but per UTF-8 byte: comparable across tokenizers,
    because every tokenizer must ultimately explain the same bytes."""
    return total_nll_nats / LN2 / n_bytes


def token_ppl_to_bpb(token_ppl: float, n_tokens: int, n_bytes: int) -> float:
    """Convert a per-TOKEN perplexity to bits per byte: total bits = n_tokens·log2(ppl)."""
    return n_tokens * math.log2(token_ppl) / n_bytes


# ----------------------------------------------------------------------------
# Calibration (Guo et al. 2017, "On Calibration of Modern Neural Networks")
# ----------------------------------------------------------------------------
def bin_edges(conf: np.ndarray, n_bins: int = 10, binning: str = "width") -> np.ndarray:
    """Equal-WIDTH bins split [0, 1] into n_bins intervals. Equal-MASS bins put
    the same number of predictions in every bin (quantiles of the confidences),
    so no bin is nearly empty: less noisy when confidences pile up near 1."""
    if binning == "width":
        return np.linspace(0.0, 1.0, n_bins + 1)
    if binning == "mass":
        edges = np.quantile(conf, np.linspace(0.0, 1.0, n_bins + 1))
        edges[0], edges[-1] = 0.0, 1.0
        return edges
    raise ValueError(binning)


def reliability_data(conf, correct, n_bins: int = 10, binning: str = "width") -> List[Dict]:
    """Per bin: [lo, hi), count, mean confidence, accuracy. The reliability
    diagram plots accuracy against confidence; perfect calibration = diagonal."""
    conf = np.asarray(conf, dtype=float)
    correct = np.asarray(correct, dtype=float)
    edges = bin_edges(conf, n_bins, binning)
    if binning == "mass":
        # Assign by rank so equal-mass bins stay equal even with tied confidences.
        order = np.argsort(conf, kind="stable")
        which = np.empty(len(conf), dtype=int)
        which[order] = np.minimum(np.arange(len(conf)) * n_bins // len(conf), n_bins - 1)
    else:
        which = np.clip(np.searchsorted(edges, conf, side="right") - 1, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = which == b
        n = int(m.sum())
        rows.append(dict(lo=float(edges[b]), hi=float(edges[b + 1]), count=n,
                         conf=float(conf[m].mean()) if n else float("nan"),
                         acc=float(correct[m].mean()) if n else float("nan")))
    return rows


def ece(conf, correct, n_bins: int = 10, binning: str = "width") -> float:
    """Expected calibration error = Σ_bins (n_b / N) · |accuracy_b - confidence_b|."""
    rows = reliability_data(conf, correct, n_bins, binning)
    N = sum(r["count"] for r in rows)
    return float(sum(r["count"] / N * abs(r["acc"] - r["conf"]) for r in rows if r["count"]))


def brier_multiclass(probs: np.ndarray, targets: np.ndarray) -> float:
    """Mean over examples of Σ_classes (p_c - onehot_c)². 0 = perfect, 2 = worst."""
    probs = np.asarray(probs, dtype=float)
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(targets)), targets] = 1.0
    return float(((probs - onehot) ** 2).sum(axis=1).mean())


def brier_binary(conf, correct) -> float:
    """Mean of (confidence - correct)² for a yes/no prediction."""
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    return float(((conf - correct) ** 2).mean())


def softmax_rows(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=-1, keepdims=True)


def temperature_nll(logits: np.ndarray, targets: np.ndarray, beta: float):
    """Mean NLL of softmax(beta · z) and its first two derivatives in beta = 1/T.

        NLL(β)    = mean_i [ -β z_i,y + logsumexp(β z_i) ]
        dNLL/dβ   = mean_i [ -z_i,y + E_p[z_i] ]           (p = softmax(β z_i))
        d²NLL/dβ² = mean_i [ Var_p[z_i] ]  >= 0            convex in β: Newton is safe
    """
    z = np.asarray(logits, float)
    targets = np.asarray(targets)
    p = softmax_rows(z * beta)
    zy = z[np.arange(len(targets)), targets]
    m = (z * beta).max(axis=1)
    lse = m + np.log(np.exp(z * beta - m[:, None]).sum(axis=1))
    nll = float((-beta * zy + lse).mean())
    ez = (p * z).sum(axis=1)
    grad = float((-zy + ez).mean())
    hess = float(((p * z * z).sum(axis=1) - ez ** 2).mean())
    return nll, grad, hess


def fit_temperature(logits: np.ndarray, targets: np.ndarray, iters: int = 50) -> float:
    """Temperature scaling (Guo et al. 2017): one scalar T, fit on held-out
    data by minimizing NLL. It changes confidences but never the argmax, so
    accuracy is unchanged. Newton's method on beta = 1/T."""
    beta = 1.0
    for _ in range(iters):
        _, g, h = temperature_nll(logits, targets, beta)
        step = g / max(h, 1e-12)
        beta = max(beta - step, 1e-3)
        if abs(step) < 1e-10:
            break
    return 1.0 / beta


# ----------------------------------------------------------------------------
# Agreement between two raters (judge vs human, human vs human)
# ----------------------------------------------------------------------------
def raw_agreement(a: Sequence, b: Sequence) -> float:
    return float(np.mean([x == y for x, y in zip(a, b)]))


def cohens_kappa(a: Sequence, b: Sequence, categories: Optional[Sequence] = None) -> float:
    """kappa = (p_o - p_e) / (1 - p_e): agreement beyond what two raters with
    the same label frequencies would reach by chance. 1 = perfect, 0 = chance."""
    cats = list(categories) if categories is not None else sorted(set(a) | set(b))
    idx = {c: i for i, c in enumerate(cats)}
    table = np.zeros((len(cats), len(cats)))
    for x, y in zip(a, b):
        table[idx[x], idx[y]] += 1
    n = table.sum()
    p_o = np.trace(table) / n
    p_e = float((table.sum(axis=1) / n) @ (table.sum(axis=0) / n))
    if p_e == 1.0:
        return 1.0 if p_o == 1.0 else 0.0
    return float((p_o - p_e) / (1.0 - p_e))


# ----------------------------------------------------------------------------
# Error bars for an accuracy
# ----------------------------------------------------------------------------
def accuracy_se(p: float, n: int) -> float:
    """Standard error of a mean of n independent 0/1 scores: sqrt(p(1-p)/n)."""
    return math.sqrt(p * (1.0 - p) / n)


def wilson_interval(k: int, n: int, z: float = 1.959964) -> tuple:
    """Better than p ± z·SE when p is near 0 or 1 or n is small (never leaves [0, 1])."""
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return center - half, center + half


def main():
    print("Answer matching (SQuAD normalization: lowercase, no punctuation, no articles):")
    for pred, gold in [("The Lighthouse.", "lighthouse"), ("a jar of honey", "honey"),
                       ("the", "a"), ("bread and honey", "honey and bread")]:
        print(f"  pred {pred!r:20s} gold {gold!r:18s} EM {exact_match(pred, gold):.0f}  "
              f"normalized EM {normalized_exact_match(pred, gold):.0f}  F1 {token_f1(pred, gold):.2f}")
    print("\npass@k from n = 20 samples with c correct (unbiased, Chen et al. 2021):")
    for c in (1, 4, 10):
        print(f"  c = {c:2d}: " + "  ".join(f"pass@{k} {pass_at_k(20, c, k):.3f}" for k in (1, 5, 10))
              + f"   pass^5 {pass_hat_k(20, c, 5):.4f}")
    print("\nLikelihood: a model that is uniform over 44 characters has")
    print(f"  {math.log(44):.3f} nats/char = {bits_per_char(math.log(44), 1):.3f} bits/char "
          f"= perplexity {perplexity(math.log(44)):.0f}")
    print(f"  a per-token perplexity of 20 at 4 bytes/token = "
          f"{token_ppl_to_bpb(20, 1, 4):.3f} bits per byte")
    rng = np.random.default_rng(0)
    conf = rng.uniform(0.3, 1.0, 5000)
    correct = rng.random(5000) < conf - 0.1
    print("\nCalibration of a simulated model whose confidence is 0.1 too high:")
    print(f"  ECE equal-width {ece(conf, correct):.3f}, equal-mass "
          f"{ece(conf, correct, binning='mass'):.3f}, Brier {brier_binary(conf, correct):.3f}")
    print(f"\nkappa of two raters [A A B B] vs [A B B B]: {cohens_kappa('AABB', 'ABBB'):.2f} "
          f"(raw agreement {raw_agreement('AABB', 'ABBB'):.2f})")
    lo, hi = wilson_interval(70, 100)
    print(f"accuracy 0.70 on n = 100: SE {accuracy_se(0.7, 100):.3f}, 95% CI "
          f"±{1.96 * accuracy_se(0.7, 100):.3f}, Wilson [{lo:.3f}, {hi:.3f}]")


if __name__ == "__main__":
    main()
