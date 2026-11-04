"""
Statistics for evals from scratch: how sure are you that the number is real?
(README sections 5-6; Miller 2024, "Adding Error Bars to Evals".)

  error bars        CLT standard error, percentile bootstrap CI (Efron & Tibshirani),
                    clustered standard errors and the cluster bootstrap when
                    questions come in groups (passages, documents, conversations)
  A vs B            paired bootstrap CI of the difference, paired permutation
                    (sign-flip) test, McNemar's test for 0/1 scores, and the
                    unpaired tests you should NOT use when both models saw the
                    same questions (two-proportion z-test, unpaired permutation)
  planning          minimum detectable effect and required sample size for a
                    given accuracy, significance level and power (paired and not)
  many comparisons  Bonferroni and Holm corrections

Run (prints a worked example on simulated scores):
    python evals/stats.py
"""
from __future__ import annotations

import math
import warnings
from statistics import NormalDist
from typing import Callable, Dict, Optional, Sequence

import numpy as np

warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)

Z = NormalDist()


def z_of(p: float) -> float:
    """The standard-normal quantile, e.g. z_of(0.975) = 1.96."""
    return Z.inv_cdf(p)


# ----------------------------------------------------------------------------
# Error bars for one model
# ----------------------------------------------------------------------------
def mean_se(x: Sequence[float]) -> float:
    """Standard error of the mean, SE = s / sqrt(n) (sample std, ddof=1)."""
    x = np.asarray(x, float)
    return float(x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 1 else float("nan")


def clt_ci(x: Sequence[float], alpha: float = 0.05):
    """mean ± z_{1-alpha/2} · SE: what Miller 2024 recommends reporting by default."""
    m, se = float(np.mean(x)), mean_se(x)
    z = z_of(1 - alpha / 2)
    return m - z * se, m + z * se


def bootstrap_stats(x: Sequence[float], stat: Callable = np.mean, n_boot: int = 10000,
                    seed: int = 0, chunk: int = 1000) -> np.ndarray:
    """Resample the n examples WITH replacement n_boot times; return the statistic
    of every resample. `stat` must accept a (rows, n) array and axis=1."""
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    out = []
    for start in range(0, n_boot, chunk):
        m = min(chunk, n_boot - start)
        idx = rng.integers(0, len(x), size=(m, len(x)))  # (m, n) resampled indices
        out.append(stat(x[idx], axis=1))
    return np.concatenate(out)


def bootstrap_ci(x: Sequence[float], stat: Callable = np.mean, n_boot: int = 10000,
                 alpha: float = 0.05, seed: int = 0):
    """Percentile bootstrap: the alpha/2 and 1-alpha/2 quantiles of the resampled
    statistic. Works for any statistic (median, F1, ratio of sums), no formula needed."""
    boots = bootstrap_stats(x, stat, n_boot, seed)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi), boots


def cluster_bootstrap_stats(x: Sequence[float], clusters: Sequence, n_boot: int = 10000,
                            seed: int = 0) -> np.ndarray:
    """Bootstrap that resamples whole CLUSTERS (passages, documents, conversations)
    with replacement and keeps all of each cluster's examples together. The mean is
    a ratio (sum of scores / number of examples) because clusters differ in size."""
    x = np.asarray(x, float)
    keys = sorted(set(clusters), key=str)
    idx = {k: i for i, k in enumerate(keys)}
    cid = np.array([idx[c] for c in clusters])
    sums = np.bincount(cid, weights=x, minlength=len(keys))
    sizes = np.bincount(cid, minlength=len(keys)).astype(float)
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(keys), size=(n_boot, len(keys)))
    return sums[pick].sum(axis=1) / sizes[pick].sum(axis=1)


def clustered_se(x: Sequence[float], clusters: Sequence) -> float:
    """Cluster-robust SE of the mean (Miller 2024, eq. 4): questions that share a
    passage / document / task are correlated, so they carry less information than
    independent ones. Sum the deviations INSIDE each cluster first:

        SE_clustered = sqrt( Σ_c ( Σ_{i in c} (x_i - x̄) )² ) / n

    With one example per cluster this is the usual SE (population form, ddof=0).
    The cross terms (x_i - x̄)(x_j - x̄) for i != j in the same cluster are what
    make it bigger when cluster members fail or succeed together."""
    x = np.asarray(x, float)
    dev = x - x.mean()
    sums: Dict = {}
    for d, c in zip(dev, clusters):
        sums[c] = sums.get(c, 0.0) + d
    return float(math.sqrt(sum(s * s for s in sums.values())) / len(x))


# ----------------------------------------------------------------------------
# Comparing two models on the SAME examples
# ----------------------------------------------------------------------------
def paired_bootstrap(a: Sequence[float], b: Sequence[float], n_boot: int = 10000,
                     alpha: float = 0.05, seed: int = 0):
    """CI of mean(b - a), resampling EXAMPLES (both models' scores move together,
    so the per-question difficulty cancels). Returns (diff, lo, hi, boot_diffs)."""
    d = np.asarray(b, float) - np.asarray(a, float)
    lo, hi, boots = bootstrap_ci(d, np.mean, n_boot, alpha, seed)
    return float(d.mean()), lo, hi, boots


def paired_permutation_test(a: Sequence[float], b: Sequence[float], n_perm: int = 10000,
                            seed: int = 0) -> float:
    """Two-sided sign-flip test. Under H0 "A and B are exchangeable", the label of
    each pair is arbitrary, so every per-example difference d_i could have had
    either sign. p = fraction of random sign patterns whose |mean| is at least the
    observed one (with the +1 correction so p is never exactly 0)."""
    d = np.asarray(b, float) - np.asarray(a, float)
    obs = abs(d.mean())
    rng = np.random.default_rng(seed)
    hits = 0
    for start in range(0, n_perm, 1000):
        m = min(1000, n_perm - start)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(m, len(d)))
        hits += int((np.abs((signs * d).mean(axis=1)) >= obs - 1e-12).sum())
    return (hits + 1) / (n_perm + 1)


def unpaired_permutation_test(a: Sequence[float], b: Sequence[float], n_perm: int = 10000,
                              seed: int = 0) -> float:
    """Shuffle the A/B labels across ALL 2n scores, ignoring which question each
    came from. Valid but wasteful when the scores are paired (README section 5)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    pooled = np.concatenate([a, b])
    obs = abs(b.mean() - a.mean())
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n_perm):
        p = rng.permutation(pooled)
        hits += abs(p[len(a):].mean() - p[:len(a)].mean()) >= obs - 1e-12
    return (hits + 1) / (n_perm + 1)


def two_proportion_ztest(a: Sequence[float], b: Sequence[float]) -> float:
    """The classic unpaired test for two accuracies (pooled variance), two-sided p."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    p = (a.sum() + b.sum()) / (len(a) + len(b))
    se = math.sqrt(p * (1 - p) * (1 / len(a) + 1 / len(b)))
    if se == 0:
        return 1.0
    z = (b.mean() - a.mean()) / se
    return 2 * (1 - Z.cdf(abs(z)))


def _binom_cdf(k: int, n: int, p: float = 0.5) -> float:
    """P(X <= k) for X ~ Binomial(n, p), summed in log space so n can be large."""
    if k < 0:
        return 0.0
    logs = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
            + i * math.log(p) + (n - i) * math.log(1 - p) for i in range(0, k + 1)]
    m = max(logs)
    return min(1.0, math.exp(m) * sum(math.exp(v - m) for v in logs))


def mcnemar(b: int, c: int, exact: bool = True) -> float:
    """McNemar's test for paired 0/1 outcomes. Only the DISCORDANT pairs matter:
        b = A right, B wrong        c = A wrong, B right
    Under H0 each discordant pair is a fair coin, so b ~ Binomial(b + c, 1/2).
    exact=True:  p = 2 · P(X <= min(b, c))                   (capped at 1)
    exact=False: chi² = (|b - c| - 1)² / (b + c) with 1 dof (continuity-corrected)."""
    n = b + c
    if n == 0:
        return 1.0
    if exact:
        return min(1.0, 2 * _binom_cdf(min(b, c), n))
    chi2 = (abs(b - c) - 1) ** 2 / n
    return math.erfc(math.sqrt(chi2 / 2))  # survival function of chi² with 1 dof


def mcnemar_table(a: Sequence[float], b: Sequence[float]):
    """The 2x2 table of paired 0/1 scores: (both right, A only, B only, both wrong)."""
    a, b = np.asarray(a) > 0.5, np.asarray(b) > 0.5
    return int((a & b).sum()), int((a & ~b).sum()), int((~a & b).sum()), int((~a & ~b).sum())


# ----------------------------------------------------------------------------
# Planning: how many examples do you need?
# ----------------------------------------------------------------------------
def mde_unpaired(n: int, p: float, alpha: float = 0.05, power: float = 0.8) -> float:
    """Smallest accuracy difference an UNPAIRED test detects with the given power,
    with n examples per model at accuracy ≈ p:  (z_a + z_b) · sqrt(2 p (1-p) / n)."""
    return (z_of(1 - alpha / 2) + z_of(power)) * math.sqrt(2 * p * (1 - p) / n)


def mde_paired(n: int, sd_diff: float, alpha: float = 0.05, power: float = 0.8) -> float:
    """Same for a PAIRED test: only the spread of the per-question differences
    matters. For 0/1 scores, sd_diff² ≈ discordant rate (fraction of questions on
    which the models disagree), usually far below 2p(1-p)."""
    return (z_of(1 - alpha / 2) + z_of(power)) * sd_diff / math.sqrt(n)


def required_n_unpaired(p1: float, p2: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Examples per model for a two-proportion z-test to detect p1 vs p2 (Fleiss)."""
    pbar = (p1 + p2) / 2
    num = (z_of(1 - alpha / 2) * math.sqrt(2 * pbar * (1 - pbar))
           + z_of(power) * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2)))
    return math.ceil(num ** 2 / (p1 - p2) ** 2)


def required_n_paired(delta: float, discordant: float, alpha: float = 0.05,
                      power: float = 0.8) -> int:
    """Examples (shared by both models) for McNemar to detect an accuracy gain
    delta when a fraction `discordant` of questions get different outcomes
    (Connor 1987): n = (z_a sqrt(ψ) + z_b sqrt(ψ - δ²))² / δ²."""
    num = z_of(1 - alpha / 2) * math.sqrt(discordant) + z_of(power) * math.sqrt(
        max(discordant - delta ** 2, 1e-12))
    return math.ceil(num ** 2 / delta ** 2)


# ----------------------------------------------------------------------------
# Many comparisons
# ----------------------------------------------------------------------------
def bonferroni(pvals: Sequence[float], alpha: float = 0.05):
    """Reject p_i if p_i < alpha / m. Controls the chance of ANY false alarm."""
    m = len(pvals)
    return [p < alpha / m for p in pvals]


def holm(pvals: Sequence[float], alpha: float = 0.05):
    """Holm (1979) step-down: sort p-values, compare the i-th smallest with
    alpha / (m - i), stop at the first failure. Same guarantee as Bonferroni,
    never less powerful."""
    m = len(pvals)
    order = np.argsort(pvals)
    reject = [False] * m
    for rank, i in enumerate(order):
        if pvals[i] < alpha / (m - rank):
            reject[i] = True
        else:
            break
    return reject


def compare_paired(a: Sequence[float], b: Sequence[float], alpha: float = 0.05,
                   n_boot: int = 10000, seed: int = 0, binary: Optional[bool] = None) -> Dict:
    """Everything the regression gate needs about B - A on the same examples."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if binary is None:
        binary = bool(np.isin(a, (0.0, 1.0)).all() and np.isin(b, (0.0, 1.0)).all())
    diff, lo, hi, _ = paired_bootstrap(a, b, n_boot, alpha, seed)
    out = dict(n=len(a), mean_a=float(a.mean()), mean_b=float(b.mean()), diff=diff,
               ci=(lo, hi), se_paired=mean_se(b - a),
               se_unpaired=math.sqrt(mean_se(a) ** 2 + mean_se(b) ** 2),
               corr=float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else 0.0)
    if binary:
        both, a_only, b_only, neither = mcnemar_table(a, b)
        out.update(test="McNemar (exact)", p=mcnemar(a_only, b_only),
                   table=(both, a_only, b_only, neither),
                   p_unpaired=two_proportion_ztest(a, b))
    else:
        out.update(test="paired permutation", p=paired_permutation_test(a, b, seed=seed),
                   p_unpaired=unpaired_permutation_test(a, b, n_perm=2000, seed=seed))
    return out


def main():
    rng = np.random.default_rng(0)
    n = 200
    difficulty = rng.random(n)  # shared by both models: this is why pairing helps
    a = (rng.random(n) > difficulty * 0.9).astype(float)
    u = rng.random(n)
    b = np.where((a == 0) & (u < 0.3), 1.0, np.where((a == 1) & (u < 0.05), 0.0, a))
    print(f"Simulated: {n} questions, model A acc {a.mean():.3f}, model B acc {b.mean():.3f}")
    lo, hi = clt_ci(a)
    blo, bhi, _ = bootstrap_ci(a)
    print(f"  A: CLT 95% CI [{lo:.3f}, {hi:.3f}], bootstrap [{blo:.3f}, {bhi:.3f}]")
    r = compare_paired(a, b)
    print(f"  B - A = {r['diff']:+.3f}, paired 95% CI [{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}]")
    print(f"  table (both, A only, B only, neither) = {r['table']}")
    print(f"  {r['test']} p = {r['p']:.2g}; unpaired z-test p = {r['p_unpaired']:.2g}")
    print(f"  SE of the difference: paired {r['se_paired']:.4f} vs unpaired {r['se_unpaired']:.4f}")
    clusters = np.argsort(np.argsort(difficulty)) // 10  # 20 groups of similar difficulty
    print(f"  if questions came in 20 groups of 10 similar ones: SE {mean_se(a):.4f} -> clustered "
          f"{clustered_se(a, clusters):.4f}")
    print(f"\nPlanning at 70% accuracy, alpha 0.05, power 0.8:")
    print(f"  unpaired: n = {required_n_unpaired(0.70, 0.75)} per model to detect +5 points")
    print(f"  paired, 10% of questions discordant: n = {required_n_paired(0.05, 0.10)}")
    print(f"  with n = 500: MDE unpaired {mde_unpaired(500, 0.7):.3f}, paired "
          f"{mde_paired(500, math.sqrt(0.10)):.3f}")
    ps = [0.01, 0.015, 0.03, 0.2]
    print(f"\nFour tasks with p = {ps}: Bonferroni rejects {bonferroni(ps)}, Holm {holm(ps)}")


if __name__ == "__main__":
    main()
