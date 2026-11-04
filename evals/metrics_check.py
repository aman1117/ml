"""
Metrics check: every metric and statistical test in metrics.py and stats.py
against an independent answer (brute force, simulation, or a known value).
The analog of ../transformer/gradient_check.py: if a formula is wrong, a check
fails. Try it: break a line (e.g. drop the "+ 1" in pass_at_k's np.arange, or
use ddof=1 in clustered_se) and re-run.

    EM / F1              hand-computed SQuAD cases, including the article trap
    pass@k, pass^k       == enumerating every k-subset of n samples (small n),
                         == Monte Carlo, and stable where C(n, k) overflows floats
    likelihood           nats -> bits -> per char / per byte / per token conversions
    calibration          ECE of a perfectly calibrated simulator ≈ 0, of an
                         overconfident one ≈ its gap; equal-mass bins are equal;
                         Brier on known values; temperature-scaling derivatives
                         vs finite differences, and the fit recovers a known T
    agreement            Cohen's kappa on textbook tables
    bootstrap            95% CIs cover the true accuracy ≈ 95% of the time
    permutation test     p-values uniform under H0 (KS distance), small under an
                         effect, and equal to exact enumeration for small n
    McNemar              exact binomial and chi² p-values on known tables
    clustered SE         == plain SE for singleton clusters; matches the true
                         spread of the mean on simulated clustered data
    sample size          a test run at the computed n has ≈ the promised power

Run (~5 s):
    python evals/metrics_check.py
"""
from __future__ import annotations

import itertools
import math
import time
import warnings

import numpy as np

import metrics as M
import stats as S

warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)

RESULTS = []


def check(name: str, ok: bool, detail: str = ""):
    RESULTS.append(bool(ok))
    print(f"  {name:62s} {detail:32s} {'OK' if ok else 'FAIL'}")


def close(a, b, tol=1e-9):
    return abs(a - b) <= tol


# ----------------------------------------------------------------------------
def check_matching():
    print("Answer matching (SQuAD definitions)")
    check("normalized EM: 'The Cat!' == 'cat'", M.normalized_exact_match("The Cat!", "cat") == 1.0)
    check("strict EM: 'The Cat!' != 'cat'", M.exact_match("The Cat!", "cat") == 0.0)
    f1 = M.token_f1("the cat sat", "a cat sat on the mat")  # 2 shared; P = 2/2, R = 2/4
    check("F1('the cat sat', 'a cat sat on the mat') = 2/3", close(f1, 2 / 3), f"{f1:.4f}")
    f1 = M.token_f1("honey honey bread", "honey bread bread")  # Counter overlap = 2
    check("F1 counts multiplicities: 2 shared of 3 and 3 -> 2/3", close(f1, 2 / 3), f"{f1:.4f}")
    check("article trap: 'the' vs 'a' match after SQuAD normalization",
          M.normalized_exact_match("the", "a") == 1.0
          and M.normalized_exact_match("the", "a", remove_articles=False) == 0.0)
    check("F1 of two empty answers is 1, empty vs non-empty is 0",
          M.token_f1("the", "a") == 1.0 and M.token_f1("", "honey") == 0.0)
    check("max over several gold answers",
          M.best_over_golds(M.normalized_exact_match, "grey", ["gray", "grey"]) == 1.0)


def check_pass_at_k():
    print("\npass@k and pass^k (Chen et al. 2021)")
    worst = 0.0
    worst_hat = 0.0
    for n in range(1, 9):
        for c in range(n + 1):
            samples = [1] * c + [0] * (n - c)
            for k in range(1, n + 1):
                subsets = list(itertools.combinations(samples, k))
                brute = np.mean([any(s) for s in subsets])
                brute_hat = np.mean([all(s) for s in subsets])
                worst = max(worst, abs(M.pass_at_k(n, c, k) - brute))
                worst_hat = max(worst_hat, abs(M.pass_hat_k(n, c, k) - brute_hat))
    check("pass@k == enumerate all k-subsets (every n <= 8, c, k)", worst < 1e-12,
          f"max |diff| {worst:.1e}")
    check("pass^k == enumerate all k-subsets (every n <= 8, c, k)", worst_hat < 1e-12,
          f"max |diff| {worst_hat:.1e}")
    rng = np.random.default_rng(0)
    n, c, k, trials = 20, 3, 5, 200000
    keys = rng.random((trials, n)).argsort(axis=1)[:, :k]  # random k-subsets of 20 samples
    mc = float((keys < c).any(axis=1).mean())  # samples 0..c-1 are the correct ones
    check(f"pass@{k} (n={n}, c={c}) == Monte Carlo over {trials:,} draws",
          abs(mc - M.pass_at_k(n, c, k)) < 0.005, f"{M.pass_at_k(n, c, k):.4f} vs {mc:.4f}")
    n, c, k = 3000, 20, 300
    exact = 1 - math.comb(n - c, k) / math.comb(n, k)  # Python's exact big integers
    got = M.pass_at_k(n, c, k)
    try:
        float(math.comb(n, k))
        overflow = "no"
    except OverflowError:
        overflow = "C(n,k) overflows"
    check(f"stable at n={n}, k={k}: product form == exact integers", close(got, exact, 1e-12),
          f"{got:.6f}; float {overflow}")
    check("pass@k with 1 sample is the plain success rate", close(M.pass_at_k(10, 3, 1), 0.3))


def check_likelihood():
    print("\nLikelihood conversions")
    n = 1000
    total = n * math.log(44)  # a uniform model over 44 characters
    check("uniform over 44 chars: log2(44) bits/char, perplexity 44",
          close(M.bits_per_char(total, n), math.log2(44)) and close(M.perplexity(total / n), 44.0),
          f"{M.bits_per_char(total, n):.4f} bits")
    text = "héllo wörld"  # 11 characters, 13 UTF-8 bytes
    nll = 17.3
    nb = len(text.encode("utf-8"))
    check("bits/char · chars == bits/byte · bytes (same total bits)",
          close(M.bits_per_char(nll, len(text)) * len(text), M.bits_per_byte(nll, nb) * nb),
          f"{len(text)} chars, {nb} bytes")
    lp = np.log(np.random.default_rng(1).uniform(0.05, 0.9, 25))  # 25 token log-probs
    ppl = math.exp(-lp.mean())
    check("token perplexity -> bits/byte == total NLL -> bits/byte",
          close(M.token_ppl_to_bpb(ppl, 25, 80), M.bits_per_byte(-lp.sum(), 80)))


def check_calibration():
    print("\nCalibration")
    rng = np.random.default_rng(0)
    conf = rng.uniform(0.0, 1.0, 200000)
    correct = rng.random(200000) < conf  # right exactly as often as it claims
    e_w, e_m = M.ece(conf, correct), M.ece(conf, correct, binning="mass")
    check("perfectly calibrated simulator: ECE ≈ 0 (both binnings)", e_w < 0.005 and e_m < 0.005,
          f"width {e_w:.4f}, mass {e_m:.4f}")
    conf2 = rng.uniform(0.3, 1.0, 200000)
    correct2 = rng.random(200000) < conf2 - 0.1
    e2 = M.ece(conf2, correct2)
    check("overconfident by 0.10 everywhere: ECE ≈ 0.10", abs(e2 - 0.10) < 0.005, f"{e2:.4f}")
    counts = [r["count"] for r in M.reliability_data(conf2[:1000], correct2[:1000], 10, "mass")]
    check("equal-mass bins hold equal counts", len(set(counts)) == 1, f"{counts[:3]}...")
    b = M.brier_multiclass(np.array([[1.0, 0.0], [0.5, 0.5]]), np.array([0, 1]))
    check("Brier([[1,0],[.5,.5]], targets [0,1]) = 0.25", close(b, 0.25), f"{b:.4f}")
    check("binary Brier: always say 0.5 -> 0.25", close(M.brier_binary([0.5] * 4, [1, 0, 1, 0]), 0.25))
    z = rng.standard_normal((500, 6)) * 3
    t = rng.integers(0, 6, 500)
    beta, h = 0.7, 1e-5
    f, g, hess = M.temperature_nll(z, t, beta)
    fp, gp, _ = M.temperature_nll(z, t, beta + h)
    fm, gm, _ = M.temperature_nll(z, t, beta - h)
    num_g, num_h = (fp - fm) / (2 * h), (gp - gm) / (2 * h)
    check("temperature scaling: dNLL/dβ, d²NLL/dβ² vs finite diff",
          abs(g - num_g) < 1e-7 and abs(hess - num_h) < 1e-6,
          f"{abs(g - num_g):.1e}, {abs(hess - num_h):.1e}")
    logits = rng.standard_normal((20000, 8)) * 4
    probs = M.softmax_rows(logits / 2.0)  # the TRUE distribution has temperature 2
    labels = (probs.cumsum(1) < rng.random((20000, 1))).sum(1)
    T = M.fit_temperature(logits, labels)
    check("fit_temperature recovers the true temperature 2.0", abs(T - 2.0) < 0.05, f"T = {T:.3f}")


def check_agreement():
    print("\nAgreement")
    # The textbook example: 50 items, yes/yes 20, yes/no 5, no/yes 10, no/no 15.
    a = ["y"] * 25 + ["n"] * 25
    b = ["y"] * 20 + ["n"] * 5 + ["y"] * 10 + ["n"] * 15
    k = M.cohens_kappa(a, b)
    check("Cohen's kappa on the 20/5/10/15 table = 0.4", close(k, 0.4), f"{k:.4f}")
    check("raw agreement on that table = 0.7", close(M.raw_agreement(a, b), 0.7))
    check("identical raters: kappa = 1", close(M.cohens_kappa(a, a), 1.0))
    rng = np.random.default_rng(0)
    x, y = rng.integers(0, 3, 20000), rng.integers(0, 3, 20000)
    k0 = M.cohens_kappa(x.tolist(), y.tolist())
    check("independent raters: kappa ≈ 0", abs(k0) < 0.02, f"{k0:+.4f}")


def check_bootstrap():
    print("\nBootstrap confidence intervals")
    rng = np.random.default_rng(0)
    p, n, sims = 0.7, 100, 400
    cover = 0
    for i in range(sims):
        x = (rng.random(n) < p).astype(float)
        lo, hi, _ = S.bootstrap_ci(x, n_boot=1000, seed=i)
        cover += lo <= p <= hi
    check(f"95% CI covers the true accuracy ({sims} datasets, n={n})",
          0.92 <= cover / sims <= 0.975, f"coverage {cover / sims:.3f}")
    x = rng.standard_normal(400)
    lo, hi, _ = S.bootstrap_ci(x, n_boot=4000)
    clo, chi = S.clt_ci(x)
    check("bootstrap CI ≈ CLT CI for the mean of 400 normal draws",
          abs(lo - clo) < 0.02 and abs(hi - chi) < 0.02, f"[{lo:.3f}, {hi:.3f}] vs [{clo:.3f}, {chi:.3f}]")


def check_permutation():
    print("\nPaired permutation test")
    rng = np.random.default_rng(0)
    ps = []
    for i in range(300):
        a = rng.standard_normal(40)
        b = a + rng.standard_normal(40)  # same mean: H0 is true
        ps.append(S.paired_permutation_test(a, b, n_perm=500, seed=i))
    ps = np.sort(ps)
    ks = float(np.max(np.abs(ps - (np.arange(1, 301) / 300))))
    check("H0 true: p-values ~ Uniform(0, 1) (KS distance, 300 tests)", ks < 0.1, f"KS D = {ks:.3f}")
    rate = float(np.mean(ps < 0.05))
    check("H0 true: false-alarm rate at alpha = 0.05 ≈ 5%", 0.02 <= rate <= 0.09, f"{rate:.3f}")
    eff = [S.paired_permutation_test(a, a + 0.6 + rng.standard_normal(40), n_perm=500, seed=i)
           for i, a in enumerate(rng.standard_normal((50, 40)))]
    check("real effect (+0.6 sd, n=40): p-values small", np.median(eff) < 0.01,
          f"median p {np.median(eff):.4f}")
    d = rng.standard_normal(10) + 0.5
    signs = np.array(list(itertools.product([-1, 1], repeat=10)))
    exact = float(np.mean(np.abs((signs * d).mean(1)) >= abs(d.mean()) - 1e-12))
    mc = S.paired_permutation_test(np.zeros(10), d, n_perm=20000)
    check("n=10: Monte Carlo p == exact enumeration of 1024 sign flips",
          abs(mc - exact) < 0.01, f"{mc:.4f} vs {exact:.4f}")


def check_mcnemar():
    print("\nMcNemar's test")
    p = S.mcnemar(10, 2)
    check("exact, b=10 c=2: p = 2·(1+12+66)/4096 = 0.03857", close(p, 2 * 79 / 4096, 1e-12), f"{p:.5f}")
    p = S.mcnemar(10, 2, exact=False)
    check("chi², b=10 c=2: (|10-2|-1)²/12 = 4.083 -> p = 0.04331", abs(p - 0.043308) < 1e-5, f"{p:.5f}")
    check("exact, b=0 c=5: p = 2/32 = 0.0625", close(S.mcnemar(0, 5), 0.0625))
    check("no disagreement or b == c: p = 1", S.mcnemar(0, 0) == 1.0 and S.mcnemar(7, 7) == 1.0)
    big = S.mcnemar(600, 540)
    check("large table (b+c = 1140) without overflow", 0.07 < big < 0.09, f"p = {big:.4f}")
    a, b = np.array([1, 1, 0, 0, 1, 0]), np.array([1, 0, 1, 1, 1, 0])
    check("2x2 table from paired scores", S.mcnemar_table(a, b) == (2, 1, 2, 1))


def check_clustered():
    print("\nClustered standard errors (Miller 2024)")
    rng = np.random.default_rng(0)
    x = rng.random(200)
    se1 = S.clustered_se(x, np.arange(200))
    check("one example per cluster == plain SE (ddof = 0)", close(se1, x.std() / math.sqrt(200), 1e-12))
    G, m, sims = 40, 10, 2000
    means, se_naive, se_cl = [], [], []
    for _ in range(sims):
        u = np.repeat(rng.standard_normal(G) * 0.15, m)  # shared passage difficulty
        y = (rng.random(G * m) < 0.6 + u).astype(float)
        means.append(y.mean())
        se_naive.append(S.mean_se(y))
        se_cl.append(S.clustered_se(y, np.repeat(np.arange(G), m)))
    true_sd = float(np.std(means))
    r_cl, r_nv = np.mean(se_cl) / true_sd, np.mean(se_naive) / true_sd
    check("clustered data: clustered SE ≈ true spread of the mean", 0.9 < r_cl < 1.05,
          f"ratio {r_cl:.2f}")
    check("clustered data: naive SE underestimates it", r_nv < 0.75, f"ratio {r_nv:.2f}")


def check_power():
    print("\nSample size and power")
    rng = np.random.default_rng(0)
    n = S.required_n_unpaired(0.70, 0.80)
    sims = 2000
    rej = 0
    for _ in range(sims):
        a = (rng.random(n) < 0.70).astype(float)
        b = (rng.random(n) < 0.80).astype(float)
        rej += S.two_proportion_ztest(a, b) < 0.05
    check(f"unpaired: n = {n}/model -> 80% power for 70% vs 80%", 0.76 <= rej / sims <= 0.85,
          f"simulated power {rej / sims:.3f}")
    delta, psi = 0.05, 0.15
    n = S.required_n_paired(delta, psi)
    rej = 0
    for _ in range(sims):
        u = rng.random(n)
        b_only = int((u < (psi + delta) / 2).sum())  # new right, old wrong
        a_only = int(((u >= (psi + delta) / 2) & (u < psi)).sum())
        rej += S.mcnemar(a_only, b_only, exact=False) < 0.05
    check(f"paired: n = {n} -> 80% power, +5 pts, 15% discordant",
          0.74 <= rej / sims <= 0.86, f"simulated power {rej / sims:.3f}")
    mde = S.mde_unpaired(n=S.required_n_unpaired(0.70, 0.80), p=0.75)
    check("MDE at the required n ≈ the effect it was sized for", abs(mde - 0.10) < 0.01, f"{mde:.4f}")
    check("Holm rejects 2, Bonferroni 1 of p = [.01, .015, .03, .2]",
          S.holm([0.01, 0.015, 0.03, 0.2]) == [True, True, False, False]
          and S.bonferroni([0.01, 0.015, 0.03, 0.2]) == [True, False, False, False])


def main():
    t0 = time.time()
    for fn in (check_matching, check_pass_at_k, check_likelihood, check_calibration,
               check_agreement, check_bootstrap, check_permutation, check_mcnemar,
               check_clustered, check_power):
        fn()
    ok = all(RESULTS)
    print(f"\n{sum(RESULTS)}/{len(RESULTS)} checks passed in {time.time() - t0:.1f} s")
    print("Metrics check", "PASSED" if ok else "FAILED")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
