"""
Production monitoring from the request logs, from scratch: what an on-call
engineer (and the dashboards and pagers behind them) computes from
serving/logs/requests.jsonl.

  1. SLIs and SLOs: availability, latency p50/p95/p99, TTFT, tokens/s, and the
     ERROR BUDGET: with a 99% availability SLO, 1% of requests may fail; the burn
     rate is (observed error rate) / 1%. Burn rate 1 spends the budget exactly at
     the end of the SLO period; 14.4 spends 2% of a 30-day budget in one hour.
  2. A FAILURE-MODE TAXONOMY: every request is classified (timeouts, 4xx validation,
     429 overload, slow clients, disconnects, empty output, degenerate repetition,
     max-token truncation, unknown-character inputs, context truncation, latency
     outliers) with counts, first-seen times and examples.
  3. QUALITY BY MODEL VERSION: failure rates per version with a two-proportion
     z-test, which is how a bad canary gets caught.
  4. DRIFT of the input distribution: prompt length and character histogram of a
     reference window vs later windows, with PSI and KL divergence.
  5. ALERT RULES: thresholds and multi-window burn rates (Google SRE workbook),
     printed as FIRING / ok.
  6. A markdown report (serving/logs/report.md) and an export of failing requests
     as eval cases (serving/logs/eval_cases.jsonl) for ../evals/: production
     failures become regression tests.

Run (after python serving/load_test.py --spawn-server):
    python serving/monitor.py
    python serving/monitor.py --log serving/logs/requests.jsonl --slo 0.995
    python serving/monitor.py --synthetic        # on a generated log with known failures
"""
from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter, OrderedDict
from datetime import datetime

import numpy as np

from common import LOG_DIR, loop_period, percentile, repetition_ratio

# Failure modes: (name, description, counts against the availability SLO?)
MODES = OrderedDict([
    ("timeout", ("request exceeded its timeout_s (504)", True)),
    ("validation_4xx", ("rejected by request validation (400/411/413)", False)),
    ("overload_429", ("queue full, backpressure (429)", True)),
    ("slow_client", ("body not received within the read timeout (408)", False)),
    ("client_disconnect", ("client hung up mid-request (499)", False)),
    ("server_error_5xx", ("internal / engine error (500/503)", True)),
    ("empty_output", ("200 but the completion is empty or whitespace", False)),
    ("degenerate_repetition", ("output stuck in a loop (repeating suffix or 4-gram "
                               "repetition >= 0.35)", False)),
    ("max_token_truncation", ("a stop sequence was requested but max_tokens hit first",
                              False)),
    ("unknown_chars", ("prompt contains characters the model never saw (dropped)", False)),
    ("context_truncation", ("prompt longer than the 32-token context (only the end used)",
                            False)),
    ("latency_outlier", ("latency per output token > median + 4 robust SDs (>= 10 tokens)",
                         False)),
])
REP_THRESHOLD = 0.35


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
def parse_ts(s):
    return datetime.fromisoformat(s).timestamp()


def load_logs(path):
    """Records sorted by time. Malformed lines are counted, not fatal: a monitoring
    job that crashes on one bad line is blind exactly when things go wrong."""
    recs, bad = [], 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
                r["t"] = parse_ts(r["ts"])
                recs.append(r)
            except (ValueError, KeyError, TypeError):
                bad += 1
    recs.sort(key=lambda r: r["t"])
    return recs, bad


def ok(r):
    return r["status"] == 200 and not r.get("error_type")


# ----------------------------------------------------------------------------
# 1. SLIs, SLOs, error budget
# ----------------------------------------------------------------------------
def counts_against_slo(r):
    """Valid for the availability SLI = everything except the client's own faults."""
    return r["status"] not in (400, 408, 411, 413, 499)


def slis(recs, slo=0.99, latency_slo_s=0.5):
    valid = [r for r in recs if counts_against_slo(r)]
    good = [r for r in valid if ok(r)]
    lat = [r["latency_s"] for r in good]
    ttft = [r["ttft_s"] for r in good if r.get("ttft_s") is not None]
    span = max(recs[-1]["t"] - recs[0]["t"], 1e-9)
    toks = sum(r.get("output_tokens") or 0 for r in good)
    avail = len(good) / max(1, len(valid))
    err = 1 - avail
    return dict(n=len(recs), valid=len(valid), good=len(good), availability=avail,
                error_rate=err, slo=slo, budget_burn=err / (1 - slo),
                budget_remaining=1 - err / (1 - slo), span_s=span,
                req_s=len(recs) / span, tok_s=toks / span,
                latency={q: percentile(lat, q) for q in (50, 95, 99)},
                ttft={q: percentile(ttft, q) for q in (50, 95, 99)},
                latency_slo_s=latency_slo_s,
                latency_good=float(np.mean([x <= latency_slo_s for x in lat])) if lat else 1.0)


def burn_rates(recs, window_s, slo, step_s=0.5):
    """Burn rate of the availability SLO over a sliding window, evaluated every step_s."""
    valid = [r for r in recs if counts_against_slo(r)]
    t = np.array([r["t"] for r in valid])
    bad = np.array([not ok(r) for r in valid], dtype=float)
    t0 = recs[0]["t"]
    times = np.arange(t0 + step_s, recs[-1]["t"] + step_s, step_s)
    out = []
    for now in times:
        m = (t > now - window_s) & (t <= now)
        out.append(bad[m].mean() / (1 - slo) if m.any() else 0.0)
    return times - t0, np.array(out)


# ----------------------------------------------------------------------------
# 2. Failure-mode taxonomy
# ----------------------------------------------------------------------------
def latency_outlier_threshold(recs):
    """Robust: median + 4·1.4826·MAD of log(latency per output token) over successes."""
    x = np.log([r["latency_s"] / r["output_tokens"] for r in recs
                if ok(r) and (r.get("output_tokens") or 0) >= 10])
    if len(x) < 10:
        return float("inf")
    med = np.median(x)
    mad = 1.4826 * np.median(np.abs(x - med))
    return float(np.exp(med + 4 * max(mad, 1e-6)))


def classify(r, lat_thr):
    """All failure modes of one request (a request can have several)."""
    modes, s, e = [], r["status"], r.get("error_type")
    if e == "timeout" or s == 504:
        modes.append("timeout")
    elif s in (400, 411, 413):
        modes.append("validation_4xx")
    elif s == 429:
        modes.append("overload_429")
    elif s == 408:
        modes.append("slow_client")
    elif s == 499:
        modes.append("client_disconnect")
    elif s >= 500:
        modes.append("server_error_5xx")
    out = r.get("output")
    if ok(r):
        if out is not None and not out.strip():
            modes.append("empty_output")
        n = r.get("output_tokens") or 0
        rep = r.get("rep4") if r.get("rep4") is not None else repetition_ratio(out or "")
        loop = r.get("loop_period") if r.get("loop_period") is not None else loop_period(out or "")
        if n >= 20 and (loop or rep >= REP_THRESHOLD):
            modes.append("degenerate_repetition")
        if r.get("stop") and r.get("finish_reason") == "length":
            modes.append("max_token_truncation")
        if n >= 10 and r["latency_s"] / n > lat_thr:
            modes.append("latency_outlier")
    if (r.get("unknown_chars") or 0) > 0:
        modes.append("unknown_chars")
    if r.get("prompt_truncated"):
        modes.append("context_truncation")
    return modes


def taxonomy(recs):
    thr = latency_outlier_threshold(recs)
    t0 = recs[0]["t"]
    tax = OrderedDict((m, dict(count=0, first_seen=None, examples=[], subtypes=Counter()))
                      for m in MODES)
    for r in recs:
        r["modes"] = classify(r, thr)
        for m in r["modes"]:
            d = tax[m]
            d["count"] += 1
            if d["first_seen"] is None:
                d["first_seen"] = r["t"] - t0
            if m in ("validation_4xx", "server_error_5xx"):
                d["subtypes"][r.get("error_type")] += 1
            if len(d["examples"]) < 3:
                d["examples"].append(r)
    return tax, thr


# ----------------------------------------------------------------------------
# 3. Quality by model version
# ----------------------------------------------------------------------------
def two_proportion_z(k1, n1, k2, n2):
    """z for H0: p1 = p2 (pooled), and the two-sided p-value."""
    if n1 == 0 or n2 == 0:
        return 0.0, 1.0
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(max(p * (1 - p) * (1 / n1 + 1 / n2), 1e-300))
    z = (k1 / n1 - k2 / n2) / se
    return z, math.erfc(abs(z) / math.sqrt(2))


def by_version(recs):
    out = OrderedDict()
    for v in sorted({r["model_version"] for r in recs}):
        mine = [r for r in recs if r["model_version"] == v]
        good = [r for r in mine if ok(r)]
        rest = [r for r in recs if r["model_version"] != v and ok(r)]
        k = sum("degenerate_repetition" in r["modes"] for r in good)
        k2 = sum("degenerate_repetition" in r["modes"] for r in rest)
        z, p = two_proportion_z(k, len(good), k2, len(rest))
        out[v] = dict(n=len(mine), ok=len(good), first=min(r["t"] for r in mine) - recs[0]["t"],
                      p95=percentile([r["latency_s"] for r in good], 95),
                      mean_rep4=float(np.mean([r["rep4"] for r in good if
                                               (r.get("output_tokens") or 0) >= 20] or [0])),
                      degenerate=k, degenerate_rate=k / max(1, len(good)), z=z, p=p)
    return out


# ----------------------------------------------------------------------------
# 4. Drift
# ----------------------------------------------------------------------------
def psi(ref, cur, eps=1e-4):
    """Population Stability Index: Σ (c - r) ln(c / r) over bins (proportions).
    It is the symmetrized KL divergence KL(c||r) + KL(r||c). Rule of thumb:
    < 0.1 stable, 0.1-0.25 moderate shift, > 0.25 major shift."""
    r = np.asarray(ref, float) + eps
    c = np.asarray(cur, float) + eps
    r, c = r / r.sum(), c / c.sum()
    return float(((c - r) * np.log(c / r)).sum())


def kl(p, q, eps=1e-4):
    p = np.asarray(p, float) + eps
    q = np.asarray(q, float) + eps
    p, q = p / p.sum(), q / q.sum()
    return float((p * np.log(p / q)).sum())


VOCAB_GROUPS = ("lower", "upper", "space", "punct", "digit", "other")


def char_group(c):
    if c.islower() and c.isascii():
        return "lower"
    if c.isupper() and c.isascii():
        return "upper"
    if c.isspace():
        return "space"
    if c.isdigit():
        return "digit"
    if c in ".,;:!?'\"-()":
        return "punct"
    return "other"


def length_bins(ref_lengths, n_bins=10):
    """Bin edges at the reference quantiles: each bin holds ~10% of the reference."""
    edges = np.unique(np.quantile(ref_lengths, np.linspace(0, 1, n_bins + 1)[1:-1]))
    return edges


def features(recs, edges, alphabet):
    lengths = [r["prompt_chars"] for r in recs if r.get("prompt_chars")]
    len_hist = np.bincount(np.searchsorted(edges, lengths, side="right"),
                           minlength=len(edges) + 1)
    text = "".join(r.get("prompt") or "" for r in recs)
    idx = {c: i for i, c in enumerate(alphabet)}
    char_hist = np.zeros(len(alphabet) + 1)
    for c in text:
        char_hist[idx.get(c, len(alphabet))] += 1  # last bin: never seen in the reference
    groups = Counter(char_group(c) for c in text)
    return len_hist, char_hist, np.array([groups[g] for g in VOCAB_GROUPS], float)


def drift(recs, ref_frac=0.3, n_windows=10):
    """Reference = the first ref_frac of the run; compare it with the last window and
    with every window over time."""
    t0, t1 = recs[0]["t"], recs[-1]["t"]
    inputs = [r for r in recs if r.get("prompt_chars")]
    ref = [r for r in inputs if r["t"] <= t0 + ref_frac * (t1 - t0)]
    edges = length_bins([r["prompt_chars"] for r in ref])
    alphabet = sorted(set("".join(r.get("prompt") or "" for r in ref)))
    ref_f = features(ref, edges, alphabet)
    windows = []
    bounds = np.linspace(t0, t1 + 1e-6, n_windows + 1)
    for a, b in zip(bounds[:-1], bounds[1:]):
        w = [r for r in inputs if a <= r["t"] < b]
        if not w:
            continue
        f = features(w, edges, alphabet)
        windows.append(dict(t=(a + b) / 2 - t0, n=len(w), psi_len=psi(ref_f[0], f[0]),
                            psi_chars=psi(ref_f[1], f[1]), kl_chars=kl(f[1], ref_f[1]),
                            psi_groups=psi(ref_f[2], f[2]),
                            mean_len=float(np.mean([r["prompt_chars"] for r in w])),
                            unknown_rate=float(np.mean([(r.get("unknown_chars") or 0) > 0
                                                        for r in w]))))
    cur = [r for r in inputs if r["t"] >= bounds[-2]]
    cur_f = features(cur, edges, alphabet)
    return dict(ref_n=len(ref), cur_n=len(cur), edges=edges, windows=windows,
                psi_len=psi(ref_f[0], cur_f[0]), psi_chars=psi(ref_f[1], cur_f[1]),
                kl_chars=kl(cur_f[1], ref_f[1]), psi_groups=psi(ref_f[2], cur_f[2]),
                ref_len=ref_f[0], cur_len=cur_f[0], ref_groups=ref_f[2], cur_groups=cur_f[2],
                ref_mean_len=float(np.mean([r["prompt_chars"] for r in ref])),
                cur_mean_len=float(np.mean([r["prompt_chars"] for r in cur])))


# ----------------------------------------------------------------------------
# 5. Alerts
# ----------------------------------------------------------------------------
# (name, long window, short window, burn-rate threshold). The SRE workbook pages on
# 1 h & 5 min > 14.4 and on 6 h & 30 min > 6; our logs span seconds, so the windows
# are scaled down (the thresholds are not).
BURN_RULES = (("fast burn (page)", 5.0, 1.0, 14.4), ("slow burn (page)", 15.0, 3.0, 6.0))


def alerts(recs, s, tax, versions, dr, slo, rules=BURN_RULES):
    """Each rule: (name, firing, detail). Multi-window burn rate: fire only if BOTH the
    long window (the problem is big enough to matter) and the short window (it is
    still happening) burn faster than the threshold."""
    out = []
    for name, long_w, short_w, thr in rules:
        tl, bl = burn_rates(recs, long_w, slo)
        _, bs = burn_rates(recs, short_w, slo)
        both = (bl > thr) & (bs > thr)
        first = f"first at t = {tl[both.argmax()]:.1f} s; " if both.any() else ""
        out.append((f"availability {name}", bool(both.any()),
                    f"{first}max burn {bl.max():.1f} over {long_w:g} s, {bs.max():.1f} over "
                    f"{short_w:g} s; threshold {thr}"))
    out.append(("error budget exhausted (ticket)", s["budget_burn"] > 1,
                f"{s['budget_burn']:.1f}x the budget used over the whole log"))
    out.append(("p95 latency > SLO", s["latency"][95] > s["latency_slo_s"],
                f"p95 {s['latency'][95] * 1e3:.0f} ms vs {s['latency_slo_s'] * 1e3:.0f} ms"))
    n = len(recs)
    r429 = tax["overload_429"]["count"] / n
    out.append(("overload (429 rate > 2%)", r429 > 0.02, f"{r429:.1%} of requests"))
    r5 = tax["server_error_5xx"]["count"] / n
    out.append(("server errors (5xx > 0.5%)", r5 > 0.005, f"{r5:.1%}"))
    for v, d in versions.items():
        if len(versions) > 1:
            fire = d["degenerate_rate"] > 0.02 and d["z"] > 0 and d["p"] < 0.01
            out.append((f"quality regression {v}", fire,
                        f"degenerate {d['degenerate']}/{d['ok']} = {d['degenerate_rate']:.1%}, "
                        f"z = {d['z']:+.1f}, p = {d['p']:.1g} vs other versions"))
    worst = max(dr["psi_len"], dr["psi_chars"])
    out.append(("input drift (PSI > 0.25)", worst > 0.25,
                f"prompt length PSI {dr['psi_len']:.2f}, characters PSI {dr['psi_chars']:.2f}"))
    unk = np.mean([(r.get("unknown_chars") or 0) > 0 for r in recs[-max(1, n // 5):]])
    out.append(("unseen characters (> 10% of recent prompts)", unk > 0.10,
                f"{unk:.0%} of the last 20% of requests"))
    return out


# ----------------------------------------------------------------------------
# 6. Report + eval export
# ----------------------------------------------------------------------------
EVAL_CHECKS = {
    "degenerate_repetition": {"type": "max_repetition_ratio", "value": REP_THRESHOLD},
    "empty_output": {"type": "non_empty_output"},
    "max_token_truncation": {"type": "finish_reason", "value": "stop"},
    "unknown_chars": {"type": "no_dropped_characters"},
    "context_truncation": {"type": "prompt_fits_context"},
    "validation_4xx": {"type": "expect_status", "value": 400},
    "timeout": {"type": "max_latency_s"},
    "latency_outlier": {"type": "max_latency_per_token_s"},
}


def export_eval_cases(recs, path, per_mode=20):
    """One eval case per (failure mode, prompt), at most per_mode per mode (the first
    ones seen): the prompt and params to replay, what we saw, and the check that
    should pass once it is fixed. Overload, slow clients and disconnects are
    capacity/client issues, not model behaviour: not exported."""
    seen, cases, n_mode = set(), [], Counter()
    for r in recs:
        for m in r["modes"]:
            if m not in EVAL_CHECKS or (m, r.get("prompt_sha")) in seen or n_mode[m] >= per_mode:
                continue
            n_mode[m] += 1
            seen.add((m, r.get("prompt_sha")))
            check = dict(EVAL_CHECKS[m])
            if m == "timeout":
                check["value"] = r.get("timeout_s")
            if m == "latency_outlier":
                check["value"] = round(r["latency_s"] / max(1, r.get("output_tokens") or 1), 6)
            cases.append(dict(
                id=f"prod-{m}-{r.get('prompt_sha') or r['request_id']}",
                source="production", failure_mode=m, request_id=r["request_id"], ts=r["ts"],
                model_version=r["model_version"], input=dict(
                    prompt=r.get("prompt"), max_tokens=r.get("max_tokens"),
                    temperature=r.get("temperature"), top_k=r.get("top_k"),
                    seed=r.get("seed"), stop=r.get("stop")),
                params_raw=r.get("params"), observed=dict(
                    status=r["status"], error_type=r.get("error_type"),
                    output=r.get("output"), finish_reason=r.get("finish_reason"),
                    rep4=r.get("rep4"), latency_s=r.get("latency_s")),
                check=check))
    with open(path, "w", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    return cases


def snippet(r, n=50):
    for key in ("output", "prompt", "params"):
        if r.get(key):
            return f"{key}: {r[key][:n]!r}"
    return ""


def write_report(path, log_path, s, tax, versions, dr, al, n_bad, n_cases, lat_thr):
    L = [f"# Serving report: `{os.path.relpath(log_path)}`", "",
         f"{s['n']} requests over {s['span_s']:.1f} s ({s['req_s']:.1f} req/s), "
         f"{n_bad} malformed log lines.", "", "## SLIs", "",
         "| SLI | value |", "|---|---|",
         f"| availability (good / valid) | {s['availability']:.2%} ({s['good']}/{s['valid']}) |",
         f"| error budget ({s['slo']:.1%} SLO) | {s['budget_burn']:.1f}x used |",
         f"| latency p50 / p95 / p99 | {s['latency'][50] * 1e3:.0f} / "
         f"{s['latency'][95] * 1e3:.0f} / {s['latency'][99] * 1e3:.0f} ms |",
         f"| TTFT p50 / p95 / p99 | {s['ttft'][50] * 1e3:.1f} / {s['ttft'][95] * 1e3:.1f} / "
         f"{s['ttft'][99] * 1e3:.1f} ms |",
         f"| successes under {s['latency_slo_s'] * 1e3:.0f} ms | {s['latency_good']:.1%} |",
         f"| throughput | {s['tok_s']:,.0f} tokens/s |", "",
         "## Alerts", "", "| rule | state | detail |", "|---|---|---|"]
    L += [f"| {n} | {'**FIRING**' if f else 'ok'} | {d} |" for n, f, d in al]
    L += ["", "## Failure modes", "",
          "| mode | count | % | first seen | counts vs SLO | example |", "|---|---|---|---|---|---|"]
    for m, d in tax.items():
        ex = f"`{d['examples'][0]['request_id']}` {snippet(d['examples'][0])}" \
            if d["examples"] else ""
        fs = f"{d['first_seen']:.1f} s" if d["first_seen"] is not None else "-"
        L.append(f"| {m} | {d['count']} | {d['count'] / s['n']:.1%} | {fs} | "
                 f"{'yes' if MODES[m][1] else 'no'} | {ex.replace('|', '/')} |")
    L += ["", f"Latency outlier threshold: {lat_thr * 1e3:.2f} ms per output token.", "",
          "## By model version", "",
          "| version | requests | ok | first seen | p95 latency | mean rep4 | degenerate | z | p |",
          "|---|---|---|---|---|---|---|---|---|"]
    for v, d in versions.items():
        L.append(f"| {v} | {d['n']} | {d['ok']} | {d['first']:.1f} s | {d['p95'] * 1e3:.0f} ms | "
                 f"{d['mean_rep4']:.3f} | {d['degenerate']} ({d['degenerate_rate']:.1%}) | "
                 f"{d['z']:+.1f} | {d['p']:.1g} |")
    L += ["", "## Input drift (reference: first 30% of the log, current: last 10%)", "",
          f"- prompt length: mean {dr['ref_mean_len']:.0f} -> {dr['cur_mean_len']:.0f} chars, "
          f"PSI {dr['psi_len']:.2f}",
          f"- characters: PSI {dr['psi_chars']:.2f}, KL(current || reference) "
          f"{dr['kl_chars']:.2f}, character-class PSI {dr['psi_groups']:.2f}", "",
          "| window (s) | requests | mean length | unknown-char rate | PSI length | PSI chars |",
          "|---|---|---|---|---|---|"]
    for w in dr["windows"]:
        L.append(f"| {w['t']:.1f} | {w['n']} | {w['mean_len']:.0f} | {w['unknown_rate']:.0%} | "
                 f"{w['psi_len']:.2f} | {w['psi_chars']:.2f} |")
    L += ["", f"## Eval cases", "",
          f"{n_cases} failing requests exported to `eval_cases.jsonl` (one per failure mode "
          f"and prompt) for `../evals/`: replay them after a fix and keep them as a "
          f"regression gate."]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


# ----------------------------------------------------------------------------
# A synthetic log with KNOWN failures (serving_check.py tests the analyzer on it)
# ----------------------------------------------------------------------------
SYNTHETIC_EXPECTED = dict(timeout=4, validation_4xx=6, overload_429=5, slow_client=1,
                          client_disconnect=2, server_error_5xx=1, empty_output=3,
                          degenerate_repetition=4, max_token_truncation=2, unknown_chars=3,
                          context_truncation=2, latency_outlier=1)


def synthetic_log(n_normal=200, seed=0):
    rng = np.random.default_rng(seed)
    t0 = 1.7e9
    words = "keeper climbed stairs and girl saw light over the sea".split()

    def base(i, **kw):
        out = " ".join(rng.permutation(words)[:6])  # varied text, no loops
        r = dict(ts=datetime.utcfromtimestamp(t0 + i * 0.05).isoformat() + "+00:00",
                 request_id=f"syn-{i}", model_version="gpt-test", status=200,
                 error_type=None, prompt="The keeper", prompt_chars=10, prompt_sha=f"s{i}",
                 unknown_chars=0, prompt_truncated=False, max_tokens=40, temperature=0.8,
                 top_k=None, seed=i, stop=None, output=out, output_tokens=len(out),
                 finish_reason="length", latency_s=0.02 * (1 + 0.05 * rng.random()),
                 ttft_s=0.001, timeout_s=10.0)
        r["rep4"], r["loop_period"] = repetition_ratio(out), loop_period(out)
        r.update(kw)
        return r

    recs = [base(i) for i in range(n_normal)]
    i = n_normal
    E = SYNTHETIC_EXPECTED

    def add(k, **kw):
        nonlocal i
        for _ in range(k):
            recs.append(base(i, **kw))
            i += 1

    add(E["timeout"], status=504, error_type="timeout", output="", output_tokens=0)
    add(E["validation_4xx"] - 1, status=400, error_type="invalid_temperature", output=None,
        output_tokens=0)
    add(1, status=413, error_type="body_too_large", output=None, output_tokens=0)
    add(E["overload_429"], status=429, error_type="overloaded", output=None, output_tokens=0)
    add(E["slow_client"], status=408, error_type="slow_client", output=None, output_tokens=0)
    add(E["client_disconnect"], status=499, error_type="client_disconnected", output="the",
        output_tokens=3)
    add(E["server_error_5xx"], status=500, error_type="internal_error", output=None,
        output_tokens=0)
    add(E["empty_output"], output="   ", output_tokens=3, finish_reason="stop", stop=[" "],
        rep4=0.0, loop_period=0)
    loop = "the light and " * 6
    add(E["degenerate_repetition"], output=loop, output_tokens=len(loop),
        rep4=repetition_ratio(loop), loop_period=loop_period(loop))
    add(E["max_token_truncation"], stop=["."], finish_reason="length")
    add(E["unknown_chars"], prompt="Hello! café", prompt_chars=11, unknown_chars=3)
    add(E["context_truncation"], prompt="x" * 100, prompt_chars=100, prompt_truncated=True)
    add(E["latency_outlier"], latency_s=5.0)
    return recs


def analyze(recs, slo=0.99, latency_slo_s=0.5):
    s = slis(recs, slo, latency_slo_s)
    tax, thr = taxonomy(recs)
    versions = by_version(recs)
    dr = drift(recs)
    al = alerts(recs, s, tax, versions, dr, slo)
    return s, tax, thr, versions, dr, al


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", default=os.path.join(LOG_DIR, "requests.jsonl"))
    ap.add_argument("--slo", type=float, default=0.99, help="availability SLO")
    ap.add_argument("--latency-slo", type=float, default=0.5, help="seconds")
    ap.add_argument("--synthetic", action="store_true", help="analyze a generated log")
    args = ap.parse_args()

    if args.synthetic:
        recs, n_bad = synthetic_log(), 0
        for r in recs:
            r["t"] = parse_ts(r["ts"])
        out_dir = os.path.join(LOG_DIR, "synthetic")
    else:
        if not os.path.exists(args.log):
            raise SystemExit(f"No log at {args.log}. Generate one with: "
                             "python serving/load_test.py --spawn-server")
        recs, n_bad = load_logs(args.log)
        out_dir = os.path.dirname(os.path.abspath(args.log))
    os.makedirs(out_dir, exist_ok=True)
    s, tax, thr, versions, dr, al = analyze(recs, args.slo, args.latency_slo)

    print(f"{s['n']} requests over {s['span_s']:.1f} s ({s['req_s']:.1f} req/s), "
          f"{n_bad} malformed lines\n")
    print("== SLIs ==")
    print(f"  availability {s['availability']:.2%} ({s['good']} good / {s['valid']} valid; "
          f"400/408/499 are the client's fault and excluded)")
    print(f"  error budget for a {s['slo']:.1%} SLO: {s['budget_burn']:.1f}x the budget used "
          f"(burn rate over the whole log)")
    print(f"  latency p50 {s['latency'][50] * 1e3:.0f} ms, p95 {s['latency'][95] * 1e3:.0f} ms, "
          f"p99 {s['latency'][99] * 1e3:.0f} ms; {s['latency_good']:.1%} under "
          f"{s['latency_slo_s'] * 1e3:.0f} ms")
    print(f"  TTFT p50 {s['ttft'][50] * 1e3:.1f} ms, p95 {s['ttft'][95] * 1e3:.1f} ms, "
          f"p99 {s['ttft'][99] * 1e3:.1f} ms; throughput {s['tok_s']:,.0f} tokens/s")

    print("\n== Failure modes ==")
    for m, d in tax.items():
        fs = f"first at {d['first_seen']:5.1f} s" if d["first_seen"] is not None else ""
        sub = (" (" + ", ".join(f"{k}: {v}" for k, v in d["subtypes"].most_common()) + ")"
               if d["subtypes"] else "")
        print(f"  {m:22s} {d['count']:4d} {d['count'] / s['n']:6.1%}  {fs}{sub}")
        if d["examples"] and m in ("degenerate_repetition", "empty_output", "timeout",
                                   "max_token_truncation"):
            print(f"      e.g. {d['examples'][0]['request_id']}: {snippet(d['examples'][0], 60)}")

    print("\n== By model version ==")
    for v, d in versions.items():
        print(f"  {v}: {d['n']} requests from t = {d['first']:.1f} s, p95 {d['p95'] * 1e3:.0f} ms, "
              f"mean rep4 {d['mean_rep4']:.3f}, degenerate {d['degenerate']}/{d['ok']} = "
              f"{d['degenerate_rate']:.1%} (z = {d['z']:+.1f}, p = {d['p']:.1g})")

    print(f"\n== Input drift (reference: first 30%, {dr['ref_n']} prompts; current: last 10%, "
          f"{dr['cur_n']}) ==")
    print(f"  prompt length: mean {dr['ref_mean_len']:.0f} -> {dr['cur_mean_len']:.0f} chars, "
          f"PSI {dr['psi_len']:.2f}")
    print(f"  characters: PSI {dr['psi_chars']:.2f}, KL(current || reference) "
          f"{dr['kl_chars']:.2f}; character classes PSI {dr['psi_groups']:.2f}")
    print("  PSI over time (length / chars): " + "  ".join(
        f"{w['t']:.0f}s {w['psi_len']:.2f}/{w['psi_chars']:.2f}" for w in dr["windows"]))

    print("\n== Alerts ==")
    for name, firing, detail in al:
        print(f"  [{'FIRING' if firing else '  ok  '}] {name}: {detail}")

    report = os.path.join(out_dir, "report.md")
    cases = export_eval_cases(recs, os.path.join(out_dir, "eval_cases.jsonl"))
    write_report(report, args.log if not args.synthetic else "synthetic", s, tax, versions, dr,
                 al, n_bad, len(cases), thr)
    print(f"\nwrote {os.path.relpath(report)} and {len(cases)} eval cases to "
          f"{os.path.relpath(os.path.join(out_dir, 'eval_cases.jsonl'))}")


if __name__ == "__main__":
    main()
