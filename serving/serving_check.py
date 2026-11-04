"""
Correctness checks for everything in serving/ (the analog of
../transformer/gradient_check.py): each fast implementation is compared with a
brute-force or reference computation. Tiny random-weight models in float64 keep
it exact and fast (~5 s).

  1. static padded batching == one-at-a-time generation (generate.py), greedy and
     seeded sampling, including prompts longer than the window and window slides
  2. continuous batching (+ preemption, stop sequences, late arrivals) == one at a time
  3. paged KV cache reads == the contiguous KV cache, bit for bit
  4. int8 / int4 dequantization error <= scale / 2 elementwise; int4 pack/unpack
     round-trip exact; the dequantize-on-the-fly matmul == x @ dequant(W)
  5. speculative decoding: the accept/reject rule reproduces p (and a naive rule
     fails the same test), greedy speculative == greedy target exactly, and the
     2nd/3rd sampled tokens match the target's exact marginals
  6. the server: request validation, response schema, SSE stream and /metrics over
     real HTTP (a tiny model on a free port)
  7. percentile() == np.percentile; Prometheus histogram buckets; PSI/KL identities
  8. monitor.py's failure taxonomy on a synthetic log with known failures

Run:
    python serving/serving_check.py
"""
from __future__ import annotations

import json
import os
import shutil
import time
import urllib.error
import urllib.request

import numpy as np

from batching import (ContinuousBatcher, PagedKVCache, Request, reference_generate,
                      static_generate)
from common import LOG_DIR, decode_step, percentile, prefill, tiny_model
from monitor import SYNTHETIC_EXPECTED, kl, parse_ts, psi, synthetic_log, taxonomy
from quantize_numpy import (QLinear, pack_int4, quant_int4_group, quant_int8_channel,
                            quant_int8_tensor, unpack_int4)
from server import (ApiError, Histogram, InferenceServer, check_response_schema,
                    parse_prometheus, validate_completion)
from speculative import distribution_test, rule_test, speculative_generate

RESULTS = []


def report(name, ok, detail=""):
    RESULTS.append(ok)
    print(f"  {'OK  ' if ok else 'FAIL'} {name}" + (f"  ({detail})" if detail else ""))


def requests_for(V, T, rng, n=6, temperature=0.0):
    """Prompts of 1..T+6 tokens (some longer than the window), outputs long enough to
    slide the window more than once."""
    out = []
    for i in range(n):
        L = int(rng.integers(1, T + 7))
        out.append(Request(rng.integers(0, V, L).tolist(), int(rng.integers(3, 2 * T + 5)),
                           temperature, seed=100 + i))
    return out


def fresh(reqs, **kw):
    return [Request(r.prompt, r.max_tokens, r.temperature, r.top_k, seed=100 + i,
                    stop=r.stop, **kw) for i, r in enumerate(reqs)]


def check_batching():
    print("\n1-3. Batching and the paged KV cache (tiny GPT, V=12, T=16, float64)")
    model = tiny_model()
    V, T = model.config["vocab_size"], model.config["block_size"]
    rng = np.random.default_rng(0)
    for temp, label in ((0.0, "greedy"), (0.9, "sampled, per-request seeds")):
        reqs = requests_for(V, T, rng, temperature=temp)
        refs = [reference_generate(model, r) for r in fresh(reqs)]
        static_generate(model, reqs)
        same = all(r.out == ref for r, ref in zip(reqs, refs))
        report(f"static padded batch == unbatched ({label})", same,
               f"{len(reqs)} requests, prompts {min(len(r.prompt) for r in reqs)}-"
               f"{max(len(r.prompt) for r in reqs)} tokens, up to "
               f"{max(len(r.out) for r in reqs)} new")

    for temp, label in ((0.0, "greedy"), (0.9, "sampled")):
        reqs = requests_for(V, T, rng, n=8, temperature=temp)
        reqs[1].stop, reqs[5].stop = [(reqs[1].prompt[-1],)], [(3, 4)]
        refs = [reference_generate(model, r) for r in fresh(reqs)]
        kv = PagedKVCache.for_model(model, num_blocks=7, block_size=4)  # tight: forces preemption
        eng = ContinuousBatcher(model, kv, max_batch=4)
        step = 0
        pending = list(reqs)
        while pending or eng.has_work():
            if pending and step % 3 == 0:  # one new arrival every 3 steps
                eng.add(pending.pop(0))
            eng.step()
            step += 1
        same = all(r.out == ref for r, ref in zip(reqs, refs))
        freed = len(kv.free) == kv.num_blocks
        report(f"continuous batching == unbatched ({label}), all KV blocks freed", same and freed,
               f"{len(reqs)} staggered requests, {step} steps, "
               f"{sum(r.n_preempted for r in reqs)} preemptions, freed: {freed}")

    ids = rng.integers(0, V, T).tolist()
    _, cache = prefill(model, ids[:5])
    kv = PagedKVCache.for_model(model, num_blocks=8, block_size=3)
    kv.free.rotate(3)  # non-contiguous physical blocks
    kv.ensure("s", 5)
    for l in range(len(model.blocks)):
        kv.write_seq("s", l, cache.k[l], cache.v[l])
    for t, tok in enumerate(ids[5:], start=5):
        decode_step(model, tok, cache)
        kv.ensure("s", t + 1)
        for l in range(len(model.blocks)):
            kv.write_tokens(l, ["s"], [t], cache.k[l][None, :, t], cache.v[l][None, :, t])
    worst = 0.0
    for l in range(len(model.blocks)):
        K, Vv, mask = kv.gather(l, ["s"], [T])
        worst = max(worst, np.abs(K[0][:, mask[0]] - cache.k[l]).max(),
                    np.abs(Vv[0][:, mask[0]] - cache.v[l]).max())
    report("paged KV gather == contiguous KV cache", worst == 0.0,
           f"{T} tokens in blocks {kv.tables['s']}, max |diff| = {worst:.1e}")


def check_quantization():
    print("\n4. Quantization")
    rng = np.random.default_rng(1)
    W = (rng.standard_normal((128, 96)) * 0.05).astype(np.float32)
    W[3, 7] = 1.5  # an outlier
    q, s = quant_int8_tensor(W)
    e1 = np.abs(W - q * s) <= s / 2 + 1e-7
    q, s = quant_int8_channel(W)
    e2 = np.abs(W - q * s) <= s / 2 + 1e-7
    ok3 = True
    for g in (32, 64):
        q, s = quant_int4_group(W, g)
        Wd = (q.reshape(-1, g, 96) * s[:, None]).reshape(W.shape)
        bound = np.repeat(s, g, axis=0) / 2 + 1e-7
        ok3 &= bool((np.abs(W - Wd) <= bound).all()) and q.min() >= -8 and q.max() <= 7
    report("int8 per-tensor |W - Ŵ| <= s/2 everywhere", bool(e1.all()))
    report("int8 per-channel |W - Ŵ| <= s_j/2 everywhere", bool(e2.all()))
    report("int4 group 32/64 |W - Ŵ| <= s_g/2 everywhere, q in [-8, 7]", ok3)
    allq = np.array([[a, b] for a in range(-8, 8) for b in range(-8, 8)], np.int8).T  # (2, 256)
    p = pack_int4(allq)
    every_byte = np.arange(256, dtype=np.uint8)[None]
    ok = (np.array_equal(unpack_int4(p), allq) and p.dtype == np.uint8 and p.shape == (1, 256)
          and np.array_equal(pack_int4(unpack_int4(every_byte)), every_byte))
    report("int4 pack/unpack round-trip (all 256 nibble pairs, all 256 bytes)", ok)

    class Lin:
        params = {"W": W.astype(np.float64), "b": rng.standard_normal(96)}

    x = rng.standard_normal((5, 3, 128))
    worst = 0.0
    for scheme, g in (("int8-tensor", None), ("int8", None), ("int4", 32), ("fp16", None)):
        ql = QLinear(Lin, scheme, g)
        worst = max(worst, np.abs(ql.forward(x) - (x @ ql.dequant() + Lin.params["b"])).max())
    report("on-the-fly dequant matmul == x @ dequant(W) + b (4 schemes)", worst < 1e-10,
           f"max |diff| = {worst:.1e}")


def check_speculative():
    print("\n5. Speculative decoding")
    r = rule_test(N=100_000)
    ex, nv = r["exact rule"], r["naive rule"]
    report("accept/reject rule: output ~ p (chi-square, 100k draws)",
           ex["chi2"] < ex["crit"] and ex["tv"] < 0.01,
           f"TV {ex['tv']:.4f}, chi² {ex['chi2']:.1f} < {ex['crit']:.1f}")
    report("...and the same test rejects a naive rule", nv["chi2"] > nv["crit"],
           f"TV {nv['tv']:.3f}, chi² {nv['chi2']:.0f}")
    target = tiny_model(V=8, block_size=12, seed=1)
    draft = tiny_model(V=8, n_layer=1, d_model=8, block_size=12, seed=2, std=0.2)
    rng = np.random.default_rng(3)
    same = True
    for g in (1, 3, 5):
        for _ in range(3):
            ids = rng.integers(0, 8, int(rng.integers(1, 14))).tolist()
            a = speculative_generate(target, draft, ids, 40, g, temperature=0.0)
            b = reference_generate(target, Request(ids, 40, 0.0))
            same &= a == b
    report("greedy speculative == greedy target (γ = 1, 3, 5; window slides)", same)
    res, st = distribution_test(target, draft, [1, 2, 3], N=3000, gamma=3, seed=0)
    ok = all(d["chi2"] < d["crit"] for d in res)
    report("sampled tokens 1-3 match the target's exact marginals (3,000 runs)", ok,
           ", ".join(f"TV {d['tv']:.3f} chi² {d['chi2']:.1f}<{d['crit']:.1f}" for d in res)
           + f"; {st['accepted']}/{st['proposed']} drafts accepted")


def check_server():
    print("\n6. Server")
    good = {"prompt": "abc", "max_tokens": 5, "temperature": 0.5, "top_k": 3, "seed": 1,
            "stop": ".", "stream": False, "timeout_s": 2}
    cases = [({"prompt": ""}, "invalid_prompt"), ({"prompt": "a" * 600}, "prompt_too_long"),
             ({"prompt": "a", "max_tokens": 0}, "invalid_max_tokens"),
             ({"prompt": "a", "max_tokens": True}, "invalid_max_tokens"),
             ({"prompt": "a", "temperature": -1}, "invalid_temperature"),
             ({"prompt": "a", "temperature": float("nan")}, "invalid_temperature"),
             ({"prompt": "a", "top_k": 99}, "invalid_top_k"),
             ({"prompt": "a", "max_token": 5}, "unknown_field"),
             ({"prompt": "a", "stream": "yes"}, "invalid_stream"),
             ({"prompt": "a", "stop": ["x"] * 5}, "invalid_stop"),
             ({"prompt": "a", "timeout_s": 0}, "invalid_timeout"), (["a"], "invalid_body")]
    ok = validate_completion(good, 12)["stop"] == ["."]
    for body, want in cases:
        try:
            validate_completion(body, 12)
            ok = False
        except ApiError as e:
            ok &= e.status == 400 and e.type == want
    report(f"request validation: 1 valid + {len(cases)} invalid bodies -> 400 + error type", ok)

    model = tiny_model()
    chars = list("abcdefghij .")
    log = os.path.join(LOG_DIR, "check", "requests.jsonl")
    srv = InferenceServer(model, chars, "gpt-tiny", port=0, log_path=log, quiet=True,
                          max_batch=4, queue_size=4).start()
    try:
        def post(body):
            req = urllib.request.Request(srv.url + "/v1/completions",
                                         data=json.dumps(body).encode())
            try:
                with urllib.request.urlopen(req, timeout=10) as r:
                    return r.status, r.headers.get("Content-Type"), r.read()
            except urllib.error.HTTPError as e:
                return e.code, None, e.read()

        status, _, body = post({"prompt": "abc!", "max_tokens": 7, "seed": 3})
        obj = json.loads(body)
        errs = check_response_schema(obj)
        ref = reference_generate(model, Request([0, 1, 2], 7, 0.8, seed=3))
        same = obj["choices"][0]["text"] == "".join(chars[i] for i in ref)
        report("POST /v1/completions: 200, schema valid, text == unbatched generation",
               status == 200 and not errs and same and len(obj["warnings"]) == 1,
               f"usage {obj['usage']}, warnings {obj['warnings']}" if not errs else str(errs))
        status, _, body = post({"prompt": "abc", "temperature": 7})
        report("invalid request over HTTP -> 400 invalid_temperature",
               status == 400 and json.loads(body)["error"]["type"] == "invalid_temperature")
        status, ctype, body = post({"prompt": "ab", "max_tokens": 5, "stream": True, "seed": 4})
        events = [line[6:] for line in body.decode().split("\n") if line.startswith("data: ")]
        pieces = [json.loads(e)["choices"][0]["text"] for e in events[:-2]]
        final = json.loads(events[-2])
        report("stream=true: SSE events, pieces concatenate to the final text",
               ctype == "text/event-stream" and events[-1] == "[DONE]" and len(pieces) == 5
               and "".join(pieces) == final["choices"][0]["text"])
        with urllib.request.urlopen(srv.url + "/metrics") as r:
            m = parse_prometheus(r.read().decode())
        report("/metrics counts the requests by status",
               m.get('serving_requests_total{status="200"}') == 2
               and m.get('serving_requests_total{status="400"}') == 1
               and m.get("serving_generated_tokens_total") == 12, "200: 2, 400: 1, 12 tokens")
    finally:
        srv.shutdown(drain_timeout=2)
    lines = [json.loads(x) for x in open(log)]
    report("one structured JSON log line per request",
           [x["status"] for x in lines] == [200, 400, 200] and all(
               "latency_s" in x and "model_version" in x for x in lines))
    shutil.rmtree(os.path.dirname(log))


def check_stats():
    print("\n7-8. Metrics, drift statistics and the log analyzer")
    rng = np.random.default_rng(5)
    worst = 0.0
    for n in (1, 2, 3, 10, 101, 1000):
        x = rng.exponential(size=n).tolist() + [0.5] * (n // 3)  # ties too
        for q in (0, 1, 25, 50, 90, 95, 99, 99.9, 100):
            worst = max(worst, abs(percentile(x, q) - np.percentile(x, q)))
    report("percentile() == np.percentile (n = 1..1000, q = 0..100, ties)", worst < 1e-12,
           f"max |diff| = {worst:.1e}")
    h = Histogram("h", "test", buckets=(0.1, 0.5, 1.0))
    xs = rng.exponential(0.5, 500)
    for x in xs:
        h.observe(float(x))
    m = parse_prometheus("\n".join(h.render()))
    ok = all(m[f'h_bucket{{le="{b}"}}'] == (xs <= b).sum() for b in (0.1, 0.5, 1))
    ok &= m['h_bucket{le="+Inf"}'] == 500 == m["h_count"] and abs(m["h_sum"] - xs.sum()) < 1e-3
    report("Prometheus histogram: cumulative buckets == brute-force counts", ok)
    p, q = rng.dirichlet(np.ones(6)), rng.dirichlet(np.ones(6))
    ok = abs(psi(p, p)) < 1e-12 and abs(psi(p, q) - (kl(p, q) + kl(q, p))) < 1e-9
    report("PSI(p, p) = 0 and PSI(p, q) = KL(p||q) + KL(q||p)", ok)
    recs = synthetic_log()
    for r in recs:
        r["t"] = parse_ts(r["ts"])
    tax, _ = taxonomy(recs)
    got = {m: d["count"] for m, d in tax.items()}
    report("failure taxonomy on a synthetic log == the injected failures", got == SYNTHETIC_EXPECTED,
           ", ".join(f"{m} {c}" for m, c in got.items() if c) if got == SYNTHETIC_EXPECTED
           else f"got {got}")


def main():
    t0 = time.time()
    check_batching()
    check_quantization()
    check_speculative()
    check_server()
    check_stats()
    ok = all(RESULTS)
    print(f"\n{len(RESULTS)} checks in {time.time() - t0:.1f} s")
    print("Serving check", "PASSED" if ok else "FAILED")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
