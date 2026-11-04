"""
Load-test the inference server like real traffic would, with injected faults,
and produce the request logs that monitor.py analyzes.

The load is OPEN-LOOP: requests arrive on a Poisson schedule (fixed seed) whether
or not earlier ones have finished, like independent users. (A closed loop, where
N clients wait for each answer before sending the next, hides overload: when the
server slows down, the clients slow down with it.)

Traffic profile (fractions are per request, all configurable):
  normal       story-like prompts (5-40 chars of the training text), mixed sampling
               settings, some streamed, some with a stop sequence ('.', '\\n' or ' ')
  oversized    prompts of 600-2,000 chars                    -> 400 prompt_too_long
  invalid      bad params: temperature -1, max_tokens 0, a typo'd field, top_k 999,
               stream "yes", broken JSON                     -> 400
  unseen       prompts with characters the model never saw (digits, '!', 'é', '✓')
  timeout      timeout_s of 5-20 ms for long generations     -> 504
  slow         a client that sends half its body, stalls past the server's read
               timeout, then sends the rest                  -> 408 slow_client
  disconnect   a streaming client that hangs up after 3 tokens -> 499 in the logs
  burst        at 60% of the run, `--burst` requests at once  -> 429 (queue full)
  drift        in the second half, half the clients send LONG ALL-CAPS prompts
               (context truncation + unknown characters: the input distribution
               moves, which monitor.py should detect)
  greedy       temperature 0 with long outputs
  canary       at 40% of the run, a new model version (the 6x smaller draft model
               from speculative.py, "to cut cost") starts getting 30% of the traffic.
               It loops ("ten steps, ten steps, ...") far more often: monitor.py
               should catch the regression by model_version.

Run (~25 s, self-contained: starts the server in this process on a free port):
    python serving/load_test.py --spawn-server
    python serving/load_test.py --spawn-server --duration 60 --rate 60 --no-faults
    python serving/load_test.py --url http://127.0.0.1:8000     # an already running server
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from common import DEFAULT_TEXT, HELDOUT_PATH, LOG_DIR, percentile

DEFAULT_FAULTS = dict(oversized=0.02, invalid=0.04, unseen=0.04, timeout=0.02, slow=0.01,
                      disconnect=0.02, greedy=0.08)
UNSEEN = ["Hello! ", "Room 101: ", "café ", "✓ done. ", "Q&A: ", "x = 42; ", "#tag ", "@user "]


def word_slice(text, rng, lo, hi):
    """A random slice of text that starts at a word boundary."""
    for _ in range(100):
        s = int(rng.integers(0, len(text) - hi))
        if s == 0 or text[s - 1] in " \n":
            return text[s:s + int(rng.integers(lo, hi + 1))]
    return text[:lo]


def make_schedule(duration, rate, seed=0, faults=None, burst=120, drift=True):
    """[(t_send, kind, payload)] for the whole run, fixed by the seed."""
    rng = np.random.default_rng(seed)
    faults = DEFAULT_FAULTS if faults is None else faults
    story = open(DEFAULT_TEXT, encoding="utf-8").read()
    held = open(HELDOUT_PATH, encoding="utf-8").read()
    t, sched = 0.0, []
    while True:
        t += rng.exponential(1.0 / rate)
        if t >= duration:
            break
        u, kind = rng.random(), "normal"
        acc = 0.0
        for name, frac in faults.items():
            acc += frac
            if u < acc:
                kind = name
                break
        payload = dict(prompt=word_slice(story, rng, 5, 40),
                       max_tokens=int(np.clip(rng.lognormal(np.log(48), 0.6), 4, 200)),
                       temperature=float(rng.choice([0.8, 0.8, 0.8, 1.0, 1.2])),
                       seed=int(rng.integers(0, 2 ** 31)))
        if rng.random() < 0.3:
            payload["stream"] = True
        if rng.random() < 0.25:
            payload["stop"] = [str(rng.choice([".", "\n", " "]))]
        if drift and t > duration / 2 and rng.random() < 0.5 and kind == "normal":
            kind = "drift"
            payload["prompt"] = word_slice(held, rng, 60, 300).upper()
        if kind == "oversized":
            payload["prompt"] = (story * 2)[:int(rng.integers(600, 2000))]
        elif kind == "invalid":
            bad = int(rng.integers(0, 6))
            if bad == 0:
                payload["temperature"] = -1
            elif bad == 1:
                payload["max_tokens"] = 0
            elif bad == 2:
                payload["max_token"] = payload.pop("max_tokens")
            elif bad == 3:
                payload["top_k"] = 999
            elif bad == 4:
                payload["stream"] = "yes"
            else:
                payload = "{'prompt': 'single quotes are not JSON'"
        elif kind == "unseen":
            payload["prompt"] = str(rng.choice(UNSEEN)) + payload["prompt"]
        elif kind == "timeout":
            payload.update(max_tokens=200, timeout_s=float(rng.uniform(0.005, 0.02)))
        elif kind == "disconnect":
            payload.update(max_tokens=200, stream=True)
        elif kind == "greedy":
            payload.update(temperature=0.0, max_tokens=int(rng.integers(100, 200)))
            payload.pop("stop", None)
        sched.append((t, kind, payload))
    if burst:
        tb = 0.6 * duration
        for i in range(burst):
            p = dict(prompt=word_slice(story, rng, 5, 40), max_tokens=120,
                     seed=int(rng.integers(0, 2 ** 31)))
            sched.append((tb + i * 1e-4, "burst", p))
        sched.sort(key=lambda x: x[0])
    return sched


# ----------------------------------------------------------------------------
# Clients
# ----------------------------------------------------------------------------
def post(url, payload, timeout=30.0):
    """One request with urllib. Streams are read line by line (SSE over chunked
    encoding, which urllib decodes). Returns a result dict measured client-side."""
    data = payload.encode() if isinstance(payload, str) else json.dumps(payload).encode()
    req = urllib.request.Request(url + "/v1/completions", data=data,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    res = dict(status=None, ttft=None, tokens=0, error_type=None)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            res["status"] = r.status
            if r.headers.get("Content-Type", "").startswith("text/event-stream"):
                for line in r:
                    if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                        continue
                    ev = json.loads(line[6:])
                    if "error" in ev:
                        res["error_type"] = ev["error"]["type"]
                    elif ev["choices"][0]["finish_reason"] is None:
                        res["ttft"] = res["ttft"] or time.perf_counter() - t0
                    else:
                        res["tokens"] = ev["usage"]["completion_tokens"]
            else:
                body = json.loads(r.read())
                res["tokens"] = body["usage"]["completion_tokens"]
                res["ttft"] = body["timing"]["ttft_s"]
    except urllib.error.HTTPError as e:
        res["status"] = e.code
        try:
            res["error_type"] = json.loads(e.read())["error"]["type"]
        except Exception:
            pass
    except (urllib.error.URLError, OSError) as e:
        res["status"], res["error_type"] = -1, type(e).__name__
    res["latency"] = time.perf_counter() - t0
    return res


def raw_request(url, payload, mode, stall=0.0):
    """Raw-socket clients for the misbehaving cases urllib can't express:
    'slow' sends half the body, stalls, then the rest; 'disconnect' reads three
    stream events and hangs up."""
    u = urllib.parse.urlparse(url)
    body = json.dumps(payload).encode()
    head = (f"POST /v1/completions HTTP/1.1\r\nHost: {u.hostname}\r\nContent-Type: "
            f"application/json\r\nContent-Length: {len(body)}\r\nConnection: close\r\n\r\n")
    t0 = time.perf_counter()
    res = dict(status=None, ttft=None, tokens=0, error_type=None)
    try:
        with socket.create_connection((u.hostname, u.port), timeout=30) as s:
            if mode == "slow":
                s.sendall(head.encode() + body[:len(body) // 2])
                time.sleep(stall)
                try:
                    s.sendall(body[len(body) // 2:])
                except OSError:
                    pass
                data = b""
                while True:
                    chunk = s.recv(65536)
                    if not chunk:
                        break
                    data += chunk
                res["status"] = int(data.split(b" ", 2)[1]) if data else -1
            else:
                s.sendall(head.encode() + body)
                f = s.makefile("rb")
                res["status"] = int(f.readline().split(b" ", 2)[1])
                events = 0
                for line in f:
                    if line.startswith(b"data: "):
                        events += 1
                        res["ttft"] = res["ttft"] or time.perf_counter() - t0
                        if events == 3:
                            break
                res["error_type"] = "hung_up"
    except OSError as e:
        res["status"], res["error_type"] = res["status"] or -1, type(e).__name__
    res["latency"] = time.perf_counter() - t0
    return res


def run_load(url, schedule, read_timeout=2.0, workers=256, verbose=True, events=()):
    """Fire the schedule open-loop with a thread pool. Returns per-request results.
    events: [(t, callback)] run at time t (e.g. a canary rollout)."""
    results = [None] * len(schedule)
    t_start = time.perf_counter()
    events = sorted(events, key=lambda e: e[0])

    def fire(i, kind, payload):
        if kind == "slow":
            r = raw_request(url, payload, "slow", stall=read_timeout + 0.5)
        elif kind == "disconnect":
            r = raw_request(url, payload, "disconnect")
        else:
            r = post(url, payload)
        r.update(kind=kind, t_send=time.perf_counter() - t_start)
        results[i] = r

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, (t, kind, payload) in enumerate(schedule):
            while events and events[0][0] <= t:
                events.pop(0)[1]()
            delay = t - (time.perf_counter() - t_start)
            if delay > 0:
                time.sleep(delay)
            pool.submit(fire, i, kind, payload)
        if verbose:
            print(f"  all {len(schedule)} requests sent after "
                  f"{time.perf_counter() - t_start:.1f} s; waiting for the last answers...")
    return results, time.perf_counter() - t_start


def report(results, wall, metrics_text=None):
    from server import parse_prometheus
    n = len(results)
    by_status, by_kind = {}, {}
    for r in results:
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1
        k = by_kind.setdefault(r["kind"], {})
        k[r["status"]] = k.get(r["status"], 0) + 1
    ok = [r for r in results if r["status"] == 200 and not r["error_type"]]
    lat = [r["latency"] for r in ok]
    ttft = [r["ttft"] for r in ok if r["ttft"] is not None]
    toks = sum(r["tokens"] for r in ok)
    summary = dict(n=n, wall=wall, ok=len(ok), error_rate=1 - len(ok) / n,
                   req_s=len(ok) / wall, tok_s=toks / wall, by_status=by_status,
                   by_kind=by_kind, latency={q: percentile(lat, q) for q in (50, 95, 99)},
                   ttft={q: percentile(ttft, q) for q in (50, 95, 99)})
    print(f"\n== Load test: {n} requests in {wall:.1f} s ==")
    print(f"  successful: {len(ok)} ({len(ok) / wall:.1f} req/s, {toks / wall:,.0f} tokens/s); "
          f"error rate {summary['error_rate']:.1%}")
    print("  by status: " + ", ".join(f"{s}: {c}" for s, c in sorted(by_status.items(),
                                                                       key=lambda x: str(x[0]))))
    print("  by kind:")
    for kind, st in sorted(by_kind.items()):
        print(f"    {kind:10s} " + ", ".join(f"{s}: {c}" for s, c in
                                            sorted(st.items(), key=lambda x: str(x[0]))))
    print(f"  client-side latency of successful requests: p50 {summary['latency'][50] * 1e3:.0f} "
          f"ms, p95 {summary['latency'][95] * 1e3:.0f} ms, p99 "
          f"{summary['latency'][99] * 1e3:.0f} ms")
    print(f"  time to first token: p50 {summary['ttft'][50] * 1e3:.1f} ms, p95 "
          f"{summary['ttft'][95] * 1e3:.1f} ms, p99 {summary['ttft'][99] * 1e3:.1f} ms")
    if metrics_text:
        m = parse_prometheus(metrics_text)
        served = {k.split('"')[1]: int(v) for k, v in m.items()
                  if k.startswith("serving_requests_total")}
        print(f"  server's /metrics agrees: requests by status {dict(sorted(served.items()))}, "
              f"{int(m.get('serving_generated_tokens_total', 0)):,} tokens generated, "
              f"{int(sum(v for k, v in m.items() if k.startswith('serving_engine_steps'))):,} "
              f"engine steps")
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=None, help="server to test (default: --spawn-server)")
    ap.add_argument("--spawn-server", action="store_true", help="start the server in-process")
    ap.add_argument("--duration", type=float, default=20.0, help="seconds of traffic")
    ap.add_argument("--rate", type=float, default=40.0, help="mean requests per second")
    ap.add_argument("--burst", type=int, default=150, help="requests fired at once at 60%%")
    ap.add_argument("--no-faults", action="store_true", help="only normal traffic")
    ap.add_argument("--no-drift", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quantize", choices=["none", "int8", "int4"], default="none")
    ap.add_argument("--canary-frac", type=float, default=0.3,
                    help="(with --spawn-server) traffic share of the canary model")
    ap.add_argument("--canary-at", type=float, default=0.4, help="fraction of the run")
    ap.add_argument("--no-canary", action="store_true")
    ap.add_argument("--log-file", default=os.path.join(LOG_DIR, "requests.jsonl"),
                    help="(with --spawn-server) where the server logs; overwritten")
    args = ap.parse_args()
    if not args.url and not args.spawn_server:
        args.spawn_server = True

    read_timeout = 2.0
    srv = None
    if args.spawn_server:
        from server import build_server
        os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
        open(args.log_file, "w").close()  # a fresh log for this run
        srv = build_server(args.quantize, port=0, quiet=True, max_batch=16, queue_size=32,
                           read_timeout=read_timeout, log_path=args.log_file,
                           canary=None if args.no_canary else "draft").start()
        url = srv.url
        print(f"started the server in-process on {url} (logs -> {os.path.relpath(args.log_file)})")
    else:
        url = args.url.rstrip("/")
    faults = {} if args.no_faults else DEFAULT_FAULTS
    sched = make_schedule(args.duration, args.rate, args.seed, faults,
                          0 if args.no_faults else args.burst, drift=not args.no_drift)
    print(f"schedule: {len(sched)} requests over {args.duration:.0f} s "
          f"(Poisson {args.rate:g}/s" + ("" if args.no_faults else
                                         f" + a burst of {args.burst} at "
                                         f"{0.6 * args.duration:.0f} s") + ")")
    events = []
    if srv is not None and not args.no_canary:
        t_c = args.canary_at * args.duration
        events.append((t_c, lambda: srv.set_canary(args.canary_frac)))
        print(f"canary: {srv.canary_version} gets {args.canary_frac:.0%} of traffic from "
              f"{t_c:.0f} s (stable: {srv.version})")
    results, wall = run_load(url, sched, read_timeout, events=events)
    with urllib.request.urlopen(url + "/metrics") as r:
        metrics_text = r.read().decode()
    report(results, wall, metrics_text)
    if srv is not None:
        srv.shutdown()
        print(f"\nserver stopped; {sum(1 for _ in open(args.log_file))} log lines in "
              f"{os.path.relpath(args.log_file)}. Next: python serving/monitor.py")


if __name__ == "__main__":
    main()
