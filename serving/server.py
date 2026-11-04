"""
A real HTTP inference server for the mini-GPT, from the standard library only
(http.server, threading, queue, json, logging): the moving parts of vLLM / TGI
without the framework.

    client ──HTTP──> handler thread ──validate──> bounded queue ──> engine thread
                     (one per connection)          (429 when full)   (continuous
                     waits on the job's events  <── tokens ──────────  batching over
                     and writes JSON or SSE                            a paged KV cache)
    every request -> one JSON line in serving/logs/requests.jsonl  (monitor.py reads it)
    GET /metrics  -> Prometheus text format, hand-written            (dashboards, alerts)

Endpoints
  POST /v1/completions  {"prompt": str, "max_tokens": int, "temperature": float,
                         "top_k": int|null, "seed": int|null, "stop": str|[str]|null,
                         "stream": bool, "timeout_s": float}
                        -> {"id", "object", "model", "created", "choices": [{"text",
                            "index", "finish_reason"}], "usage", "timing", "warnings"}
                        stream=true: server-sent events over chunked transfer encoding
  GET  /health          liveness/readiness (503 while draining on shutdown)
  GET  /metrics         request counters by status, latency/TTFT histograms, tokens,
                        queue depth, in-flight, batch size, KV blocks in use

Production behaviour: strict request validation (400 with an error type), max prompt
length, per-request timeout (504), bounded queue with backpressure (429 +
Retry-After), client disconnects (499 in the logs), graceful shutdown on
Ctrl+C / SIGTERM (stop accepting, finish in-flight work, flush logs).

Run:
    python serving/server.py                         # http://127.0.0.1:8000
    python serving/server.py --port 8080 --quantize int8
    curl -s localhost:8000/v1/completions -d '{"prompt": "The keeper", "max_tokens": 60}'
    curl -sN localhost:8000/v1/completions -d '{"prompt": "The sea", "stream": true}'
    curl -s localhost:8000/metrics
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import queue
import signal
import socket
import sys
import threading
import time
import traceback
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from batching import ContinuousBatcher, PagedKVCache, Request
from common import LOG_DIR, CharTokenizer, get_model, loop_period, repetition_ratio

LATENCY_BUCKETS = (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)


# ----------------------------------------------------------------------------
# Metrics in the Prometheus text exposition format (no client library)
# ----------------------------------------------------------------------------
def _labels(labels):
    if not labels:
        return ""
    return "{" + ",".join(f'{k}="{v}"' for k, v in labels) + "}"


class Metric:
    def __init__(self, name, help_, kind):
        self.name, self.help, self.kind = name, help_, kind
        self.values = {}  # sorted label tuple -> value
        self.lock = threading.Lock()

    def inc(self, n=1.0, **labels):
        key = tuple(sorted(labels.items()))
        with self.lock:
            self.values[key] = self.values.get(key, 0.0) + n

    def set(self, v, **labels):
        with self.lock:
            self.values[tuple(sorted(labels.items()))] = float(v)

    def render(self):
        lines = [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} {self.kind}"]
        with self.lock:
            for key, v in sorted(self.values.items()):
                lines.append(f"{self.name}{_labels(key)} {v:g}")
        return lines


class Histogram:
    """Cumulative buckets: _bucket{le="x"} counts observations <= x; +Inf = _count."""

    def __init__(self, name, help_, buckets=LATENCY_BUCKETS):
        self.name, self.help, self.buckets = name, help_, tuple(buckets)
        self.counts = [0] * (len(self.buckets) + 1)  # per bucket, last = above all
        self.sum, self.count = 0.0, 0
        self.lock = threading.Lock()

    def observe(self, x):
        with self.lock:
            i = next((i for i, b in enumerate(self.buckets) if x <= b), len(self.buckets))
            self.counts[i] += 1
            self.sum += x
            self.count += 1

    def render(self):
        lines = [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} histogram"]
        with self.lock:
            cum = 0
            for b, c in zip(self.buckets + (float("inf"),), self.counts):
                cum += c
                le = "+Inf" if b == float("inf") else f"{b:g}"
                lines.append(f'{self.name}_bucket{{le="{le}"}} {cum}')
            lines.append(f"{self.name}_sum {self.sum:.6f}")
            lines.append(f"{self.name}_count {self.count}")
        return lines


class Metrics:
    def __init__(self):
        self.requests = Metric("serving_requests_total",
                               "Completion requests by HTTP status code.", "counter")
        self.errors = Metric("serving_errors_total", "Failed requests by error type.", "counter")
        self.tokens = Metric("serving_generated_tokens_total", "Tokens generated.", "counter")
        self.prompt_tokens = Metric("serving_prompt_tokens_total", "Prompt tokens read.",
                                    "counter")
        self.steps = Metric("serving_engine_steps_total", "Continuous-batching engine steps.",
                            "counter")
        self.queue = Metric("serving_queue_depth", "Requests waiting for a batch slot.", "gauge")
        self.inflight = Metric("serving_inflight_requests",
                               "Accepted requests not finished yet.", "gauge")
        self.batch = Metric("serving_batch_size", "Sequences in the running batch.", "gauge")
        self.kv_used = Metric("serving_kv_blocks_used", "Paged KV-cache blocks in use.", "gauge")
        self.kv_total = Metric("serving_kv_blocks_total", "Paged KV-cache blocks.", "gauge")
        self.info = Metric("serving_model_info", "The model being served.", "gauge")
        self.uptime = Metric("serving_uptime_seconds", "Seconds since start.", "gauge")
        self.latency = Histogram("serving_request_latency_seconds",
                                 "End-to-end latency of successful completions.")
        self.ttft = Histogram("serving_time_to_first_token_seconds",
                              "Arrival to first generated token.")

    def render(self):
        lines = []
        for m in (self.requests, self.errors, self.tokens, self.prompt_tokens, self.steps,
                  self.queue, self.inflight, self.batch, self.kv_used, self.kv_total,
                  self.info, self.uptime, self.latency, self.ttft):
            lines += m.render()
        return "\n".join(lines) + "\n"


def parse_prometheus(text):
    """{'name{labels}': value} from exposition text (for tests and the load tester)."""
    out = {}
    for line in text.splitlines():
        if line and not line.startswith("#"):
            key, _, v = line.rpartition(" ")
            out[key] = float(v)
    return out


# ----------------------------------------------------------------------------
# Request validation and the response schema
# ----------------------------------------------------------------------------
class ApiError(Exception):
    def __init__(self, status, type_, message):
        super().__init__(message)
        self.status, self.type, self.message = status, type_, message


DEFAULT_LIMITS = dict(max_prompt_chars=512, max_tokens_cap=256, default_max_tokens=64,
                      default_timeout=10.0, max_timeout=60.0, max_body_bytes=65536)
FIELDS = {"prompt", "max_tokens", "temperature", "top_k", "seed", "stop", "stream", "timeout_s"}


def _is_int(x):
    return isinstance(x, int) and not isinstance(x, bool)


def _is_num(x):
    return (isinstance(x, (int, float)) and not isinstance(x, bool)
            and x == x and abs(x) != float("inf"))


def validate_completion(body, vocab_size, limits=None):
    """The request schema. Returns clean params or raises ApiError(400, type, msg).
    Strict: unknown fields are rejected, so a client typo ("max_token") fails loudly
    instead of silently using a default."""
    L = dict(DEFAULT_LIMITS, **(limits or {}))

    def bad(type_, msg):
        raise ApiError(400, type_, msg)

    if not isinstance(body, dict):
        bad("invalid_body", "the body must be a JSON object")
    unknown = sorted(set(body) - FIELDS)
    if unknown:
        bad("unknown_field", f"unknown field(s): {unknown}")
    prompt = body.get("prompt")
    if not isinstance(prompt, str) or not prompt:
        bad("invalid_prompt", "'prompt' must be a non-empty string")
    if len(prompt) > L["max_prompt_chars"]:
        bad("prompt_too_long", f"prompt has {len(prompt)} chars, max {L['max_prompt_chars']}")
    p = dict(prompt=prompt)
    p["max_tokens"] = body.get("max_tokens", L["default_max_tokens"])
    if not _is_int(p["max_tokens"]) or not 1 <= p["max_tokens"] <= L["max_tokens_cap"]:
        bad("invalid_max_tokens", f"'max_tokens' must be an integer in [1, {L['max_tokens_cap']}]")
    p["temperature"] = body.get("temperature", 0.8)
    if not _is_num(p["temperature"]) or not 0 <= p["temperature"] <= 2:
        bad("invalid_temperature", "'temperature' must be a number in [0, 2]")
    p["top_k"] = body.get("top_k")
    if p["top_k"] is not None and (not _is_int(p["top_k"]) or not 1 <= p["top_k"] <= vocab_size):
        bad("invalid_top_k", f"'top_k' must be null or an integer in [1, {vocab_size}]")
    p["seed"] = body.get("seed")
    if p["seed"] is not None and (not _is_int(p["seed"]) or p["seed"] < 0):
        bad("invalid_seed", "'seed' must be null or a non-negative integer")
    stop = body.get("stop")
    stop = [stop] if isinstance(stop, str) else stop
    if stop is not None and (not isinstance(stop, list) or len(stop) > 4 or not all(
            isinstance(s, str) and 1 <= len(s) <= 16 for s in stop)):
        bad("invalid_stop", "'stop' must be null, a string or a list of <= 4 strings (1-16 chars)")
    p["stop"] = stop or []
    p["stream"] = body.get("stream", False)
    if not isinstance(p["stream"], bool):
        bad("invalid_stream", "'stream' must be true or false")
    p["timeout_s"] = body.get("timeout_s", L["default_timeout"])
    if not _is_num(p["timeout_s"]) or not 0 < p["timeout_s"] <= L["max_timeout"]:
        bad("invalid_timeout", f"'timeout_s' must be in (0, {L['max_timeout']}]")
    return p


RESPONSE_SCHEMA = {"id": str, "object": str, "model": str, "created": int, "choices": list,
                   "usage": dict, "timing": dict, "warnings": list}


def check_response_schema(obj):
    """List of problems with a completion response (empty = valid)."""
    errs = [f"{k}: expected {t.__name__}" for k, t in RESPONSE_SCHEMA.items()
            if not isinstance(obj.get(k), t)]
    if errs:
        return errs
    ch = obj["choices"]
    if len(ch) != 1 or not isinstance(ch[0].get("text"), str) or \
            ch[0].get("finish_reason") not in ("length", "stop"):
        errs.append("choices: expected one {text, index, finish_reason in length|stop}")
    u = obj["usage"]
    if not all(_is_int(u.get(k)) for k in ("prompt_tokens", "completion_tokens", "total_tokens")):
        errs.append("usage: expected integer prompt/completion/total_tokens")
    elif u["total_tokens"] != u["prompt_tokens"] + u["completion_tokens"]:
        errs.append("usage: total != prompt + completion")
    return errs


# ----------------------------------------------------------------------------
# The engine thread: continuous batching over the bounded queue
# ----------------------------------------------------------------------------
class Job:
    """What a handler thread and the engine thread share for one request."""

    def __init__(self, rid, req, deadline, t_arrival, engine):
        self.rid, self.req, self.deadline, self.t_arrival = rid, req, deadline, t_arrival
        self.engine = engine
        self.events = queue.Queue()  # ("token", id, t) ... ("done", reason, t)
        self.cancelled = False
        self.t_admit = self.t_first = self.t_done = None


class Engine:
    """One model + its scheduler + its worker thread. A canary model gets its own."""

    def __init__(self, model, version, max_batch, kv_blocks, block_size, queue_size, metrics,
                 log):
        self.version = version
        self.kv = PagedKVCache.for_model(model, kv_blocks, block_size)
        self.batcher = ContinuousBatcher(model, self.kv, max_batch)
        self.queue = queue.Queue(maxsize=queue_size)
        self.metrics, self.log = metrics, log
        self.jobs = {}  # req.rid -> Job (engine thread only)
        self.inflight = 0
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._loop, name="engine", daemon=True)
        metrics.kv_total.set(kv_blocks, model=version)

    def start(self):
        self.thread.start()

    def submit(self, job):
        """Backpressure: never queue more than the queue holds. False -> 429."""
        try:
            self.queue.put_nowait(job)
        except queue.Full:
            return False
        with self.lock:
            self.inflight += 1
        return True

    def depth(self):
        return self.queue.qsize() + len(self.batcher.waiting)

    def busy(self):
        return self.inflight > 0

    def _finish(self, job, reason):
        job.t_done = time.time()
        job.events.put(("done", reason, job.t_done))
        self.jobs.pop(job.req.rid, None)
        with self.lock:
            self.inflight -= 1

    def _loop(self):
        b = self.batcher
        while not self.stop_event.is_set():
            # 1. pull new requests while the batch has room (they wait in OUR bounded
            #    queue, not in an unbounded list inside the scheduler)
            while len(b.running) + len(b.waiting) < b.max_batch:
                try:
                    job = self.queue.get_nowait() if b.has_work() else self.queue.get(timeout=0.02)
                except queue.Empty:
                    break
                if job.cancelled:  # the client gave up while it was queued
                    self._finish(job, "cancelled")
                    continue
                self.jobs[job.req.rid] = job
                b.add(job.req)
            # 2. timeouts / cancellations free their KV blocks right away
            now = time.time()
            for job in list(self.jobs.values()):
                if job.cancelled or now > job.deadline:
                    b.abort(job.req, "timeout")
                    self._finish(job, "cancelled" if job.cancelled else "timeout")
            self.metrics.queue.set(self.depth(), model=self.version)
            self.metrics.inflight.set(self.inflight, model=self.version)
            self.metrics.batch.set(len(b.running), model=self.version)
            self.metrics.kv_used.set(self.kv.num_blocks - len(self.kv.free), model=self.version)
            if not b.has_work():
                continue
            # 3. one engine step
            try:
                res = b.step()
            except Exception:  # never leave handlers hanging on a dead engine
                self.log.error("engine step failed:\n%s", traceback.format_exc())
                for job in list(self.jobs.values()):
                    b.abort(job.req, "error")
                    self._finish(job, "error")
                continue
            now = time.time()
            self.metrics.steps.inc(model=self.version)
            for r, _ in res.prefill:
                job = self.jobs[r.rid]
                job.t_admit = job.t_admit or now
            for r, tok in res.tokens:
                job = self.jobs[r.rid]
                job.t_first = job.t_first or now
                job.events.put(("token", tok, now))
            for r in res.finished:
                self._finish(self.jobs[r.rid], r.finish_reason)
        # shutdown (after the drain period): whatever is left gets a clean 503
        for job in list(self.jobs.values()):
            self._finish(job, "shutdown")
        while not self.queue.empty():
            self._finish(self.queue.get_nowait(), "shutdown")


# ----------------------------------------------------------------------------
# HTTP
# ----------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"  # keep-alive; streams use chunked transfer encoding
    server_version = "MiniGPTServe/1.0"
    timeout = 5.0  # socket read timeout: a client that sends its body too slowly gets a 408

    def log_message(self, fmt, *args):  # we write structured logs instead
        pass

    @property
    def app(self):
        return self.server.app

    def _send(self, status, body, ctype="application/json", headers=()):
        data = body.encode() if isinstance(body, str) else body
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        for k, v in headers:
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, status, obj, headers=()):
        self._send(status, json.dumps(obj), headers=headers)

    def do_GET(self):
        app = self.app
        if self.path == "/health":
            ok = not app.draining
            self._send_json(200 if ok else 503, dict(
                status="ok" if ok else "draining", model=app.version,
                quantization=app.quantization, uptime_s=round(time.time() - app.t_start, 1),
                canary=app.canary_version, canary_frac=app.canary_frac,
                queue_depth=app.depth(), inflight=app.inflight()))
        elif self.path == "/metrics":
            app.metrics.uptime.set(time.time() - app.t_start)
            self._send(200, app.metrics.render(), "text/plain; version=0.0.4")
        else:
            self._send_json(404, {"error": {"type": "not_found", "message": self.path}})

    def do_POST(self):
        if self.path != "/v1/completions":
            self._send_json(404, {"error": {"type": "not_found", "message": self.path}})
            return
        app = self.app
        t_arrival = time.time()
        rid = "cmpl-" + uuid.uuid4().hex[:12]
        rec = dict(request_id=rid, t_arrival=t_arrival, client=self.client_address[0],
                   queue_depth=app.depth(), inflight=app.inflight(),
                   status=None, error_type=None)
        job = None
        try:
            if app.draining:
                raise ApiError(503, "shutting_down", "server is shutting down")
            body = self._read_json()
            if isinstance(body, dict):  # keep what we can for the log, even if invalid
                if isinstance(body.get("prompt"), str):
                    rec.update(prompt=body["prompt"], prompt_chars=len(body["prompt"]))
                rec["params"] = json.dumps({k: v for k, v in body.items() if k != "prompt"})[:200]
            p = validate_completion(body, len(app.tok), app.limits)
            ids, unknown = app.tok.encode(p["prompt"])
            rec.update(prompt=p["prompt"], prompt_chars=len(p["prompt"]),
                       prompt_tokens=len(ids), unknown_chars=len(unknown),
                       max_tokens=p["max_tokens"], temperature=p["temperature"],
                       top_k=p["top_k"], seed=p["seed"], stream=p["stream"],
                       stop=p["stop"] or None, timeout_s=p["timeout_s"],
                       prompt_truncated=len(ids) > app.block_size)
            if not ids:
                raise ApiError(400, "no_known_characters",
                               "the prompt has no characters the model knows")
            stop_ids = [app.tok.encode(s)[0] for s in p["stop"]
                        if not app.tok.encode(s)[1]]  # a stop string with unseen chars never matches
            req = Request(ids, p["max_tokens"], p["temperature"], p["top_k"], p["seed"],
                          stop=stop_ids, arrival=t_arrival)
            job = Job(rid, req, t_arrival + p["timeout_s"], t_arrival, app.route(rid))
            if not job.engine.submit(job):
                raise ApiError(429, "overloaded", "queue is full, retry later")
            warnings = []
            if unknown:
                warnings.append(f"dropped {len(unknown)} unknown character(s): "
                                f"{sorted(set(unknown))[:10]}")
            if rec["prompt_truncated"]:
                warnings.append(f"prompt longer than the {app.block_size}-token context: "
                                f"only the last {app.block_size} characters are used")
            if p["stream"]:
                self._stream(job, rec, warnings)
            else:
                self._respond(job, rec, warnings)
        except ApiError as e:
            if job is not None:
                job.cancelled = True
            rec.update(status=e.status, error_type=e.type, error=e.message)
            headers = [("Retry-After", "1")] if e.status == 429 else []
            try:
                self._send_json(e.status, {"error": {"type": e.type, "message": e.message}},
                                headers=[("X-Request-Id", rid)] + headers)
            except OSError:
                pass
        except OSError as e:  # BrokenPipe / ConnectionReset: the client went away
            if job is not None:
                job.cancelled = True
            rec.update(status=499, error_type="client_disconnected", error=type(e).__name__)
            self.close_connection = True
        except Exception as e:
            if job is not None:
                job.cancelled = True
            app.log.error("internal error:\n%s", traceback.format_exc())
            rec.update(status=500, error_type="internal_error", error=repr(e)[:200])
            try:
                self._send_json(500, {"error": {"type": "internal_error", "message": "oops"}})
            except OSError:
                pass
        finally:
            app.record(rec, job)

    def _read_json(self):
        n = self.headers.get("Content-Length")
        if n is None:
            raise ApiError(411, "missing_content_length", "Content-Length is required")
        n = int(n)
        if n > self.app.limits["max_body_bytes"]:
            self.close_connection = True
            raise ApiError(413, "body_too_large", f"body is {n} bytes")
        try:
            raw = self.rfile.read(n)
        except socket.timeout:
            self.close_connection = True
            raise ApiError(408, "slow_client", f"body not received within {self.timeout:g} s")
        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            raise ApiError(400, "invalid_json", str(e)[:100])

    def _events(self, job):
        """Yield ("token", id) / ("done", reason) from the engine, enforcing the deadline."""
        while True:
            remaining = job.deadline - time.time()
            try:
                kind, val, _ = job.events.get(timeout=max(remaining, 0.001))
            except queue.Empty:
                if time.time() >= job.deadline:
                    job.cancelled = True
                    yield "done", "timeout"
                    return
                continue
            yield kind, val
            if kind == "done":
                return

    def _result(self, job, rec, out, reason, warnings):
        if reason == "stop":  # like OpenAI-style APIs, the stop sequence is not returned
            stop = max((s for s in job.req.stop if tuple(out[-len(s):]) == s), key=len)
            out = out[:len(out) - len(stop)]
        text = self.app.tok.decode(out)
        rec.update(output=text, output_tokens=len(out), finish_reason=reason)
        return dict(id=job.rid, object="text_completion", model=job.engine.version,
                    created=int(job.t_arrival),
                    choices=[dict(text=text, index=0, finish_reason=reason)],
                    usage=dict(prompt_tokens=rec["prompt_tokens"], completion_tokens=len(out),
                               total_tokens=rec["prompt_tokens"] + len(out)),
                    timing=dict(queue_s=round((job.t_admit or time.time()) - job.t_arrival, 6),
                                ttft_s=round((job.t_first or time.time()) - job.t_arrival, 6),
                                latency_s=round(time.time() - job.t_arrival, 6)),
                    warnings=warnings)

    @staticmethod
    def _error_for(reason):
        return {"timeout": ApiError(504, "timeout", "request exceeded its timeout_s"),
                "shutdown": ApiError(503, "shutting_down", "server is shutting down"),
                "error": ApiError(500, "engine_error", "the engine failed"),
                "cancelled": ApiError(499, "client_disconnected", "cancelled")}[reason]

    def _respond(self, job, rec, warnings):
        out = []
        for kind, val in self._events(job):
            if kind == "token":
                out.append(val)
            elif val not in ("length", "stop"):
                rec.update(output_tokens=len(out), output=self.app.tok.decode(out))
                raise self._error_for(val)
            else:
                body = self._result(job, rec, out, val, warnings)
                self._send_json(200, body, headers=[("X-Request-Id", job.rid)])
                rec["status"] = 200

    def _chunk(self, data):
        """One piece of a chunked body: '<hex length>\\r\\n<data>\\r\\n'."""
        data = data.encode()
        self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
        self.wfile.flush()

    def _stream(self, job, rec, warnings):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.send_header("X-Request-Id", job.rid)
        self.end_headers()
        rec["status"] = 200
        out = []
        for kind, val in self._events(job):
            if kind == "token":
                out.append(val)
                ev = dict(id=job.rid, choices=[dict(text=self.app.tok.chars[val], index=0,
                                                    finish_reason=None)])
                self._chunk(f"data: {json.dumps(ev)}\n\n")
            elif val in ("length", "stop"):  # (streamed pieces may include the stop
                body = self._result(job, rec, out, val, warnings)  # text; this has it cut)
                self._chunk(f"data: {json.dumps(body)}\n\n")
            else:  # the status line is already sent: report the error in-band
                err = self._error_for(val)
                rec.update(error_type=err.type, error=err.message, output_tokens=len(out),
                           output=self.app.tok.decode(out), status=err.status)
                self._chunk(f"data: {json.dumps({'error': {'type': err.type}})}\n\n")
        self._chunk("data: [DONE]\n\n")
        self.wfile.write(b"0\r\n\r\n")  # the zero-length chunk ends the body
        self.wfile.flush()


class HTTPServer(ThreadingHTTPServer):
    # The kernel's queue of not-yet-accepted connections. The default (5) makes a
    # burst of clients fail with "connection reset" before our code ever sees them;
    # we want them inside, where the bounded queue can answer 429 + Retry-After.
    request_queue_size = 512
    daemon_threads = True


# ----------------------------------------------------------------------------
# The application: model + engine + HTTP server + logs
# ----------------------------------------------------------------------------
def make_request_logger(path):
    """A `logging` logger that writes each record's message (a JSON string) as one line."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    logger = logging.getLogger(f"serving.requests.{path}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for h in list(logger.handlers):
        logger.removeHandler(h)
    h = logging.FileHandler(path, encoding="utf-8")
    h.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(h)
    return logger


class InferenceServer:
    def __init__(self, model, chars, version, quantization="none", host="127.0.0.1", port=8000,
                 max_batch=16, kv_blocks=None, block_size=8, queue_size=32, limits=None,
                 log_path=os.path.join(LOG_DIR, "requests.jsonl"), log_text=True, read_timeout=5.0,
                 canary=None, canary_frac=0.0, quiet=False):
        """canary = (model, version): a second model that gets `canary_frac` of the
        traffic (set_canary() changes it live), the way new versions are rolled out."""
        T = model.config["block_size"]
        kv_blocks = kv_blocks or max_batch * -(-T // block_size)
        self.tok, self.version, self.quantization = CharTokenizer(chars), version, quantization
        self.block_size, self.limits = T, dict(DEFAULT_LIMITS, **(limits or {}))
        self.log_text = log_text
        self.log = logging.getLogger("serving.server")
        if not self.log.handlers:
            h = logging.StreamHandler(sys.stderr)
            h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
            self.log.addHandler(h)
        self.log.setLevel(logging.WARNING if quiet else logging.INFO)
        self.metrics = Metrics()
        self.metrics.info.set(1, version=version, quantization=quantization, role="stable")
        self.engines = {version: Engine(model, version, max_batch, kv_blocks, block_size,
                                        queue_size, self.metrics, self.log)}
        self.canary_version, self.canary_frac = None, 0.0
        if canary is not None:
            cm, self.canary_version = canary
            self.engines[self.canary_version] = Engine(
                cm, self.canary_version, max_batch, kv_blocks, block_size, queue_size,
                self.metrics, self.log)
            self.metrics.info.set(1, version=self.canary_version, quantization=quantization,
                                  role="canary")
            self.set_canary(canary_frac)
        self.log_path = log_path
        self.requests_log = make_request_logger(log_path)
        handler = type("ConfiguredHandler", (Handler,), {"timeout": read_timeout})
        self.httpd = HTTPServer((host, port), handler)
        self.httpd.app = self
        self.draining = False
        self.t_start = time.time()
        self._serve_thread = None

    def set_canary(self, frac):
        assert self.canary_version is not None or frac == 0
        self.canary_frac = float(frac)
        if frac:
            self.log.info("canary %s now gets %.0f%% of traffic", self.canary_version, 100 * frac)

    def route(self, rid):
        """Sticky by request id: hash -> [0, 1) -> canary if below canary_frac."""
        u = int(hashlib.sha256(rid.encode()).hexdigest()[:8], 16) / 16 ** 8
        if self.canary_version and u < self.canary_frac:
            return self.engines[self.canary_version]
        return self.engines[self.version]

    def depth(self):
        return sum(e.depth() for e in self.engines.values())

    def inflight(self):
        return sum(e.inflight for e in self.engines.values())

    @property
    def url(self):
        host, port = self.httpd.server_address[:2]
        return f"http://{host}:{port}"

    def start(self):
        for e in self.engines.values():
            e.start()
        self._serve_thread = threading.Thread(target=self.httpd.serve_forever,
                                              kwargs=dict(poll_interval=0.1), daemon=True)
        self._serve_thread.start()
        self.log.info("serving %s (%s) on %s", self.version, self.quantization, self.url)
        return self

    def shutdown(self, drain_timeout=10.0):
        """Graceful: refuse new work (503, /health -> draining), let in-flight requests
        finish (up to drain_timeout), stop the engine, close the socket, flush logs."""
        self.draining = True
        self.log.info("draining: %d in flight", self.inflight())
        t0 = time.time()
        while self.inflight() and time.time() - t0 < drain_timeout:
            time.sleep(0.05)
        for e in self.engines.values():
            e.stop_event.set()
            e.thread.join(timeout=5)
        time.sleep(0.1)  # let handler threads write their last responses and log lines
        self.httpd.shutdown()
        self.httpd.server_close()
        for h in self.requests_log.handlers:
            h.flush()
            h.close()
        self.log.info("stopped after %.1f s", time.time() - self.t_start)

    def record(self, rec, job):
        """Finish one request: metrics + one JSON log line."""
        now = time.time()
        status = rec["status"] or 500
        m = self.metrics
        m.requests.inc(status=str(status))
        if rec.get("error_type"):
            m.errors.inc(type=rec["error_type"])
        out = rec.get("output") or ""
        if job is not None and job.t_admit:
            m.prompt_tokens.inc(rec.get("prompt_tokens", 0))
            m.tokens.inc(rec.get("output_tokens", 0))
            if job.t_first:
                m.ttft.observe(job.t_first - job.t_arrival)
        latency = now - rec["t_arrival"]
        if status == 200 and not rec.get("error_type"):
            m.latency.observe(latency)
        n_out = rec.get("output_tokens", 0)
        line = dict(
            ts=datetime.fromtimestamp(rec["t_arrival"], timezone.utc).isoformat(
                timespec="milliseconds"),
            request_id=rec["request_id"],
            model_version=job.engine.version if job else self.version,
            quantization=self.quantization, status=status, error_type=rec.get("error_type"),
            error=rec.get("error"), client=rec["client"],
            params=rec.get("params") if status == 400 else None,
            prompt_chars=rec.get("prompt_chars"), prompt_tokens=rec.get("prompt_tokens"),
            unknown_chars=rec.get("unknown_chars"),
            unknown_frac=(round(rec["unknown_chars"] / rec["prompt_chars"], 4)
                          if rec.get("prompt_chars") and rec.get("unknown_chars") is not None
                          else None),
            prompt_truncated=rec.get("prompt_truncated"),
            max_tokens=rec.get("max_tokens"), temperature=rec.get("temperature"),
            top_k=rec.get("top_k"), seed=rec.get("seed"), stop=rec.get("stop"),
            stream=rec.get("stream"), timeout_s=rec.get("timeout_s"),
            output_tokens=n_out, finish_reason=rec.get("finish_reason"),
            queue_s=_r(job.t_admit - job.t_arrival) if job and job.t_admit else None,
            ttft_s=_r(job.t_first - job.t_arrival) if job and job.t_first else None,
            latency_s=_r(latency),
            tpot_s=(_r((job.t_done - job.t_first) / (n_out - 1))
                    if job and job.t_first and job.t_done and n_out > 1 else None),
            queue_depth=rec["queue_depth"], inflight=rec["inflight"],
            preemptions=job.req.n_preempted if job else 0,
            rep4=round(repetition_ratio(out), 4), loop_period=loop_period(out),
            prompt_sha=hashlib.sha256(rec.get("prompt", "").encode()).hexdigest()[:12],
            # The text itself is what makes failures debuggable and turnable into eval
            # cases, and also what can carry PII: --no-log-text keeps only the hash.
            prompt=rec.get("prompt", "")[:300] if self.log_text else None,
            output=out[:300] if self.log_text else None)
        self.requests_log.info(json.dumps(line, ensure_ascii=False))


def _r(x):
    return round(float(x), 6)


def build_server(quantize="none", port=8000, quiet=False, canary=None, **kw):
    """The mini-GPT (optionally quantized); canary="draft" adds the 6x smaller draft
    model from speculative.py as a canary version (it gets 0% until set_canary)."""
    import numpy as np
    from common import model_version
    model, chars, version = get_model(np.float32, verbose=not quiet)
    if quantize != "none":
        from quantize_numpy import quantize_model
        model = quantize_model(model, quantize, group=32)
    if canary == "draft":
        from speculative import get_draft
        draft = get_draft(model, chars, verbose=not quiet)
        kw["canary"] = (draft, model_version(draft))
    return InferenceServer(model, chars, version, quantize, port=port, quiet=quiet, **kw)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--quantize", choices=["none", "int8", "int4"], default="none",
                    help="serve int8 per-channel or int4 group-32 weights")
    ap.add_argument("--max-batch", type=int, default=16)
    ap.add_argument("--kv-blocks", type=int, default=None, help="default: max_batch x 4")
    ap.add_argument("--queue-size", type=int, default=32)
    ap.add_argument("--timeout", type=float, default=10.0, help="default per-request timeout")
    ap.add_argument("--max-prompt-chars", type=int, default=512)
    ap.add_argument("--log-file", default=os.path.join(LOG_DIR, "requests.jsonl"))
    ap.add_argument("--no-log-text", action="store_true", help="log prompt hashes, not text")
    ap.add_argument("--canary", choices=["draft"], default=None,
                    help="also serve the small draft model as a canary version")
    ap.add_argument("--canary-frac", type=float, default=0.1)
    args = ap.parse_args()

    srv = build_server(args.quantize, args.port, host=args.host, max_batch=args.max_batch,
                       kv_blocks=args.kv_blocks, queue_size=args.queue_size,
                       limits=dict(default_timeout=args.timeout,
                                   max_prompt_chars=args.max_prompt_chars),
                       log_path=args.log_file, log_text=not args.no_log_text,
                       canary=args.canary,
                       canary_frac=args.canary_frac if args.canary else 0.0).start()
    print(f"\nListening on {srv.url}  (logs -> {os.path.relpath(args.log_file)})")
    print(f"  curl -s {srv.url}/v1/completions -d '{{\"prompt\": \"The keeper\", "
          f"\"max_tokens\": 60}}'")
    print(f"  curl -sN {srv.url}/v1/completions -d '{{\"prompt\": \"The sea\", "
          f"\"stream\": true}}'")
    print(f"  curl -s {srv.url}/metrics     curl -s {srv.url}/health")
    print("Ctrl+C for a graceful shutdown.\n")

    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    while not stop.is_set():
        stop.wait(0.5)
    print("\nshutting down gracefully...")
    srv.shutdown()


if __name__ == "__main__":
    main()
