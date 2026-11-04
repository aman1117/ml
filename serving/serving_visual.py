"""
Watch the inference server work, step by step, in your terminal.

Part 1, the life of ONE request, with the real mini-GPT:
  raw HTTP bytes -> parse -> validate -> tokenize -> bounded queue -> admission
  (KV blocks from the free list) -> PREFILL (the prompt's K, V written into
  paged blocks) -> DECODE steps (one token each; a new block when one fills up)
  -> server-sent events on the wire -> the JSON log line.

Part 2, CONTINUOUS BATCHING: six requests arrive at different times; the
scheduler admits them as slots and KV blocks free up. You see each request's
state per engine step, and which request owns each KV block, side by side with
static batching of the same requests.

Uses transformer/model.npz if it exists, otherwise trains it first (~10 s).

Run:
    python serving/serving_visual.py            (press Enter for the next step)
    python serving/serving_visual.py --no-pause
    python serving/serving_visual.py --prompt "She climbed" --tokens 12
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from batching import ContinuousBatcher, PagedKVCache, Request, static_generate
from common import CharTokenizer, get_model, repetition_ratio, softmax
from server import ApiError, validate_completion

TTY = sys.stdout.isatty()
PAUSE = True
LETTERS = "ABCDEF"


def color(code, s):
    return f"\033[{code}m{s}\033[0m" if TTY else s


def cyan(s): return color("36;1", s)      # requests / inputs
def yellow(s): return color("33;1", s)    # scheduler / memory
def green(s): return color("32;1", s)     # outputs
def red(s): return color("31;1", s)       # rejections
def dim(s): return color("2", s)


def pause():
    if PAUSE and sys.stdin.isatty():
        try:
            input(dim("\n  [Enter] next step "))
        except EOFError:
            print()
    print("\n" + "=" * 78 + "\n")


def ch(c):
    return {" ": "␣", "\n": "↵"}.get(c, c)


def bar(p, width=20):
    n = int(round(p * width))
    return "█" * n + "·" * (width - n)


def show_blocks(kv, seqs, tok, ids_of, names=None):
    """Each block as [c c c c]: the characters whose K, V sit in its slots."""
    names = names or {}
    for s in seqs:
        table = kv.tables.get(s, [])
        ids = ids_of(s)
        cells = []
        for b, blk in enumerate(table):
            slots = [ch(tok.chars[ids[b * kv.block_size + i]])
                     if b * kv.block_size + i < kv.lengths.get(s, 0) else "·"
                     for i in range(kv.block_size)]
            cells.append(f"#{blk:<2d}[" + " ".join(slots) + "]")
        print(f"    {names.get(s, s)}: block table {table}   " + "  ".join(cells))


def part1(model, tok, version, prompt, n_tokens):
    body = {"prompt": prompt, "max_tokens": n_tokens, "temperature": 0.8, "seed": 7,
            "stream": True}
    raw = json.dumps(body)
    http = (f"POST /v1/completions HTTP/1.1\r\nHost: 127.0.0.1:8000\r\n"
            f"Content-Type: application/json\r\nContent-Length: {len(raw)}\r\n\r\n{raw}")
    print(cyan("Step 1. The bytes that arrive on the socket"))
    for line in http.split("\r\n"):
        print("    " + (line if line else dim("(empty line: end of headers)")))
    print(dim("\n  A handler thread (one per connection, ThreadingHTTPServer) reads the request"
              "\n  line and headers, then exactly Content-Length bytes of body."))
    pause()

    print(cyan("Step 2. Parse + validate (strict schema; 400 with an error type if wrong)"))
    p = validate_completion(json.loads(raw), len(tok))
    for k, v in p.items():
        print(f"    {k:12s} = {v!r:28s} {green('✓')}")
    for bad in ({"prompt": prompt, "temperature": -1}, {"prompt": prompt, "max_token": 5},
                {"prompt": "x" * 600}):
        try:
            validate_completion(bad, len(tok))
        except ApiError as e:
            shown = {k: (v if len(str(v)) < 20 else str(v)[:12] + "...") for k, v in bad.items()}
            print(f"    {red('rejected')} {json.dumps(shown)} -> {e.status} {e.type}")
    pause()

    print(cyan("Step 3. Tokenize: characters -> ids (unknown characters are dropped + logged)"))
    ids, unknown = tok.encode(prompt)
    print("    " + "  ".join(f"{ch(c)}={i}" for c, i in zip(prompt, ids)))
    _, unk = tok.encode(prompt + "!")
    print(f"    (if the client had sent {prompt + '!'!r}: {unk} is not in the {len(tok)}-char "
          f"vocabulary -> dropped, a warning in the response, unknown_chars=1 in the log)")
    pause()

    print(cyan("Step 4. The bounded queue and admission"))
    print("    queue: " + yellow("[■ ■ □ □ □ □ □ □]") + "  2 waiting of 8 allowed; ours is #3")
    print("    (if it were full: 429 Too Many Requests + Retry-After: backpressure, not an"
          "\n     ever-growing queue whose requests would all time out anyway)")
    kv = PagedKVCache.for_model(model, num_blocks=12, block_size=4)
    kv.free.rotate(5)  # pretend other requests used blocks before: physical ids are scattered
    eng = ContinuousBatcher(model, kv, max_batch=4)
    req = Request(ids, n_tokens, 0.8, seed=7, rid="req")
    print(f"    free list: {list(kv.free)}")
    eng.add(req)
    res = eng.step()  # admission + prefill + first token
    print(f"    admitted: needs ceil({len(ids)} / {kv.block_size}) = {len(kv.tables['req'])} "
          f"blocks -> block table {kv.tables['req']}; free list now {list(kv.free)}")
    pause()

    print(cyan(f"Step 5. PREFILL: all {len(ids)} prompt tokens in ONE forward pass"))
    print("    Every layer's keys and values for every prompt token go into the request's"
          "\n    blocks (2 layers x 4 heads x 16 dims each, for K and for V):")
    show_blocks(kv, ["req"], tok, lambda s: req.ids, {"req": "req"})
    print(f"    The last position's logits give the first token: {green(repr(tok.chars[req.out[0]]))}"
          f"   (TTFT = queue wait + this one pass)")
    print(dim("    Prefill is compute-bound: many tokens share each weight read (batching.py)."))
    pause()

    print(cyan("Step 6. DECODE: one token per engine step, reading the whole cache"))
    T = model.config["block_size"]
    snapshot = None
    for step in range(1, n_tokens):
        if step == n_tokens - 1:  # remember the cache before the last step frees it
            snapshot = ({"req": list(kv.tables["req"])}, dict(kv.lengths))
        before = list(kv.tables["req"])
        pos = req.cached
        # the logits this step will sample from (same computation as the engine's decode)
        x = model.wte.params["weight"][req.ids[-1]]
        res = eng.step()
        probs = None
        if res.tokens:
            tok_id = res.tokens[0][1]
            new_block = [b for b in kv.tables.get("req", []) if b not in before]
            note = yellow(f"  block {new_block[0]} allocated (slot {pos} starts a new block)") \
                if new_block else ""
            print(f"    step {step:2d}: feed {ch(tok.chars[req.ids[-2]])!s:2s} at position "
                  f"{pos:2d}, attend over {pos + 1:2d} cached tokens -> "
                  f"{green(repr(tok.chars[tok_id]))}{note}")
        del x, probs
    if snapshot:
        print("    the cache before the last step:")
        kv.tables, kv.lengths = snapshot
        show_blocks(kv, ["req"], tok, lambda s: req.ids, {"req": "req"})
        kv.tables, kv.lengths = {}, {}
    print(dim(f"    Each decode step reads ALL weights for ONE token: memory-bound. That is why"
              f"\n    the server batches many requests into each step (part 2). Window: {T}."))
    if not eng.has_work():
        print(f"    finished ({req.finish_reason}): blocks returned to the free list -> "
              f"{len(kv.free)} free")
    pause()

    print(cyan("Step 7. Streaming: server-sent events in HTTP chunks"))
    print("    HTTP/1.1 200 OK\n    Content-Type: text/event-stream\n"
          "    Transfer-Encoding: chunked\n")
    for i, t in enumerate(req.out[:4]):
        data = f'data: {{"choices": [{{"text": {json.dumps(tok.chars[t])}, "index": 0}}]}}\n\n'
        print(f"    {len(data.encode()):x}\\r\\n{data.strip()}\\n\\n\\r\\n")
    print("    ...\n    e\\r\\ndata: [DONE]\\n\\n\\r\\n\n    0\\r\\n\\r\\n   <- zero-length chunk ends the body")
    pause()

    print(cyan("Step 8. One structured log line (what monitor.py reads later)"))
    text = tok.decode(req.out)
    line = dict(ts="2026-11-04T10:00:00.000+00:00", request_id="cmpl-3f9a1c2b7d10",
                model_version=version, status=200, error_type=None, prompt_chars=len(prompt),
                prompt_tokens=len(ids), unknown_chars=len(unknown), max_tokens=n_tokens,
                temperature=0.8, seed=7, stream=True, output_tokens=len(req.out),
                finish_reason=req.finish_reason, ttft_s=0.0006, latency_s=0.0031,
                rep4=round(repetition_ratio(text), 4), prompt=prompt, output=text)
    print("    " + json.dumps(line))
    print(green(f"\n    {prompt}") + green(text))
    pause()


def timeline(model, specs, policy, max_batch=3, num_blocks=14, block_size=4, steps=60):
    """Run a few requests step by step; record each one's state and the block owners."""
    reqs = [Request(p, n, 0.0, rid=LETTERS[i]) for i, (p, n, _) in enumerate(specs)]
    arrive = {r.rid: a for r, (_, _, a) in zip(reqs, specs)}
    rows = {r.rid: [] for r in reqs}
    blocks = []
    if policy == "continuous":
        kv = PagedKVCache.for_model(model, num_blocks, block_size)
        eng = ContinuousBatcher(model, kv, max_batch)
        for s in range(steps):
            for r in reqs:
                if arrive[r.rid] == s:
                    eng.add(r)
            res = eng.step() if eng.has_work() else None
            pre = {r.rid for r, _ in res.prefill} if res else set()
            pree = {r.rid for r in res.preempted} if res else set()
            owner = ["·"] * num_blocks
            for sid, table in kv.tables.items():
                for b in table:
                    owner[b] = sid
            blocks.append("".join(owner))
            for r in reqs:
                if s < arrive[r.rid]:
                    c = " "
                elif r.rid in pree:
                    c = "x"
                elif r.rid in pre:
                    c = "P"
                elif r in eng.running:
                    c = "█" if res and r in res.decoded else "·"
                elif r in eng.waiting:
                    c = "."
                elif r.finish_reason and res and r in res.finished:
                    c = "█"
                else:
                    c = " "
                rows[r.rid].append(c)
            if not eng.has_work() and s > max(arrive.values()):
                break
    else:  # static: take up to max_batch waiting requests, run them to the longest end
        queue, done, active, s = [], set(), None, 0
        while len(done) < len(reqs) and s < steps:
            queue += [r for r in reqs if arrive[r.rid] == s]
            if active is None and queue:
                batch, queue = queue[:max_batch], queue[max_batch:]
                static_generate(model, batch)
                active = (batch, s, max(r.max_tokens for r in batch))
            for r in reqs:
                if active and r in active[0]:
                    k = s - active[1]
                    c = "P" if k == 0 else ("█" if k < r.max_tokens else "░")
                else:
                    c = "." if r in queue else " "
                rows[r.rid].append(c)
            blocks.append("")
            s += 1
            if active and s - active[1] >= active[2]:
                done |= {r.rid for r in active[0]}
                active = None
    return rows, blocks, reqs


def part2(model, tok):
    texts = ["The keeper", "She", "One winter night", "The sea", "Good morning", "The girl"]
    lens = [6, 14, 4, 10, 8, 5]
    arrivals = [0, 0, 1, 2, 5, 6]
    specs = [(tok.encode(t)[0], n, a) for t, n, a in zip(texts, lens, arrivals)]
    print(cyan("Part 2. Continuous batching vs static batching, same six requests"))
    for (t, n, a), L in zip(zip(texts, lens, arrivals), LETTERS):
        print(f"    {L}: {t!r:18s} wants {n:2d} tokens, arrives at step {a}")
    print("    max batch 3, KV pool of 14 blocks x 4 tokens\n")
    print(dim("    legend:  . waiting   P prefill (+ first token)   █ decode step   x preempted"
              "\n             ░ static: finished but its slot still computes"))
    pause()

    rows, blocks, reqs = timeline(model, specs, "continuous")
    n = len(blocks)
    print(cyan("Continuous batching (iteration-level scheduling)") + "   step:")
    print("        " + "".join(f"{s % 10}" for s in range(n)))
    for L in LETTERS[:len(specs)]:
        print(f"     {L}  " + "".join(rows[L]))
    print(yellow("\n    KV block owners per step (rows = steps, columns = the 14 physical "
                 "blocks):"))
    for s in range(0, n, 2):
        print(f"    step {s:2d}  " + " ".join(blocks[s]))
    print(dim("    Blocks are taken from the free list as a request grows and returned the\n"
              "    moment it finishes; a new arrival starts as soon as a slot AND blocks are\n"
              "    free. When the pool runs dry, the newest request is preempted (x): its\n"
              "    blocks are freed and it is recomputed later (vLLM's recompute policy)."))
    cont_end = n
    pause()

    rows, blocks, _ = timeline(model, specs, "static")
    n = len(blocks)
    print(cyan("Static batching (request-level): a batch runs until its LONGEST request ends"))
    print("        " + "".join(f"{s % 10}" for s in range(n)))
    for L in LETTERS[:len(specs)]:
        print(f"     {L}  " + "".join(rows[L]))
    print(f"\n    Continuous: all done after {cont_end} steps. Static: {n} steps, and the ░ cells"
          f"\n    are slots burning compute on finished requests while others wait (.).")
    pause()


def main():
    global PAUSE
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompt", default="The keeper")
    ap.add_argument("--tokens", type=int, default=10)
    ap.add_argument("--no-pause", action="store_true")
    args = ap.parse_args()
    PAUSE = not args.no_pause
    model, chars, version = get_model()
    tok = CharTokenizer(chars)
    part1(model, tok, version, args.prompt, args.tokens)
    part2(model, tok)
    print(green("Done. Next: python serving/server.py, then python serving/load_test.py "
                "--spawn-server"))


if __name__ == "__main__":
    main()
