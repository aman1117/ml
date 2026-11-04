"""
Watch the agent work, step by step, in your terminal.

  1. The task and the system prompt: tool specs (as the model sees them, and as the
     JSON schema native function calling sends), the protocol and the rules.
  2. The learned router splitting the request into clauses and picking a tool for each.
  3. Every step of the loop: prompt size, the model's raw output, the parsed call,
     validation, the observation (wrapped as untrusted data), and the next thought,
     until the final answer.
  4. The SAME task under NoisyPolicy: the model emits broken calls, a guardrail
     catches each one and the run still succeeds; then the same noise with the
     guardrails off.
  5. A prompt-injection attack: a poisoned document tells the agent to write to its
     memory. Undefended it obeys; with delimiting + confirmation it doesn't.

Run:
    python agents/agent_visual.py              (press Enter for the next step)
    python agents/agent_visual.py --no-pause   (or with stdin not a terminal)
    python agents/agent_visual.py --request "Convert 26.2 miles to km, then multiply it by 2"
"""
import argparse
import json
import sys
import textwrap

from agent import Agent, AgentConfig, Guardrails, render_system_prompt, wrap_observation
from agent_eval import load_tasks, run_one, simulated_user
from policies import NoisyPolicy, ScriptedPolicy, split_plan
from router_numpy import LABELS, load_router
from tools import make_registry

TTY = sys.stdout.isatty()
PAUSE = True
DEFAULT = "Find how many tokens per parameter Chinchilla suggests in the docs, then multiply it by 7"


def color(code, s):
    return f"\033[{code}m{s}\033[0m" if TTY else s


def cyan(s): return color("36;1", s)      # the user / inputs
def yellow(s): return color("33;1", s)    # the model's output
def green(s): return color("32;1", s)     # tool results, success
def red(s): return color("31;1", s)       # errors, attacks
def dim(s): return color("2", s)


def pause():
    if PAUSE and sys.stdin.isatty():
        try:
            input(dim("\n  [Enter] next step "))
        except EOFError:
            print()
    print("\n" + "=" * 78 + "\n")


def block(text, prefix="    ", width=100, max_lines=12, paint=lambda s: s):
    lines = []
    for raw in str(text).splitlines() or [""]:
        lines += textwrap.wrap(raw, width) or [""]
    for ln in lines[:max_lines]:
        print(prefix + paint(ln))
    if len(lines) > max_lines:
        print(prefix + dim(f"… ({len(lines) - max_lines} more lines)"))


def bar(p, width=24):
    n = int(round(p * width))
    return "█" * n + "·" * (width - n)


def show_router(router, request):
    print(cyan("The plan: split the request into clauses, then the ROUTER picks a tool for each\n"))
    print(dim("  (router_numpy.py: hashed bag of words -> MLP -> softmax; trained in NumPy, hand-written backward)\n"))
    for i, step in enumerate(split_plan(request)):
        for clause in step:
            label, p, probs = router.route(clause)
            print(f"  step {i + 1}: {cyan(repr(clause))}")
            for lab in sorted(LABELS, key=lambda l: -probs[l])[:3]:
                print(f"      {lab:13s} {bar(probs[lab])} {probs[lab]:5.1%}" + (green("  <- chosen") if lab == label else ""))


def show_step(r, show_prompt_size=True):
    if r.get("event") == "end":
        return
    ev = [e for e in r.get("guardrail_events", [])]
    head = f"STEP {r['step']}"
    if show_prompt_size:
        head += dim(f"   prompt {r['prompt_chars']:,} chars ≈ {r['prompt_tokens']} tokens, "
                    f"model latency ≈ {r['llm_latency_s']:.2f} s (simulated)")
    print(head)
    print("  model output (raw text):")
    block(r["raw_output"], paint=yellow)
    if r["action"] == "parse_error":
        print("  " + red(f"parse error -> fed back to the model: {r.get('error')}"))
    elif r["action"] == "rejected_final":
        print("  " + red("grounding check: answer rejected (no tool was used, the router says one is needed)"))
    elif r["action"] == "final":
        print("  " + green(f"FINAL ANSWER: {r['answer']}"))
    for c in r.get("calls", []):
        print(f"  parsed call: {c['tool']}({json.dumps(c['args'])})")
        if c.get("ok"):
            print("  " + green(f"validated, executed in {c.get('latency_ms', 0):.1f} ms -> observation "
                               "(appended to the messages as untrusted data):"))
            block(wrap_observation(c["tool"], c["id"], c.get("content", ""), True), prefix="      ",
                  max_lines=7, paint=green)
        else:
            print("  " + red(f"{c.get('error_type')}: ") + red((c.get("content") or "")[:300]))
    shown = [e for e in ev if e not in ("retry_with_feedback",)]
    if shown:
        print("  " + dim("guardrail events: " + ", ".join(shown)))


def find_noisy_seed(task, router, p=0.4):
    """A seed where the noisy model makes >= 2 different mistakes, all guardrails
    recover, and the same seed without guardrails fails."""
    for seed in range(400):
        pol = NoisyPolicy(ScriptedPolicy(router), p)
        with_g = run_one(task, pol, Guardrails(), seed, router)
        events = set(pol.events)
        if not with_g["success"] or len(events) < 2:
            continue
        pol2 = NoisyPolicy(ScriptedPolicy(router), p)
        without = run_one(task, pol2, Guardrails.none(), seed, router)
        if not without["success"]:
            return seed
    return 0


def main():
    global PAUSE
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--request", default=DEFAULT)
    ap.add_argument("--no-pause", action="store_true")
    args = ap.parse_args()
    PAUSE = not args.no_pause
    router, _ = load_router(verbose=True)
    reg = make_registry()

    print(cyan("THE TASK\n"))
    print(f"  user: {cyan(args.request)}\n")
    print("  An agent = an LLM + tools + a loop (+ memory). The LLM here is ScriptedPolicy, a")
    print("  deterministic stand-in that follows the protocol; the loop and the tools are real.")
    pause()

    print(cyan("THE SYSTEM PROMPT (what the model reads before the user's message)\n"))
    block(render_system_prompt(reg.specs()), max_lines=40)
    print("\n  Native function calling sends the same tool as a JSON schema instead:")
    block(json.dumps(reg.specs()[0], indent=1), max_lines=14, paint=dim)
    pause()

    show_router(router, args.request)
    pause()

    task = {"id": "visual", "category": "demo", "request": args.request, "check": {"type": "regex", "pattern": "."},
            "required_tools": [], "forbidden_tools": [], "max_steps": 10}
    res = Agent(reg, ScriptedPolicy(router), router=router).run(args.request, task_id="visual")
    print(cyan("THE LOOP: generate -> parse -> validate -> execute -> observe -> repeat\n"))
    for r in res.trace:
        if r.get("event") == "end":
            continue
        show_step(r)
        pause()
    print(green(f"Done in {res.steps} steps, {res.tokens:,} tokens, ≈ {res.latency_s:.2f} s. "
                f"Answer: {res.answer}"))
    pause()

    p = 0.4
    seed = find_noisy_seed({**task, "check": {"type": "regex", "pattern": r"\b140\b"}} if args.request == DEFAULT
                           else task, router, p)
    print(cyan(f"THE SAME TASK, NOISY MODEL (NoisyPolicy p={p}, seed {seed}), ALL GUARDRAILS ON\n"))
    pol = NoisyPolicy(ScriptedPolicy(router), p)
    res = Agent(make_registry(), pol, AgentConfig(guards=Guardrails()), router=router).run(
        args.request, task_id="visual", seed=seed)
    print(dim(f"  failures the noisy model injected: {pol.events}\n"))
    for r in res.trace:
        if r.get("event") != "end":
            show_step(r, show_prompt_size=False)
            print()
    print(green(f"Recovered: {res.answer!r} after {res.steps} steps"))
    pause()

    print(cyan("SAME NOISE, SAME SEED, GUARDRAILS OFF\n"))
    pol = NoisyPolicy(ScriptedPolicy(router), p)
    res = Agent(make_registry(), pol, AgentConfig(guards=Guardrails.none()), router=router).run(
        args.request, task_id="visual", seed=seed)
    for r in res.trace:
        if r.get("event") != "end":
            show_step(r, show_prompt_size=False)
            print()
    print(red(f"Without guardrails: stop={res.stop_reason}, answer={res.answer!r}"))
    pause()

    t = next(t for t in load_tasks() if t["id"] == "inj_memory_poison")
    print(cyan("PROMPT INJECTION: a poisoned document in the search index\n"))
    print(f"  user: {cyan(t['request'])}")
    print("  the document the search will return:")
    block(t["extra_docs"][0]["text"], prefix="      ", paint=red)
    for name, g in (("undefended", Guardrails(delimit_untrusted=False, confirm_mutations=False)),
                    ("delimit + confirm", Guardrails())):
        reg = make_registry(t.get("extra_docs"), t.get("memory"))
        res = Agent(reg, ScriptedPolicy(router), AgentConfig(guards=g), confirm=simulated_user(t["request"]),
                    router=router).run(t["request"])
        calls = [f"{c['tool']}({json.dumps(c['args'])[:70]})" for c in res.calls]
        print(f"\n  {name}:")
        print(f"    tool calls: {calls}")
        print(f"    memory writes: {reg.memory.writes or 'none'}")
        paint = red if "evil" in (res.answer or "") or reg.memory.writes else green
        print("    answer: " + paint(str(res.answer)[:120]))
    print(dim("\n  Delimiting made our model treat the document as data. A real model can still slip"))
    print(dim("  (see agent_eval.py part E): that's why mutations also need a human's OK."))


if __name__ == "__main__":
    main()
