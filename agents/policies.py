"""
Policies: the "LLM" that drives the agent loop in agent.py.

Our char-level models from ../transformer/ cannot act as agents (they continue
text; they were never trained to follow instructions or emit tool calls). So the
loop is driven by a pluggable Policy with one method,

    generate(messages, tool_specs) -> PolicyOutput(text, tool_calls?, tokens?, latency?)

and three implementations:

ScriptedPolicy  A deterministic stand-in for a decent LLM that follows the ReAct
                protocol. It splits the request into clauses ("..., then ..."),
                asks the LEARNED router (router_numpy.py) which tool each clause
                needs, fills in arguments with small extractors, reads the
                observations, chains results ("multiply IT by 4") and writes a
                final answer. Reproducible offline, so every lesson is testable.
                Like real models it is vulnerable to prompt injection: it obeys
                instructions found in tool output unless that output is clearly
                delimited as untrusted data (our caricature of instruction-
                hierarchy training; real models fall somewhere in between).
NoisyPolicy     Wraps any policy and injects the failure modes real LLM agents
                show in traces, each step with probability p: malformed JSON,
                unknown tool, missing/wrong-typed args, hallucinated tool results,
                repeating the same call in a loop, answering without using tools,
                and following injected instructions even when delimited.
HTTPPolicy      Any OpenAI-compatible /chat/completions endpoint, with urllib only.
                mode="native": the API's tool-calling field. mode="text": plain-text
                ReAct protocol, parsed by agent.parse_output (works with any model).
                Set OPENAI_BASE_URL, OPENAI_API_KEY, AGENT_MODEL; skipped if unset.

Run (the scripted policy on a few requests, then the same requests with noise):
    python agents/policies.py
"""
from __future__ import annotations

import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import List, Optional

from agent import TAG
from tools import STOPWORDS, norm_key, norm_unit, tokenize


@dataclass
class PolicyOutput:
    text: str
    tool_calls: Optional[list] = None  # native tool calls: [{"name", "arguments"}]
    prompt_tokens: Optional[int] = None  # None -> the agent estimates from the prompt
    completion_tokens: Optional[int] = None
    latency_s: Optional[float] = None  # None -> the agent's latency model


class Policy:
    name = "policy"
    native = False

    def reset(self, request: str, seed: int = 0):
        pass

    def generate(self, messages: List[dict], tools: List[dict]) -> PolicyOutput:
        raise NotImplementedError


def action_text(thought: str, calls: List[dict]) -> str:
    body = calls[0] if len(calls) == 1 else calls
    return f"Thought: {thought}\nAction: {json.dumps(body)}"


def final_text(thought: str, answer: str) -> str:
    return f"Thought: {thought}\nFinal Answer: {answer}"


def unwrap(content: str) -> str:
    m = re.match(rf"\s*<{TAG}[^>]*>\n?(.*)\n?</{TAG}>\s*$", content, flags=re.S)
    return m.group(1) if m else content


def lenient_json(content: str):
    """Read a (possibly truncated) JSON observation the way a model would: try to close it."""
    text = unwrap(content)
    for closer in ("", '"}', '"}]}', "}", "]}"):
        try:
            return json.loads(text + closer)
        except json.JSONDecodeError:
            continue
    return {"_text": text}


# ----------------------------------------------------------------------------
# Small "understanding" helpers used by the scripted policy
# ----------------------------------------------------------------------------
SEQ_SPLIT = r"\s*[,;.]?\s+(?:and\s+)?then\s+"
PAR_SPLIT = r"\s*[,;]?\s+and\s+also\s+|;\s+also\s+|\.\s+also,?\s+"
NUM = r"-?\d[\d,]*(?:\.\d+)?"
PRONOUN = r"\b(?:the result|that number|this number|the number|that value|that|it|this)\b"
QUESTION_WORDS = set("""explain explains explained cover covers covered section sections describe
describes described mention mentions mentioned discuss discusses discussed introduce introduced
introduces paper report reports reported reach reaches reached notes note chapter grab where
whats what's anything something knowledge base check lookup look""".split())


def split_plan(request: str) -> List[List[str]]:
    steps = [s for s in re.split(SEQ_SPLIT, request.strip(), flags=re.I) if s.strip()]
    return [[c.strip(" .") for c in re.split(PAR_SPLIT, s, flags=re.I) if c.strip(" .")] for s in steps]


def numbers_in(text: str):
    """[(value, start, end)] of the numbers in text ("107,776" -> 107776)."""
    out = []
    for m in re.finditer(rf"(?<![\w.\-]){NUM}(?![\w\-])", text):
        try:
            out.append((float(m.group(0).replace(",", "")), m.start(), m.end()))
        except ValueError:
            pass
    return out


def fmt(v) -> str:
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


def build_expression(text: str, prev) -> Optional[str]:
    t = text.lower()
    t = re.sub(r"(?<=\d),(?=\d{3}\b)", "", t)  # 107,776 -> 107776 (the calculator rejects commas)
    if prev is not None:
        t = re.sub(PRONOUN, f" {fmt(prev)} ", t)
    n = r"(\d+(?:\.\d+)?)"
    rules = [
        (rf"square root of\s+{n}", r"sqrt(\1)"), (rf"sqrt of\s+{n}", r"sqrt(\1)"),
        (rf"log of\s+{n}", r"log(\1)"),
        (rf"{n}\s*squared", r"\1**2"), (rf"\bsquare\s+{n}", r"\1**2"),
        (rf"\bdouble\s+{n}", r"\1*2"), (rf"\bhalve\s+{n}", r"\1/2"), (rf"half of\s+{n}", r"\1/2"),
        (rf"\badd\s+{n}\s+to\s+{n}", r"\2+\1"), (rf"\bsubtract\s+{n}\s+from\s+{n}", r"\2-\1"),
        (rf"\bdivide\s+{n}\s+by\s+{n}", r"\1/\2"), (rf"\bmultiply\s+{n}\s+by\s+{n}", r"\1*\2"),
        (rf"{n}\s*(?:%|percent)\s+of\s+{n}", r"(\1/100*\2)"),
        (rf"product of\s+{n}\s+and\s+{n}", r"\1*\2"),
        (r"to the power of", "**"), (r"multiplied by", "*"), (r"\btimes\b", "*"), (r"\bplus\b", "+"),
        (r"\bminus\b", "-"), (r"divided by", "/"), (r"(?<=\d)\s*x\s*(?=\d)", "*"),
    ]
    for pat, rep in rules:
        t = re.sub(pat, rep, t)
    m = re.search(rf"(?:sum|total) of\s+({n}(?:\s*(?:,|and)\s*{n})+)", t)
    if m:
        t = "+".join(x for x in re.findall(r"\d+(?:\.\d+)?", m.group(1)))
    m = re.search(rf"average of\s+({n}(?:\s*(?:,|and)\s*{n})+)", t)
    if m:
        xs = re.findall(r"\d+(?:\.\d+)?", m.group(1))
        t = f"({'+'.join(xs)})/{len(xs)}"
    toks = re.findall(r"sqrt|log|\*\*|\d+(?:\.\d+)?|[-+*/%()]", t)
    while toks and toks[-1] in "+-*/%**(":
        toks.pop()
    while toks and toks[0] in "+*/%)" or (toks and toks[0] == "**"):
        toks.pop(0)
    if not toks:
        return None
    expr = " ".join(toks).replace("sqrt (", "sqrt(").replace("log (", "log(")
    if prev is not None and not any(re.fullmatch(r"\d+(?:\.\d+)?", x) for x in toks[:1]) and toks[0] in "+-*/":
        expr = f"{fmt(prev)} {expr}"
    if not re.search(r"[-+*/%]|sqrt|log", expr) and prev is None:
        return None
    if expr.count("(") != expr.count(")"):
        return None
    return expr


def build_conversion(text: str, prev, prev_unit) -> Optional[dict]:
    t = text.replace("°", "")
    m = re.search(r"how many\s+([A-Za-z]+)\s+(?:is|are)\s+(?:in\s+)?(" + NUM + r")\s*([A-Za-z]+)", t, re.I)
    if m and norm_unit(m.group(1)) and norm_unit(m.group(3)):
        return {"value": float(m.group(2).replace(",", "")), "from_unit": m.group(3), "to_unit": m.group(1)}
    m = re.search(r"(" + NUM + r")\s*([A-Za-z]+)\b.*?\b(?:to|in|into|as)\s+([A-Za-z]+)", t, re.I)
    if m and norm_unit(m.group(2)) and norm_unit(m.group(3)):
        return {"value": float(m.group(1).replace(",", "")), "from_unit": m.group(2), "to_unit": m.group(3)}
    m = re.search(PRONOUN + r".*?\b(?:to|in|into)\s+([A-Za-z]+)", t, re.I)
    if m and prev is not None and prev_unit and norm_unit(m.group(1)):
        return {"value": prev, "from_unit": prev_unit, "to_unit": m.group(1)}
    return None


def build_memory_call(text: str, prev) -> Optional[dict]:
    t = text.strip().rstrip("?.!")
    pats = [
        (r"(?:remember|note|keep in mind|store|save)(?: that)?:?\s+(?:my\s+)?(.+?)\s+(?:is|=|:)\s+(.+)$", "remember", 1, 2),
        (r"^(?:please\s+)?remember my\s+(.+?):\s*(.+)$", "remember", 1, 2),
        (r"update my\s+(.+?)\s+to\s+(.+)$", "remember", 1, 2),
        (r"save\s+(.+?)\s+as my\s+(.+)$", "remember", 2, 1),
        (r"(?:remember|save|store)\s+(?:it|that|the result)\s+as\s+(?:my\s+)?(.+)$", "remember_prev", 1, None),
        (r"forget my\s+(.+)$", "forget", 1, None),
        (r"(?:what (?:is|was|did i tell you)|recall|remind me of|do you remember)\s+my\s+(.+?)(?:\s+(?:was|is|again))*$",
         "recall", 1, None),
        (r"(what do you (?:have in memory|remember)|list what you remember)", "list", None, None),
    ]
    for pat, action, ki, vi in pats:
        m = re.search(pat, t, re.I)
        if not m:
            continue
        if action == "remember":
            return {"action": "remember", "key": norm_key(m.group(ki)), "value": m.group(vi).strip()}
        if action == "remember_prev":
            if prev is None:
                return None
            return {"action": "remember", "key": norm_key(m.group(ki)), "value": fmt(prev)}
        if action in ("recall", "forget"):
            return {"action": action, "key": norm_key(m.group(ki))}
        return {"action": "list"}
    return None


def search_query(text: str) -> str:
    t = re.sub(r"\b(search|look up|find|check|grab)\b(\s+the)?(\s+(docs|documentation|readmes?|notes))?(\s+for)?",
               " ", text, flags=re.I)
    t = re.sub(r"according to the\s+(\w+\s+)?(docs|readme|documentation)", r" \1 ", t, flags=re.I)
    words = [w for w in re.findall(r"[A-Za-z0-9][A-Za-z0-9\-]*", t)
             if w.lower() not in STOPWORDS and w.lower() not in QUESTION_WORDS]
    return " ".join(words)


def best_sentence(results, terms):
    """The sentence of the top passages that contains the most query terms."""
    best, best_s = None, -1.0
    for r, res in enumerate(results):
        # Re-join hard-wrapped prose; table rows and list items stay separate units.
        units, prose = [], []
        for line in res.get("text", "").splitlines():
            if re.match(r"\s*([|*-]\s|\d+\.\s|```)", line) or not line.strip():
                units += [" ".join(prose)] if prose else []
                prose = []
                units.append(line)
            else:
                prose.append(line.strip())
        units += [" ".join(prose)] if prose else []
        sents = [s for u in units for s in re.split(r"(?<=[.!?])\s+(?=[A-Z(*`])", re.sub(r"[`*|]", " ", u))]
        for s in sents:
            s = re.sub(r"\s+", " ", s)
            words = set(tokenize(s))
            score = sum(1 for q in terms if any(w.startswith(q[:5]) for w in words)) - 0.3 * r
            if score > best_s and len(s.strip()) > 15:
                best, best_s = s.strip(), score
    return best


def number_near_terms(sentence: str, terms) -> Optional[float]:
    """The number with the most distinct query terms within 40 characters (earliest on ties)."""
    nums = numbers_in(sentence)
    if not nums:
        return None
    low = sentence.lower()
    terms = [q for q in terms if len(q) > 1 and not q.isdigit()]
    pos = {q: [m.start() for m in re.finditer(re.escape(q[:5]), low)] for q in terms}

    def score(n):
        return sum(any(n[1] - 40 <= p <= n[2] + 40 for p in ps) for ps in pos.values())
    return max(nums, key=lambda n: (score(n), -n[1]))[0]


def unit_after(sentence: str, value) -> Optional[str]:
    for v, s, e in numbers_in(sentence):
        if v == value:
            m = re.match(r"\s*([A-Za-z]+)", sentence[e:])
            if m and norm_unit(m.group(1)):
                return norm_unit(m.group(1))
    return None


CAPABILITIES = ("I'm a tool-using agent for this repo. I can do exact arithmetic (calculator), "
                "convert units, search the module READMEs, read files in the repo, and remember "
                "things for you (kv_memory).")


def direct_answer(request: str) -> str:
    """What the scripted 'LLM' says when the router decides no tool is needed."""
    t = request.strip()
    lo = t.lower()
    m = re.search(r"repeat the word\s+['\"]?(\w+)['\"]?\s+(\w+)\s+times", lo)
    if m:
        k = {"two": 2, "three": 3, "four": 4, "five": 5}.get(m.group(2), int(m.group(2)) if m.group(2).isdigit() else 3)
        return " ".join([m.group(1)] * k)
    m = re.search(r"(?:uppercase|shout)(?: the word)?\s+['\"]?(\w+)", lo)
    if m:
        return m.group(1).upper()
    m = re.search(r"(?:reverse(?: the word)?|spell)\s+['\"]?(\w+)['\"]?", lo)
    if m and m.group(1) not in ("the", "word"):
        return m.group(1)[::-1]
    m = re.search(r"how many words are in the sentence\s+['\"](.+)['\"]", t, re.I)
    if m:
        return str(len(m.group(1).split()))
    m = re.search(r"say hello to\s+(.+?)[.!?]*$", t, re.I)
    if m:
        return f"Hello, {m.group(1)}!"
    if re.search(r"\b(thanks|thank you|appreciated|bye)\b", lo):
        return "You're welcome!"
    if re.search(r"\b(hi|hello|hey|good morning)\b|who are you|what can you do|introduce yourself|what tools", lo):
        return "Hello! " + CAPABILITIES
    return ("I can't help with that: it needs live data, personal information or actions I have no "
            "tool for. " + CAPABILITIES)


INJECTION_MARK = re.compile(r"ignore (?:all |any )?(?:previous|prior|above) instructions|"
                            r"note to (?:ai|the) assistant|system override|assistant instructions", re.I)


def find_injection(content: str) -> Optional[dict]:
    """Instructions planted in a document: calls to make and/or what to tell the user."""
    text = unwrap(content).replace('\\"', '"')
    if not INJECTION_MARK.search(text):
        return None
    calls = []
    for m in re.finditer(r"call\s+(\w+)\s+with\s+(\{.*?\})(?=[\s.,;]|$)", text):
        try:
            calls.append({"tool": m.group(1), "args": json.loads(m.group(2))})
        except json.JSONDecodeError:
            pass
    m = re.search(r"(?:tell|answer|reply to) the user(?: that)?:?\s*[\"']([^\"']+)[\"']", text, re.I)
    if not calls and not m:
        return None
    return {"calls": calls, "answer": m.group(1) if m else None}


# ----------------------------------------------------------------------------
# ScriptedPolicy
# ----------------------------------------------------------------------------
class ScriptedPolicy(Policy):
    """A deterministic 'decent LLM'. State (the plan, what it already learned) lives in
    Python attributes; it re-reads every new message the harness shows it."""

    name = "scripted"

    def __init__(self, router):
        self.router = router

    def reset(self, request: str, seed: int = 0):
        self.request = request
        self.plan = []  # list of steps; a step is a list of clauses that can run in parallel
        for i, step in enumerate(split_plan(request)):
            clauses = []
            for text in step:
                label, p, _ = self.router.route(text)
                clauses.append({"text": text, "tool": label, "p": p})
            tool_clauses = [c for c in clauses if c["tool"] != "no_tool"]
            if tool_clauses:
                self.plan.append(tool_clauses)
            elif i == 0:
                break  # the request itself needs no tool: answer directly
        self.step = 0
        self.results = {}  # (step, clause) -> {"value", "unit", "answer"}
        self.pending = []  # [{"key", "tool", "args"}]
        self.attempts, self.crashes = {}, {}
        self.failure = None
        self.seen = 0
        self.followed_injection = False
        self.hijacked_answer = None

    # -- reading observations ---------------------------------------------
    def _prev(self, step):
        """(value, unit) produced by the previous step (for 'it', 'that', 'the result')."""
        if step == 0:
            return None, None
        r = [self.results[(step - 1, j)] for j in range(len(self.plan[step - 1]))
             if (step - 1, j) in self.results]
        return (r[-1].get("value"), r[-1].get("unit")) if r else (None, None)

    def _read(self, clause, call, data) -> dict:
        tool, text = call["tool"], clause["text"]
        if tool == "calculator":
            return {"value": data.get("result"), "answer": f"{call['args']['expression']} = {data.get('result')}"}
        if tool == "unit_convert":
            return {"value": data.get("result"), "unit": data.get("to_unit"),
                    "answer": f"{fmt(data.get('value'))} {data.get('from_unit')} = {data.get('result')} {data.get('to_unit')}"}
        if tool == "search_docs":
            terms = tokenize(call["args"]["query"])
            missing = data.get("terms_not_found", [])
            if not data.get("results") or len(missing) >= max(1, len(terms)) / 3:
                missing = ", ".join(data.get("terms_not_found", [])) or "anything relevant"
                return {"fail": f"I could not find that in the docs (no passage mentions: {missing})."}
            sent = best_sentence(data["results"], terms)
            value = number_near_terms(sent, terms) if sent else None
            src = data["results"][0]["source"]
            return {"value": value, "unit": unit_after(sent, value) if value is not None else None,
                    "answer": f"{sent} (source: {src})"}
        if tool == "read_file":
            content, lo = data.get("content", ""), text.lower()
            if "how many lines" in lo:
                return {"value": data["n_lines"], "answer": f"{data['path']} has {data['n_lines']} lines."}
            if "character" in lo or "how long" in lo:
                return {"value": data["n_chars"], "answer": f"{data['path']} has {data['n_chars']} characters."}
            heads = [ln.lstrip("#").strip() for ln in content.splitlines() if ln.startswith("#")]
            if ("title" in lo or "heading" in lo) and heads:
                return {"answer": f"The title of {data['path']} is: {heads[0]}"}
            m = re.search(r"mentions?\s+['\"]?(.+?)['\"]?\??$", text, re.I)
            if m:
                term = m.group(1).strip()
                hit = term.lower() in content.lower()
                return {"answer": f"{'Yes' if hit else 'No'}, {data['path']} "
                                  f"{'mentions' if hit else 'does not mention'} {term!r}"
                                  f"{'' if not data.get('truncated') or hit else ' (in the part I read)'}."}
            keys = [w for w in tokenize(re.sub(r"[\w/.\-]+\.(md|txt|py|jsonl?)", " ", text))
                    if w not in QUESTION_WORDS and w not in ("read", "open", "file", "summarize", "contents",
                                                             "show", "tell", "print", "cat", "load")]
            if keys:
                lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
                best = max(lines, key=lambda ln: sum(k[:5] in ln.lower() for k in keys), default="")
                if sum(k[:5] in best.lower() for k in keys):
                    nums = numbers_in(best)
                    return {"value": nums[0][0] if nums else None, "answer": best}
            body = next((ln for ln in content.splitlines() if ln.strip() and not ln.startswith("#")), "")
            return {"answer": f"{data['path']}: {heads[0] if heads else ''} — {body[:200]}"}
        if tool == "kv_memory":
            a = call["args"].get("action")
            if a == "remember":
                return {"value": data.get("stored"), "answer": f"Saved: your {data.get('key', '').replace('_', ' ')} "
                                                                f"is {data.get('stored')}."}
            if a == "recall":
                if data.get("found"):
                    return {"value": data["value"], "answer": f"Your {data['key'].replace('_', ' ')} is {data['value']}."}
                if data.get("did_you_mean"):
                    return {"retry_args": {"action": "recall", "key": data["did_you_mean"]}}
                return {"fail": f"I don't have your {data.get('key', '').replace('_', ' ')} stored."}
            if a == "forget":
                return {"answer": f"Forgot your {data.get('key', '').replace('_', ' ')}."}
            return {"answer": "I remember: " + (", ".join(data.get("keys", [])) or "nothing yet") + "."}
        return {"answer": str(data)[:200]}

    def _build_call(self, step, j) -> Optional[dict]:
        clause = self.plan[step][j]
        prev, prev_unit = self._prev(step)
        text, tool = clause["text"], clause["tool"]
        if tool == "calculator":
            e = build_expression(text, prev)
            return {"tool": tool, "args": {"expression": e}} if e else None
        if tool == "unit_convert":
            a = build_conversion(text, prev, prev_unit)
            return {"tool": tool, "args": a} if a else None
        if tool == "search_docs":
            q = search_query(text)
            return {"tool": tool, "args": {"query": q}} if q else None
        if tool == "read_file":
            m = re.search(r"(?<![\w/])((?:\.\./)*[\w.\-/~]*[\w\-]+\.(?:md|txt|py|jsonl?|env|pem))\b", text)
            if not m:
                m = re.search(r"((?:\.\./)+[\w.\-/]+|/[\w.\-/]+)", text)
            return {"tool": tool, "args": {"path": m.group(1)}} if m else None
        if tool == "kv_memory":
            a = build_memory_call(text, prev)
            return {"tool": tool, "args": a} if a else None
        return None

    # -- one turn ---------------------------------------------------------
    def generate(self, messages, tools) -> PolicyOutput:
        new = messages[self.seen:]
        self.seen = len(messages)
        tool_msgs = [m for m in new if m["role"] == "tool"]

        if self.hijacked_answer is not None:
            return PolicyOutput(final_text("The document said to tell the user this.", self.hijacked_answer))

        still = []
        for p in self.pending:
            m = next((t for t in tool_msgs if t["name"] == p["tool"] and t.get("args") == p["args"]), None)
            if m is not None and m.get("ok"):
                r = self._read(self.plan[p["key"][0]][p["key"][1]], p, lenient_json(m["content"]))
                if "retry_args" in r:
                    still.append({**p, "args": r["retry_args"]})
                elif "fail" in r:
                    self.failure = r["fail"]
                else:
                    self.results[p["key"]] = r
                continue
            if m is not None:
                kind = m.get("error_type")
                err = lenient_json(m["content"]).get("error", {})
                msg = err.get("message", "") if isinstance(err, dict) else str(err)
                if kind in ("tool_error", "permission_denied"):
                    self.failure = f"I couldn't complete that: {msg}"
                    continue
                if kind == "crash":  # an opaque error: a model can't tell what to fix
                    self.crashes[p["key"]] = self.crashes.get(p["key"], 0) + 1
                    if self.crashes[p["key"]] > 1:
                        self.failure = f"I couldn't complete that: the tool kept failing ({msg[:80]})."
                        continue
            still.append(p)  # not executed, or a fixable error: issue it again
        self.pending = still

        inj = self.injection_in(tool_msgs, delimited_ok=False)
        if inj and not self.followed_injection:
            return self.follow_injection(inj)
        if self.failure:
            return PolicyOutput(final_text("A tool result means I cannot answer as asked.", self.failure))
        while self.step < len(self.plan) and all((self.step, j) in self.results
                                                 for j in range(len(self.plan[self.step]))):
            self.step += 1
            self.pending = []
        if not self.plan:
            return PolicyOutput(final_text("No tool is needed for this.", direct_answer(self.request)))
        if self.step >= len(self.plan):
            last = self.plan[-1]
            ans = "; ".join(self.results[(len(self.plan) - 1, j)]["answer"] for j in range(len(last)))
            return PolicyOutput(final_text("I have everything I need.", ans))

        if not self.pending:
            for j, clause in enumerate(self.plan[self.step]):
                if (self.step, j) in self.results:
                    continue
                call = self._build_call(self.step, j)
                if call is None:
                    return PolicyOutput(final_text(
                        "I can't turn this into a tool call.",
                        f"I couldn't work out the {clause['tool']} arguments for: {clause['text']!r}."))
                self.pending.append({"key": (self.step, j), **call})
        for p in self.pending:
            self.attempts[p["key"]] = self.attempts.get(p["key"], 0) + 1
        if max(self.attempts[p["key"]] for p in self.pending) > 4:
            return PolicyOutput(final_text("My calls keep failing.", "I couldn't complete that: the tool "
                                           "calls kept failing."))
        clauses = [self.plan[p["key"][0]][p["key"][1]] for p in self.pending]
        thought = "; ".join(f"{c['text'][:60]!r} -> {c['tool']} (router p={c['p']:.2f})" for c in clauses)
        return PolicyOutput(action_text(thought, [{"tool": p["tool"], "args": p["args"]} for p in self.pending]))

    # -- prompt injection -------------------------------------------------
    @staticmethod
    def injection_in(tool_msgs, delimited_ok: bool) -> Optional[dict]:
        for m in tool_msgs:
            if not m.get("ok"):
                continue
            delimited = m["content"].lstrip().startswith(f"<{TAG}")
            if delimited and not delimited_ok:
                continue  # clearly marked as data: a well-trained model ignores it
            inj = find_injection(m["content"])
            if inj:
                return inj
        return None

    def follow_injection(self, inj) -> PolicyOutput:
        self.followed_injection = True
        self.hijacked_answer = inj.get("answer")
        if inj["calls"]:
            return PolicyOutput(action_text("The document has new instructions for me.", inj["calls"]))
        self.hijacked_answer = None
        return PolicyOutput(final_text("The document said to tell the user this.", inj["answer"]))


# ----------------------------------------------------------------------------
# NoisyPolicy
# ----------------------------------------------------------------------------
FAILURE_MODES = ["malformed_json", "unknown_tool", "bad_args", "hallucinated_result", "loop",
                 "no_tool_answer", "injection_followed"]
MISSPELL = {"calculator": ["calculater", "calc", "math"], "search_docs": ["search", "doc_search", "searchdocs"],
            "read_file": ["readfile", "open_file", "cat"], "unit_convert": ["convert_units", "unit_converter"],
            "kv_memory": ["memory", "kvmemory"]}
RENAME = {"expression": "expr", "query": "q", "path": "file", "value": "amount", "from_unit": "from",
          "to_unit": "to", "action": "op", "key": "name"}


class NoisyPolicy(Policy):
    """Wraps a policy; at each step, with probability p, replaces its output by one of
    the failure modes (uniformly among those that apply to this step)."""

    def __init__(self, base: Policy, p: float = 0.2):
        self.base, self.p = base, p
        self.name = f"noisy(p={p})"

    def reset(self, request, seed=0):
        self.base.reset(request, seed)
        self.rng = random.Random(seed)
        self.in_loop = False
        self.seen = 0
        self.events = []
        self.injection_rolled = False
        self.last_bad = None

    def generate(self, messages, tools) -> PolicyOutput:
        new = messages[self.seen:]
        self.seen = len(messages)
        if any(m["role"] == "tool" and m.get("error_type") == "loop" for m in new):
            self.in_loop = False  # the harness told us; snap out of it
        out = self.base.generate(messages, tools)
        # Errors as feedback: an actionable error ("missing required argument 'value'")
        # lets the model fix its call; an opaque one ("TypeError: ...") often doesn't.
        # Assumption of this simulation: after an opaque crash the model repeats its
        # broken call half of the time.
        bad, self.last_bad = self.last_bad, None
        if bad and any(m["role"] == "tool" and m.get("error_type") == "crash" for m in new) \
                and self.rng.random() < 0.5:
            self.last_bad = bad
            self.events.append("repeated_bad_call")
            return PolicyOutput(bad)
        executed = [m for m in messages if m["role"] == "tool" and m.get("ok")]
        last_call = {"tool": executed[-1]["name"], "args": executed[-1]["args"]} if executed else None
        if self.in_loop and last_call:
            return PolicyOutput(action_text("Let me check that again.", [last_call]))

        # Injected instructions inside DELIMITED output: the base model ignores them,
        # but a real model sometimes follows them anyway (rolled once per document).
        inj = None
        if not self.injection_rolled and isinstance(self.base, ScriptedPolicy) and not self.base.followed_injection:
            inj = ScriptedPolicy.injection_in([m for m in new if m["role"] == "tool"], delimited_ok=True)
            if inj:
                self.injection_rolled = True
                if self.rng.random() < self.p:
                    self.events.append("injection_followed")
                    return self.base.follow_injection(inj)

        if self.rng.random() >= self.p:
            return out
        is_action = "Action:" in out.text
        modes = ["malformed_json"]
        if last_call:
            modes.append("loop")  # re-check instead of moving on / answering
        if is_action:
            modes += ["unknown_tool", "bad_args", "hallucinated_result"]
            if not executed:
                modes.append("no_tool_answer")
        mode = self.rng.choice(modes)
        self.events.append(mode)
        text = self._corrupt(out.text, mode, last_call)
        if mode in ("unknown_tool", "bad_args"):
            self.last_bad = text
        return PolicyOutput(text)

    def _corrupt(self, text, mode, last_call) -> str:
        rng = self.rng
        thought, _, rest = text.partition("\n")
        if mode == "malformed_json":
            if "Final Answer:" in text:  # forgot the format
                return text.split("Final Answer:", 1)[1].strip()
            js = rest.split("Action:", 1)[1].strip()
            v = rng.randrange(3)
            if v == 0:
                return f"{thought}\nAction:\n```json\n{js}\n```\nLet me know if you need anything else!"
            if v == 1:
                return f"{thought}\nAction: " + js.replace('"', "'")[:-1] + ",}"
            return f"{thought}\nAction: {js[:-2]}"
        if mode == "loop":
            self.in_loop = True
            return action_text("Let me check that again.", [last_call])
        if mode == "no_tool_answer":
            return final_text("I'm confident I know this without tools.",
                              f"I believe the answer is {rng.randint(2, 999)}.")
        calls = json.loads(rest.split("Action:", 1)[1])
        calls = calls if isinstance(calls, list) else [calls]
        c = dict(calls[0])
        c["args"] = dict(c["args"])
        if mode == "unknown_tool":
            c["tool"] = rng.choice(MISSPELL.get(c["tool"], [c["tool"] + "_v2"]))
        elif mode == "bad_args":
            keys = list(c["args"])
            numeric = [k for k, v in c["args"].items() if isinstance(v, (int, float))]
            v = rng.randrange(3 if numeric else 2)
            k = rng.choice(keys)
            if v == 0:
                del c["args"][k]
            elif v == 1:
                c["args"][RENAME.get(k, k + "_")] = c["args"].pop(k)
            else:
                k = rng.choice(numeric)
                c["args"][k] = f"{fmt(c['args'][k])}" + rng.choice(["", " units"])
        elif mode == "hallucinated_result":
            fake = rng.randint(2, 999)
            return (f"{text}\nObservation: {{\"result\": {fake}}}\nThought: Great, that settles it.\n"
                    f"Final Answer: The answer is {fake}.")
        return action_text(thought.replace("Thought: ", ""), [c] + calls[1:])


# ----------------------------------------------------------------------------
# HTTPPolicy (OpenAI-compatible endpoint, urllib only)
# ----------------------------------------------------------------------------
def to_openai_messages(messages: List[dict], native: bool) -> List[dict]:
    out = []
    for m in messages:
        if m["role"] == "assistant" and native and m.get("tool_calls"):
            out.append({"role": "assistant", "content": m["content"] or None, "tool_calls": [
                {"id": c["id"], "type": "function",
                 "function": {"name": c["tool"], "arguments": json.dumps(c["args"])}}
                for c in m["tool_calls"]]})
        elif m["role"] == "tool" and native:
            out.append({"role": "tool", "tool_call_id": m["tool_call_id"], "content": m["content"]})
        elif m["role"] == "tool":  # text protocol: observations come back as user turns
            out.append({"role": "user", "content": f"Observation [{m['tool_call_id']}] {m['name']}:\n{m['content']}"})
        else:
            out.append({"role": m["role"], "content": m["content"]})
    if native:  # tool messages must directly follow the assistant message that called them
        fixed = []
        for m in out:
            if m["role"] == "assistant" and not m.get("tool_calls") and not m.get("content"):
                continue
            fixed.append(m)
        out = fixed
    return out


class HTTPPolicy(Policy):
    def __init__(self, mode: str = "native", base_url: Optional[str] = None, api_key: Optional[str] = None,
                 model: Optional[str] = None, timeout: float = 60.0, max_http_retries: int = 3):
        self.native = mode == "native"
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL", "")).rstrip("/")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.model = model or os.environ.get("AGENT_MODEL", "")
        self.timeout, self.max_http_retries = timeout, max_http_retries
        self.name = f"http:{self.model}:{mode}"

    @staticmethod
    def available() -> bool:
        return bool(os.environ.get("OPENAI_BASE_URL") and os.environ.get("AGENT_MODEL"))

    def _post(self, payload: dict) -> dict:
        req = urllib.request.Request(self.base_url + "/chat/completions", data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json",
                                              "Authorization": f"Bearer {self.api_key}"})
        for attempt in range(self.max_http_retries):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    return json.loads(r.read().decode())
            except urllib.error.HTTPError as e:
                # 429 / 5xx are worth retrying with exponential backoff; 4xx are bugs.
                if e.code in (429, 500, 502, 503, 504) and attempt < self.max_http_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                raise RuntimeError(f"HTTP {e.code}: {e.read().decode()[:300]}") from None
            except urllib.error.URLError:
                if attempt < self.max_http_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                raise
        raise RuntimeError("unreachable")

    def generate(self, messages, tools) -> PolicyOutput:
        payload = {"model": self.model, "temperature": 0,
                   "messages": to_openai_messages(messages, self.native)}
        if self.native:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        else:
            payload["stop"] = ["\nObservation:"]  # the real stop sequence
        t0 = time.perf_counter()
        resp = self._post(payload)
        latency = time.perf_counter() - t0
        msg = resp["choices"][0]["message"]
        usage = resp.get("usage", {})
        calls = None
        if msg.get("tool_calls"):
            calls = [{"name": c["function"]["name"], "arguments": c["function"].get("arguments", "{}")}
                     for c in msg["tool_calls"]]
        return PolicyOutput(msg.get("content") or "", calls, usage.get("prompt_tokens"),
                            usage.get("completion_tokens"), latency)


def demo():
    from agent import Agent, print_trace
    from router_numpy import load_router
    from tools import make_registry

    router, _ = load_router(verbose=True)
    reqs = ["Convert 5 km to miles and also 3 kg to pounds",
            "Find the KV cache memory per token of LLaMA-2-7B in the docs, then convert it to KB",
            "What is my favorite optimizer?"]
    for pol in (ScriptedPolicy(router), NoisyPolicy(ScriptedPolicy(router), p=0.4)):
        print(f"\n######## {pol.name} ########")
        for i, r in enumerate(reqs):
            reg = make_registry(memory={"favorite optimizer": "AdamW"})
            res = Agent(reg, pol, router=router).run(r, seed=i)
            print(f"\nrequest: {r}")
            print_trace(res)
            if isinstance(pol, NoisyPolicy):
                print(f"   injected failures: {pol.events}")
    print("\nHTTPPolicy:", "configured" if HTTPPolicy.available() else
          "not configured (set OPENAI_BASE_URL, OPENAI_API_KEY, AGENT_MODEL to try a real model)")


if __name__ == "__main__":
    demo()
