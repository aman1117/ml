"""
Tools for the agent: specs, validation, sandboxing, and five real tools.

A "tool" is just a Python function plus a SPEC the model can read: a name, a
description, and a JSON-schema description of its parameters. The model never
runs code; it writes a JSON object {"tool": name, "args": {...}} and the HARNESS
(this file) decides whether and how to run it:

    model text -> parsed call -> validate + coerce args -> permission check
               -> run with a timeout in a sandbox -> JSON observation (or error)

Everything the model gets back is a string. Errors are written FOR THE MODEL
("value: expected number, got string '5 km'. Example: {...}"), because the model
can only fix what the error message tells it (the agent-computer interface idea
from SWE-agent).

Tools (side-effect class in brackets):
    calculator    [read-only]  safe arithmetic with `ast` (never eval)
    search_docs   [read-only]  BM25 over the six existing READMEs of this repo
    read_file     [read-only]  repo files only: no traversal, no hidden files, size cap
    unit_convert  [read-only]  length / mass / time / data / temperature
    kv_memory     [mutating for remember/forget]  a key-value memory for the session

Mutating calls go through a `confirm(tool, args)` callback (a human in the loop),
see agent.py for when the agent asks.

Run (a small demo of every tool and of the error messages):
    python agents/tools.py
"""
from __future__ import annotations

import ast
import difflib
import json
import math
import operator
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
DOC_MODULES = ["mlp", "cnn", "rnn", "lstm", "attention", "transformer"]


# ----------------------------------------------------------------------------
# Errors and results
# ----------------------------------------------------------------------------
class ToolError(Exception):
    """An error the MODEL will read. `kind` feeds the failure taxonomy in agent_eval.py.

    kinds: unknown_tool, invalid_args, tool_error, timeout, transient,
           permission_denied, crash (an unexpected Python exception)."""

    def __init__(self, kind: str, message: str, retryable: bool = False):
        super().__init__(message)
        self.kind = kind
        self.message = message
        self.retryable = retryable


@dataclass
class ToolResult:
    ok: bool
    content: str  # JSON string shown to the model
    error_kind: Optional[str] = None
    latency_ms: float = 0.0
    data: Any = None  # the Python value (handy for tests; a real LLM only ever sees `content`)


def to_json(x) -> str:
    return json.dumps(x, ensure_ascii=False, separators=(", ", ": "))


def error_payload(e: ToolError) -> str:
    return to_json({"error": {"type": e.kind, "message": e.message, "retryable": e.retryable}})


# ----------------------------------------------------------------------------
# Tool spec + argument validation / coercion
# ----------------------------------------------------------------------------
@dataclass
class Tool:
    name: str
    description: str
    parameters: dict  # JSON schema: {"type": "object", "properties": {...}, "required": [...]}
    fn: Callable[..., Any]
    side_effect: str = "read_only"  # "read_only" | "mutating"
    mutating_if: Optional[Callable[[dict], bool]] = None  # per-call refinement
    timeout_s: float = 1.0
    example: Optional[dict] = None  # shown in error messages: the fastest way to fix a call

    def spec(self) -> dict:
        """OpenAI-style function spec (what native tool calling sends to the model)."""
        return {"type": "function",
                "function": {"name": self.name, "description": self.description,
                             "parameters": self.parameters}}

    def is_mutating(self, args: dict) -> bool:
        if self.mutating_if is not None:
            return bool(self.mutating_if(args))
        return self.side_effect == "mutating"


def _coerce(value, schema: dict, path: str):
    """Coerce one value to its schema type, or raise ValueError with a model-readable reason.

    Coercions are the ones a model's near-misses need: "5" -> 5 for numbers,
    "true" -> True, an enum value in the wrong case -> the right case. Anything
    else is an error (silently guessing would hide bugs)."""
    t = schema.get("type", "string")
    if t == "string":
        if isinstance(value, bool) or isinstance(value, (dict, list)) or value is None:
            raise ValueError(f"{path}: expected string, got {type(value).__name__} {value!r}")
        value = str(value) if isinstance(value, (int, float)) else value
        if "enum" in schema:
            for opt in schema["enum"]:
                if value.strip().lower() == opt.lower():
                    return opt
            raise ValueError(f"{path}: {value!r} is not one of {schema['enum']}")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            raise ValueError(f"{path}: string longer than {schema['maxLength']} characters")
        return value
    if t in ("number", "integer"):
        if isinstance(value, bool):
            raise ValueError(f"{path}: expected {t}, got boolean {value!r}")
        if isinstance(value, str):
            s = value.strip().replace("_", "")
            try:
                f = float(s)
            except ValueError:
                raise ValueError(f"{path}: expected {t}, got string {value!r} "
                                 f"(pass just the number, e.g. 5)") from None
            value = f
        if not isinstance(value, (int, float)):
            raise ValueError(f"{path}: expected {t}, got {type(value).__name__} {value!r}")
        if not math.isfinite(float(value)):
            raise ValueError(f"{path}: must be a finite number")
        if t == "integer":
            if float(value) != int(value):
                raise ValueError(f"{path}: expected integer, got {value!r}")
            value = int(value)
        if "minimum" in schema and value < schema["minimum"]:
            raise ValueError(f"{path}: must be >= {schema['minimum']}, got {value}")
        if "maximum" in schema and value > schema["maximum"]:
            raise ValueError(f"{path}: must be <= {schema['maximum']}, got {value}")
        return value
    if t == "boolean":
        if isinstance(value, str) and value.strip().lower() in ("true", "false"):
            return value.strip().lower() == "true"
        if not isinstance(value, bool):
            raise ValueError(f"{path}: expected boolean, got {value!r}")
        return value
    if t == "array":
        if not isinstance(value, list):
            raise ValueError(f"{path}: expected array, got {type(value).__name__} {value!r}")
        return [_coerce(v, schema.get("items", {}), f"{path}[{i}]") for i, v in enumerate(value)]
    raise ValueError(f"{path}: unsupported schema type {t}")


def validate_args(tool: Tool, args) -> dict:
    """Check `args` against tool.parameters. Returns cleaned args or raises ToolError
    listing EVERY problem at once (one round trip instead of several)."""
    props = tool.parameters.get("properties", {})
    required = tool.parameters.get("required", [])
    hint = f" Example: {to_json({'tool': tool.name, 'args': tool.example})}" if tool.example else ""
    if not isinstance(args, dict):
        raise ToolError("invalid_args", f"{tool.name}: args must be a JSON object, got "
                        f"{type(args).__name__}.{hint}")
    problems, clean = [], {}
    for k in args:
        if k not in props:
            close = difflib.get_close_matches(k, list(props), n=1)
            problems.append(f"unexpected argument {k!r}" +
                            (f" (did you mean {close[0]!r}?)" if close else "") +
                            f"; allowed: {sorted(props)}")
    for k in required:
        if k not in args:
            problems.append(f"missing required argument {k!r} ({props[k].get('type')}: "
                            f"{props[k].get('description', '')})")
    for k, v in args.items():
        if k in props:
            try:
                clean[k] = _coerce(v, props[k], k)
            except ValueError as e:
                problems.append(str(e))
    if problems:
        raise ToolError("invalid_args", f"{tool.name}: " + "; ".join(problems) + "." + hint)
    for k, s in props.items():
        if k not in clean and "default" in s:
            clean[k] = s["default"]
    return clean


# ----------------------------------------------------------------------------
# Tool 1: calculator (ast whitelist, never eval)
# ----------------------------------------------------------------------------
_BINOPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
           ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
           ast.Pow: operator.pow}
_UNOPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS = {"sqrt": math.sqrt, "log": math.log, "log2": math.log2, "log10": math.log10,
          "exp": math.exp, "abs": abs, "round": round, "min": min, "max": max,
          "sin": math.sin, "cos": math.cos, "ceil": math.ceil, "floor": math.floor}
_CONSTS = {"pi": math.pi, "e": math.e}


def _eval_node(node):
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
            and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        a, b = _eval_node(node.left), _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and (abs(b) > 64 or abs(a) > 1e6):
            # 9**9**9 would hang the process: a classic DoS on "safe" calculators.
            raise ToolError("tool_error", "exponent too large (|base| <= 1e6, |exponent| <= 64)")
        try:
            return _BINOPS[type(node.op)](a, b)
        except ZeroDivisionError:
            raise ToolError("tool_error", "division by zero") from None
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNOPS:
        return _UNOPS[type(node.op)](_eval_node(node.operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
            and node.func.id in _FUNCS and not node.keywords:
        try:
            return _FUNCS[node.func.id](*[_eval_node(a) for a in node.args])
        except (ValueError, TypeError) as e:
            raise ToolError("tool_error", f"{node.func.id}(): {e}") from None
    if isinstance(node, ast.Name) and node.id in _CONSTS:
        return _CONSTS[node.id]
    if isinstance(node, ast.Tuple) and all(isinstance(x, ast.Constant) for x in node.elts):
        raise ToolError("tool_error", "a ',' inside a number? Write 107776, not 107,776.")
    raise ToolError("tool_error", f"unsupported syntax {type(node).__name__!r}. Allowed: numbers, "
                    f"+ - * / // % ** ( ), {sorted(_FUNCS)}, pi, e.")


def fmt_number(x) -> Any:
    """Integers stay integers; floats get 10 significant digits."""
    if isinstance(x, float) and x.is_integer() and abs(x) < 1e15:
        return int(x)
    if isinstance(x, float):
        return float(f"{x:.10g}")
    return x


def calculator(expression: str) -> dict:
    expr = expression.replace("^", "**").replace("×", "*").replace("÷", "/").strip()
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        raise ToolError("tool_error", f"could not parse {expression!r} as arithmetic. "
                        "Use only numbers and + - * / ** ( ), e.g. '(3 + 4) * 2'.") from None
    return {"expression": expression, "result": fmt_number(_eval_node(tree))}


# ----------------------------------------------------------------------------
# Tool 2: search_docs (BM25 over this repo's READMEs)
# ----------------------------------------------------------------------------
STOPWORDS = set("""a an the of to in on for and or is are was were be been it its this that
these those with as by at from into what which who whom how why when where does do did can could
should would will shall may might must about than then there their they them our we you your i
me my he she his her not no yes if so but all any each some such only also just very more most
much many say says said tell show give find search look up docs doc documentation readme
readmes module folder please according use using get number value""".split())


def tokenize(text: str) -> List[str]:
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS]


def chunk_readme(path: str, module: str) -> List[dict]:
    """Split a README into paragraph chunks, each tagged with its section heading.
    Code fences stay whole; very short paragraphs are merged with the next one."""
    with open(path, encoding="utf-8") as f:
        lines = f.read().splitlines()
    chunks, section, buf, in_code = [], "", [], False

    def flush():
        text = "\n".join(buf).strip()
        if text:
            chunks.append({"source": f"{module}/README.md § {section}".strip(), "text": text})
        buf.clear()

    for line in lines:
        if line.startswith("```"):
            in_code = not in_code
        if not in_code and re.match(r"^#{1,3} ", line):
            flush()
            section = line.lstrip("#").strip()
            continue
        if not in_code and not line.strip():
            if sum(len(b) for b in buf) >= 120:
                flush()
            continue
        buf.append(line)
    flush()
    out = []
    for c in chunks:  # split very long chunks so a result fits in a context budget
        t = c["text"]
        while len(t) > 900:
            cut = t.rfind("\n", 0, 700)
            cut = cut if cut > 200 else 700
            out.append({"source": c["source"], "text": t[:cut].strip()})
            t = t[cut:]
        out.append({"source": c["source"], "text": t.strip()})
    return out


class BM25:
    """Okapi BM25: score(q, d) = Σ_t idf(t) · tf·(k1+1) / (tf + k1·(1 − b + b·|d|/avgdl)).
    See ../rag/ for the full retrieval story; this is the 30-line version."""

    def __init__(self, docs: List[dict], k1: float = 1.2, b: float = 0.75):
        self.docs, self.k1, self.b = docs, k1, b
        # Index the heading too: "Shapes and parameter count" is a strong signal.
        self.toks = [tokenize(d["source"].split("§")[-1] + " " + d["text"]) for d in docs]
        self.tfs = []
        for t in self.toks:
            tf: Dict[str, int] = {}
            for w in t:
                tf[w] = tf.get(w, 0) + 1
            self.tfs.append(tf)
        self.avgdl = sum(len(t) for t in self.toks) / max(1, len(self.toks))
        self.df: Dict[str, int] = {}
        for t in self.toks:
            for w in set(t):
                self.df[w] = self.df.get(w, 0) + 1
        self.N = len(docs)

    def idf(self, w):
        n = self.df.get(w, 0)
        return math.log(1 + (self.N - n + 0.5) / (n + 0.5))

    def search(self, query: str, k: int = 3):
        q = tokenize(query)
        scores = []
        for toks, tf in zip(self.toks, self.tfs):
            s = 0.0
            for w in set(q):
                if w in tf:
                    f = tf[w]
                    s += self.idf(w) * f * (self.k1 + 1) / (
                        f + self.k1 * (1 - self.b + self.b * len(toks) / self.avgdl))
            scores.append(s)
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [(i, scores[i]) for i in order if scores[i] > 0], q


_BASE_DOCS: Optional[List[dict]] = None


def base_docs() -> List[dict]:
    global _BASE_DOCS
    if _BASE_DOCS is None:
        _BASE_DOCS = []
        for m in DOC_MODULES:
            _BASE_DOCS += chunk_readme(os.path.join(REPO, m, "README.md"), m)
    return _BASE_DOCS


_INDEX_CACHE: Dict[str, BM25] = {}


def make_search_docs(extra_docs: Optional[List[dict]] = None):
    key = to_json(extra_docs or [])
    if key not in _INDEX_CACHE:  # building the index takes ~20 ms; evals build hundreds
        _INDEX_CACHE[key] = BM25(base_docs() + list(extra_docs or []))
    index = _INDEX_CACHE[key]

    def search_docs(query: str, k: int = 3) -> dict:
        hits, q = index.search(query, k)
        results = []
        for i, s in hits:
            d = index.docs[i]
            text = d["text"] if len(d["text"]) <= 600 else d["text"][:600] + " …"
            results.append({"source": d["source"], "score": round(s, 2), "text": text})
        # Telling the model which words appear in NO document is cheap and prevents
        # "confidently answer from an irrelevant hit" (an ACI choice).
        return {"query": query, "terms_not_found": sorted(w for w in set(q) if w not in index.df),
                "results": results}
    return search_docs, index


# ----------------------------------------------------------------------------
# Tool 3: read_file (sandboxed to the repo)
# ----------------------------------------------------------------------------
READ_EXTS = {".md", ".txt", ".py", ".json", ".jsonl"}
READ_CAP = 2000  # characters returned to the model (context is the scarce resource)


def safe_repo_path(path: str) -> str:
    """Resolve `path` inside the repo or raise permission_denied.

    Blocks: absolute paths outside the repo, `..` traversal, symlinks that escape
    (realpath), hidden files/dirs (.env, .git, .venv), and non-text extensions."""
    if not path or "\x00" in path:
        raise ToolError("invalid_args", "read_file: empty or invalid path")
    p = os.path.expanduser(path)
    full = os.path.realpath(p if os.path.isabs(p) else os.path.join(REPO, p))
    if os.path.commonpath([full, REPO]) != REPO:
        raise ToolError("permission_denied", f"read_file: {path!r} is outside the repository; "
                        "only files inside the repo can be read.")
    rel = os.path.relpath(full, REPO)
    if any(part.startswith(".") for part in rel.split(os.sep)):
        raise ToolError("permission_denied", f"read_file: hidden files/directories ({rel}) "
                        "are not readable.")
    if os.path.splitext(full)[1] not in READ_EXTS:
        raise ToolError("permission_denied", f"read_file: only {sorted(READ_EXTS)} files "
                        "can be read.")
    if not os.path.isfile(full):
        raise ToolError("tool_error", f"read_file: no such file {rel!r}. Paths are relative "
                        "to the repo root, e.g. 'mlp/README.md'.")
    return full


def read_file(path: str, max_chars: int = READ_CAP) -> dict:
    full = safe_repo_path(path)
    with open(full, encoding="utf-8", errors="replace") as f:
        text = f.read(200_000)
    n = min(max_chars, READ_CAP)
    n_lines = text.count("\n") + (1 if text and not text.endswith("\n") else 0)
    return {"path": os.path.relpath(full, REPO), "n_lines": n_lines, "n_chars": len(text),
            "truncated": len(text) > n, "content": text[:n]}


# ----------------------------------------------------------------------------
# Tool 4: unit_convert
# ----------------------------------------------------------------------------
UNITS = {  # unit -> (dimension, factor to the base unit of that dimension)
    "m": ("length", 1.0), "km": ("length", 1000.0), "cm": ("length", 0.01),
    "mm": ("length", 0.001), "mi": ("length", 1609.344), "ft": ("length", 0.3048),
    "in": ("length", 0.0254), "yd": ("length", 0.9144),
    "kg": ("mass", 1.0), "g": ("mass", 0.001), "lb": ("mass", 0.45359237),
    "oz": ("mass", 0.028349523125),
    "s": ("time", 1.0), "ms": ("time", 0.001), "min": ("time", 60.0), "h": ("time", 3600.0),
    "day": ("time", 86400.0),
    "B": ("data", 1.0), "KB": ("data", 1e3), "MB": ("data", 1e6), "GB": ("data", 1e9),
    "TB": ("data", 1e12), "KiB": ("data", 1024.0), "MiB": ("data", 1024.0 ** 2),
    "GiB": ("data", 1024.0 ** 3),
    "C": ("temperature", None), "F": ("temperature", None), "K": ("temperature", None),
}
ALIASES = {
    "meter": "m", "meters": "m", "metre": "m", "metres": "m", "kilometer": "km",
    "kilometers": "km", "kilometre": "km", "kilometres": "km", "centimeters": "cm",
    "centimeter": "cm", "millimeters": "mm", "millimeter": "mm", "mile": "mi", "miles": "mi",
    "foot": "ft", "feet": "ft", "inch": "in", "inches": "in", "yard": "yd", "yards": "yd",
    "kilogram": "kg", "kilograms": "kg", "kgs": "kg", "gram": "g", "grams": "g",
    "pound": "lb", "pounds": "lb", "lbs": "lb", "ounce": "oz", "ounces": "oz",
    "second": "s", "seconds": "s", "sec": "s", "secs": "s", "millisecond": "ms",
    "milliseconds": "ms", "minute": "min", "minutes": "min", "mins": "min", "hour": "h",
    "hours": "h", "hr": "h", "hrs": "h", "days": "day", "d": "day",
    "byte": "B", "bytes": "B", "kilobytes": "KB", "megabytes": "MB", "gigabyte": "GB",
    "gigabytes": "GB", "terabytes": "TB", "terabyte": "TB", "gib": "GiB", "mib": "MiB",
    "kib": "KiB", "celsius": "C", "fahrenheit": "F", "kelvin": "K", "degc": "C", "degf": "F",
}


def norm_unit(u: str) -> Optional[str]:
    u = u.strip().replace("°", "")
    if u in UNITS:
        return u
    lo = u.lower()
    if lo in ALIASES:
        return ALIASES[lo]
    for k in UNITS:
        if k.lower() == lo:
            return k
    return None


def _to_kelvin(v, u):
    return {"K": v, "C": v + 273.15, "F": (v - 32) * 5 / 9 + 273.15}[u]


def _from_kelvin(v, u):
    return {"K": v, "C": v - 273.15, "F": (v - 273.15) * 9 / 5 + 32}[u]


def unit_convert(value: float, from_unit: str, to_unit: str) -> dict:
    a, b = norm_unit(from_unit), norm_unit(to_unit)
    for raw, u in ((from_unit, a), (to_unit, b)):
        if u is None:
            raise ToolError("tool_error", f"unknown unit {raw!r}. Known: {sorted(UNITS)}")
    if UNITS[a][0] != UNITS[b][0]:
        raise ToolError("tool_error", f"cannot convert {a} ({UNITS[a][0]}) to {b} ({UNITS[b][0]})")
    if UNITS[a][0] == "temperature":
        out = _from_kelvin(_to_kelvin(value, a), b)
    else:
        out = value * UNITS[a][1] / UNITS[b][1]
    return {"value": fmt_number(value), "from_unit": a, "to_unit": b,
            "result": fmt_number(round(out, 6))}


# ----------------------------------------------------------------------------
# Tool 5: kv_memory (the only mutating tool)
# ----------------------------------------------------------------------------
def norm_key(k: str) -> str:
    k = re.sub(r"^(my|the|our)\s+", "", str(k).strip().lower())
    return re.sub(r"[^a-z0-9]+", "_", k).strip("_")


class Memory:
    """Session memory. `seed` simulates long-term memory written in earlier sessions."""

    def __init__(self, seed: Optional[dict] = None):
        self.store: Dict[str, str] = {norm_key(k): v for k, v in (seed or {}).items()}
        self.writes: List[dict] = []  # audit log of every mutation

    def __call__(self, action: str, key: str = "", value: Optional[str] = None) -> dict:
        k = norm_key(key)
        if action in ("remember", "recall", "forget") and not k:
            raise ToolError("invalid_args", f"kv_memory: action {action!r} needs a non-empty 'key'")
        if action == "remember":
            if value is None or value == "":
                raise ToolError("invalid_args", "kv_memory: 'remember' needs a 'value'")
            self.store[k] = value
            self.writes.append({"action": "remember", "key": k, "value": value})
            return {"ok": True, "key": k, "stored": value}
        if action == "recall":
            if k in self.store:
                return {"found": True, "key": k, "value": self.store[k]}
            close = difflib.get_close_matches(k, list(self.store), n=1, cutoff=0.6)
            return {"found": False, "key": k, "known_keys": sorted(self.store),
                    **({"did_you_mean": close[0]} if close else {})}
        if action == "forget":
            existed = self.store.pop(k, None) is not None
            self.writes.append({"action": "forget", "key": k})
            return {"ok": True, "key": k, "existed": existed}
        if action == "list":
            return {"keys": sorted(self.store)}
        raise ToolError("tool_error", f"kv_memory: unknown action {action!r}")


# ----------------------------------------------------------------------------
# The registry: specs + validation + timeouts + faults + confirmation
# ----------------------------------------------------------------------------
_POOL = ThreadPoolExecutor(max_workers=8, thread_name_prefix="tool")


class ToolRegistry:
    """Holds the tools of ONE agent session.

    call() is the only way a tool runs. `faults` injects environment failures for
    evals, e.g. {"search_docs": ["transient"]} makes the first search fail with a
    retryable 503, ["slow"] makes it exceed its timeout."""

    def __init__(self, tools: List[Tool], faults: Optional[Dict[str, List[str]]] = None):
        self.tools = {t.name: t for t in tools}
        self.faults = {k: list(v) for k, v in (faults or {}).items()}
        self.n_calls: Dict[str, int] = {}
        self.memory: Optional[Memory] = None

    def specs(self) -> List[dict]:
        return [t.spec() for t in self.tools.values()]

    def get(self, name: str) -> Tool:
        if name not in self.tools:
            close = difflib.get_close_matches(str(name), list(self.tools), n=1, cutoff=0.4)
            raise ToolError("unknown_tool", f"there is no tool named {name!r}" +
                            (f"; did you mean {close[0]!r}?" if close else ".") +
                            f" Available tools: {sorted(self.tools)}.")
        return self.tools[name]

    def call(self, name: str, args, validate: bool = True,
             confirm: Optional[Callable[[str, dict], bool]] = None) -> ToolResult:
        t0 = time.perf_counter()
        try:
            if validate:
                tool = self.get(name)
                args = validate_args(tool, args)
            else:
                tool = self.tools[name]  # KeyError -> an opaque crash, on purpose
            if confirm is not None and isinstance(args, dict) and tool.is_mutating(args) \
                    and not confirm(name, args):
                raise ToolError("permission_denied", f"the user declined {name}({to_json(args)}). "
                                "Do not retry this action; continue without it or ask the user.")
            self.n_calls[name] = self.n_calls.get(name, 0) + 1
            fault = self.faults[name].pop(0) if self.faults.get(name) else None
            if fault == "transient":
                raise ToolError("transient", "503 Service Unavailable (temporary). Retry the same "
                                "call.", retryable=True)
            fn = tool.fn
            if fault == "slow":
                def fn(*a, _f=tool.fn, _t=tool.timeout_s, **kw):
                    time.sleep(_t * 1.5)
                    return _f(*a, **kw)
            fut = _POOL.submit(fn, **args) if isinstance(args, dict) else _POOL.submit(fn, args)
            try:
                data = fut.result(timeout=tool.timeout_s)
            except FutureTimeout:
                # Python threads can't be killed: the late result is just discarded. Real
                # sandboxes run tools in a subprocess/container they CAN kill.
                raise ToolError("timeout", f"{name} timed out after {tool.timeout_s:.2f}s; "
                                "it may succeed if retried.", retryable=True) from None
            return ToolResult(True, to_json(data), None, (time.perf_counter() - t0) * 1e3, data)
        except ToolError as e:
            return ToolResult(False, error_payload(e), e.kind, (time.perf_counter() - t0) * 1e3)
        except Exception as e:  # noqa: BLE001  (a bug or bad input in a tool: report, don't die)
            msg = f"{type(e).__name__}: {e}"
            return ToolResult(False, to_json({"error": {"type": "crash", "message": msg}}),
                              "crash", (time.perf_counter() - t0) * 1e3)


def make_registry(extra_docs: Optional[List[dict]] = None, memory: Optional[dict] = None,
                  faults: Optional[Dict[str, List[str]]] = None,
                  timeout_s: float = 1.0) -> ToolRegistry:
    search_fn, _ = make_search_docs(extra_docs)
    mem = Memory(memory)
    tools = [
        Tool("calculator",
             "Evaluate an arithmetic expression exactly. Use for ANY arithmetic instead of "
             "doing it in your head. Supports + - * / // % ** ( ), sqrt, log, exp, min, max, "
             "round, pi, e. No variables, no thousands separators.",
             {"type": "object", "properties": {
                 "expression": {"type": "string", "maxLength": 200,
                                "description": "e.g. '(107776 * 4) / 2'"}},
              "required": ["expression"]},
             calculator, example={"expression": "17 * 23 + 4"}),
        Tool("search_docs",
             "Keyword (BM25) search over the READMEs of this repo's modules (mlp, cnn, rnn, "
             "lstm, attention, transformer). Returns the top passages with their source "
             "section, and the query words that appear in no document. Use for facts and numbers "
             "stated in the docs.",
             {"type": "object", "properties": {
                 "query": {"type": "string", "maxLength": 300,
                           "description": "a few specific keywords, e.g. 'mini-GPT parameters'"},
                 "k": {"type": "integer", "minimum": 1, "maximum": 5, "default": 3,
                       "description": "number of passages"}},
              "required": ["query"]},
             search_fn, example={"query": "KV cache memory per token", "k": 3}),
        Tool("read_file",
             "Read a text file inside this repository (path relative to the repo root, "
             f"e.g. 'transformer/README.md'). Returns line/char counts and up to {READ_CAP} "
             "characters. Cannot read outside the repo or hidden files.",
             {"type": "object", "properties": {
                 "path": {"type": "string", "description": "e.g. 'mlp/README.md'"},
                 "max_chars": {"type": "integer", "minimum": 1, "maximum": READ_CAP,
                               "default": READ_CAP}},
              "required": ["path"]},
             read_file, example={"path": "mlp/README.md"}),
        Tool("unit_convert",
             "Convert a value between units of length (m km cm mm mi ft in yd), mass (kg g lb "
             "oz), time (s ms min h day), data (B KB MB GB TB KiB MiB GiB) or temperature "
             "(C F K).",
             {"type": "object", "properties": {
                 "value": {"type": "number", "description": "the number to convert"},
                 "from_unit": {"type": "string", "description": "e.g. 'km'"},
                 "to_unit": {"type": "string", "description": "e.g. 'mi'"}},
              "required": ["value", "from_unit", "to_unit"]},
             unit_convert, example={"value": 5, "from_unit": "km", "to_unit": "mi"}),
        Tool("kv_memory",
             "Long-term key-value memory for this user. action='remember' stores key=value "
             "(MUTATING: may need user confirmation), 'recall' fetches a key, 'list' shows keys, "
             "'forget' deletes a key.",
             {"type": "object", "properties": {
                 "action": {"type": "string", "enum": ["remember", "recall", "list", "forget"]},
                 "key": {"type": "string", "description": "e.g. 'favorite_optimizer'"},
                 "value": {"type": "string", "description": "only for remember"}},
              "required": ["action"]},
             mem, side_effect="mutating",
             mutating_if=lambda a: a.get("action") in ("remember", "forget"),
             example={"action": "recall", "key": "favorite_optimizer"}),
    ]
    for t in tools:
        t.timeout_s = timeout_s
    reg = ToolRegistry(tools, faults)
    reg.memory = mem
    return reg


def demo():
    reg = make_registry(memory={"favorite optimizer": "AdamW"})
    print("Side-effect classes:", {n: t.side_effect for n, t in reg.tools.items()}, "\n")
    cases = [
        ("calculator", {"expression": "107776 * 4 / 2"}),
        ("calculator", {"expression": "__import__('os').system('ls')"}),
        ("calculator", {"expression": "9**9**9"}),
        ("calculator", {"expr": "2+2"}),
        ("search_docs", {"query": "KV cache memory LLaMA", "k": 1}),
        ("read_file", {"path": "../../etc/passwd"}),
        ("read_file", {"path": ".gitignore"}),
        ("read_file", {"path": "mlp/README.md", "max_chars": 80}),
        ("unit_convert", {"value": "5", "from_unit": "kilometers", "to_unit": "miles"}),
        ("unit_convert", {"value": "5 km", "from_unit": "km", "to_unit": "kg"}),
        ("kv_memory", {"action": "Recall", "key": "my favorite optimizer"}),
        ("calculater", {"expression": "1+1"}),
    ]
    for name, args in cases:
        r = reg.call(name, args)
        print(f"{name}({to_json(args)})\n  -> {'OK ' if r.ok else 'ERR'} {r.content[:240]}\n")


if __name__ == "__main__":
    demo()
