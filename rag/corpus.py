"""
The corpus and the eval set: the six READMEs of this repo, parsed into a markdown
heading tree, and the hand-written questions in rag/qa.jsonl with their gold evidence.

This file teaches one thing that most RAG tutorials skip: how to LABEL a retrieval
eval so the labels survive re-chunking. A gold label like "chunk 17 is relevant"
breaks the moment you change the chunk size. So each question in qa.jsonl points at
the SOURCE instead: a file, a section heading and a short evidence substring that
must appear verbatim in that file. We locate the substring's character span once,
and then any chunk from any chunker is relevant if it covers at least half of that
span. Change the chunker, the labels still mean the same thing.

    {"id": "k07", "type": "keyword", "question": "...", "answer": "...",
     "evidence": [{"file": "transformer/README.md",
                   "section": "10. Generation: sampling and the KV cache",
                   "text": "0.5 MB per token for LLaMA-2-7B in fp16",
                   "alt": [ ...other places that also answer it... ]}]}

`evidence` is a list of SLOTS (multi-hop questions have 2: both are needed).
A slot may list alternatives (`alt`): any one of them satisfies the slot.
Unanswerable questions have no slots: the right behaviour is to abstain.

Run (prints corpus statistics and the eval set breakdown):
    python rag/corpus.py
"""
from __future__ import annotations

import json
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DOC_NAMES = ["mlp", "cnn", "rnn", "lstm", "attention", "transformer"]
QA_PATH = os.path.join(HERE, "qa.jsonl")
QTYPES = ["keyword", "semantic", "multihop", "unanswerable"]

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*$")


@dataclass
class Section:
    start: int            # char offset of the heading line in the doc
    end: int              # char offset where the next heading of the same or higher level starts
    level: int            # 1 = "#", 2 = "##", ...
    title: str            # heading text without the #'s
    path: Tuple[str, ...]  # titles from the document title down to this heading


@dataclass
class Doc:
    name: str             # "transformer/README.md"
    text: str
    sections: List[Section] = field(default_factory=list)

    @property
    def title(self):
        return self.sections[0].title if self.sections else self.name

    def section_at(self, pos: int) -> Section:
        """The innermost section containing char offset `pos`."""
        best = self.sections[0]
        for s in self.sections:
            if s.start <= pos < s.end:
                best = s  # sections are in document order, so the last match is innermost
        return best


def parse_sections(text: str) -> List[Section]:
    """Markdown headings -> sections with their heading path. Lines inside ``` code
    fences are never headings (a '# comment' in a code block is not a title)."""
    heads = []  # (start, level, title)
    pos, in_code = 0, False
    for line in text.splitlines(keepends=True):
        if line.lstrip().startswith("```"):
            in_code = not in_code
        elif not in_code:
            m = HEADING_RE.match(line)
            if m:
                heads.append((pos, len(m.group(1)), m.group(2)))
        pos += len(line)
    sections, stack = [], []
    for i, (start, level, title) in enumerate(heads):
        end = len(text)
        for start2, level2, _ in heads[i + 1:]:
            if level2 <= level:
                end = start2
                break
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
        sections.append(Section(start, end, level, title, tuple(t for _, t in stack)))
    return sections


_CORPUS_CACHE: Dict[str, List[Doc]] = {}


def load_corpus() -> List[Doc]:
    """The six READMEs, read at runtime (so edits to them flow into the index)."""
    if "docs" not in _CORPUS_CACHE:
        docs = []
        for name in DOC_NAMES:
            path = os.path.join(REPO, name, "README.md")
            with open(path, encoding="utf-8") as f:
                text = f.read()
            docs.append(Doc(f"{name}/README.md", text, parse_sections(text)))
        _CORPUS_CACHE["docs"] = docs
    return _CORPUS_CACHE["docs"]


def doc_by_name(docs: List[Doc]) -> Dict[str, Doc]:
    return {d.name: d for d in docs}


# ----------------------------------------------------------------------------
# Locating evidence strings: whitespace-insensitive exact match
# ----------------------------------------------------------------------------
def _normalize_with_map(text: str):
    """Collapse every run of whitespace to one space; remember where each kept char
    came from. Evidence strings are written on one line, but the README wraps them."""
    out, idx, prev_space = [], [], False
    for i, c in enumerate(text):
        if c.isspace():
            if not prev_space:
                out.append(" ")
                idx.append(i)
            prev_space = True
        else:
            out.append(c)
            idx.append(i)
            prev_space = False
    return "".join(out), idx


_NORM_CACHE: Dict[str, Tuple[str, List[int]]] = {}


def find_all(doc: Doc, evidence: str) -> List[Tuple[int, int]]:
    """Every (start, end) char span of `evidence` in the doc, ignoring whitespace layout."""
    if doc.name not in _NORM_CACHE:
        _NORM_CACHE[doc.name] = _normalize_with_map(doc.text)
    norm, idx = _NORM_CACHE[doc.name]
    needle = " ".join(evidence.split())
    spans, i = [], norm.find(needle)
    while i >= 0 and needle:
        spans.append((idx[i], idx[i + len(needle) - 1] + 1))
        i = norm.find(needle, i + 1)
    return spans


# ----------------------------------------------------------------------------
# The eval set
# ----------------------------------------------------------------------------
@dataclass
class Evidence:
    file: str
    section: str
    text: str
    span: Optional[Tuple[int, int]] = None  # filled in by load_qa()


@dataclass
class Question:
    id: str
    type: str
    question: str
    answer: str
    slots: List[List[Evidence]]  # each slot: alternatives, any one is enough

    @property
    def answerable(self):
        return len(self.slots) > 0


def load_qa(path: str = QA_PATH, docs: Optional[List[Doc]] = None) -> List[Question]:
    docs = docs or load_corpus()
    by_name = doc_by_name(docs)
    qs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            slots = []
            for ev in r["evidence"]:
                slot = []
                for a in [ev] + ev.get("alt", []):
                    e = Evidence(a["file"], a["section"], a["text"])
                    spans = find_all(by_name[e.file], e.text) if e.file in by_name else []
                    e.span = spans[0] if len(spans) == 1 else None
                    slot.append(e)
                slots.append(slot)
            qs.append(Question(r["id"], r["type"], r["question"], r["answer"], slots))
    return qs


def validate_qa(qs: List[Question], docs: Optional[List[Doc]] = None) -> List[str]:
    """Every evidence string must occur EXACTLY ONCE in its file, inside the named
    section. Returns a list of problems (empty = all good)."""
    by_name = doc_by_name(docs or load_corpus())
    problems, seen = [], set()
    for q in qs:
        if q.id in seen:
            problems.append(f"{q.id}: duplicate id")
        seen.add(q.id)
        if q.type not in QTYPES:
            problems.append(f"{q.id}: unknown type {q.type!r}")
        if (q.type == "unanswerable") != (not q.answerable):
            problems.append(f"{q.id}: unanswerable questions (and only they) have no evidence")
        if q.type == "multihop" and len(q.slots) < 2:
            problems.append(f"{q.id}: a multi-hop question needs >= 2 evidence slots")
        if q.answerable and not q.answer.strip():
            problems.append(f"{q.id}: missing gold answer")
        for slot in q.slots:
            for e in slot:
                if e.file not in by_name:
                    problems.append(f"{q.id}: unknown file {e.file}")
                    continue
                doc = by_name[e.file]
                spans = find_all(doc, e.text)
                if len(spans) != 1:
                    problems.append(f"{q.id}: evidence found {len(spans)}x in {e.file}: {e.text!r}")
                    continue
                sec = doc.section_at(spans[0][0])
                if sec.title != e.section:
                    problems.append(f"{q.id}: evidence is in section {sec.title!r}, "
                                    f"not {e.section!r}")
    return problems


def main():
    docs = load_corpus()
    print("Corpus: the six READMEs of this repo\n")
    print(f"  {'file':24s} {'chars':>7s} {'lines':>6s} {'sections':>9s}  title")
    for d in docs:
        print(f"  {d.name:24s} {len(d.text):7,d} {d.text.count(chr(10)):6,d} "
              f"{len(d.sections):9d}  {d.title[:44]}")
    total = sum(len(d.text) for d in docs)
    lines = sum(d.text.count("\n") for d in docs)
    print(f"  {'total':24s} {total:7,d} {lines:6,d}")
    qs = load_qa(docs=docs)
    print(f"\nEval set: {len(qs)} questions in {os.path.relpath(QA_PATH)}")
    c = Counter(q.type for q in qs)
    for t in QTYPES:
        print(f"  {t:13s} {c[t]:3d}")
    files = Counter(slot[0].file.split("/")[0] for q in qs for slot in q.slots)
    print("  gold evidence slots by file: " + ", ".join(f"{k} {v}" for k, v in sorted(files.items())))
    problems = validate_qa(qs, docs)
    n_ev = sum(len(slot) for q in qs for slot in q.slots)
    if not problems:
        print(f"\nevidence check: all {n_ev} evidence strings found exactly once, in the named section")
    for p in problems:
        print("  PROBLEM:", p)


if __name__ == "__main__":
    main()
