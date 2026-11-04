"""
Chunking: how to cut documents into the pieces a retriever can find.

A retriever never returns "a document", it returns a CHUNK. The chunk is the unit of
scoring (BM25 and embeddings see only its words), and the unit the generator reads.
Too small and a chunk lacks the context to be found or understood; too big and its
vector or term counts are diluted by unrelated text, and you waste the LLM's context.
Cut in the wrong place and the answer is split across two chunks that each look
half-relevant.

Four strategies, all returning chunks with exact character spans into the source:

  fixed-chars   every `size` characters with `overlap` (cuts mid-word, mid-table)
  fixed-words   every `n` words with overlap (cuts mid-sentence)
  sentence      pack whole sentences up to `max_chars`, overlap by one sentence
  markdown      follow the heading tree: never cross a section, pack whole paragraphs /
                code blocks, split only oversized blocks by sentence
  markdown+ctx  the same chunks, each PREFIXED with its file and heading path
                ("contextual headers"): "lstm/README.md | LSTM and GRU ... > 8. Practical tips"

The prefix is the cheapest trick in RAG: a chunk that says "set it to 1" is
unfindable; "LSTM ... > The forget-gate bias = 1 trick: set it to 1" is not. Anthropic's
"Contextual Retrieval" (2024) does the same with an LLM-written context sentence.

Run (stats per strategy, and one passage cut by every strategy):
    python rag/chunking.py
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

from corpus import Doc, load_corpus, load_qa

Span = Tuple[int, int]


@dataclass
class Chunk:
    id: int
    doc: str       # "lstm/README.md"
    start: int     # char span of the body in the doc text
    end: int
    body: str      # doc.text[start:end]
    heading: str   # "LSTM and GRU ... > 8. Practical tips" (heading path at `start`)
    text: str      # what gets indexed: body, or header + body for contextual chunks

    @property
    def section(self):
        return self.heading.split(" > ")[-1]


# ----------------------------------------------------------------------------
# Splitting text into units: blocks (paragraphs / code fences) and sentences
# ----------------------------------------------------------------------------
LIST_LINE = re.compile(r"^\s*(?:[-*>|]|\d+\.\s)")
SENT_END = re.compile(r"(?<=[.!?])[\"')\]*]*\s+(?=[A-Z0-9`*(\[\"'])")
NO_SPLIT_AFTER = ("al.", "e.g.", "i.e.", "vs.", "etc.", "Fig.", "cf.")
LIST_NUMBER = re.compile(r"^\s*(?:[-*]\s*)?\d+\.$")  # "1." of a numbered list is not a sentence


def _trim(text: str, s: int, e: int) -> Span:
    while s < e and text[s].isspace():
        s += 1
    while e > s and text[e - 1].isspace():
        e -= 1
    return s, e


def blocks(text: str, s: int, e: int) -> List[Tuple[int, int, str]]:
    """Paragraphs (separated by blank lines) and whole ``` code blocks in text[s:e]."""
    out, pos, cur, in_code = [], s, None, False
    for line in text[s:e].splitlines(keepends=True):
        lstart, pos = pos, pos + len(line)
        fence = line.lstrip().startswith("```")
        if in_code:
            if fence:
                out.append((cur, pos, "code"))
                cur, in_code = None, False
            continue
        if fence:
            if cur is not None:
                out.append((cur, lstart, "text"))
            cur, in_code = lstart, True
        elif not line.strip():
            if cur is not None:
                out.append((cur, lstart, "text"))
                cur = None
        elif cur is None:
            cur = lstart
    if cur is not None:
        out.append((cur, pos, "code" if in_code else "text"))
    return [(a, b, k) for a, b, k in ((*_trim(text, a, b), k) for a, b, k in out) if b > a]


def sentence_spans(text: str, s: int, e: int, kind: str = "text") -> List[Span]:
    """Sentences of a text block. Code blocks split by line; list items, table rows and
    quote lines start a new unit; abbreviations like "et al." don't end a sentence."""
    lines, pos = [], s
    for line in text[s:e].splitlines(keepends=True):
        a, b = _trim(text, pos, pos + len(line))
        pos += len(line)
        if b > a:
            if kind == "code" or not lines or LIST_LINE.match(text[a:b]):
                lines.append([a, b])
            else:
                lines[-1][1] = b  # a wrapped line continues the previous one
    spans = []
    for a, b in lines:
        if kind == "code" or text[a] == "#":  # code lines and headings stay whole
            spans.append((a, b))
            continue
        cuts = [a]
        for m in SENT_END.finditer(text, a, b):
            before = text[cuts[-1]:m.start()].rstrip("\"')]*")
            if not before.endswith(NO_SPLIT_AFTER) and not LIST_NUMBER.match(before):
                cuts.append(m.end())
        cuts.append(b)
        spans += [_trim(text, x, y) for x, y in zip(cuts[:-1], cuts[1:])]
    return [sp for sp in spans if sp[1] > sp[0]]


def _pack(units: List[Span], max_chars: int, overlap_units: int = 0) -> List[Span]:
    """Greedily pack consecutive units into spans of at most max_chars. A unit that is
    longer than max_chars on its own is cut into fixed pieces (the last resort)."""
    flat = []
    for a, b in units:
        while b - a > max_chars:
            flat.append((a, a + max_chars))
            a += max_chars
        flat.append((a, b))
    out, i = [], 0
    while i < len(flat):
        j = i + 1
        while j < len(flat) and flat[j][1] - flat[i][0] <= max_chars:
            j += 1
        out.append((flat[i][0], flat[j - 1][1]))
        i = max(j - overlap_units, i + 1) if j < len(flat) else j
    return out


# ----------------------------------------------------------------------------
# The chunkers
# ----------------------------------------------------------------------------
def heading_path(doc: Doc, pos: int) -> str:
    return " > ".join(doc.section_at(pos).path)


def header(doc: Doc, heading: str) -> str:
    return f"{doc.name} | {heading}\n"


def _make(doc: Doc, spans: List[Span], contextual: bool, out: List[Chunk]):
    for a, b in spans:
        a, b = _trim(doc.text, a, b)
        if b <= a:
            continue
        h = heading_path(doc, a)
        body = doc.text[a:b]
        out.append(Chunk(len(out), doc.name, a, b, body, h,
                         header(doc, h) + body if contextual else body))


def chunk_fixed_chars(docs: List[Doc], size: int = 1000, overlap: int = 200) -> List[Chunk]:
    out: List[Chunk] = []
    for d in docs:
        starts = range(0, max(1, len(d.text) - overlap), size - overlap)
        _make(d, [(s, min(s + size, len(d.text))) for s in starts], False, out)
    return out


def chunk_fixed_words(docs: List[Doc], n: int = 160, overlap: int = 40) -> List[Chunk]:
    out: List[Chunk] = []
    for d in docs:
        words = [m.span() for m in re.finditer(r"\S+", d.text)]
        spans = [(words[i][0], words[min(i + n, len(words)) - 1][1])
                 for i in range(0, max(1, len(words) - overlap), n - overlap)]
        _make(d, spans, False, out)
    return out


def chunk_sentences(docs: List[Doc], max_chars: int = 1000, overlap: int = 1) -> List[Chunk]:
    out: List[Chunk] = []
    for d in docs:
        units = [sp for a, b, k in blocks(d.text, 0, len(d.text))
                 for sp in sentence_spans(d.text, a, b, k)]
        _make(d, _pack(units, max_chars, overlap), False, out)
    return out


def own_regions(doc: Doc) -> List[Span]:
    """Each section's own text: from its heading to the next heading of ANY level."""
    starts = [s.start for s in doc.sections] + [len(doc.text)]
    return [(a, b) for a, b in zip(starts[:-1], starts[1:])]


def chunk_markdown(docs: List[Doc], max_chars: int = 1000, contextual: bool = True) -> List[Chunk]:
    out: List[Chunk] = []
    for d in docs:
        if d.sections and d.sections[0].start > 0:  # text before the first heading
            _make(d, _pack([(0, d.sections[0].start)], max_chars), contextual, out)
        for a, b in own_regions(d):
            bl = blocks(d.text, a, b)
            if len(bl) <= 1:
                continue  # just a heading line: its title lives on in the subsections' paths
            units = []
            for s, e, kind in bl:
                if e - s <= max_chars:
                    units.append((s, e))
                else:  # oversized paragraph / code block: fall back to sentences / lines
                    units += sentence_spans(d.text, s, e, kind)
            _make(d, _pack(units, max_chars), contextual, out)
    return out


CHUNKERS: Dict[str, Callable[[List[Doc]], List[Chunk]]] = {
    "fixed-chars": chunk_fixed_chars,
    "fixed-words": chunk_fixed_words,
    "sentence": chunk_sentences,
    "markdown": lambda docs: chunk_markdown(docs, contextual=False),
    "markdown+ctx": lambda docs: chunk_markdown(docs, contextual=True),
}
DEFAULT_CHUNKER = "markdown+ctx"

_CHUNK_CACHE: Dict[str, List[Chunk]] = {}


def get_chunks(name: str = DEFAULT_CHUNKER, docs=None) -> List[Chunk]:
    if name not in _CHUNK_CACHE:
        _CHUNK_CACHE[name] = CHUNKERS[name](docs or load_corpus())
    return _CHUNK_CACHE[name]


# ----------------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------------
def chunk_stats(chunks: List[Chunk], docs: List[Doc], qs=None) -> Dict[str, float]:
    """Sizes, plus two failure counts that matter for retrieval:
    - cross: chunks whose body spans two sections (mixes topics)
    - mid: chunks that end in the middle of a sentence (or a line of code)
    - split: gold evidence strings that no single chunk contains completely"""
    by = {d.name: d for d in docs}
    lens = [len(c.body) for c in chunks]
    words = [len(c.text.split()) for c in chunks]
    cross = sum(by[c.doc].section_at(c.start).start != by[c.doc].section_at(c.end - 1).start
                for c in chunks)
    ends = {d.name: {b for a, b0, k in blocks(d.text, 0, len(d.text))
                     for _, b in sentence_spans(d.text, a, b0, k)} for d in docs}
    mid = sum(c.end not in ends[c.doc] for c in chunks)
    split, total = 0, 0
    for q in qs or []:
        for slot in q.slots:
            for e in slot:
                total += 1
                split += not any(c.doc == e.file and c.start <= e.span[0] and e.span[1] <= c.end
                                 for c in chunks)
    return dict(n=len(chunks), mean_chars=sum(lens) / len(lens), max_chars=max(lens),
                mean_words=sum(words) / len(words), cross=cross, mid=mid, split=split, evidence=total,
                index_chars=sum(len(c.text) for c in chunks))


def main():
    docs = load_corpus()
    qs = load_qa(docs=docs)
    print(f"Corpus: {len(docs)} READMEs, {sum(len(d.text) for d in docs):,} chars\n")
    print(f"  {'strategy':13s} {'chunks':>6s} {'mean chars':>10s} {'max':>5s} {'mean words':>10s}"
          f" {'cross-section':>13s} {'mid-sentence':>12s} {'evidence split':>15s}"
          f" {'indexed chars':>13s}")
    for name in CHUNKERS:
        st = chunk_stats(get_chunks(name, docs), docs, qs)
        print(f"  {name:13s} {st['n']:6d} {st['mean_chars']:10.0f} {st['max_chars']:5d} "
              f"{st['mean_words']:10.0f} {st['cross']:13d} {st['mid']:12d} "
              f"{st['split']:6d} of {st['evidence']:3d} {st['index_chars']:13,d}")
    print("\n  cross-section = chunks whose text spans two sections (mixes topics)")
    print("  mid-sentence = chunks whose last character is not the end of a sentence")
    print("  evidence split = gold evidence strings that no single chunk contains in full")
    print("  indexed chars > corpus chars because of overlap and contextual headers\n")

    d = next(x for x in docs if x.name == "lstm/README.md")
    sec = next(s for s in d.sections if s.title.startswith("The forget-gate bias"))
    lo, hi = sec.start, sec.start + 1400
    print(f"How each strategy cuts lstm/README.md around '{sec.title}' "
          f"(chars {lo}-{hi}):\n")
    for name in CHUNKERS:
        print(f"  {name}:")
        for c in get_chunks(name, docs):
            if c.doc == d.name and c.end > lo and c.start < hi:
                first = " ".join(c.text.split())[:70]
                last = " ".join(c.body.split())[-30:]
                print(f"    [{c.start:5d}-{c.end:5d}] {first!r} ... {last!r}")
        print()


if __name__ == "__main__":
    main()
