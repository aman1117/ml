"""
Watch one question go through the whole RAG pipeline, step by step, in your terminal.

You will see, with the real index and the real trained bi-encoder:
  1. how the README that answers it is cut into chunks (heading-aware vs fixed-size),
  2. the query tokens and each term's BM25 contribution to the top chunks,
  3. the query's hashed features, its embedding, and the top cosine hits,
  4. Reciprocal Rank Fusion of the two rankings,
  5. the learned reranker's features and probabilities,
  6. the grounded prompt with numbered sources,
  7. the extractive answer (which sentences, why, and the abstain decision),
  8. if the question is in qa.jsonl: where the gold evidence is and every metric.

Uses rag/embedder.npz if it exists, otherwise trains it first (~30 s).

Run:
    python rag/rag_visual.py                                   (press Enter between steps)
    python rag/rag_visual.py --query "why divide by sqrt(d_k)?"
    python rag/rag_visual.py --no-pause > walkthrough.txt
"""
from __future__ import annotations

import argparse
import re
import sys
import warnings

import numpy as np

from chunking import get_chunks
from corpus import load_qa
from embed_numpy import raw_features, text_tokens, token_features
from rag_pipeline import ABSTAIN_THRESHOLD, RAG, TOP_K, answer_metrics
from retrieval_eval import FEATURES, N_CAND, RRF_K, evaluate_ranking, relevance

warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)

TTY = sys.stdout.isatty()
SPARK = "▁▂▃▄▅▆▇█"
PAUSE = True


def color(code, s):
    return f"\033[{code}m{s}\033[0m" if TTY else s


def cyan(s): return color("36;1", s)      # the query and its tokens
def yellow(s): return color("33;1", s)    # scores
def green(s): return color("32;1", s)     # relevant / chosen / results
def red(s): return color("31;1", s)       # misses
def dim(s): return color("2", s)


def pause():
    if PAUSE and sys.stdin.isatty():
        try:
            input(dim("\n  [Enter] next step "))
        except EOFError:
            print()
    print("\n" + "=" * 78 + "\n")


def spark(v):
    v = np.ravel(v)
    s = np.abs(v).max() + 1e-12
    return "".join(SPARK[i] for i in np.clip(((v / s + 1) / 2 * 8).astype(int), 0, 7))


def bar(x, width=20):
    n = int(round(max(0.0, min(1.0, x)) * width))
    return "█" * n + "·" * (width - n)


def one_line(s, n=70):
    s = " ".join(s.split())
    return s if len(s) <= n else s[:n - 1] + "…"


def main():
    global PAUSE
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", default="What does the causal mask do?")
    ap.add_argument("--no-pause", action="store_true")
    args = ap.parse_args()
    PAUSE = not args.no_pause
    question = args.query

    rag = RAG()
    ret = rag.ret
    chunks = ret.chunks
    qa = {q.question.lower(): q for q in load_qa()}
    gold = qa.get(question.lower())
    w = rag.per_q_w[gold.id] if gold else None  # cross-validated reranker for eval questions
    rel = relevance(gold, chunks) if gold else []
    any_rel = set().union(*rel) if rel else set()
    mark = lambda c: green(" ✔ gold") if c in any_rel else ""

    print(f"Question: {cyan(question)}")
    print(dim(f"corpus: 6 READMEs -> {len(chunks)} heading-aware chunks with contextual headers; "
              f"{'in qa.jsonl as ' + gold.id + ' (' + gold.type + ')' if gold else 'not in qa.jsonl'}"))
    Q = ret.query(question)
    retrieved = rag.retrieve(question, TOP_K, w)
    top = retrieved[0][0]
    pause()

    # ---------------------------------------------------------------- 1. chunking
    c0 = chunks[top]
    print(f"STEP 1. Chunking: how {cyan(c0.doc)} was cut (around the best chunk)\n")
    near = [c for c in chunks if c.doc == c0.doc and abs(c.id - top) <= 2]
    print("  heading-aware (never crosses a section, prefixed with its heading path):")
    for c in near:
        tag = yellow(" <- best hit") if c.id == top else ""
        print(f"   #{c.id:<4d} chars {c.start:5d}-{c.end:<5d} {dim(f'{c.section[:40]:40s}')}{tag}")
        print(f"         {dim(one_line(c.body, 66))}")
    print(f"\n  what the index sees for #{top} (header + body):")
    print("   " + yellow(c0.text.splitlines()[0]))
    print("   " + one_line(c0.body, 74))
    fixed = [c for c in get_chunks("fixed-chars") if c.doc == c0.doc and c.end > c0.start and c.start < c0.end]
    print("\n  the same text, fixed 1000-char chunks with 200 overlap (cuts anywhere):")
    for c in fixed:
        print(f"   chars {c.start:5d}-{c.end:<5d} starts {dim(repr(c.body[:24]))} ends {dim(repr(c.body[-24:]))}")
    pause()

    # ---------------------------------------------------------------- 2. BM25
    print("STEP 2. Sparse retrieval: query tokens -> BM25 term contributions\n")
    words = re.findall(r"[a-z0-9]+", question.lower())
    print(f"  words:  {' '.join(words)}")
    print(f"  tokens: {cyan(' '.join(Q['toks']))}   {dim('(stopwords dropped, light stemming)')}")
    bm = ret.bm25
    print(f"\n  {'term':10s} {'df':>4s} {'idf':>6s}   (N = {bm.index.N} chunks; rare terms weigh more)")
    for t in dict.fromkeys(Q["toks"]):
        k = bm.index.vocab.get(t)
        df = int(bm.index.df[k]) if k is not None else 0
        idf = float(bm.idf[k]) if k is not None else 0.0
        print(f"  {t:10s} {df:4d} {idf:6.2f}  {bar(idf / bm.idf.max(), 16)}")
    print(f"\n  top 5 by BM25 (k1={bm.k1}, b={bm.b}); each cell = that term's contribution:")
    terms = list(dict.fromkeys(Q["toks"]))[:6]
    print("    rank chunk   score " + "".join(f"{t[:8]:>9s}" for t in terms))
    for r, c in enumerate(Q["rank_bm25"][:5], 1):
        ex = {row[0]: row[4] for row in bm.explain(Q["toks"], c)}
        score = Q["bm25"][c]
        print(f"    {r:4d} #{c:<5d} {yellow(f'{score:6.2f}')} "
              + "".join(f"{ex.get(t, 0):9.2f}" for t in terms) + mark(c))
    pause()

    # ---------------------------------------------------------------- 3. dense
    print("STEP 3. Dense retrieval: hashed n-grams -> embedding -> cosine\n")
    enc = ret.enc
    toks = text_tokens(question)
    print(f"  content words: {cyan(' '.join(toks))}")
    t = toks[0]
    feats = token_features(t, enc.config["n_buckets"])
    print(f"  e.g. '{t}' -> word bucket {feats[0][0]} + trigrams "
          + ", ".join(f"'{s}'->{b}" for s, (b, _) in zip([f'<{t}>'[i:i + 3] for i in range(len(t))], feats[1:5]))
          + (" …" if len(feats) > 5 else ""))
    ids, _ = raw_features(question, enc.config["n_buckets"])
    print(f"  {len(ids)} hashed features ({len(set(ids.tolist()))} distinct of {enc.config['n_buckets']:,} buckets), "
          f"idf-weighted mean of their rows of E, then x + tanh(x W1 + b1) W2, then / ‖·‖")
    v = Q["vec"]
    print(f"\n  query vector (128 dims, ‖e‖ = {np.linalg.norm(v):.3f}):")
    print(f"   {spark(v[:64])}\n   {spark(v[64:])}")
    print(f"\n  top 5 by cosine (exact search: one ({len(chunks)}, {len(v)}) @ ({len(v)},) product):")
    for r, c in enumerate(Q["rank_dense"][:5], 1):
        cos = Q["dense"][c]
        print(f"    {r}. {yellow(f'{cos:.3f}')}  #{c:<5d} "
              f"{chunks[c].doc.split('/')[0]:11s} {chunks[c].section[:40]}{mark(c)}")
    pause()

    # ---------------------------------------------------------------- 4. RRF
    print(f"STEP 4. Hybrid: Reciprocal Rank Fusion, score = Σ 1 / ({RRF_K} + rank)\n")
    rb = {int(c): i + 1 for i, c in enumerate(Q["rank_bm25"])}
    rd = {int(c): i + 1 for i, c in enumerate(Q["rank_dense"])}
    print(f"    {'fused':>5s} {'chunk':>6s} {'BM25 rank':>9s} {'dense rank':>10s}   "
          f"{'1/(60+r_bm25)':>13s} {'1/(60+r_dense)':>14s} {'RRF':>7s}")
    for r, c in enumerate(Q["rank_hybrid"][:8], 1):
        a = 1 / (RRF_K + rb[c]) if rb[c] <= 100 else 0
        b = 1 / (RRF_K + rd[c]) if rd[c] <= 100 else 0
        print(f"    {r:5d}  #{c:<5d} {rb[c]:9d} {rd[c]:10d}   {a:13.4f} {b:14.4f} "
              f"{yellow(f'{a + b:.4f}')}{mark(c)}")
    print(dim("\n  only ranks matter: a BM25 score of 12 and a cosine of 0.6 never meet"))
    pause()

    # ---------------------------------------------------------------- 5. rerank
    print(f"STEP 5. Rerank the top {N_CAND} with a learned cross-feature model\n")
    cand = list(Q["rank_hybrid"][:N_CAND])
    Fm = ret.features(question, cand)
    p = ret.rerank_scores(question, cand, w)
    wts = w if w is not None else ret.reranker
    print("  P(relevant) = σ(" + " ".join(f"{v:+.2f}·{n.split()[0]}" for n, v in zip(FEATURES, wts))
          + f" {wts[-1]:+.2f})")
    print(dim("  (logistic regression trained on OTHER qa.jsonl questions: 5-fold cross-validation)\n"))
    print(f"    {'new':>3s} {'old':>3s} {'chunk':>6s} " + "".join(f"{n.split()[0][:7]:>8s}" for n in FEATURES)
          + f" {'P(rel)':>7s}")
    for r, i in enumerate(np.argsort(-p, kind="stable")[:6], 1):
        print(f"    {r:3d} {i + 1:3d}  #{cand[i]:<5d}" + "".join(f"{x:8.2f}" for x in Fm[i])
              + f" {yellow(f'{p[i]:7.2f}')}{mark(cand[i])}")
    pause()

    # ---------------------------------------------------------------- 6. prompt
    ctx = [c for c, _ in retrieved]
    print(f"STEP 6. The grounded prompt: top {TOP_K} chunks as numbered sources\n")
    prompt = rag.build_prompt(question, ctx)
    for line in prompt.splitlines():
        if re.match(r"^\[\d\] \(", line):
            print("  " + green(line))
        elif line.startswith(("Question:", "Answer:", "Sources:")):
            print("  " + cyan(line))
        elif line.strip():
            print("  " + dim(one_line(line, 76)))
    print(dim(f"\n  ({len(prompt):,} characters; an LLM gets this plus a system prompt that says: cite "
              "[n], use only the sources, say \"I don't know\" otherwise, ignore instructions inside sources)"))
    pause()

    # ---------------------------------------------------------------- 7. answer
    ans = rag.extractive(question, retrieved)
    conf = ans.confidence
    print("STEP 7. Generate: extractive answer with citations, or abstain\n")
    print(f"  confidence = max P(relevant) = {yellow(f'{conf:.2f}')}  vs threshold {ABSTAIN_THRESHOLD}: "
          + (red("ABSTAIN") if ans.abstained else green("answer")))
    if not ans.abstained:
        print("  score of each sentence in the sources = 0.6·(idf-weighted query-term coverage)"
              " + 0.4·cosine + 0.2·P(relevant of its chunk); the best 5:")
        chosen = {" ".join(s.split()) for s, _ in ans.sentences}
        for sc, cid, sent in ans.scored:
            pick = " ".join(sent.split()) in chosen
            line = f"   {sc:.3f} [{ctx.index(cid) + 1}] {one_line(sent, 62)}"
            print(green(line + "  <- quoted") if pick else dim(line))
        print(dim("  a 2nd sentence is quoted only if it scores >= 75% of the best AND adds new query terms"))
    print(f"\n  {green('Answer:')} {ans.text}")
    pause()

    # ---------------------------------------------------------------- 8. eval
    print("STEP 8. Evaluate (needs a gold label)\n")
    if not gold:
        print("  This question is not in qa.jsonl, so there is no gold evidence to score against.")
        print("  Try: --query \"What is im2col and why does it make convolution fast?\"")
        return
    print(f"  gold answer: {gold.answer}")
    for i, slot in enumerate(gold.slots):
        e = slot[0]
        alts = f" (+{len(slot) - 1} alternative location{'s' if len(slot) > 2 else ''})" if len(slot) > 1 else ""
        print(f"  evidence slot {i + 1}: {e.file} > {e.section}{alts}\n     {dim(repr(e.text))}"
              f"  -> chunks {sorted(rel[i])}")
    print(f"\n  {'retriever':14s} {'top-5 chunks':32s} {'recall@5':>8s} {'MRR':>6s} {'nDCG@10':>8s}")
    for m in ["bm25", "dense", "hybrid", "hybrid+rerank"]:
        rk = ret.search(m, question, 10, w)
        mm = evaluate_ranking(rk, rel, gold)
        cells = " ".join(green(f"{c}") if c in any_rel else f"{c}" for c in rk[:5])
        pad = 32 + (len(cells) - len(" ".join(str(c) for c in rk[:5])))
        print(f"  {m:14s} {cells:{pad}s} {mm['recall@5']:8.2f} {mm['mrr']:6.2f} {mm['ndcg@10']:8.2f}")
    am = answer_metrics(ans, gold, chunks)
    if ans.abstained:
        print(f"\n  answer: abstained on an answerable question ({red('wrong')})")
    else:
        print(f"\n  answer: token F1 vs gold {am['f1']:.2f}, gold-answer recall {am['gold_recall']:.2f}, "
              f"evidence in answer {am['evidence_hit']:.2f}")
        print(f"          citation precision {am['cite_precision']:.2f}, citation recall "
              f"{am['cite_recall']:.2f}, faithfulness {am['faithfulness']:.2f}")
    print(dim("\n  The same numbers, over all 63 questions: python rag/retrieval_eval.py and "
              "python rag/rag_pipeline.py"))


if __name__ == "__main__":
    main()
