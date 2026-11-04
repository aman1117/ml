"""
The domain dataset: the (fictional) Grey Coast Lighthouse Service handbook.

Our base model (../transformer/) was pretrained on a 3.9 KB story about a
lighthouse keeper. It can continue English text in that style, but it knows no
FACTS about any lighthouse service and has never seen a chat. This file builds
a small, deterministic "domain" dataset to fine-tune it on:

    12 lighthouses x 4 attributes (light colour, supply-boat day, keeper, fog signal)
    = 48 facts, each asked with 3 different question phrasings.

Splits (README section 3), chosen to separate FORMAT from FACTS from JUDGMENT:
    sft_train.jsonl
      train         40 facts x 2 of their 3 phrasings = 80 chats
      refusal_demo  36 demonstrations of declining: 12 out-of-scope questions
                    ("How tall is the tower at Gull Rock?" -> "The handbook does not say.")
                    about 3 lighthouses, and 24 questions about 6 lighthouses that are
                    not in the handbook (-> "I do not know that lighthouse.")
    sft_test.jsonl (never trained on)
      paraphrase    the same 40 facts, their 3rd phrasing = 40 chats
                    -> the fact was trained, this wording of it was not
      unseen_fact   8 facts never trained, all 3 phrasings = 24 chats
                    -> nobody can know these from the training data; a good model
                       still answers in the right FORMAT, but the value is a guess
      out_of_scope  the 4 out-of-scope questions about the last 6 lighthouses = 24 chats
      unknown_place 6 more lighthouses that are not in the handbook x 4 = 24 chats
    prefs.jsonl: (prompt, chosen, rejected) pairs, never on test prompts
      160 on the train prompts: chosen = the correct answer, rejected = a wrong value,
          a wrong kind of answer, a rambling answer or a needless refusal
      24 out-of-scope questions (first 6 lighthouses) and 48 questions about 12
          unknown lighthouses: chosen = the right refusal, rejected = a confident
          made-up answer (a hallucination)

Everything is written in the base model's alphabet (no digits, no q/x/z, only
the capitals that occur in the story), so the only NEW tokens are the three
reserved chat characters below.

Chat template. A chat model sees one flat string. We reserve three characters
that never occur in normal text (ASCII literally has "start of heading", "start
of text" and "end of text" control codes for this) and render a conversation as

    <|user|>What colour is the light at Gull Rock?<|assistant|>The light is amber.<|end|>

The model is trained to produce everything after <|assistant|> including <|end|>
(so it learns to STOP); generation feeds "<|user|>question<|assistant|>" and
stops at <|end|>. Real models do exactly this with multi-character special
tokens (e.g. "<|im_start|>assistant\\n" ... "<|im_end|>" in SmolLM2/Qwen).

Run:
    python finetuning/domain_data.py          (writes the 3 jsonl files and prints examples)
"""
import json
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
STORY = os.path.join(HERE, "..", "rnn", "input.txt")
SEED = 0

# The three reserved single-character special tokens of our char-level chat template.
USER, ASSISTANT, END = "\x01", "\x02", "\x03"
SPECIAL_NAMES = {USER: "<|user|>", ASSISTANT: "<|assistant|>", END: "<|end|>"}

LIGHTHOUSES = ["Gull Rock", "Seal Head", "Wolf Isle", "Storm Bay", "Iron Gate", "Dark Sound",
               "Fog Hill", "Swan Reef", "Mist Head", "Hare Island", "Moon Harbour", "Old Tower"]

# Not in the handbook: UNKNOWN_TRAIN[:6] get SFT refusal demos, all 12 appear in
# preference pairs, UNKNOWN_TEST only in the test.
UNKNOWN_TRAIN = ["Bird Rock", "Sand Isle", "Grey Head", "Ash Bay", "Tide Gate", "Eagle Reef",
                 "Dawn Hill", "Ember Sound", "Wave Tower", "Morning Reef", "Fern Island",
                 "Deer Harbour"]
UNKNOWN_TEST = ["Frost Bay", "Hawk Isle", "Owl Head", "Shell Rock", "Thorn Hill", "Raven Sound"]
REFUSAL = "I do not know that lighthouse."

# Questions the handbook does not answer. SFT demos use LIGHTHOUSES[:3], prefs
# LIGHTHOUSES[:6], the test LIGHTHOUSES[6:].
OUT_OF_SCOPE = ["How tall is the tower at {e}?", "How old is the light at {e}?",
                "How many stairs does {e} have?", "Who built the light at {e}?"]
NOT_IN_HANDBOOK = "The handbook does not say."

# attribute -> (possible values, 3 question phrasings, answer template)
ATTRIBUTES = {
    "colour": (["red", "green", "white", "blue", "amber"],
               ["What colour is the light at {e}?",
                "Which colour does {e} show?",
                "Tell me the colour of the {e} light."],
               "The light is {v}."),
    "boat": (["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
             ["When does the boat come to {e}?",
              "Which day is the boat at {e}?",
              "On what day does {e} get the boat?"],
             "The boat comes on {v}."),
    "keeper": (["Anna", "Ben", "Mara", "Tom", "Ida", "Rosa", "Hugo", "Otto"],
               ["Who keeps the light at {e}?",
                "Who is the keeper of {e}?",
                "Tell me who keeps {e}."],
               "The keeper is {v}."),
    "signal": (["bell", "horn", "gong", "whistle", "siren"],
               ["What fog signal does {e} use?",
                "Which sound warns ships at {e}?",
                "How does {e} warn ships in fog?"],
               "It uses a {v}."),
}

# Sentences from the story, used as "rambling" rejected answers (the failure mode
# of a model that never learned to stop).
RAMBLE = ["The wind was cold and the sea was grey.",
          "The gulls were loud and the boats were slow.",
          "She knocked on the door and the keeper opened it."]

N_UNSEEN_PER_ATTRIBUTE = 2


def build_facts(seed=SEED):
    """{(entity, attribute): value}, deterministic."""
    rng = np.random.default_rng(seed)
    facts = {}
    for e in LIGHTHOUSES:
        for a, (values, _, _) in ATTRIBUTES.items():
            facts[(e, a)] = values[int(rng.integers(len(values)))]
    return facts


def question(e, a, k):
    return ATTRIBUTES[a][1][k].format(e=e)


def answer(a, v):
    return ATTRIBUTES[a][2].format(v=v)


def answer_regex(a):
    """What a well-FORMATTED answer to an attribute-a question looks like (any valid value)."""
    values, _, tmpl = ATTRIBUTES[a]
    pre, post = tmpl.split("{v}")
    return re.compile("^" + re.escape(pre) + "(" + "|".join(values) + ")" + re.escape(post) + "$")


def format_ok(row, text):
    """Right SHAPE of answer for this question? A fact question needs the attribute's
    template with any valid value; a refusal question needs exactly the refusal."""
    if row["value"] is None:
        return text == row["messages"][1]["content"]
    return bool(answer_regex(row["attribute"]).match(text))


def chat(q, a_text=None):
    return {"messages": [{"role": "user", "content": q}]
            + ([] if a_text is None else [{"role": "assistant", "content": a_text}])}


def build_splits(seed=SEED):
    rng = np.random.default_rng(seed + 1)
    facts = build_facts(seed)
    # Hold out N_UNSEEN_PER_ATTRIBUTE facts per attribute, on different lighthouses.
    order = [int(i) for i in rng.permutation(len(LIGHTHOUSES))]
    unseen = set()
    for i, a in enumerate(ATTRIBUTES):
        for j in range(N_UNSEEN_PER_ATTRIBUTE):
            unseen.add((LIGHTHOUSES[order[i * N_UNSEEN_PER_ATTRIBUTE + j]], a))

    train, test = [], []
    n = 0
    for e in LIGHTHOUSES:
        for a in ATTRIBUTES:
            v = facts[(e, a)]
            meta = {"entity": e, "attribute": a, "value": v}
            if (e, a) in unseen:
                for k in range(3):
                    test.append({**chat(question(e, a, k), answer(a, v)), **meta,
                                 "template": k, "split": "unseen_fact"})
                continue
            held = n % 3  # rotate which phrasing is held out, so every phrasing is trained
            n += 1
            for k in range(3):
                row = {**chat(question(e, a, k), answer(a, v)), **meta, "template": k}
                if k == held:
                    test.append({**row, "split": "paraphrase"})
                else:
                    train.append({**row, "split": "train"})
    for e in LIGHTHOUSES[:3]:
        for k, q in enumerate(OUT_OF_SCOPE):
            train.append({**chat(q.format(e=e), NOT_IN_HANDBOOK), "entity": e,
                          "attribute": None, "value": None, "template": k,
                          "split": "refusal_demo"})
    for i, e in enumerate(UNKNOWN_TRAIN[:6]):
        for j, a in enumerate(ATTRIBUTES):
            k = (i + j) % 3
            train.append({**chat(question(e, a, k), REFUSAL), "entity": e, "attribute": a,
                          "value": None, "template": k, "split": "refusal_demo"})
    for e in LIGHTHOUSES[6:]:
        for k, q in enumerate(OUT_OF_SCOPE):
            test.append({**chat(q.format(e=e), NOT_IN_HANDBOOK), "entity": e,
                         "attribute": None, "value": None, "template": k,
                         "split": "out_of_scope"})
    for i, e in enumerate(UNKNOWN_TEST):
        for j, a in enumerate(ATTRIBUTES):
            k = (i + j) % 3
            test.append({**chat(question(e, a, k), REFUSAL), "entity": e, "attribute": a,
                         "value": None, "template": k, "split": "unknown_place"})
    return facts, train, test


def build_prefs(train, seed=SEED):
    """Two preference pairs per training chat: chosen = the correct answer,
    rejected = a wrong value in the right format (a factual error), and then in
    turn a wrong KIND of answer (another attribute's template), a correct answer
    that keeps rambling with story text instead of stopping, or a needless refusal.
    Plus refusal pairs: out-of-scope questions about LIGHTHOUSES[:6], and one
    question per (unknown lighthouse, attribute): chosen = the right refusal,
    rejected = a confident made-up answer (what the SFT model actually does)."""
    rng = np.random.default_rng(seed + 2)
    prefs = []
    for i, row in enumerate(r for r in train if r["split"] == "train"):
        a, v = row["attribute"], row["value"]
        prompt = row["messages"][:1]
        chosen = answer(a, v)
        wrong = [x for x in ATTRIBUTES[a][0] if x != v]
        prefs.append({"prompt": prompt, "chosen": chosen,
                      "rejected": answer(a, wrong[int(rng.integers(len(wrong)))]),
                      "kind": "wrong_value"})
        if i % 3 == 0:
            others = [b for b in ATTRIBUTES if b != a]
            b = others[int(rng.integers(len(others)))]
            vals = ATTRIBUTES[b][0]
            rejected, kind = answer(b, vals[int(rng.integers(len(vals)))]), "wrong_kind"
        elif i % 3 == 1:
            rejected, kind = chosen + " " + RAMBLE[int(rng.integers(len(RAMBLE)))], "rambling"
        else:
            rejected, kind = REFUSAL, "needless_refusal"
        prefs.append({"prompt": prompt, "chosen": chosen, "rejected": rejected, "kind": kind})
    for e in LIGHTHOUSES[:6]:
        for q in OUT_OF_SCOPE:
            b = list(ATTRIBUTES)[int(rng.integers(len(ATTRIBUTES)))]
            vals = ATTRIBUTES[b][0]
            prefs.append({"prompt": chat(q.format(e=e))["messages"], "chosen": NOT_IN_HANDBOOK,
                          "rejected": answer(b, vals[int(rng.integers(len(vals)))]),
                          "kind": "out_of_scope"})
    for i, e in enumerate(UNKNOWN_TRAIN):
        for j, a in enumerate(ATTRIBUTES):
            vals = ATTRIBUTES[a][0]
            prefs.append({"prompt": chat(question(e, a, (i + j + 1) % 3))["messages"],
                          "chosen": REFUSAL,
                          "rejected": answer(a, vals[int(rng.integers(len(vals)))]),
                          "kind": "hallucination"})
    return prefs


# ----------------------------------------------------------------------------
# Rendering the chat template and the loss mask (for the char-level models)
# ----------------------------------------------------------------------------
def render(messages, add_generation_prompt=False):
    """messages -> one flat string with our reserved characters."""
    s = ""
    for m in messages:
        tag = USER if m["role"] == "user" else ASSISTANT
        s += tag + m["content"] + (END if m["role"] == "assistant" else "")
    if add_generation_prompt:
        s += ASSISTANT
    return s


def render_with_mask(messages):
    """The rendered string and a 0/1 per character: 1 = the model must learn to
    PRODUCE this character (assistant content + its <|end|>), 0 = context only."""
    s, m = "", []
    for msg in messages:
        if msg["role"] == "user":
            part = USER + msg["content"]
            s += part
            m += [0] * len(part)
        else:
            s += ASSISTANT
            m += [0]  # the template inserts this tag; the model never has to write it
            part = msg["content"] + END
            s += part
            m += [1] * len(part)
    return s, m


def show(s):
    """Printable version: reserved characters as <|user|> / <|assistant|> / <|end|>."""
    for c, name in SPECIAL_NAMES.items():
        s = s.replace(c, name)
    return s


def build_vocab(story_text):
    """Story characters + the 3 reserved ones. Ids 0..V0-1 are the base model's,
    so the pretrained embeddings keep their meaning; the new tokens come last."""
    chars = sorted(set(story_text))
    return chars + [USER, ASSISTANT, END]


def load_story():
    with open(STORY, encoding="utf-8") as f:
        return f.read()


def load_jsonl(name):
    with open(os.path.join(HERE, name), encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(name, rows):
    with open(os.path.join(HERE, name), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_all():
    """(train, test, prefs) from the committed jsonl files, regenerating them if missing."""
    names = ("sft_train.jsonl", "sft_test.jsonl", "prefs.jsonl")
    if not all(os.path.exists(os.path.join(HERE, n)) for n in names):
        main(verbose=False)
    return tuple(load_jsonl(n) for n in names)


def main(verbose=True):
    facts, train, test = build_splits()
    prefs = build_prefs(train)
    story_chars = set(load_story())
    for r in train + test + prefs:
        text = "".join(m["content"] for m in r.get("messages", r.get("prompt", [])))
        text += r.get("chosen", "") + r.get("rejected", "")
        missing = set(text) - story_chars
        assert not missing, f"characters the base model never saw: {missing}"
    write_jsonl("sft_train.jsonl", train)
    write_jsonl("sft_test.jsonl", test)
    write_jsonl("prefs.jsonl", prefs)
    if not verbose:
        return
    count = lambda rows, key: {s: sum(r[key] == s for r in rows)
                               for s in dict.fromkeys(r[key] for r in rows)}
    print(f"{len(facts)} facts about {len(LIGHTHOUSES)} lighthouses\n"
          f"  sft_train.jsonl: {len(train)} chats {count(train, 'split')}\n"
          f"  sft_test.jsonl:  {len(test)} chats {count(test, 'split')}\n"
          f"  prefs.jsonl:     {len(prefs)} pairs {count(prefs, 'kind')}\n")
    print("the handbook (* = held out, never trained):")
    print(f"  {'lighthouse':14s}" + "".join(f"{a:>12s}" for a in ATTRIBUTES))
    unseen = {(r["entity"], r["attribute"]) for r in test if r["split"] == "unseen_fact"}
    for e in LIGHTHOUSES:
        cells = [f"{facts[(e, a)] + ('*' if (e, a) in unseen else ''):>12s}" for a in ATTRIBUTES]
        print(f"  {e:14s}" + "".join(cells))
    s, m = render_with_mask(train[0]["messages"])
    print("\none training chat as the model sees it (^ = trained on):")
    print("  " + show(s))
    marks = "".join(("^" if k else " ") * len(SPECIAL_NAMES.get(c, c)) for c, k in zip(s, m))
    print("  " + marks)
    lens = [len(render(r["messages"])) for r in train + test]
    print(f"\nlengths: {min(lens)}-{max(lens)} characters per chat")
    for p in (prefs[1], prefs[-1]):
        print(f"a preference pair: prompt {p['prompt'][0]['content']!r}\n"
              f"   chosen   {p['chosen']!r}\n   rejected {p['rejected']!r} ({p['kind']})")


if __name__ == "__main__":
    main()
