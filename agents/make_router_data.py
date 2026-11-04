"""
Generate the synthetic training/test data for the tool router (deterministic).

The router answers ONE question: "which tool, if any, should handle this request
(or this clause of a request)?". Labels:

    calculator, search_docs, read_file, unit_convert, kv_memory, no_tool

Each label has hand-written templates with slots ({n}, {topic}, {path}, ...).
To measure generalization honestly, the TEST set uses templates the training set
never saw (held-out phrasings) and partly unseen slot values; a router that only
memorized the training sentences would do badly on it.

Writes (committed, so everyone trains on identical data):
    agents/router_train.jsonl   {"text": ..., "label": ...} per line
    agents/router_test.jsonl

Run:
    python agents/make_router_data.py
"""
import json
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
LABELS = ["calculator", "search_docs", "read_file", "unit_convert", "kv_memory", "no_tool"]

# Slot values. The second list of each pair appears ONLY in the test set.
TOPICS = (["the KV cache", "LayerNorm backward", "vanishing gradients", "im2col",
           "the forget gate bias", "truncated BPTT", "dropout", "padding and stride",
           "the receptive field", "GRU vs LSTM", "the causal mask", "weight tying",
           "scaling laws", "softmax cross-entropy", "data augmentation", "beam search",
           "Bahdanau attention", "max pooling", "gradient clipping", "warmup and cosine decay",
           "multi-head attention", "positional embeddings", "the residual stream",
           "batch normalization", "L2 regularization", "the bottleneck problem"],
          ["induction heads", "the GELU activation", "cross-attention", "the attention backward pass",
           "translation equivariance", "the parameter count of the mini-GPT", "sampling temperature",
           "the char-RNN loss", "the LSTM cell state", "AdamW"])
MODULES = ["mlp", "cnn", "rnn", "lstm", "attention", "transformer"]
PATHS = (["mlp/README.md", "transformer/gradient_check.py", "rnn/input.txt", "cnn/README.md",
          "lstm/lstm_numpy.py", "attention/tasks.py", "transformer/generate.py",
          "agents/fixtures/meeting_notes.txt", "mlp/mlp_numpy.py", "README.md"],
         ["transformer/README.md", "attention/README.md", "agents/fixtures/release_notes.txt",
          "rnn/rnn_numpy.py", "requirements.txt"])
UNITS = ([("km", "miles"), ("miles", "km"), ("kg", "pounds"), ("pounds", "kg"), ("C", "F"),
          ("F", "C"), ("GB", "MB"), ("GiB", "GB"), ("hours", "minutes"), ("feet", "meters"),
          ("inches", "cm"), ("ounces", "grams"), ("days", "hours"), ("MB", "KB")],
         [("meters", "feet"), ("grams", "ounces"), ("TB", "GB"), ("minutes", "seconds"),
          ("kelvin", "celsius"), ("yards", "meters")])
MEM = ([("favorite optimizer", "AdamW"), ("GPU budget", "8 A100s"), ("preferred batch size", "64"),
        ("project name", "tiny-gpt"), ("deadline", "Friday"), ("team lead", "Priya"),
        ("default learning rate", "3e-4"), ("dataset path", "/data/wiki"), ("favorite paper", "ReAct")],
       [("cluster name", "atlas"), ("seed", "1337"), ("manager", "Sam"), ("eval set", "tau-bench")])
OPS = [("times", "*"), ("multiplied by", "*"), ("plus", "+"), ("minus", "-"), ("divided by", "/")]

T = {  # (train templates, held-out test templates)
    "calculator": ([
        "What is {a} {op} {b}?", "Compute {a} {op} {b}", "calculate {a}*{b}+{c}",
        "{a} + {b} * {c} = ?", "what's ({a} - {b}) / {c}", "Evaluate {a} ** 2",
        "How much is {a} {op} {b}", "sqrt of {a}", "What is {p}% of {a}?",
        "multiply it by {b}", "then divide that by {b}", "add {b} to the result",
        "square it", "then multiply the result by {c}", "subtract {b} from it",
        "If I have {a} GPUs and each costs {b} dollars, what is the total cost?",
        "Sum of {a}, {b} and {c}?", "average of {a} and {b}", "{a} times {b}",
        "what is {a} to the power of 3", "work out {a} divided by {b} for me",
        "how many tokens is {a} sequences of {b} tokens each", "double it",
        "what is half of {a}", "then add {c}", "{a}/{b}"],
        ["Quick math: {a} {op} {b}", "I need the product of {a} and {b}",
         "take that number and multiply by {c}", "Can you crunch ({a} + {b}) * {c}?",
         "what do you get if you divide {a} by {b}", "then halve it",
         "Total for {a} items at {b} each?", "log of {a}"]),
    "search_docs": ([
        "What does the {module} README say about {topic}?", "Search the docs for {topic}",
        "Explain {topic} according to the docs", "Find where the repo explains {topic}",
        "Which module covers {topic}?", "Look up {topic} in the documentation",
        "What is {topic}? check the readmes", "How does the {module} module describe {topic}?",
        "What paper introduced {topic}?", "find the section on {topic}",
        "search for {topic}", "According to the docs, why does {topic} matter?",
        "In the {module} notes, what is said about {topic}", "docs on {topic}",
        "What loss did the {module} model reach according to its README?",
        "How many parameters does the {module} model have according to the docs?",
        "What accuracy does the {module} README report?", "what is the {topic} formula in the docs",
        "Look up the parameter count in the docs", "find {topic} in the readmes"],
        ["Is there anything in the notes about {topic}?", "Where is {topic} discussed?",
         "Grab the explanation of {topic} from the readmes",
         "What did the {module} chapter conclude about {topic}?",
         "Search the knowledge base: {topic}", "according to the transformer docs, what is {topic}"]),
    "read_file": ([
        "Read {path}", "Open {path} and summarize it", "Show me the contents of {path}",
        "How many lines does {path} have?", "What is the title of {path}?",
        "What is the first heading in {path}?", "Read the file {path}",
        "print {path}", "cat {path}", "Does {path} mention {topic}?",
        "Open the file at {path}", "What's in {path}?", "load {path}",
        "read {path} and tell me what it is about", "how long is {path} in characters?",
        "check the file {path}"],
        ["Peek into {path}", "Could you go through {path}?",
         "What does the file {path} contain?", "Display {path} for me"]),
    "unit_convert": ([
        "Convert {x} {u1} to {u2}", "How many {u2} is {x} {u1}?", "{x} {u1} in {u2}",
        "What is {x} {u1} in {u2}?", "convert that to {u2}", "change {x} {u1} into {u2}",
        "{x} {u1} to {u2} please", "express {x} {u1} in {u2}", "then convert it to {u2}",
        "How many {u2} are in {x} {u1}", "turn {x} {u1} into {u2}",
        "is {x} {u1} more than 10 {u2}? convert it"],
        ["What's {x} {u1} expressed as {u2}?", "{u1} to {u2}: {x}",
         "I have {x} {u1}, how much is that in {u2}?", "and in {u2}?"]),
    "kv_memory": ([
        "Remember that my {k} is {v}", "Please remember my {k}: {v}", "What is my {k}?",
        "Recall my {k}", "Do you remember my {k}?", "Save {v} as my {k}",
        "Store my {k} = {v}", "Forget my {k}", "What do you have in memory?",
        "remember it as my {k}", "then save that as my {k}", "What did I tell you my {k} was?",
        "Note that my {k} is {v}", "list what you remember about me", "remind me of my {k}",
        "update my {k} to {v}"],
        ["Keep in mind: my {k} is {v}", "Don't forget that my {k} is {v}",
         "What was my {k} again?", "Make a note: {k} = {v}", "store the result as my {k}"]),
    "no_tool": ([
        "Hi!", "Hello there", "Thanks, that's all", "Thank you!", "Who are you?",
        "What can you do?", "What tools do you have?", "What will the weather be in Paris tomorrow?",
        "What is the stock price of NVDA right now?", "What's my home address?",
        "Write a haiku about gradients", "Tell me a joke", "Good morning",
        "Repeat the word {w} three times", "Uppercase the word {w}", "Reverse the word {w}",
        "How many words are in the sentence '{sent}'?", "Say hello to {name}",
        "What is your favorite color?", "Who will win the next election?",
        "Book me a flight to Tokyo", "Send an email to my boss", "Are you conscious?",
        "What time is it in Tokyo right now?", "Order a pizza for me", "ok", "cool, bye"],
        ["Hey, how's it going?", "Much appreciated!", "Predict next week's bitcoin price",
         "Spell {w} backwards", "Can you introduce yourself?", "What's the news today?",
         "Shout the word {w}", "Please transfer $100 to Bob"]),
}
WORDS = ["agent", "banana", "tensor", "router", "gradient", "token", "loop"]
SENTS = ["agents call tools in a loop", "the model reads the observation",
         "every step is traced", "retries need good error messages"]
NAMES = ["Ada", "Alan", "Grace", "Yann", "Fei-Fei"]


def fill(template, rng, test):
    pick = lambda pair: rng.choice(pair[0] + pair[1] if test else pair[0])  # noqa: E731
    op_word, _ = rng.choice(OPS)
    u1, u2 = pick(UNITS)
    k, v = pick(MEM)
    return template.format(
        a=rng.randint(2, 999), b=rng.randint(2, 99), c=rng.randint(2, 50), p=rng.choice([5, 10, 15, 20, 25]),
        op=op_word, topic=pick(TOPICS), module=rng.choice(MODULES), path=pick(PATHS),
        x=rng.choice([1, 2, 3.5, 5, 10, 12, 42, 100, 250]), u1=u1, u2=u2, k=k, v=v,
        w=rng.choice(WORDS), sent=rng.choice(SENTS), name=rng.choice(NAMES))


def casing(s, rng):
    """Real users don't capitalize consistently."""
    r = rng.random()
    if r < 0.15:
        return s.lower()
    if r < 0.2:
        return s.rstrip("?.!")
    return s


def make(n_per_label, test, seed):
    rng = random.Random(seed)
    rows = []
    for label in LABELS:
        train_t, test_t = T[label]
        # Test: half held-out templates, half seen templates with (partly) unseen slots.
        pool = (test_t + train_t) if test else train_t
        for i in range(n_per_label):
            tpl = test_t[i % len(test_t)] if (test and i % 2 == 0) else rng.choice(pool if not test else train_t)
            rows.append({"text": casing(fill(tpl, rng, test), rng), "label": label})
    rng.shuffle(rows)
    return rows


def main():
    train = make(300, test=False, seed=0)
    test = make(100, test=True, seed=1)
    for name, rows in (("router_train.jsonl", train), ("router_test.jsonl", test)):
        with open(os.path.join(HERE, name), "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"wrote agents/{name}: {len(rows)} examples")
    for r in test[:6]:
        print(f"  {r['label']:13s} {r['text']}")


if __name__ == "__main__":
    main()
