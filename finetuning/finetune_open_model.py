"""
OPTIONAL: LoRA fine-tuning of a real open model on the lighthouse chats, with
OUR LoRALinear, OUR masked loss and OUR training/decoding loops.

Hugging Face `transformers` is used for exactly three things: download the
weights into an nn.Module, the tokenizer, and the model's chat template. No
Trainer, no PEFT, no TRL, no model.generate():

  1. load the model (default HuggingFaceTB/SmolLM2-135M-Instruct) in float32 on CPU,
  2. render every chat with tokenizer.apply_chat_template(); the loss mask is 1
     only on the tokens after the generation prompt (the answer + <|im_end|>),
  3. evaluate BEFORE: greedy decoding with our own KV-cache loop, exact match and
     format adherence on sft_test.jsonl (same splits as the char-level model),
  4. inject finetune_pytorch.LoRALinear into q_proj, k_proj, v_proj, o_proj
     (every other weight frozen), train with our loop,
  5. evaluate AFTER, save ONLY the adapter weights (finetuning/open_lora.pt),
  6. merge the adapters into the base weights, check the logits did not change,
     and evaluate the merged model.

`transformers` is NOT a dependency of this repo. Without it, this script prints
how to get it and exits cleanly. Do not install it into the main .venv; use a
separate environment, e.g.:
    python -m venv ~/ft_env && ~/ft_env/bin/pip install torch transformers numpy
    ~/ft_env/bin/python finetuning/finetune_open_model.py

Run (CPU, ~3 min for the 135M model, plus the one-time ~270 MB download):
    python finetuning/finetune_open_model.py
    python finetuning/finetune_open_model.py --model Qwen/Qwen2.5-0.5B-Instruct --steps 100
    python finetuning/finetune_open_model.py --targets q_proj,v_proj --rank 4
"""
import argparse
import math
import os
import sys
import time
import warnings

import torch

import domain_data as dd

warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "open_lora.pt")


def encode_chat(tok, messages):
    """(token ids, loss mask) using the model's OWN chat template. The mask is 1 on
    the assistant's answer and the end-of-turn token(s) after it."""
    prompt = tok.apply_chat_template(messages[:1], tokenize=False, add_generation_prompt=True)
    full = tok.apply_chat_template(messages, tokenize=False)
    assert full.startswith(prompt), "chat template is not prefix-consistent"
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    f_ids = tok(full, add_special_tokens=False)["input_ids"]
    assert f_ids[:len(p_ids)] == p_ids, "prompt tokenizes differently inside the full chat"
    # Train on the answer up to and including the first end-of-turn token; whatever the
    # template appends after it (a newline) is formatting, not something to generate.
    stop = set(stop_ids(tok))
    end = next((i + 1 for i in range(len(p_ids), len(f_ids)) if f_ids[i] in stop), len(f_ids))
    ids = f_ids[:end]
    return ids, [0] * len(p_ids) + [1] * (end - len(p_ids))


def stop_ids(tok):
    ids = {tok.eos_token_id}
    for t in ("<|im_end|>", "<|eot_id|>", "<|end|>"):
        i = tok.convert_tokens_to_ids(t)
        if isinstance(i, int) and i != tok.unk_token_id and i is not None and i >= 0:
            ids.add(i)
    return [i for i in ids if i is not None]


def batch_tensors(examples, pad_id):
    """Right-pad; inputs = ids[:-1], targets = ids[1:], mask on targets."""
    L = max(len(i) for i, _ in examples)
    ids = torch.full((len(examples), L), pad_id, dtype=torch.long)
    att = torch.zeros((len(examples), L), dtype=torch.long)
    msk = torch.zeros((len(examples), L))
    for b, (i, m) in enumerate(examples):
        ids[b, :len(i)] = torch.tensor(i)
        att[b, :len(i)] = 1
        msk[b, :len(i)] = torch.tensor(m, dtype=torch.float)
    return ids[:, :-1], att[:, :-1], ids[:, 1:], msk[:, 1:]


@torch.no_grad()
def greedy(model, tok, messages, max_new=24):
    """Our own decode loop: one prefill pass, then one token at a time with the KV cache."""
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = torch.tensor([tok(prompt, add_special_tokens=False)["input_ids"]])
    out = model(input_ids=ids, use_cache=True)
    past, logits = out.past_key_values, out.logits[:, -1]
    stops, gen = set(stop_ids(tok)), []
    for _ in range(max_new):
        nxt = int(logits.argmax(-1))
        if nxt in stops:
            return tok.decode(gen, skip_special_tokens=True).strip(), True
        gen.append(nxt)
        out = model(input_ids=torch.tensor([[nxt]]), past_key_values=past, use_cache=True)
        past, logits = out.past_key_values, out.logits[:, -1]
    return tok.decode(gen, skip_special_tokens=True).strip(), False


def evaluate(model, tok, rows, label):
    model.eval()
    t0 = time.time()
    answers = [greedy(model, tok, r["messages"][:1]) for r in rows]
    by = {}
    for r, (a, stopped) in zip(rows, answers):
        d = by.setdefault(r["split"], [0, 0, 0])
        d[0] += 1
        d[1] += int(stopped and a == r["messages"][1]["content"])
        d[2] += int(stopped and dd.format_ok(r, a))
    print(f"  {label:22s}" + "  ".join(f"{k} {100 * e / n:5.1f}%/{100 * f / n:5.1f}%"
                                       for k, (n, e, f) in by.items())
          + f"   ({time.time() - t0:.0f} s)")
    return answers


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    ap.add_argument("--targets", default="q_proj,k_proj,v_proj,o_proj")
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=16.0)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--eval-train", type=int, default=24, help="train chats to evaluate on")
    args = ap.parse_args()

    try:
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("`transformers` is not installed, so this optional script is skipped.\n"
              "It is not a dependency of this repo (everything else runs without it). To try it,\n"
              "use a SEPARATE environment, e.g.:\n"
              "    python -m venv ~/ft_env && ~/ft_env/bin/pip install torch transformers numpy\n"
              "    ~/ft_env/bin/python finetuning/finetune_open_model.py")
        return 0
    from finetune_pytorch import inject_lora, lora_modules, masked_ce, merge_and_unload

    torch.manual_seed(0)
    torch.set_num_threads(args.threads)
    t0 = time.time()
    print(f"transformers {transformers.__version__}, torch {torch.__version__}: loading "
          f"{args.model} (weights + tokenizer + chat template only)...")
    try:
        tok = AutoTokenizer.from_pretrained(args.model)
        major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
        kw = "dtype" if (major, minor) >= (4, 56) else "torch_dtype"  # renamed in 4.56
        model = AutoModelForCausalLM.from_pretrained(args.model, **{kw: torch.float32})
    except Exception as e:  # no network, unknown model, ...
        print(f"could not load {args.model}: {type(e).__name__}: {e}\nskipping.")
        return 0
    n_all = sum(p.numel() for p in model.parameters())
    print(f"  {n_all / 1e6:.1f}M params, vocab {len(tok)}, loaded in {time.time() - t0:.0f} s")

    train_rows, test_rows, _ = dd.load_all()
    example = tok.apply_chat_template(train_rows[0]["messages"], tokenize=False)
    ids, mask = encode_chat(tok, train_rows[0]["messages"])
    print(f"\nthe chat template renders a training chat as:\n{example}")
    print(f"-> {len(ids)} tokens, loss on the last {sum(mask)}: "
          f"{tok.decode([i for i, m in zip(ids, mask) if m])!r}\n")

    eval_rows = train_rows[:args.eval_train] + test_rows
    print("exact / format per split (greedy decoding):")
    evaluate(model, tok, eval_rows, "before fine-tuning")

    targets = set(args.targets.split(","))
    n = inject_lora(model, targets, args.rank, args.alpha)
    n_lora = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nLoRA r={args.rank} alpha={args.alpha:g} injected into {n} Linear layers "
          f"({','.join(sorted(targets))}): {n_lora:,} trainable params = "
          f"{100 * n_lora / n_all:.2f}% of {n_all:,}")

    data = [encode_chat(tok, r["messages"]) for r in train_rows]
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    g = torch.Generator().manual_seed(0)
    model.train()
    t0 = time.time()
    perm = []
    for step in range(args.steps):
        if len(perm) < args.batch_size:
            perm += torch.randperm(len(data), generator=g).tolist()
        b, perm = perm[:args.batch_size], perm[args.batch_size:]
        x, att, y, m = batch_tensors([data[i] for i in b], pad)
        warm = max(1, args.steps // 10)
        lr = args.lr * min(1.0, (step + 1) / warm) * (
            0.5 * (1 + math.cos(math.pi * max(0, step - warm) / max(1, args.steps - warm))))
        for grp in opt.param_groups:
            grp["lr"] = lr
        loss = masked_ce(model(input_ids=x, attention_mask=att).logits, y, m)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if step % max(1, args.steps // 6) == 0 or step == args.steps - 1:
            print(f"  step {step:3d} | masked loss {loss.item():.4f} | lr {lr:.1e} | "
                  f"{time.time() - t0:5.1f} s")

    print()
    evaluate(model, tok, eval_rows, "after LoRA SFT")

    state = {k: v.detach().clone() for k, v in model.named_parameters() if v.requires_grad}
    torch.save(dict(model=args.model, r=args.rank, alpha=args.alpha, targets=sorted(targets),
                    state=state), OUT)
    print(f"\nsaved ONLY the adapters: {os.path.relpath(OUT)} "
          f"({os.path.getsize(OUT) / 1e6:.1f} MB, vs {4 * n_all / 1e6:.0f} MB for the "
          f"float32 model)")

    probe = torch.tensor([data[0][0]])
    model.eval()
    with torch.no_grad():
        before = model(input_ids=probe).logits
        merge_and_unload(model)
        after = model(input_ids=probe).logits
    assert not lora_modules(model)
    print(f"merged the adapters into the base weights: max |logit diff| = "
          f"{(after - before).abs().max().item():.1e}")
    evaluate(model, tok, eval_rows, "merged model")
    return 0


if __name__ == "__main__":
    sys.exit(main())
