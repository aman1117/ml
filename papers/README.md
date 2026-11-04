# Papers: read one core section, implement it, log it

The daily habit: **read one core section of a paper and implement it in code the same
day**. Not skimming abstracts. A paper counts as done only when its key equation or
algorithm runs in this repo, is checked (gradient check, equivalence check or a
reproduced number), and is logged below.

> **How to use this page:** pick the next unchecked row, read only the section named in
> "What to implement", write the code in the module folder listed (or an exercise there),
> run the module's check, then add a line to the log at the bottom.

## Files

| File | What it is |
|------|-----------------|
| `README.md` | This page: the reading list mapped to code, and the implementation log. |

---

## 1. Reading list, mapped to code

"Where" is the folder that implements it (✅ already implemented by the module, so
re-derive it and compare with the code) or the exercise / addition that is your job (🛠).

### Transformers and modern LLM architecture

| Done | Paper | What to implement | Where |
|------|-------|-------------------|-------|
| [ ] | Vaswani et al. (2017), *Attention Is All You Need* | Scaled dot-product and multi-head attention, sinusoidal positions | ✅ `transformer/` |
| [ ] | Xiong et al. (2020), *On Layer Normalization in the Transformer Architecture* | Pre-LN vs post-LN stability | 🛠 `transformer/` exercise 10 |
| [ ] | Sennrich et al. (2016), *Neural Machine Translation of Rare Words with Subword Units* | The BPE merge algorithm | ✅ `tokenizer/` |
| [ ] | Radford et al. (2019), *Language Models are Unsupervised Multitask Learners* (GPT-2) | Byte-level BPE with a pre-tokenization regex | ✅ `tokenizer/` |
| [ ] | Zhang & Sennrich (2019), *Root Mean Square Layer Normalization* | RMSNorm forward and backward | ✅ `llama/` |
| [ ] | Su et al. (2021), *RoFormer: Enhanced Transformer with Rotary Position Embedding* | RoPE and its relative-position property | ✅ `llama/` |
| [ ] | Shazeer (2020), *GLU Variants Improve Transformer* | SwiGLU MLP | ✅ `llama/` |
| [ ] | Shazeer (2019), *Fast Transformer Decoding: One Write-Head is All You Need*; Ainslie et al. (2023), *GQA* | Multi-query / grouped-query attention and KV-cache size | ✅ `llama/` |
| [ ] | Touvron et al. (2023), *LLaMA: Open and Efficient Foundation Language Models* | The full architecture diff vs GPT-2 | ✅ `llama/` |
| [ ] | Fedus et al. (2021), *Switch Transformers* | Top-k routing and the load-balancing loss | ✅ `llama/moe.py` |
| [ ] | Kaplan et al. (2020), *Scaling Laws for Neural Language Models*; Hoffmann et al. (2022), *Training Compute-Optimal Large Language Models* | Fit a loss-vs-compute power law on your own small runs | 🛠 `projects/` project 4 |
| [ ] | Dao et al. (2022), *FlashAttention* | Tiled attention with online softmax (NumPy), checked against plain attention | 🛠 your addition to `transformer/` |
| [ ] | Olsson et al. (2022), *In-context Learning and Induction Heads* | Train a 2-layer model until induction heads form | 🛠 `transformer/` exercise 11 |

### Training at scale

| Done | Paper | What to implement | Where |
|------|-------|-------------------|-------|
| [ ] | Micikevicius et al. (2018), *Mixed Precision Training* | fp16 with loss scaling and fp32 master weights | ✅ `distributed/` |
| [ ] | Kalamkar et al. (2019), *A Study of BFLOAT16 for Deep Learning Training* | bf16 rounding and why it needs no loss scaling | ✅ `distributed/` |
| [ ] | Li et al. (2020), *PyTorch Distributed: Experiences on Accelerating Data Parallel Training* | DDP with gradient all-reduce (bucketing is explained, not implemented) | ✅ `distributed/` |
| [ ] | Rajbhandari et al. (2020), *ZeRO: Memory Optimizations Toward Training Trillion Parameter Models* | ZeRO stages 1/2/3 and their memory formulas | ✅ `distributed/` |
| [ ] | Shoeybi et al. (2019), *Megatron-LM* | Column/row-parallel MLP and head-parallel attention | ✅ `distributed/` |
| [ ] | Huang et al. (2019), *GPipe*; Narayanan et al. (2021), *Efficient Large-Scale Language Model Training on GPU Clusters* | Pipeline schedules and the bubble fraction | ✅ `distributed/` |
| [ ] | Chen et al. (2016), *Training Deep Nets with Sublinear Memory Cost*; Korthikanti et al. (2022), *Reducing Activation Recomputation in Large Transformer Models* | Activation checkpointing and the activation-memory formula | ✅ `distributed/` |

### Fine-tuning and alignment

| Done | Paper | What to implement | Where |
|------|-------|-------------------|-------|
| [ ] | Hu et al. (2021), *LoRA: Low-Rank Adaptation of Large Language Models* | LoRA forward/backward, init, scaling, merge | ✅ `finetuning/` |
| [ ] | Ouyang et al. (2022), *Training language models to follow instructions with human feedback* | The SFT stage with loss masking | ✅ `finetuning/` |
| [ ] | Rafailov et al. (2023), *Direct Preference Optimization* | The DPO loss and its gradient | ✅ `finetuning/` |
| [ ] | Dettmers et al. (2023), *QLoRA* | NF4 quantization of the frozen base + LoRA | 🛠 `finetuning/` exercise |
| [ ] | Zhou et al. (2023), *LIMA: Less Is More for Alignment* | Data-quality ablation on your domain set | 🛠 `projects/` project 2 |

### Evals

| Done | Paper | What to implement | Where |
|------|-------|-------------------|-------|
| [ ] | Chen et al. (2021), *Evaluating Large Language Models Trained on Code* | The unbiased pass@k estimator | ✅ `evals/` |
| [ ] | Miller (2024), *Adding Error Bars to Evals* | Standard errors, clustered SEs, paired comparisons, power | ✅ `evals/` |
| [ ] | Zheng et al. (2023), *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena* | Pairwise judging, position-swap debiasing, agreement | ✅ `evals/` |
| [ ] | Guo et al. (2017), *On Calibration of Modern Neural Networks* | ECE and reliability diagrams | ✅ `evals/` |

### Retrieval and RAG

| Done | Paper | What to implement | Where |
|------|-------|-------------------|-------|
| [ ] | Robertson & Zaragoza (2009), *The Probabilistic Relevance Framework: BM25 and Beyond* | BM25 scoring | ✅ `rag/` |
| [ ] | Karpukhin et al. (2020), *Dense Passage Retrieval* | Bi-encoder with in-batch negatives | ✅ `rag/` |
| [ ] | Oord et al. (2018), *Representation Learning with Contrastive Predictive Coding* | The InfoNCE loss and its gradient | ✅ `rag/` |
| [ ] | Lewis et al. (2020), *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* | Retrieve → generate with citations | ✅ `rag/` |
| [ ] | Jégou et al. (2011), *Product Quantization for Nearest Neighbor Search* | IVF and PQ | ✅ `rag/` |
| [ ] | Malkov & Yashunin (2018), *HNSW* | A small HNSW graph index, recall vs exact search | 🛠 `rag/` exercise |
| [ ] | Khattab & Zaharia (2020), *ColBERT* | Late-interaction (MaxSim) reranking as a new stage in `retrieval_eval.py` | 🛠 your addition to `rag/` |
| [ ] | Liu et al. (2023), *Lost in the Middle* | Measure answer quality vs position of the gold chunk | 🛠 `rag/` exercise |

### Agents

| Done | Paper | What to implement | Where |
|------|-------|-------------------|-------|
| [ ] | Yao et al. (2022), *ReAct: Synergizing Reasoning and Acting in Language Models* | The thought → action → observation loop | ✅ `agents/` |
| [ ] | Yao et al. (2024), *τ-bench* | pass^k reliability metric | ✅ `agents/` |
| [ ] | Greshake et al. (2023), *Not what you've signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection* | An injection attack through tool output, and the defense | ✅ `agents/` |
| [ ] | Yang et al. (2024), *SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering* | Tool design (ACI) ablation on your agent | 🛠 `projects/` project 3 |

### Serving

| Done | Paper | What to implement | Where |
|------|-------|-------------------|-------|
| [ ] | Pope et al. (2022), *Efficiently Scaling Transformer Inference* | Prefill vs decode cost model, memory-bandwidth bound decode | ✅ `serving/` |
| [ ] | Yu et al. (2022), *Orca* | Continuous (iteration-level) batching | ✅ `serving/` |
| [ ] | Kwon et al. (2023), *Efficient Memory Management for LLM Serving with PagedAttention* | Paged KV-cache block allocator | ✅ `serving/` |
| [ ] | Leviathan et al. (2023), *Fast Inference from Transformers via Speculative Decoding* | Draft-and-verify with exact rejection sampling | ✅ `serving/` |
| [ ] | Dettmers et al. (2022), *LLM.int8()* | Absmax int8, the outlier problem | ✅ `serving/` |
| [ ] | Frantar et al. (2022), *GPTQ* | Second-order weight quantization of one layer, compared with absmax int4 | 🛠 your addition to `serving/quantize_numpy.py` |

## 2. Implementation log

One line per session. Keep it honest: what you implemented, how you checked it, and what
confused you.

| Date | Paper / section | Code (file:function) | Check (how verified) | Notes / open questions |
|------|-----------------|----------------------|----------------------|------------------------|
| YYYY-MM-DD | | | | |
