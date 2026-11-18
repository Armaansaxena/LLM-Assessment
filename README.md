# LLM Engineering Assessment

Fine-tuning with LoRA, retrieval from scratch, and local model serving. Each task has its own folder, Colab notebook and saved results.

| Folder | Task | Notebook |
|---|---|---|
| [`task1-lora-finetuning/`](task1-lora-finetuning) | LoRA fine-tune of Qwen2.5-0.5B-Instruct (request -> JSON) | `Task1_LoRA_finetune.ipynb` |
| [`task2-rag/`](task2-rag) | RAG from scratch over the NIST AI RMF 1.0 (48 pages) | `Task2_RAG_from_scratch.ipynb` |
| [`task3-local-inference/`](task3-local-inference) | Ollama, fp16 vs q4_K_M, serving plan | `Task3_Ollama_quantisation.ipynb` |

**To reproduce:** open a notebook in Google Colab (Runtime -> T4 GPU for Tasks 1 and 3), then Runtime -> Run all. Task 2 needs an LLM key only for the optional final-answer cell.

---

# Task 1: Fine-tuning with LoRA

**Model:** Qwen2.5-0.5B-Instruct (0.5B parameters). **Task:** plain-English reminder, meeting or call request -> JSON with keys `type, title, day, time, priority`.

**Data:** 800 synthetic training examples. Held-out test set of 100: 50 in-distribution ("easy") and 50 "hard" examples using phrasing never seen in training ("noon", "midnight", "half past 4 pm", "quarter to 6 pm", "ASAP", "if you get a chance").

**Metric:** exact match (all five fields correct, no extra keys). Also reported: valid-JSON rate and per-field accuracy. Greedy decoding, same system prompt for both models, base model evaluated with the adapter disabled.

## Base vs fine-tuned (`task1-lora-finetuning/results/task1_results.csv`)

| Model | Test set | Valid JSON % | Exact match % | type | title | day | time | priority |
|---|---|---|---|---|---|---|---|---|
| Base | easy | 92 | 0 | 80 | 2 | 48 | 82 | 48 |
| Base | hard | 86 | 0 | 76 | 4 | 42 | 12 | 20 |
| Fine-tuned | easy | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| Fine-tuned | hard | 100 | 42 | 100 | 100 | 100 | 56 | 80 |

The base model almost never produced the exact schema (0% exact match; titles only 2-4% correct because it paraphrases instead of copying). Fine-tuning fixed format and field extraction on phrasing seen in training. Read the easy-set 100% with care: train and test share the same templates, so it shows the format was learned, not that the model generalises. The hard set is the honest generalisation test (42% exact match).

## LoRA settings and why

| Setting | Value | Why |
|---|---|---|
| Rank r | 16 | The task is narrow with a fixed schema, so low rank is enough; 8-16 is standard. |
| Alpha | 32 | Scale alpha/r = 2, a common default that works with lr 2e-4 without tuning. |
| Target modules | q, k, v, o, gate, up, down proj | Adapting MLP as well as attention helps learn output format and mappings, for a small parameter increase. |
| Dropout | 0.05 | Light regularisation; the data is templated and repetitive. |
| Learning rate | 2e-4 (cosine, 5% warmup) | LoRA tolerates higher rates than full fine-tuning because only small adapters train. |
| Epochs | 2 | 100 optimiser steps at effective batch 16 (8 x 2 accumulation). Loss plateaus quickly; more epochs mostly memorise templates. |

## Three failures of the fine-tuned model

29 of 100 test examples failed, all on the hard set (`results/task1_failures.json`). Type, title and day were never wrong. Breakdown: 13 "quarter to" time errors, 6 "half past" time errors, 3 "midnight" errors, and 10 priority errors (3 overlap with "quarter to").

1. **Input:** "Set up the design sync on friday at quarter to 6 pm". **Predicted** `18:00`, **gold** `17:45`. The model rounds to the next hour. "Quarter to X" means (X-1):45, which needs arithmetic it never saw; it learned the surface rule "N pm -> N+12:00". All 13 "quarter to" examples failed on time, and the 6 "half past" failures dropped the minutes (e.g. `18:30` -> `18:00`).
2. **Input:** "Call the support team on monday at midnight". **Predicted** `24:00`, **gold** `00:00`. Training times only covered 06:00-22:45, so `00:xx` never appeared as an output, and the model fell back on the pretrained habit of writing midnight as 24:00. "Noon" worked because `12:00` occurs in training, which points to missing data coverage, not a general inability.
3. **Input:** "Call Neha on friday at midnight, if you get a chance". **Predicted** priority `normal`, **gold** `low`. "If you get a chance", "only if there's time" and "drop everything for this" were absent from training, which used a small fixed vocabulary. The model defaults to the majority class for unseen paraphrases. All 10 priority errors involved unseen phrases.

**Fix:** add `00:00`, "half past", "quarter to" and a wider set of priority paraphrases to the training data (for example LLM-paraphrased requests). A 0.5B model has limited arithmetic, so more varied data helps more than more epochs.

## GPU memory to fine-tune a 7B model (rough)

**(a) Full fine-tuning (mixed precision, Adam).** Per parameter: 2 B (bf16 weights) + 2 B (bf16 gradients) + 4 B (fp32 master weights) + 8 B (Adam m and v in fp32) = 16 B. 7B x 16 B = **112 GB**, plus about 5-20 GB of activations, so **about 120-130 GB** (at least 2 x 80 GB GPUs, or sharding such as ZeRO/FSDP).

**(b) LoRA (r=16, all linear layers).**
- Frozen bf16 base: 7B x 2 B = **14 GB**.
- Adapter parameters (Llama-7B shapes, r x (d_in + d_out) per matrix): attention 4 x 16 x (4096 + 4096) = 524,288; MLP 3 x 16 x (4096 + 11008) = 724,992; per layer 1,249,280; over 32 layers **about 40M parameters**.
- Adapter weights, gradients and optimiser state: 40M x 16 B = **0.64 GB**.
- Activations (gradient checkpointing, small batch): about 2-6 GB.
- Total: **about 17-21 GB**, which fits a 24 GB GPU.

**(c) QLoRA.**
- Base in 4-bit NF4: 7B x 0.5 B = **3.5 GB**, plus about 0.4 GB of quantisation constants.
- Adapters and optimiser: 0.64 GB.
- Activations: about 2-5 GB.
- Total: **about 7-10 GB**, which fits a free Colab T4.

---

# Task 2: Retrieval (RAG) from scratch

**Document:** NIST AI Risk Management Framework 1.0 (NIST AI 100-1), 48 pages, public domain.

**Stack (no LangChain or LlamaIndex):** `pypdf` for text, word-window chunking with 20% overlap, `BAAI/bge-small-en-v1.5` embeddings, FAISS `IndexFlatIP` for dense search, my own NumPy BM25, reciprocal rank fusion for hybrid search, and an optional cross-encoder re-ranker. An LLM is used only for the final answer.

**Evaluation:** a retrieval counts as a hit when one of the top-k chunks contains the gold phrase from the source passage. Chunk sizes compared: 100 words (about 130 tokens) and 300 words (about 400 tokens); 300 words is near the embedding model's 512-token limit, so larger chunks would be truncated.

## 15 test questions with source passages (`task2-rag/questions.json`)

| # | Question | Source passage (gold phrase) | Printed page |
|---|---|---|---|
| 1 | What are the four functions that make up the AI RMF Core? | "the Core is composed of four functions: GOVERN, MAP, MEASURE, and MANAGE" | 20 |
| 2 | How does the framework define risk? | "risk refers to the composite measure of an event" | 4 |
| 3 | Does the framework tell organizations how much risk they should be willing to accept? | "While the AI RMF can be used to prioritize risk, it does not prescribe risk tolerance" | 7 |
| 4 | How does the document distinguish resilience from security? | "resilience is the ability to return to normal function after an unexpected adverse event" | 15 |
| 5 | Which categories of bias does NIST say need to be managed? | "three major categories of AI bias to be considered and managed: systemic, computational and statistical" | 18 |
| 6 | What is the Playbook that accompanies the framework for? | "the NIST AI RMF Playbook, is available to help organizations navigate the AI RMF" | 21 |
| 7 | What should happen if an AI system poses unacceptable risk? | "development and deployment should cease in a safe manner until risks" | 8 |
| 8 | Which function sets up the context for framing an AI system's risks? | "The MAP function establishes the context to frame risks related to an AI system" | 24 |
| 9 | Once governance is in place, which function do most users tackle first? | "After instituting the outcomes in GOVERN, most users of the AI RMF would start with the MAP function" | 21 |
| 10 | What can an organization do about a high-priority AI risk? | "Risk response options can include mitigating, transferring, avoiding, or accepting" | 32 |
| 11 | Is it a good goal to remove all negative risk from an AI system? | "Attempting to eliminate negative risk entirely can be counterproductive in practice" | 7 |
| 12 | Which testing and validation processes does NIST's modified lifecycle figure emphasise? | "(TEVV) processes throughout an AI lifecycle" | 9 |
| 13 | Why might AI systems need maintenance more often than traditional software? | "AI systems may require more frequent maintenance and triggers for conducting corrective maintenance due to data, model, or concept drift" | 38 |
| 14 | What does a Target Profile describe? | "A Target Profile indicates the outcomes needed to achieve the desired or target AI risk management goals" | 33 |
| 15 | Which GOVERN subcategory requires keeping a list of AI systems? | "Mechanisms are in place to inventory AI systems and are resourced according to organizational risk priorities" | 23 |

## Retrieval accuracy

_Fill these in from the notebook output (section 4)._

| Chunk size | Method | recall@1 | recall@3 | recall@5 |
|---|---|---|---|---|
| 100 words | dense | | | |
| 300 words | dense | | | |
| 100 words | hybrid (BM25 + dense) | | | |
| 300 words | hybrid (BM25 + dense) | | | |

**Which chunk size worked better and why:** _(state which won on these numbers. Typical reasoning: larger chunks contain the answer more often, so recall rises, but the embedding is a blurrier average of several topics and more tokens go to the LLM; small chunks are precise but can split an answer from its context.)_

## Two retrieval failures and what would fix them

1. _Question, what was retrieved, why it failed, fix (e.g. keyword-heavy question such as the GOVERN subcategory one -> BM25/hybrid)._
2. _Question, what was retrieved, why it failed, fix (e.g. paraphrased question -> query rewriting, re-ranker, different chunk size)._

## In my own words

**What BM25 adds to embedding search.** Embeddings match meaning but can miss exact tokens such as rare names, acronyms, IDs and section numbers (for example "GOVERN 1.6"). BM25 scores chunks by keyword overlap, giving more weight to rare terms and less to repeated terms and long chunks. Fusing the two rankings (hybrid search) covers both paraphrased and exact-match questions.

**What a re-ranker does.** The first-stage retriever embeds the query and each chunk separately, which is fast but coarse. A re-ranker is a cross-encoder that reads the query and one chunk together and outputs a relevance score. It is slower but more accurate, so it is applied only to a small candidate set (the top 20-50) and keeps the best 3-5 for the LLM.

---

# Task 3: Running a model locally and optimisation

**Model:** `qwen2.5:1.5b-instruct` served with Ollama on a Google Colab GPU (T4). **Settings compared:** fp16 vs `q4_K_M` (4-bit). Greedy decoding (temperature 0, seed 0), `num_predict` 200, 10 prompts. Raw data in `task3-local-inference/results/`.

## Tokens per second and memory

| Setting | Mean tok/s | Min - Max tok/s | Ollama model memory (GB) | `nvidia-smi` used (MB) |
|---|---|---|---|---|
| fp16 | 60.7 | 59.0 - 62.1 | 3.28 | 3263 |
| q4_K_M | 128.0 | 121.6 - 132.6 | 1.17 | 1243 |

4-bit decoding was **2.1x faster** and used **2.8x less memory** (2.6x less by `nvidia-smi`). Sanity check: 1.5B parameters x 2 B = about 3.1 GB of fp16 weights, which matches the measured 3.28 GB. Memory fell by less than the theoretical 4x because `q4_K_M` is a mixed scheme (some tensors kept at higher precision, about 5 bits per weight on average) and the KV cache and compute buffers are not 4-bit. Speed rose by less than the memory saving because decoding is memory-bandwidth bound (reading fewer weight bytes per token helps), but dequantisation and fixed overheads take some of the gain back.

## Output quality on the same 10 prompts (`results/task3_quality.csv`)

Both settings were compared side by side on every prompt. Outputs were capped at 200 tokens, so long answers (3, 4, 10) are cut off. I compare outputs rather than give numeric scores, since 10 prompts is too few for a meaningful average.

| # | Prompt | Observation | Better |
|---|---|---|---|
| 1 | Capital of Australia | Both correct. fp16 adds a founding date ("1908") that is imprecise (the site was chosen in 1908, the city founded in 1913). | Tie |
| 2 | Hash map in 4 sentences | Both break the 4-sentence limit (fp16 five, q4 seven). | fp16, slightly |
| 3 | Train journey 14:20 to 17:05 | Both reach the correct 2 h 45 min, then ramble ("borrow an hour", "convert excess hours") until cut off. | Tie |
| 4 | Iterative Fibonacci with docstring | Near-identical, correct code and docstring. | Tie |
| 5 | Extract name, city, age as JSON only | Both wrap the JSON in code fences despite "JSON only". fp16 adds an invented `"year": 2023` and a nested object; q4 is simpler with no invented data. | q4 |
| 6 | Translate to French | Identical outputs; both slightly ungrammatical ("avant le apres-midi de vendredi"). | Tie |
| 7 | One-sentence summary | q4 says "while maintaining accuracy", which contradicts the source ("a small cost in accuracy"); fp16 is faithful. | fp16 |
| 8 | Four-line poem without the letter z | Both four lines and no "z". q4 repeats "soothing" three times. | fp16, slightly |
| 9 | Syllogism (bloops, razzies, lazzies) | Both correct ("yes"); fp16 explains more fully. | Tie |
| 10 | 3 pros and 3 cons as a bulleted list | Both give 3 + 3 but as numbered lists, not bullets; q4's last item is cut off at the token limit. | fp16, slightly |

**Summary:** quantisation did not cause a clear correctness failure on this sample. The differences were small and mostly stylistic or about faithfulness (prompts 7 and 8 favour fp16; prompt 5 favours q4). With only 10 prompts, one greedy run each, this is an observation, not a statistically reliable ranking. For a 1.5B model the 2.1x speedup and 2.8x memory saving look well worth the small quality risk.

## What the KV cache is, and why quantisation saves memory (in my own words)

**KV cache.** When a transformer generates a token, attention needs the keys and values of every earlier token. Recomputing them for each new token would be wasteful, so the model stores them in the KV cache and only computes keys and values for the newest token. The price is memory that grows linearly with context length and with the number of concurrent requests. Per token it costs 2 x layers x KV-heads x head-dim x bytes. For Qwen2.5-1.5B (28 layers, 2 KV heads, head dim 128) in fp16 that is 2 x 28 x 2 x 128 x 2 B = about 28 KB per token (about 0.11 GB for a 4k context). For Llama-3.1-8B (32 layers, 8 KV heads, head dim 128) it is about 131 KB per token.

**Why quantisation saves memory.** A model's weights are mostly large tables of numbers. Storing them with fewer bits (16-bit to 4-bit) shrinks them by roughly 4x, because each number takes a quarter of the space; small groups of weights share a scale factor so the 4-bit values can be mapped back to approximate real values with little accuracy loss. It also speeds up decoding, since decoding is limited by how fast weights can be read from memory.

## Serving 20 concurrent users on one 24 GB GPU (back-of-envelope)

**Assumptions:** Llama-3.1-8B-class model, up to 4k tokens of context per user, an RTX 4090 / A5000-class 24 GB GPU, vLLM as the server.

| Item | Calculation | Memory |
|---|---|---|
| Weights, 4-bit AWQ/GPTQ | 8B x about 0.5-0.7 B | about 5.5 GB (fp16 would be 16 GB) |
| KV cache, fp16 | 20 users x 4,096 tokens x 131 KB | about 10.7 GB (about 5.4 GB with an FP8 KV cache) |
| CUDA context, activations, overhead | estimate | about 2 GB |
| **Total** | | **about 18 GB**, leaving headroom |

- fp16 weights (16 GB) plus the KV cache would need about 29 GB and would not fit, so the weights must be quantised (or the context capped at 2k).
- Use vLLM with continuous batching and PagedAttention, which avoids KV fragmentation so average usage is below the worst case. Settings: `--max-model-len 4096 --gpu-memory-utilization 0.9`, plus prefix caching if users share a system prompt.
- **Throughput:** decoding is memory-bandwidth bound. With about 1 TB/s and about 5.5 GB read per step, the ceiling is roughly 150+ batched steps per second; realistically 40-80. Shared by 20 users that is about 30-60 tokens/s per user at best, comfortable for chat.
- **Risks:** if average contexts are much longer than 4k, requests queue or get preempted. Mitigate with a concurrency cap, FP8 KV cache, or a shorter context limit.
