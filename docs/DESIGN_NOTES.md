# Design notes — what the metaphors map to

open-deepthink uses neural-net language as a **design language** for a
prompt-orchestration loop. The operations are real. The algorithms they are
named after are not running here.

This file is the honest map. The README keeps the original names so existing
docs and skills stay searchable.

## Qualitative Neural Network (QNN)

| Name in the UI / code | Actual operation |
|---|---|
| Neuron / agent | One LLM call with a persona system prompt |
| Layer | A parallel batch; layer *k* sees layer *k−1* outputs |
| Epoch | Repeat the layered pass, then optionally mutate personas |
| Weights | Natural-language system prompts, attributes, skills |
| Mirror Descent | An LLM rewrites those system prompts from the last output |
| Qualitative self-attention | Lexical overlap of persona tokens vs past text; top-k excerpts injected |
| Learning rate | How strongly the rewrite prompt asks for change |
| Forward pass | `asyncio.gather` of chat completions per layer |

There is no gradient, no softmax, no matrix multiply. Self-attention does
**not** call an LLM; it is a capped token-overlap ranker
(`deepthink/self_attention.py`).

## Qualitative Diffusion (QDAD)

| Name | Actual operation |
|---|---|
| Noun × verb basis | Two word lists of length N; cell *(i, j)* is bound to that pair |
| Forward diffusion / noise | High-temperature LLM invents a feature for that cell |
| Reverse diffusion / critic | Another LLM, same cell signature, rewrites the feature |
| Score matching | Metaphor only — critics are instructed to sharpen and implementabilize |
| Temperature | Chat-model sampling temperature |
| Decode / synthesizer | One LLM collapses the N×N matrix into an app-build prompt |

This is **grid brainstorm + iterative rewrite**, not a diffusion SDE.

## Knowledge distillation

| Name | Actual operation |
|---|---|
| Distillation | Multi-agent QA logging on a fixed 12-node topology |
| Child replacement | Mixing prompt writes a new persona; the child keeps the parent's text memory |
| Perplexity | A separate LLM scores recent QA pairs 1–100 (not model NLL) |
| Training data | JSON traces. Whether they improve a base model is **unevaluated**. |

## What we measure without a paid judge

`deepthink eval` / `deepthink.eval_structural` runs the three engines on
**mock LLMs** and checks artifact *shape*: persona count, epoch-map presence,
dataset files, grid size. A green score means the loop is wired.

It does **not** mean the strategies are better than one long prompt.

A protocol for a later paid comparison is in [`EVAL.md`](./EVAL.md).
