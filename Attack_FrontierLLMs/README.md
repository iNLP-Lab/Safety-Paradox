# Attacking Frontier LLMs with PosteriorAttack

This repository contains the code and results for evaluating **PosteriorAttack** — a jailbreak method that exploits LLM judge prompts as adversarial attack vectors — against frontier language models on the [AdvBench](https://arxiv.org/abs/2307.15043) benchmark.

---

## Overview

PosteriorAttack frames a harmful behavior as a *generation example* inside a red-teaming classifier prompt. The victim model is asked to produce a generation that would cause a classifier to output "Yes" — effectively causing it to emit harmful content while appearing to answer a meta-level question. The attack is then evaluated using multiple judge backends to compute **Attack Success Rate (ASR)**.

The pipeline has two stages:

```
[1] Generation  →  victim LLM produces a response to the attack prompt
[2] Evaluation  →  one or more judges score each response (ASR reported)
```

---

## Results

Results are stored as JSON files in the `paper_results_<effort>/` directories. Each directory corresponds to a test-time reasoning effort level. `global_results.json` inside each folder aggregates per-model ASR and token/cost statistics.

`frontier_models_exp/` contains comparison results of PosteriorAttack against other jailbreak methods.

---

## Repository Structure

```
.
├── main_api.py               # Generation script for API-based models (OpenRouter / Anthropic)
├── main_local.py             # Generation script for locally-served models (vLLM)
├── main_eval.py              # Evaluation script (all judge backends)
├── run_api.sh                # End-to-end runner: API generation + evaluation
├── run_local.sh              # End-to-end runner: local (vLLM) generation + evaluation
│
├── utils/
│   ├── posteriorattack.py     # PosteriorAttack prompt construction & response decoding
│   ├── call_OpenRouterAPI.py # Async batch inference via OpenRouter
│   ├── call_AnthropicAPI.py  # Async batch inference via Anthropic Messages API
│   ├── eval_utils.py         # Evaluator class (dict / API / HarmBench / local LLM)
│   ├── prompt_utils.py       # Chat template definitions for local models
│   └── common.py             # Dataset loading, checkpoint I/O, helper utilities
│
├── data/
│   └── harmful_behaviors_full.csv   # 520 AdvBench harmful behavior prompts
│
├── frontier_models_exp/      # ASR results: PosteriorAttack vs. baseline methods across frontier models
├── paper_results_high/       # ASR results: reasoning_effort = high
├── paper_results_medium/     # ASR results: reasoning_effort = medium
├── paper_results_low/        # ASR results: reasoning_effort = low
└── paper_results_none/       # ASR results: reasoning disabled
```

---

## Setup

### Requirements

```bash
pip install vllm transformers anthropic openai aiohttp pandas python-dotenv tqdm
```

### Environment Variables

Create a `.env` file in the project root:

```env
OPENROUTER_API_KEY=<your_openrouter_key>
ANTHROPIC_API_KEY=<your_anthropic_key>
OPENAI_API_KEY=<your_openai_key>
HF_ACCESS_TOKEN=<your_huggingface_token>
```

---

## Usage

### API-based Models (OpenRouter / Anthropic)

Edit `run_api.sh` to set the target models, reasoning effort, and judge, then run:

```bash
bash run_api.sh
```

Key parameters in `run_api.sh`:

| Parameter | Description |
|---|---|
| `models` | List of OpenRouter model IDs to attack |
| `reasoning_effort` | `none` / `minimal` / `low` / `medium` / `high` / `xhigh` |
| `judge_api` | API judge model (e.g., `openai/gpt-4o-mini`) |
| `begin` / `end` | Index range of AdvBench samples to run |
| `batch` | Parallel request batch size |

Or invoke `main_api.py` directly:

```bash
python main_api.py \
  --victim_llm "anthropic/claude-sonnet-4.6" \
  --attack_type PosteriorAttack \
  --reasoning_effort low \
  --data_path data/harmful_behaviors_full.csv \
  --begin 0 --end 520 \
  --batch 50 \
  --output_dir results/claude-sonnet-4-6/output \
  --checkpoint_dir results/claude-sonnet-4-6/checkpoint \
  --api_provider openrouter   # or "anthropic" for direct API
```

### Local Models (vLLM)

Edit `run_local.sh` to set GPU IDs and model paths, then run:

```bash
bash run_local.sh
```

Or invoke `main_local.py` directly:

```bash
python main_local.py \
  --victim_llm openai/gpt-oss-20b \
  --gpus 0 1 2 3 \
  --attack_type PosteriorAttack \
  --data_path data/harmful_behaviors_full.csv \
  --begin 0 --end 520 \
  --batch 20 \
  --output_dir results/gpt-oss-20b/output \
  --checkpoint_dir results/gpt-oss-20b/checkpoint
```

### Evaluation Only

```bash
python main_eval.py \
  --model_name claude-sonnet-4-6 \
  --attack_type PosteriorAttack \
  --result_file PosteriorAttack-harmful_behaviors_full-0_520.json \
  --output_dir results/claude-sonnet-4-6/output \
  --checkpoint_dir results/claude-sonnet-4-6/checkpoint \
  --final_result_dir results/claude-sonnet-4-6/asr_results \
  --summary_dir paper_results_low/global_results.json \
  --judge_api openai/gpt-4o-mini \
  --gpus 0 1 \
  --batch 10
```

---

## Evaluation Judges

`main_eval.py` supports four judge backends that can be combined in a single run:

| Flag | Backend | Notes |
|---|---|---|
| `--judge_dict` | Keyword matching | Fast heuristic; checks for common refusal phrases |
| `--judge_api <model>` | API-based LLM judge | GPT-4o or any OpenRouter model; scores 1–10, ASR = fraction rated 10 |
| `--judge_harm_bench <path>` | HarmBench classifier | Llama-2-13b binary classifier from HarmBench |
| `--judge_llm <path>` | Local vLLM judge | Any HuggingFace model served via vLLM |

Evaluation checkpoints are saved after every batch so runs can be resumed without re-evaluating completed samples.

---

## `utils/` Reference

### `posteriorattack.py` — `PosteriorAttack`

Constructs the PosteriorAttack prompt by injecting the harmful behavior into a red-teaming classifier template and asking the model to produce a *generation* that would be classified as "Yes".

- `generate(harm_prompt)` → returns the attack as a `[system, user]` message list
- `decode_response(text)` → extracts the bracketed generation from the model output

### `call_OpenRouterAPI.py`

Async batch client for the OpenRouter `/chat/completions` endpoint.

- `batch_chat(messages_batch, model, **kwargs)` → runs all requests concurrently, returns a list of `{output, finish_reason, reasoning, price, input_tokens, output_tokens, reasoning_tokens}` dicts
- Supports extended reasoning via `reasoning={"effort": "<level>"}` or `reasoning={"enabled": False}`
- Optional debug logging of every request/response to a directory

### `call_AnthropicAPI.py`

Async batch client for the Anthropic Messages API — mirrors `call_OpenRouterAPI` interface.

- `batch_chat(messages_batch, model, **kwargs)` → same return format
- Accepts `reasoning_effort` or `reasoning={"effort": ...}` and maps to Anthropic's `output_config.effort`
- Supported effort levels: `minimal`, `low`, `medium`, `high`, `xhigh`

### `eval_utils.py` — `Evaluator`

Multi-backend evaluator class.

- `run_evaluation_batch(harmful_prompts, attacks, responses, eval_type)` → routes to the selected backend
- `update_results(result_dicts, indices, batch_eval, eval_type)` → writes scores back into the result dictionary
- `process_evaluation_batch(...)` (module-level) → batched evaluation loop with checkpoint saving

### `prompt_utils.py`

Chat prompt templates for locally-served models (used by `main_local.py`):

---

## Data

The dataset is the standard 520-sample AdvBench split from:

```bibtex
@misc{zou2023universaltransferableadversarialattacks,
  title   = {Universal and Transferable Adversarial Attacks on Aligned Language Models},
  author  = {Andy Zou and Zifan Wang and Nicholas Carlini and Milad Nasr and J. Zico Kolter and Matt Fredrikson},
  year    = {2023},
  eprint  = {2307.15043},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url     = {https://arxiv.org/abs/2307.15043}
}
```
