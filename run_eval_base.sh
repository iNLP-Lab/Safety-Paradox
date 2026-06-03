#!/bin/bash
# Posterior Attack + Judgement Ability evaluation for base model (no RL training)
# Usage: bash run_eval_base.sh <model_name>
# Example: bash run_eval_base.sh Qwen/Qwen2.5-3B-Instruct
# export CUDA_VISIBLE_DEVICES=0
# CUDA_VISIBLE_DEVICES=1 bash run_eval_base.sh google/gemma-4-26B-A4B-it

set -e

if [ -z "$1" ]; then
    echo "Usage: bash run_eval_base.sh <model_name>"
    echo "Example: bash run_eval_base.sh Qwen/Qwen2.5-3B-Instruct"
    exit 1
fi

MODEL_NAME="$1"
MODEL_SHORT="${MODEL_NAME##*/}"  # e.g. Qwen2.5-3B-Instruct
RUN_NAME="base"

# Qwen3.6 / Qwen3_5* MoE: FlashInfer GDN prefill JIT needs newer CCCL/cuda::ptx than many CUDA 12.4
# installs. Posterior_Attack/main.py and Judgement_Ability/main.py pass LLM(gdn_prefill_backend="triton").
_model_lc="${MODEL_NAME,,}"
if [[ "${_model_lc}" == *"qwen3.6"* ]]; then
  echo "[INFO] Qwen3.6: scripts use gdn_prefill_backend=triton in Python (see main.py)."
  echo "[INFO] Qwen3.6 requires vLLM with Qwen3_5ForConditionalGeneration support (e.g. v0.19+). Upgrade vLLM if ModelConfig ValidationError mentions unsupported architectures."
fi
# Gemma 4 MoE: avoid "Cannot re-initialize CUDA in forked subprocess" when the parent process
# has already touched CUDA before vLLM's EngineCore worker starts (V1 multiprocessing default is fork).
if [[ "${_model_lc}" == *"gemma-4"* ]]; then
  export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
  echo "[INFO] Gemma 4: VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD}"
fi
unset _model_lc

echo "=============================================="
echo "Base Model Evaluation: $MODEL_NAME"
echo "=============================================="

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ---------- Step 1: Posterior Attack Evaluation ----------
echo ""
echo ">>> [1/3] Posterior Attack Evaluation"
cd Posterior_Attack
output_dir="posterior_prompt_results/${MODEL_SHORT}/${RUN_NAME}"
result_file="PosteriorAttack-harmful_behaviors-0_520.json"

python main.py \
    --gpus 0 \
    --data_path "data/harmful_behaviors.csv" \
    --gpu_memory_utilization 0.9 \
    --max_model_len 4096 \
    --victim_llm "$MODEL_NAME" \
    --victim_name "$MODEL_SHORT" \
    --begin 0 --end 520 --batch 32 \
    --output_dict "${output_dir}/output" \
    --attack_type "PosteriorAttack"

python main_eval.py \
    --gpus 0 --batch 64 \
    --result_file "$result_file" \
    --output_dir "${output_dir}/output" \
    --checkpoint_dir "${output_dir}/checkpoint" \
    --final_result_dir "${output_dir}/asr_results" \
    --judge_llm "" --judge_api "" \
    --judge_harm_bench "cais/HarmBench-Llama-2-13b-cls"
cd ..
echo "[DONE] Posterior Attack evaluation complete"
echo ""

# ---------- Step 2: Judgement Ability Evaluation ----------
echo ">>> [2/3] Judgement Ability Evaluation"
cd Judgement_Ability
# Use higher GPU memory for large models (32B+); use tensor parallelism if needed
if [[ "$MODEL_SHORT" == *"32B"* ]] || [[ "$MODEL_SHORT" == *"30B"* ]]; then
    # Large models: 8 GPUs (tensor parallel). Use 0.85 (not 0.95) so CUDA graph capture
    # has headroom; 0.95 leaves no room and causes OOM during decode graph capture.
    python main.py \
        --gpus 0 1 \
        --model "$MODEL_NAME" \
        --gpu_memory_utilization 0.9 \
        --max_model_len 4096 \
        --data_path "data/text_behaviors_val_set.json" \
        --batch_size 64 \
        --result_dir "judgement_results/${MODEL_SHORT}/${RUN_NAME}"
else
    python main.py \
        --gpus 0 \
        --model "$MODEL_NAME" \
        --gpu_memory_utilization 0.9 \
        --max_model_len 4096 \
        --data_path "data/text_behaviors_val_set.json" \
        --batch_size 64 \
        --result_dir "judgement_results/${MODEL_SHORT}/${RUN_NAME}"
fi
cd ..
echo "[DONE] Judgement Ability evaluation complete"
echo ""

# # ---------- Step 3: Utility Evaluation (GSM8k + MMLU) ----------
# echo ">>> [3/3] Utility Evaluation (GSM8k + MMLU)"
# cd "$SCRIPT_DIR"
# python Utility_Eval/eval_english.py \
#     --model "$MODEL_NAME" \
#     --gpus 7 \
#     --output_dir "Utility_Eval/utility_results" \
#     --tasks gsm8k mmlu
# echo "[DONE] Utility evaluation complete"
# echo ""

# echo "=============================================="
# echo "Base model evaluation complete for $MODEL_NAME"
# echo "  - Posterior Attack: posterior_prompt_results/${MODEL_SHORT}/${RUN_NAME}/"
# echo "  - Judgement Ability: judgement_results/${MODEL_SHORT}/${RUN_NAME}/"
# echo "  - Utility (GSM8k+MMLU): Utility_Eval/utility_results/"
# echo "=============================================="
