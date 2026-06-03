#!/bin/bash
# Full pipeline: RL training → Posterior Attack eval → Judgement Ability eval
# Usage: bash run_full_pipeline.sh <model_name> [dataset_name]
# Example: bash run_full_pipeline.sh Qwen/Qwen2.5-3B-Instruct SAD

set -e

gpus="1"


if [ -z "$1" ]; then
    echo "Usage: bash run_full_pipeline.sh <model_name> [dataset_name]"
    echo "Example: bash run_full_pipeline.sh Qwen/Qwen2.5-3B-Instruct SAD"
    exit 1
fi

MODEL_NAME="$1"
DATASET_NAME="${2:-SAD}"  # default to SAD if not provided
MODEL_SHORT="${MODEL_NAME##*/}"  # e.g. Qwen2.5-3B-Instruct
FINETUNED_PATH="finetuned_models/grpo/${MODEL_SHORT}/${DATASET_NAME}"

# GPUS="${3:-1}" 

echo "=============================================="
echo "Full Pipeline: $MODEL_NAME (dataset: $DATASET_NAME)"
echo "=============================================="

# Ensure we're in project root (directory containing this script)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ---------- Step 1: RL Training (GRPO) ----------
echo ""
echo ">>> [1/4] GRPO RL Training"
bash "train/${DATASET_NAME}.sh" "$MODEL_NAME"
echo "[DONE] Training complete. Model at ${FINETUNED_PATH}"
echo ""

# ---------- Step 2: Posterior Attack Evaluation ----------
echo ">>> [2/4] Posterior Attack Evaluation"
cd Posterior_Attack
model_path="../${FINETUNED_PATH}"
output_dir="posterior_prompt_results/${MODEL_SHORT}/${DATASET_NAME}"
result_file="Both-harmful_behaviors-0_520.json"

python main.py \
    --gpus $gpus \
    --data_path "data/harmful_behaviors.csv" \
    --gpu_memory_utilization 0.9 \
    --max_model_len 4096 \
    --victim_llm "$model_path" \
    --victim_name "$MODEL_SHORT" \
    --begin 0 --end 520 --batch 32 \
    --output_dict "${output_dir}/output" \
    --attack_type "Both"

python main_eval.py \
    --gpus $gpus --batch 64 \
    --result_file "$result_file" \
    --output_dir "${output_dir}/output" \
    --checkpoint_dir "${output_dir}/checkpoint" \
    --final_result_dir "${output_dir}/asr_results" \
    --judge_llm "" --judge_api "" \
    --judge_harm_bench "cais/HarmBench-Llama-2-13b-cls"
cd ..
echo "[DONE] Posterior Attack evaluation complete"
echo ""

# ---------- Step 3: Judgement Ability Evaluation ----------
echo ">>> [3/4] Judgement Ability Evaluation"
cd Judgement_Ability
python main.py \
    --gpus $gpus \
    --model "../${FINETUNED_PATH}" \
    --gpu_memory_utilization 0.8 \
    --max_model_len 3000 \
    --data_path "data/text_behaviors_val_set.json" \
    --batch_size 64 \
    --result_dir "judgement_results/${MODEL_SHORT}/${DATASET_NAME}"
cd ..
echo "[DONE] Judgement Ability evaluation complete"
echo ""

# ---------- Step 4: Utility Evaluation (GSM8k + MMLU) ----------
echo ">>> [4/4] Utility Evaluation (GSM8k + MMLU)"
python Utility_Eval/eval_english.py \
    --model "${FINETUNED_PATH}" \
    --gpus $gpus \
    --output_dir "Utility_Eval/utility_results" \
    --tasks gsm8k mmlu
echo "[DONE] Utility evaluation complete"
echo ""

echo "=============================================="
echo "Pipeline complete for $MODEL_NAME"
echo "  - Model: ${FINETUNED_PATH}"
echo "  - Posterior Attack: posterior_prompt_results/${MODEL_SHORT}/${DATASET_NAME}/"
echo "  - Judgement Ability: judgement_results/${MODEL_SHORT}/${DATASET_NAME}/"
echo "  - Utility (GSM8k+MMLU): Utility_Eval/utility_results/"
echo "=============================================="
