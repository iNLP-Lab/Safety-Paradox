#!/bin/bash
# Utility evaluation (GSM8k + MMLU) for base or RL-tuned models
# Run from project root: bash Utility_Eval/run.sh <model_path>
# Examples:
#   bash Utility_Eval/run.sh meta-llama/Llama-3.1-8B-Instruct
#   bash Utility_Eval/run.sh finetuned_models/grpo/Llama-3.1-8B-Instruct/SAD

set -e

if [ -z "$1" ]; then
    echo "Usage: bash Utility_Eval/run.sh <model_path>"
    echo "  Base model:  meta-llama/Llama-3.1-8B-Instruct"
    echo "  RL-tuned:    finetuned_models/grpo/Llama-3.1-8B-Instruct/SAD"
    exit 1
fi

MODEL_PATH="$1"
GPUS="${2:-0}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

python Utility_Eval/eval_english.py \
    --model "$MODEL_PATH" \
    --gpus $GPUS \
    --output_dir "utility_results" \
    --tasks gsm8k mmlu
