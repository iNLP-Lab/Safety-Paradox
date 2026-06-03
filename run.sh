#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
MODELS=(
    "meta-llama/Llama-3.1-8B-Instruct"
)


for MODEL_NAME in "${MODELS[@]}"; do
  DATASET_NAME="SAD"
  bash run_full_pipeline.sh "$MODEL_NAME" "$DATASET_NAME"

  DATASET_NAME="SAI"
  bash run_full_pipeline.sh "$MODEL_NAME" "$DATASET_NAME"

  bash run_eval_base.sh "$MODEL_NAME"
done
