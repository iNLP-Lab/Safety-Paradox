#!/bin/bash
# GRPO online RL training with RANDOM binary rewards (degradation baseline)
# Usage: bash train/SAD.sh
#
# Same setup as grpo_rl.sh but uses random 0/1 rewards instead of classification.
# Use this to verify that training without a meaningful reward signal degrades performance.

set -e

# Config: base_model can be passed as first arg, e.g. bash SAD.sh Qwen/Qwen2.5-3B-Instruct
base_model="${1:-Qwen/Qwen2.5-3B-Instruct}"
model="${base_model##*/}"
dataset_name="SAD"

# FSDP config: match model family (no cpu_ram_efficient_loading)
case "${base_model}" in
  *[Ll]lama*)    fsdp_config="train/fsdp_config_llama.json" ;;
  *[Qq]wen3*)    fsdp_config="train/fsdp_config_qwen3.json" ;;
  *[Qq]wen*)     fsdp_config="train/fsdp_config_qwen.json" ;;
  *[Mm]istral*)  fsdp_config="train/fsdp_config_mistral.json" ;;
  *[Gg]emma-2*)  fsdp_config="train/fsdp_config_gemma2.json" ;;
  *[Gg]emma*)    fsdp_config="train/fsdp_config_gemma.json" ;;
  *[Ff]alcon3*)  fsdp_config="train/fsdp_config_llama.json" ;;
  *[Ff]alcon*)   fsdp_config="train/fsdp_config_falcon.json" ;;
  *)             fsdp_config="train/fsdp_config_qwen.json" ;;
esac
# gpu_count=$(nvidia-smi -L 2>/dev/null | wc -l)
# gpu_count=${gpu_count:-1}
gpu_count=4

# Paths
input_csv="data/wildguardtrain_4096.csv"
grpo_dataset="ft_dataset/${dataset_name}"
output_dir="finetuned_models/grpo/${model}/${dataset_name}"

echo "=== Step 1: Prepare GRPO dataset ==="
python train/prepare_grpo_data.py \
    --input "${input_csv}" \
    --output "${grpo_dataset}" \
    --test_split 0.05 \
    --seed 42

echo ""
echo "=== Step 2: GRPO online RL training (RANDOM REWARD - degradation baseline) ==="
# shellcheck source=train/torchrun_master_port.inc.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/torchrun_master_port.inc.sh"
echo "torchrun MASTER_PORT: ${MASTER_PORT}"
torchrun --nproc-per-node ${gpu_count} --master_port "${MASTER_PORT}" \
    train/SAD.py \
    --model_name "${base_model}" \
    --train_file_path "${grpo_dataset}" \
    --output_dir "${output_dir}" \
    --random_seed 42 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --num_train_epochs 1 \
    --learning_rate 5e-7 \
    --max_completion_length 1024 \
    --num_generations 8 \
    --temperature 0.8 \
    --top_p 1.0 \
    --repetition_penalty 1.0 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type cosine \
    --weight_decay 0.05 \
    --bf16 True \
    --logging_steps 5 \
    --save_strategy "no" \
    --eval_strategy "epoch" \
    --remove_unused_columns False \
    --use_vllm True \
    --vllm_mode colocate \
    --vllm_max_model_length 4096 \
    --fsdp full_shard --fsdp auto_wrap \
    --fsdp_config "${fsdp_config}" \
    --gradient_checkpointing True \
    --push_to_hub False

echo ""
echo "Training complete. Model saved to ${output_dir}"
