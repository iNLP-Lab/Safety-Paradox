#!/bin/bash
set -e
# ---------- Global Config ----------
gpus="6 7"
begin=0
end=520
batch=50 # Set 20 if deepseek models

data_path="data/harmful_behaviors_full.csv"
data_name=$(basename "$data_path" .csv)

judge_llm=""
judge_api="openai/gpt-4o-mini"
judge_harm_bench=""

# ---------- Model List ----------

models=(
    "openai/gpt-4o-2024-11-20"
    "openai/gpt-5-chat"
    "openai/gpt-5"
    "anthropic/claude-3.7-sonnet"
    "anthropic/claude-sonnet-4.6"
    "qwen/qwen3-235b-a22b-2507"
    "deepseek/deepseek-v3.2"
    "google/gemma-4-26b-a4b-it"
    "qwen/qwen3.6-35b-a3b"2
    "openai/gpt-oss-20b"
    "openai/gpt-oss-120b"
)

# ---------- Other Parameters ----------
attacks=(
    "PosteriorAttack"
)
reasoning_effort="low"

# ---------- Run ----------
for attack_type in "${attacks[@]}"; do
    output_base_dir="paper_results_${reasoning_effort}/${attack_type}"
    global_result_dir="paper_results_${reasoning_effort}/global_results.json"

    # echo "$attack_type -> $output_base_dir"

    max_tokens=4096

        
    for path in "${models[@]}"; do
        model_name="${path##*/}"

        if [[ ("$attack_type" == "SelfCipher" || "$attack_type" == "ArtPrompt") && "$model_name" == "gpt-5" ]]; then
            max_tokens=8192
        fi

        output_eval_dir="$output_base_dir/$model_name"

        echo "==============================================="
        echo "Model: $model_name"
        echo "Attack: $attack_type"
        echo "==============================================="

        # -------- 1. Generation --------
        echo "[1/2] Running generation..."

        python main_api.py \
            --data_path "$data_path" \
            --victim_llm "$path" \
            --begin "$begin" \
            --end "$end" \
            --batch "$batch" \
            --output_dir "$output_eval_dir/output" \
            --checkpoint_dir "$output_eval_dir/checkpoint" \
            --attack_type "$attack_type" \
            --max_tokens $max_tokens \
            --reasoning_effort "$reasoning_effort"

        echo "[DONE] Generation complete"

        # -------- 2. Evaluation --------
        echo "[2/2] Running evaluation..."

        result_file="${attack_type}-${data_name}-${begin}_${end}.json"

        python main_eval.py \
            --gpus $gpus \
            --batch 10 \
            --attack_type "$attack_type" \
            --model_name "$model_name" \
            --result_file "$result_file" \
            --summary_dir "$global_result_dir" \
            --output_dir "$output_eval_dir/output" \
            --checkpoint_dir "$output_eval_dir/checkpoint" \
            --final_result_dir "$output_eval_dir/asr_results" \
            --judge_llm "$judge_llm" \
            --judge_api "$judge_api" \
            --judge_harm_bench "$judge_harm_bench"

        echo "[DONE] Evaluation complete for $model_name and $attack_type"
        echo ""

        # # Sleep for a short time to avoid hitting rate limits
        # echo "Sleeping for 10 seconds before the next run..."
        # sleep 10

    done
done