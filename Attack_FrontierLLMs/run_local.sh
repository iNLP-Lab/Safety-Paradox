#!/bin/bash
set -e
# ---------- Global Config ----------
gpus="0 1 2 3 4 5 6 7"
gpu_memory_utilization=0.8
begin=0
end=520
batch=20

data_path="data/harmful_behaviors_full.csv"
data_name=$(basename "$data_path" .csv)

judge_llm=""
judge_api="openai/gpt-4o-mini"
judge_harm_bench=""

# ---------- Model List (HF paths for vLLM) ----------

models=(
    "openai/gpt-oss-20b"
    "openai/gpt-oss-120b"
)

# ---------- Attack Types ----------
attacks=(
    "PosteriorAttack"
)
reasoning_effort="none"

# ---------- Run ----------
for attack_type in "${attacks[@]}"; do
    output_base_dir="paper_results_${reasoning_effort}/${attack_type}"
    global_result_dir="paper_results_${reasoning_effort}/global_results.json"


    max_tokens=4096

    for path in "${models[@]}"; do
        model_short="${path##*/}"
        model_name="${model_short}"

        output_eval_dir="$output_base_dir/$model_name"

        echo "==============================================="
        echo "Model: $model_name"
        echo "Attack: $attack_type"
        echo "==============================================="

        # -------- 1. Generation --------
        echo "[1/2] Running generation..."

        python main.py \
            --gpus $gpus \
            --gpu_memory_utilization "$gpu_memory_utilization" \
            --data_path "$data_path" \
            --victim_llm "$path" \
            --begin "$begin" \
            --end "$end" \
            --batch "$batch" \
            --output_dir "$output_eval_dir/output" \
            --checkpoint_dir "$output_eval_dir/checkpoint" \
            --attack_type "$attack_type" \
            --max_tokens $max_tokens \
            --max_model_len 8192

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

    done
done
