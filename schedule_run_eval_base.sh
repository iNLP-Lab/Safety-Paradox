#!/usr/bin/env bash
# Schedule run_eval_base.sh over all models in models.sh, using up to 4 GPUs in parallel.
# Skips models whose Posterior Attack + Judgement Ability outputs already exist.
#
# Usage:
#   bash schedule_run_eval_base.sh
#   NUM_GPUS=8 bash schedule_run_eval_base.sh   # optional: default 4
#
# Use physical GPUs 1,2,3 (not 0,1,2): set both count and id list (length must match).
#   NUM_GPUS=3 GPU_IDS=1,2,3 bash schedule_run_eval_base.sh
#
# Logs: logs/base_eval_<model_short>.log

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

NUM_GPUS="${NUM_GPUS:-4}"
if ! [[ "$NUM_GPUS" =~ ^[1-9][0-9]*$ ]]; then
  echo "NUM_GPUS must be a positive integer, got: $NUM_GPUS" >&2
  exit 1
fi

# PHYS_GPUS[slot] = CUDA device id passed to jobs for that slot (default 0..NUM_GPUS-1).
declare -a PHYS_GPUS=()
if [[ -z "${GPU_IDS:-}" ]]; then
  for ((i = 0; i < NUM_GPUS; i++)); do
    PHYS_GPUS+=("$i")
  done
else
  _gpu_ids_clean="${GPU_IDS//[[:space:]]/}"
  IFS=',' read -ra _gpu_id_parts <<< "$_gpu_ids_clean"
  for _gid in "${_gpu_id_parts[@]}"; do
    [[ -n "$_gid" ]] || continue
    PHYS_GPUS+=("$_gid")
  done
  if ((${#PHYS_GPUS[@]} != NUM_GPUS)); then
    echo "[sched] error: GPU_IDS must list exactly NUM_GPUS=${NUM_GPUS} device ids (comma-separated); got ${#PHYS_GPUS[@]} entries: ${GPU_IDS}" >&2
    exit 1
  fi
fi

# shellcheck source=models.sh
source "${SCRIPT_DIR}/models.sh"

all_models=()
for _lst in llama_lst gemma_lst mistral_lst falcon_lst qwen_lst; do
  declare -n _ref="$_lst"
  all_models+=("${_ref[@]}")
  unset -n _ref
done

model_short() {
  local m="$1"
  echo "${m##*/}"
}

# Full base eval is considered done when both final artifacts exist.
is_eval_done() {
  local m="$1"
  local s
  s="$(model_short "$m")"
  local posterior_asr="${SCRIPT_DIR}/Posterior_Attack/posterior_prompt_results/${s}/base/asr_results/asr_results.json"
  local judgement_txt="${SCRIPT_DIR}/Judgement_Ability/judgement_results/${s}/base/eval_results_${s}.txt"
  [[ -f "$posterior_asr" && -f "$judgement_txt" ]]
}

# Mirrors run_eval_base.sh: Judgement_Ability uses two GPUs for these sizes.
needs_two_gpus() {
  local s
  s="$(model_short "$1")"
  [[ "$s" == *"32B"* || "$s" == *"30B"* ]]
}

mkdir -p "${SCRIPT_DIR}/logs"

declare -a gpu_free=()
for ((i = 0; i < NUM_GPUS; i++)); do
  gpu_free[$i]=1
done

declare -A BUSY_PID_TO_GPUS=()
declare -a PIDS=()

allocate_one_gpu() {
  local i
  for ((i = 0; i < NUM_GPUS; i++)); do
    if [[ "${gpu_free[$i]:-0}" -eq 1 ]]; then
      echo "$i"
      return 0
    fi
  done
  return 1
}

allocate_two_gpus() {
  local i j
  for ((i = 0; i < NUM_GPUS; i++)); do
    [[ "${gpu_free[$i]:-0}" -eq 1 ]] || continue
    for ((j = i + 1; j < NUM_GPUS; j++)); do
      if [[ "${gpu_free[$j]:-0}" -eq 1 ]]; then
        echo "${i},${j}"
        return 0
      fi
    done
  done
  return 1
}

mark_gpus_busy() {
  local spec="$1" # slot indices, e.g. "0" or "0,1"
  local IFS=,
  read -ra _gs <<< "$spec"
  for g in "${_gs[@]}"; do
    gpu_free["$g"]=0
  done
}

release_gpus() {
  # space-separated slot indices (same keys as gpu_free)
  for g in $1; do
    gpu_free["$g"]=1
  done
}

# slot spec "0" or "0,1" -> space-separated slot indices (for gpu_free)
slots_to_slot_list() {
  echo "${1//,/ }"
}

# slot spec "0" or "0,1" -> comma-separated real CUDA device ids for the worker
slots_to_cuda_vis() {
  local spec="$1"
  local IFS=,
  local -a _slots=()
  read -ra _slots <<< "$spec"
  local _out=()
  local _s
  for _s in "${_slots[@]}"; do
    _out+=("${PHYS_GPUS[_s]}")
  done
  (IFS=','; echo "${_out[*]}")
}

start_job() {
  local model="$1"
  local slot_spec="$2" # scheduler slots: "0" or "0,1" (indices into PHYS_GPUS)
  local cuda_vis s log
  cuda_vis="$(slots_to_cuda_vis "$slot_spec")"
  s="$(model_short "$model")"
  log="${SCRIPT_DIR}/logs/base_eval_${s//[^A-Za-z0-9._-]/_}.log"

  mark_gpus_busy "$slot_spec"

  (
    cd "$SCRIPT_DIR" || exit 1
    export CUDA_VISIBLE_DEVICES="$cuda_vis"
    if [[ "$model" == *"gemma-2"* ]]; then
      export VLLM_ATTENTION_BACKEND=FLASHINFER
    fi
    bash run_eval_base.sh "$model"
  ) >"$log" 2>&1 &

  local pid=$!
  BUSY_PID_TO_GPUS[$pid]="$(slots_to_slot_list "$slot_spec")"
  PIDS+=("$pid")
  echo "[sched] started pid=$pid model=$s slots=$slot_spec CUDA_VISIBLE_DEVICES=$cuda_vis log=$log"
}

reap_finished() {
  local newpids=() pid phys ec
  for pid in "${PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      newpids+=("$pid")
      continue
    fi
    wait "$pid" || true
    ec=$?
    phys="${BUSY_PID_TO_GPUS[$pid]:-}"
    release_gpus "$phys"
    unset "BUSY_PID_TO_GPUS[$pid]"
    echo "[sched] finished pid=$pid exit=$ec (released GPUs: ${phys:-none})"
  done
  PIDS=("${newpids[@]}")
}

declare -a queue=()
skipped=0
for m in "${all_models[@]}"; do
  if is_eval_done "$m"; then
    echo "[sched] skip (already done): $(model_short "$m")"
    ((skipped++)) || true
    continue
  fi
  queue+=("$m")
done

echo "[sched] total models: ${#all_models[@]}, pending: ${#queue[@]}, skipped: ${skipped}, NUM_GPUS: ${NUM_GPUS}, GPU map: $(IFS=','; echo "${PHYS_GPUS[*]}")"

if ((${#queue[@]} == 0)); then
  echo "[sched] nothing to run."
  exit 0
fi

while ((${#queue[@]} > 0)) || ((${#PIDS[@]} > 0)); do
  reap_finished

  started=0
  while ((${#queue[@]} > 0)); do
    model="${queue[0]}"
    if needs_two_gpus "$model"; then
      pair=""
      pair="$(allocate_two_gpus)" || true
      if [[ -z "$pair" ]]; then
        break
      fi
      queue=("${queue[@]:1}")
      start_job "$model" "$pair"
    else
      g=""
      g="$(allocate_one_gpu)" || true
      if [[ -z "$g" ]]; then
        break
      fi
      queue=("${queue[@]:1}")
      start_job "$model" "$g"
    fi
    ((started++)) || true
  done

  if ((${#PIDS[@]} > 0)); then
    sleep 3
  elif ((${#queue[@]} > 0)); then
    # Should not happen often: pending work but no free GPUs pattern is handled by sleep above.
    sleep 1
  fi
done

echo "[sched] all jobs completed."
