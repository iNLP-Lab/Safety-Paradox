import os
import json
import pandas
import argparse
from tqdm import tqdm
import math

import triton_kernels_compat  # noqa: F401 — before vLLM; vendor triton_kernels if site-packages is incomplete

# Dependent modules
from utils.eval_utils import Evaluator

# VLLM for judge LLM
from vllm import LLM, SamplingParams

from utils.eval_utils import extract_name,check_current_progress, process_evaluation_batch


if __name__ == "__main__":
    parser = argparse.ArgumentParser("PosteriorAttack Evaluation")
    
    # GPU Configuration
    parser.add_argument('--gpus', type=int, nargs='+', default=[5, 6], help='GPU device IDs')
    parser.add_argument("--batch", type=int, default=32, help="Batch size for parallel processing")

    # Input/Output Directories
    parser.add_argument("--result_file", type=str, default="final_result.json", help="PosteriorAttack results file")
    parser.add_argument("--checkpoint_file", type=str, default=None, help="Checkpoint file for resuming evaluation")
    parser.add_argument("--output_dir", type=str, default="result", help="Directory containing PosteriorAttack results")
    parser.add_argument("--final_result_dir", type=str, default="final_result", help="Directory for final evaluation results")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoint", help="Directory for checkpoint files")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint if available (default: re-run from scratch)")

    # Evaluation Judges
    parser.add_argument("--judge_llm", type=str, default=None, help="Judge LLM model path")
    parser.add_argument("--judge_api", type=str, default=None, help="Judge API (e.g., GPT-4, Gemini)")
    parser.add_argument("--judge_harm_bench", type=str, default=None, help="HarmBench classifier model path")

    args = parser.parse_args()

    # Configure GPU environment (preserve CUDA_VISIBLE_DEVICES if set by scheduler/parent)
    gpu_list = ','.join(map(str, args.gpus))
    env_update = {'CUDA_DEVICE_ORDER': 'PCI_BUS_ID', 'WORLD_SIZE': '1'}
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        env_update['CUDA_VISIBLE_DEVICES'] = gpu_list
    os.environ.update(env_update)
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', gpu_list)
    print(f"[INFO] Using GPUs: {gpu_list} (CUDA_VISIBLE_DEVICES={visible})")

    # Extract judge names for file naming
    judge_llm_name = extract_name(args.judge_llm, "no_llm")
    judge_api_name = extract_name(args.judge_api, "no_api")
    judge_harm_name = extract_name(args.judge_harm_bench, "no_harm")

    attack_type = args.result_file.split("-")[0]


    if attack_type == "Both":
        attack_types = ["PosteriorAttack", "Direct"]
        result_files = [args.result_file.replace("Both", attack_type) for attack_type in attack_types]
    else:
        attack_types = [attack_type]
        result_files = [args.result_file]
        
    # Initialize evaluator with configured judges
    evaluator = Evaluator(
        judge_llm=args.judge_llm,
        judge_api=args.judge_api,
        judge_harm_bench=args.judge_harm_bench,
        tensor_parallel_size=len(args.gpus)
    )
    # Preserve original judge args (they get mutated when skipping completed evals for one attack type)
    original_judge_llm = args.judge_llm
    original_judge_api = args.judge_api
    original_judge_harm_bench = args.judge_harm_bench

    for idx, attack_type in enumerate(attack_types):
        # Reset judge args for this attack type (may have been set to None when processing previous attack type)
        args.judge_llm = original_judge_llm
        args.judge_api = original_judge_api
        args.judge_harm_bench = original_judge_harm_bench

        # Load results data
        result_file_path = os.path.join(args.output_dir, result_files[idx])
        
        # Determine checkpoint file path (None if checkpoint_dir disabled)
        use_checkpoint = args.checkpoint_dir and args.checkpoint_dir.lower() != "none"
        if use_checkpoint:
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            if args.checkpoint_file:
                checkpoint_file_path = os.path.join(args.checkpoint_dir, args.checkpoint_file)
            else:
                base_name = result_files[idx].replace('.json', '')
                checkpoint_file_path = os.path.join(
                    args.checkpoint_dir,
                    f"{base_name}_{judge_llm_name}_{judge_api_name}_{judge_harm_name}.json"
                )
        else:
            checkpoint_file_path = None

        # Load: resume from checkpoint only when --resume and checkpoint exists; else load original results
        if args.resume and use_checkpoint and checkpoint_file_path and os.path.exists(checkpoint_file_path):
            print(f"[INFO] Resuming from checkpoint: {checkpoint_file_path}")
            with open(checkpoint_file_path, "r", encoding="utf-8") as f:
                result_dicts = json.load(f)
            print(f"[INFO] Loaded {len(result_dicts)} entries from checkpoint")
        else:
            if not args.resume:
                print(f"[INFO] Re-running from scratch (use --resume to resume from checkpoint)")
            print(f"[INFO] Loading results from: {result_file_path}")
            with open(result_file_path, "r", encoding="utf-8") as f:
                result_dicts = json.load(f)
            print(f"[INFO] Loaded {len(result_dicts)} entries")

        # Normalize keys to integers if needed
        if isinstance(result_dicts, dict) and all(k.isdigit() for k in result_dicts.keys()):
            result_dicts = {int(k): v for k, v in result_dicts.items()}



        # Check evaluation progress and skip completed evaluations
        num_samples = len(result_dicts)
        eval_types = {
            "judge_dict": ("enabled", "Dictionary-based"),
            "judge_harm_bench": (args.judge_harm_bench, "HarmBench"),
            "judge_llm": (args.judge_llm, "LLM-based"),
            "judge_api": (args.judge_api, "API-based")
        }
        
        for eval_type, (attr_value, name) in eval_types.items():
            if attr_value:
                num_processed = check_current_progress(result_dicts, eval_type=eval_type)
                print(f"[INFO] {name} evaluation: {num_processed}/{num_samples} completed")
                # if num_processed == num_samples:
                #     setattr(args, eval_type.replace("judge_", "judge_"), None)
                #     print(f"[INFO] {name} evaluation already complete, skipping")
        
        # Enable dictionary evaluation by default
        if not hasattr(args, 'judge_dict') or args.judge_dict is None:
            args.judge_dict = "enabled"



        print(f"[INFO] Evaluator initialized | Batch size: {args.batch} | Samples: {len(result_dicts)}")

        # Process evaluations in order
        evaluation_pipeline = [
            ("judge_dict", args.judge_dict, "Dictionary-based"),
            ("judge_harm_bench", args.judge_harm_bench, "HarmBench"),
            ("judge_llm", args.judge_llm, "LLM-based"),
            ("judge_api", args.judge_api, "API-based")
        ]
        
        for eval_type, enabled, name in evaluation_pipeline:
            if enabled:
                print(f"\n[INFO] Starting {name} evaluation...")
                result_dicts = process_evaluation_batch(
                    result_dicts, evaluator, checkpoint_file_path, 
                    eval_type=eval_type, batch=args.batch
                )
                print(f"[INFO] Completed {name} evaluation")




        # Save final results
        os.makedirs(args.final_result_dir, exist_ok=True)
        base_name = result_files[idx].replace('.json', '')
        output_file_path = os.path.join(
            args.final_result_dir,
            f"{base_name}_{judge_llm_name}_{judge_api_name}_{judge_harm_name}.json"
        )
        
        result_list = [dict(item) for item in result_dicts.values()]
        with open(output_file_path, "w", encoding="utf-8") as f:
            json.dump(result_list, f, ensure_ascii=False, indent=4)
        
        print(f"\n[INFO] Evaluation complete. Results saved to: {output_file_path}")
        print(f"[INFO] Total samples evaluated: {len(result_list)}")

        asr = sum([item["judge_success_harm_bench"] for item in result_list]) / len(result_list)
        print(f"[INFO] ASR: {asr}")

        # Update asr_results.json: load existing, add this run's ASR, save with keys sorted alphabetically
        asr_results_path = os.path.join(
            os.path.dirname(output_file_path),
            "asr_results.json"
        )
        result_key = os.path.splitext(os.path.basename(output_file_path))[0]
        if os.path.exists(asr_results_path):
            with open(asr_results_path, "r", encoding="utf-8") as f:
                asr_data = json.load(f)
        else:
            asr_data = {}
        asr_data[result_key] = asr
        with open(asr_results_path, "w", encoding="utf-8") as f:
            json.dump(asr_data, f, ensure_ascii=False, indent=4, sort_keys=True)
        print(f"[INFO] ASR updated in {asr_results_path}")
