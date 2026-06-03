import os
import json
import sys
import pandas
import argparse
from tqdm import tqdm
import math


# Dependent modules
from utils.eval_utils import Evaluator

# VLLM for judge LLM
from vllm import LLM, SamplingParams

from utils.eval_utils import extract_name,check_current_progress, process_evaluation_batch
from utils.common import load_results, sync_checkpoint_with_output_folder, validate_sample_outputs


def skip_completed_evaluations(args, result_dicts):
    """
    Check evaluation progress and skip completed evaluations.
    
    Args:
        args: Command line arguments
        result_dicts: Dictionary of results
    """
    num_samples = len(result_dicts)
    if num_samples == 0:
        return
    eval_types = {
        "judge_dict": (args.judge_dict, "Dictionary-based"),
        "judge_harm_bench": (args.judge_harm_bench, "HarmBench"),
        "judge_llm": (args.judge_llm, "LLM-based"),
        "judge_api": (args.judge_api, "API-based")
    }
    
    for eval_type, (attr_value, name) in eval_types.items():
        if attr_value:
            num_processed = check_current_progress(result_dicts, eval_type=eval_type)
            print(f"[INFO] {name} evaluation: {num_processed}/{num_samples} completed")
            if num_processed == num_samples:
                setattr(args, eval_type, None)
                print(f"[INFO] {name} evaluation already complete, skipping")
    



def run_evaluation_pipeline(evaluator, result_dicts, checkpoint_file_path, args):
    """
    Execute the evaluation pipeline in order.
    
    Args:
        evaluator: Evaluator object
        result_dicts: Dictionary of results
        checkpoint_file_path: Path to checkpoint file
        args: Command line arguments
        
    Returns:
        dict: Updated result dictionaries with evaluation results
    """
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
    
    return result_dicts


def save_results_and_calculate_metrics(result_dicts, args, judge_llm_name, judge_api_name, judge_harm_name, number_of_invalid_samples):
    """
    Save final results and calculate metrics.
    
    Args:
        result_dicts: Dictionary of results
        args: Command line arguments
        judge_llm_name: Name of judge LLM
        judge_api_name: Name of judge API
        judge_harm_name: Name of HarmBench judge
        number_of_invalid_samples: Number of invalid samples found
    Returns:
        tuple: (output_file_path, result_list, asr, metrics_dict)
    """
    os.makedirs(args.final_result_dir, exist_ok=True)
    base_name = args.result_file.replace('.json', '')
    output_file_path = os.path.join(
        args.final_result_dir,
        f"{base_name}_{judge_llm_name}_{judge_api_name}_{judge_harm_name}.json"
    )
    
    result_list = [dict(item) for item in result_dicts.values()]
    with open(output_file_path, "w", encoding="utf-8") as f:
        json.dump(result_list, f, ensure_ascii=False, indent=4)

    print(f"\n[INFO] Evaluation complete. Results saved to: {output_file_path}")
    print(f"[INFO] Total samples evaluated: {len(result_list)}")
    print(f"[INFO] Number of invalid samples: {number_of_invalid_samples}")

    num_results = len(result_list)
    if num_results == 0:
        print("[WARN] No results to summarize; skipping metrics calculation.")
        return output_file_path, result_list, 0.0, {}

    # Calculate metrics
    asr = sum([item.get("judge_success_api", 0) for item in result_list]) / num_results
    print(f"[INFO] ASR: {asr}")

    total_input_tokens = sum([item.get("input_tokens", 0) for item in result_list])
    total_output_tokens = sum([item.get("output_tokens", 0) for item in result_list])
    total_reasoning_tokens = sum([item.get("reasoning_tokens", 0) for item in result_list])
    total_price = sum([item.get("price", 0) for item in result_list])
    total_judge_price = sum([item.get("judge_api_price", 0) for item in result_list])

    print(f"[INFO] Total Input Tokens: {total_input_tokens}")
    print(f"[INFO] Total Output Tokens: {total_output_tokens}")
    print(f"[INFO] Total Reasoning Tokens: {total_reasoning_tokens}")
    print(f"[INFO] Total Price: ${total_price:.4f}")
    print(f"[INFO] Total Judge Price: ${total_judge_price:.4f}")


    metrics_dict = {
        "asr": asr,
        "num_samples": num_results,
        "judge_api": judge_api_name,
        "input_tokens": total_input_tokens // num_results,
        "output_tokens": total_output_tokens // num_results,
        "reasoning_tokens": total_reasoning_tokens // num_results,
        "price": total_price / num_results,
        "judge_api_price": total_judge_price / num_results,
        "invalid_samples": number_of_invalid_samples
    }

    return output_file_path, result_list, asr, metrics_dict


def update_global_results(args, global_results_path, metrics_dict):
    """
    Update the global results JSON file.
    
    Args:
        args: Command line arguments
        output_file_path: Path to the output file
        metrics_dict: Dictionary containing metrics
    """
    # global_results_path = "final_paper_results/global_results.json"

    parent_dir = os.path.dirname(global_results_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    if os.path.exists(global_results_path):
        with open(global_results_path, "r", encoding="utf-8") as f:
            global_data = json.load(f)
    else:
        global_data = {}
    
    result_key = args.model_name
    
    if args.attack_type not in global_data:
        global_data[args.attack_type] = {}
        print(f"[INFO] Added new attack method to global results: {args.attack_type}")

    if result_key not in global_data[args.attack_type]:
        global_data[args.attack_type][result_key] = {}
        print(f"[INFO] Added new result key to global results: {result_key}")
        
    # Update the global results
    global_data[args.attack_type][result_key].update(metrics_dict)

    with open(global_results_path, "w", encoding="utf-8") as f:
        json.dump(global_data, f, ensure_ascii=False, indent=4, sort_keys=True)
    print(f"[INFO] Global results updated in {global_results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Attack Evaluation")
    
    # GPU Configuration
    parser.add_argument('--gpus', type=int, nargs='+', default=[5, 6], help='GPU device IDs')
    parser.add_argument("--batch", type=int, default=32, help="Batch size for parallel processing")

    # Input/Output Directories
    # parser.add_argument("--attack_type", type=str, default="PosteriorAttack", help="Attack method name for file naming")
    parser.add_argument("--attack_type", type=str, default="PosteriorAttack", 
                        choices=["PosteriorAttack", "PosteriorAttack_No_SystemPrompt", "DeepInception", "ManyShot", "SelfCipher", "CodeChameleon", "ArtPrompt", "PromptAttack", "FlipAttack", "GCG", "GCG_1", "AutoDAN"],
                       help="Type of attack model to use")
    
    parser.add_argument("--model_name", type=str, default="gpt-4-0613", help="Model name for evaluation (used in file naming)")

    parser.add_argument("--result_file", type=str, default="final_result.json", help="PosteriorAttack results file")
    parser.add_argument("--checkpoint_file", type=str, default=None, help="Checkpoint file for resuming evaluation")
    parser.add_argument("--output_dir", type=str, default="result", help="Directory containing PosteriorAttack results")
    parser.add_argument("--final_result_dir", type=str, default="final_result", help="Directory for final evaluation results")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoint", help="Directory for checkpoint files")
    parser.add_argument("--summary_dir", type=str, default="summary.json", help="Directory for summary files")

    # Evaluation Judges
    parser.add_argument("--judge_dict", type=str, default="Enabled", help="Enable dictionary-based evaluation (set to 'enabled')")
    parser.add_argument("--judge_llm", type=str, default=None, help="Judge LLM model path")
    parser.add_argument("--judge_api", type=str, default=None, help="Judge API (e.g., GPT-4, Gemini)")
    parser.add_argument("--judge_harm_bench", type=str, default=None, help="HarmBench classifier model path")

    args = parser.parse_args()

    # Configure GPU environment
    gpu_list = ','.join(map(str, args.gpus))
    os.environ.update({
        'CUDA_DEVICE_ORDER': 'PCI_BUS_ID',
        'CUDA_VISIBLE_DEVICES': gpu_list,
        'WORLD_SIZE': '1'
    })
    print(f"[INFO] Using GPUs: {gpu_list}")

    # Extract judge names for file naming
    judge_llm_name = extract_name(args.judge_llm, "no_llm")
    judge_api_name = extract_name(args.judge_api, "no_api")
    judge_harm_name = extract_name(args.judge_harm_bench, "no_harm")

    # Load results data
    result_file_path = os.path.join(args.output_dir, args.result_file)
    
    # Determine checkpoint file path
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    if args.checkpoint_file:
        checkpoint_file_path = os.path.join(args.checkpoint_dir, args.checkpoint_file)
    else:
        base_name = args.result_file.replace('.json', '')
        checkpoint_file_path = os.path.join(
            args.checkpoint_dir, 
            f"{base_name}_{judge_llm_name}_{judge_api_name}_{judge_harm_name}.json"
        )

    # Load results
    result_dicts = load_results(result_file_path, checkpoint_file_path)
    
    # Sync checkpoint with output folder if newer data exists
    result_dicts, num_synced = sync_checkpoint_with_output_folder(result_dicts, result_file_path)

    
    if num_synced > 0:
        # Save updated checkpoint after sync
        with open(checkpoint_file_path, "w", encoding="utf-8") as f:
            json.dump(result_dicts, f, ensure_ascii=False, indent=4)
        print(f"[INFO] Updated checkpoint saved to: {checkpoint_file_path}")
    
    # Validate sample outputs
    number_of_invalid_samples = validate_sample_outputs(result_dicts)
    
    # Skip completed evaluations
    skip_completed_evaluations(args, result_dicts)
    
    # Initialize evaluator with configured judges
    evaluator = Evaluator(
        judge_llm=args.judge_llm,
        judge_api=args.judge_api,
        judge_harm_bench=args.judge_harm_bench,
        tensor_parallel_size=len(args.gpus)
    )
    print(f"[INFO] Evaluator initialized | Batch size: {args.batch} | Samples: {len(result_dicts)}")

    # Run evaluation pipeline
    result_dicts = run_evaluation_pipeline(evaluator, result_dicts, checkpoint_file_path, args)
    
    # Save results and calculate metrics
    output_file_path, result_list, asr, metrics_dict = save_results_and_calculate_metrics(
        result_dicts, args, judge_llm_name, judge_api_name, judge_harm_name, number_of_invalid_samples
    )
    
    # Update global results
    if metrics_dict:
        global_results_path = args.summary_dir
        update_global_results(args, global_results_path, metrics_dict)