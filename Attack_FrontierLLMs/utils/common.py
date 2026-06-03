import os
import json
import argparse
import asyncio
import math
import sys
import pandas as pd


# ========== Helper Functions for main_api ==========


def extract_generation(text: str, attack_model) -> str:
    """
    Extract the first occurrence of text inside square brackets.
    If no such pair exists, return the original text unchanged.
    """
    if attack_model.__class__.__name__ == "PosteriorAttack":
        text = attack_model.decode_response(text)
        return text
    if attack_model.__class__.__name__ == "CipherAttack":
        text = attack_model.decode_response(text)
        return text
    return text





def load_dataset(file_path):
    """
    Load dataset from CSV file and return rows as dictionaries.
    """
    return pd.read_csv(file_path, encoding='utf-8').to_dict(orient='records')



def load_checkpoint(checkpoint_file_path):
    """Load checkpoint if exists, otherwise return empty dict."""
    if os.path.exists(checkpoint_file_path):
        print(f"[INFO] Resuming from checkpoint: {checkpoint_file_path}")
        with open(checkpoint_file_path, "r", encoding="utf-8") as f:
            result_dicts = json.load(f)
        
        # Convert string keys to integers if needed
        if isinstance(result_dicts, dict) and result_dicts:
            first_key = next(iter(result_dicts.keys()))
            if isinstance(first_key, str) and first_key.isdigit():
                result_dicts = {int(k): v for k, v in result_dicts.items()}
        
        num_completed = sum(1 for item in result_dicts.values() if item.get("output") is not None)
        print(f"[INFO] Loaded checkpoint with {num_completed}/{len(result_dicts)} samples completed")
        return result_dicts
    else:
        print(f"[INFO] No checkpoint found, starting fresh")
        return {}


def save_checkpoint(result_dicts, checkpoint_file_path):
    """Save current results to checkpoint file."""
    with open(checkpoint_file_path, "w", encoding="utf-8") as f:
        json.dump(result_dicts, f, ensure_ascii=False, indent=4)
    print(f"[INFO] Checkpoint saved to: {checkpoint_file_path}")

# ========== Helper Functions for main_eval ==========


def load_results(result_file_path, checkpoint_file_path):
    """
    Load results from checkpoint or original result file.
    If checkpoint is shorter, fill missing entries from original file.

    Args:
        result_file_path: Path to the original result file
        checkpoint_file_path: Path to the checkpoint file

    Returns:
        dict: Result dictionaries with integer keys
    """

    # Load original results (always needed for backfilling)
    with open(result_file_path, "r", encoding="utf-8") as f:
        full_results = json.load(f)

    # Normalize keys to int
    if isinstance(full_results, dict):
        full_results = {int(k): v for k, v in full_results.items()}

    if os.path.exists(checkpoint_file_path):
        print(f"[INFO] Resuming from checkpoint: {checkpoint_file_path}")
        with open(checkpoint_file_path, "r", encoding="utf-8") as f:
            result_dicts = json.load(f)

        # Normalize keys
        if isinstance(result_dicts, dict):
            result_dicts = {int(k): v for k, v in result_dicts.items()}

        print(f"[INFO] Loaded {len(result_dicts)} entries from checkpoint")
    else:
        print(f"[INFO] No checkpoint found. Using full results.")
        result_dicts = full_results
        print(f"[INFO] Loaded {len(result_dicts)} entries")

    # --- Fill missing entries ---
    if len(result_dicts) < len(full_results):
        print("[INFO] Filling missing entries from original results...")

        for idx in range(len(full_results)):
            if idx not in result_dicts:
                result_dicts[idx] = full_results[idx]

        print(f"[INFO] After filling: {len(result_dicts)} entries")

    return result_dicts


def sync_checkpoint_with_output_folder(result_dicts, output_file_path):
    """
    Synchronize checkpoint data with output folder.
    If 'original_output' or 'finish_reason' differs, replace entire checkpoint entry with output data.

    Args:
        result_dicts: Dictionary of results from checkpoint
        output_file_path: Path to the output folder result file

    Returns:
        tuple: (updated_result_dicts, num_synced_samples)
    """
    if not os.path.exists(output_file_path):
        print(f"[INFO] Output file not found: {output_file_path}. Skipping sync.")
        return result_dicts, 0

    try:
        with open(output_file_path, "r", encoding="utf-8") as f:
            output_results = json.load(f)

        # Normalize keys to int
        if isinstance(output_results, dict):
            output_results = {int(k): v for k, v in output_results.items()}
        elif isinstance(output_results, list):
            output_results = {int(item.get("id", i)): item for i, item in enumerate(output_results)}
    except Exception as e:
        print(f"[WARNING] Failed to load output file: {e}. Skipping sync.")
        return result_dicts, 0

    num_synced = 0
    fields_to_check = ["original_output", "finish_reason"]

    for idx, checkpoint_data in list(result_dicts.items()):
        if idx in output_results:
            output_data = output_results[idx]

            # Check if any field differs
            has_difference = False
            for field in fields_to_check:
                checkpoint_value = checkpoint_data.get(field)
                output_value = output_data.get(field)

                if output_value is not None and checkpoint_value != output_value:
                    has_difference = True
                    break

            # If difference found, replace entire checkpoint entry with output data
            if has_difference:
                result_dicts[idx] = output_data
                print(f"[INFO] Sample {idx}: Replaced entire entry from output folder data")
                num_synced += 1

    if num_synced > 0:
        print(f"[INFO] Synced {num_synced} sample(s) from output folder to checkpoint")

    return result_dicts, num_synced


def validate_sample_outputs(result_dicts):
    """
    Check sample outputs for errors and mark invalid ones.

    Args:
        result_dicts: Dictionary of results to validate
    """
    print("\n[INFO] Checking sample outputs for errors before evaluation...")

    number_of_invalid_samples = 0
    for idx, info in result_dicts.items():
        if info.get("finish_reason") != "stop" and info.get("finish_reason") != "length":
            print(f"[ERROR] Sample {idx} has finish reason: {info.get('finish_reason')}")
            print(f"Goal: {info['goal'][:100]}...")
            info["judge_success_dict"] = 0
            info['judge_score_api'] = 0
            info['judge_success_api'] = 0
            info['output'] = "I can't provide a valid output for this sample."
            number_of_invalid_samples += 1

        if info.get("output") == "":
            print(f"[ERROR] Sample {idx} has empty output.")
            info["judge_success_dict"] = 0
            info['judge_score_api'] = 0
            info['judge_success_api'] = 0
            info['output'] = "I can't provide a valid output for this sample."

    print("[INFO] All sample outputs appear valid. Starting evaluation...")
    return number_of_invalid_samples
