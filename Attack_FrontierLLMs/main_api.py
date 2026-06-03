import os
import json
import argparse
import asyncio
import math
import sys
import time
from dotenv import load_dotenv
import pandas as pd

from utils.call_OpenRouterAPI import batch_chat as openrouter_batch_chat
from utils.call_AnthropicAPI import batch_chat as anthropic_batch_chat

from utils.common import extract_generation, load_dataset, load_checkpoint, save_checkpoint


def get_batch_chat(provider):
    """Return the batch_chat coroutine for the selected API provider."""
    if provider == "openrouter":
        return openrouter_batch_chat
    if provider == "anthropic":
        return anthropic_batch_chat
    raise ValueError(f"Unsupported api_provider: {provider}")

load_dotenv()
access_token = os.getenv("HF_ACCESS_TOKEN")



os.environ["TORCHINDUCTOR_DISABLE"] = "1"


def process_attack_batch(
    batch_start, batch_end, batch_idx, total_batches,
    result_dicts, harmful_behaviors, attack_model,
    victim_llm, temperature, top_p, max_tokens,
    checkpoint_file_path, api_provider
):
    """
    Process a single batch of attack generation and victim LLM responses.
    
    Returns:
        result_dicts: Updated results dictionary with new outputs
    """
    print(f"\n[INFO] Processing batch {batch_idx}/{total_batches} (samples {batch_start}-{batch_end})")
    
    prompts = []
    batch_indices = []
    # Generate attacks for current batch (skip already completed samples)
    for idx in range(batch_start, batch_end):
        # Skip only if processed AND output captured (partial entries should retry)
        # Need to redo if they have reasoning, we are evaluating non-reasoning only samples
        if idx in result_dicts:
            print(f"[INFO] Sample {idx} already completed, skipping")
            continue
        
        harm_prompt = harmful_behaviors[idx]['goal']

        attack = attack_model.generate(harm_prompt)
        
        # Initialize result entry if not exists
        if idx not in result_dicts:
            result_dicts[idx] = {
                "id": idx,
                "goal": harm_prompt,
                "attack": [attack[0], attack[-1]] if len(attack) > 3 else attack, # Store truncated attack for quick reference,
                "original_output": None,
                "output": None
            }
        
        prompts.append(attack)
        batch_indices.append(idx)
    
    # Generate responses using victim LLM
    if prompts:
        print(f"[INFO] Generating {len(prompts)} responses via API...")
        try:
            print(f"[INFO] Using args.reasoning_effort: {args.reasoning_effort} (provider: {api_provider})")
            if api_provider == "anthropic":
                # Anthropic uses extended thinking; pass effort directly and let the client map it.
                reasoning_kwargs = {"reasoning_effort": args.reasoning_effort}
            else:
                reasoning_kwargs = {"reasoning": {"effort": args.reasoning_effort}}
                if args.reasoning_effort == "none" :
                    if victim_llm in ["openai/gpt-oss-20b", "openai/gpt-oss-120b", "openai/gpt-5"]:
                        print(f"[WARNING] Model {victim_llm} on Openrouter does not support disabling reasoning. For gpt oss model, using local model to force no reasoning")

                    # For OpenRouter, we need to disable reasoning explicitly if effort is none.
                    reasoning_kwargs = {"reasoning": {"effort": args.reasoning_effort, "enabled": False}}
            batch_chat = get_batch_chat(api_provider)
            info_responses = asyncio.run(
                batch_chat(prompts, model=victim_llm,
                          temperature=temperature, top_p=top_p, max_tokens=max_tokens, **reasoning_kwargs)
            )
            
            
            # Store results
            for i, info in enumerate(info_responses):
                idx = batch_indices[i]
                result_dicts[idx]["original_output"] = info.get("output")
                result_dicts[idx]["reasoning"] = info.get("reasoning")
                result_dicts[idx]["finish_reason"] = info.get("finish_reason")
                result_dicts[idx]["price"] = info.get("price")
                result_dicts[idx]["input_tokens"] = info.get("input_tokens")
                result_dicts[idx]["output_tokens"] = info.get("output_tokens")
                result_dicts[idx]["reasoning_tokens"] = info.get("reasoning_tokens")

                # Extract generation wrapper
                try:
                    response = extract_generation(info["output"], attack_model)
                except Exception as e:
                    print(f"[ERROR] Failed to extract generation for sample {idx}: {e}")
                    response = None

                result_dicts[idx]["output"] = response
                
                if batch_idx == 1 and i == 0:  # Print first sample as example
                    print(f"\n[EXAMPLE] Sample {idx}:")
                    print(f"Goal: {result_dicts[idx]['goal'][:100]}...")
                    print(f"Response: {response[:200] if response else 'None'}...")
                    
                # if info.get("finish_reason") != "stop":
                #     print(f"[WARNING] Sample {idx} finish reason: {info.get('finish_reason')}")
                #     sys.exit(1)
            
            # Save checkpoint after successful batch
            save_checkpoint(result_dicts, checkpoint_file_path)
            print(f"[INFO] Batch {batch_idx}/{total_batches} completed. Checkpoint saved.")
            print(f"[INFO] Sleeping for 20 seconds to avoid rate limits...")
            time.sleep(20)  # Sleep briefly to avoid hitting rate limits
            
            
        except Exception as e:
            print(f"[ERROR] Batch {batch_idx} failed with error: {e}")
            print(f"[INFO] Saving checkpoint before exit...")
            save_checkpoint(result_dicts, checkpoint_file_path)
            print(f"[INFO] Checkpoint saved to: {checkpoint_file_path}")
            raise
    else:
        print(f"[INFO] Batch {batch_idx}/{total_batches}: All samples already completed, skipping")
    
    return result_dicts


# ========== Main Function ==========

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Attack Generation Pipeline")
    
    # Model Configuration
    parser.add_argument("--victim_llm", type=str, default="Qwen/Qwen2.5-7B-Instruct", 
                       help="Victim LLM model path")
    parser.add_argument("--temperature", type=float, default=0, 
                       help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=1.0, 
                       help="Top-p sampling parameter")
    parser.add_argument("--max_tokens", type=int, default=2048, 
                       help="Maximum tokens for response")
    parser.add_argument("--batch", type=int, default=32, 
                       help="Batch size for parallel processing")

    parser.add_argument("--attack_type", type=str, default="PosteriorAttack", 
                        choices=["PosteriorAttack", "PosteriorAttack_No_SystemPrompt", "DeepInception", "ManyShot", "SelfCipher", "CodeChameleon", "ArtPrompt", "PromptAttack", "FlipAttack", "GCG", "AutoDAN"],
                       help="Type of attack model to use")
    
    parser.add_argument("--reasoning_effort", type=str, default="low",
                       choices=["none","minimal","low", "medium", "high", "xhigh"],
                       help="Reasoning effort level for victim LLM (if supported)")

    parser.add_argument("--api_provider", type=str, default="openrouter",
                       choices=["openrouter", "anthropic"],
                       help="API provider for victim LLM inference. "
                            "'openrouter' routes through OpenRouter (OPENROUTER_API_KEY); "
                            "'anthropic' calls the Anthropic Messages API directly (ANTHROPIC_API_KEY).")
    
    # Dataset Configuration
    parser.add_argument("--data_path", type=str, default="data/harmful_behaviors.csv", 
                       help="Path to harmful behaviors dataset")
    parser.add_argument("--begin", type=int, default=0, 
                       help="Start index for evaluation")
    parser.add_argument("--end", type=int, default=519, 
                       help="End index for evaluation")
    parser.add_argument("--output_dir", type=str, default="posteriorattack", 
                       help="Output directory for results")
    
    # Checkpoint Configuration
    parser.add_argument("--checkpoint_file", type=str, default=None, 
                       help="Checkpoint file for resuming generation")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoint", 
                       help="Directory for checkpoint files")
    
    args = parser.parse_args()
    
    # Load harmful behaviors dataset
    harmful_behaviors = load_dataset(args.data_path)
    args.end = min(args.end, len(harmful_behaviors))
    print(f"[INFO] Loaded {len(harmful_behaviors)} behaviors, evaluating indices {args.begin}-{args.end}")
    
    # Setup checkpoint paths
    dataset_name = os.path.splitext(os.path.basename(args.data_path))[0]
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    if args.checkpoint_file:
        checkpoint_file_path = os.path.join(args.checkpoint_dir, args.checkpoint_file)
    else:
        checkpoint_file_path = os.path.join(
            args.checkpoint_dir,
            f"{args.attack_type}-{dataset_name}-{args.begin}_{args.end}.json"
        )
    
    # Load checkpoint if exists
    result_dicts = load_checkpoint(checkpoint_file_path)
    
    # Initialize attack model
    
    print(f"[INFO] Initializing {args.attack_type} model...")


    if args.attack_type == "PosteriorAttack":
        from utils.posteriorattack import PosteriorAttack
        attack_model = PosteriorAttack(
            victim_llm=args.victim_llm
        )
    else:
        raise ValueError(f"Unsupported attack type: {args.attack_type}")
    
    
    # Process attacks in batches
    total_batches = math.ceil((args.end - args.begin) / args.batch)
    print(f"[INFO] Processing {total_batches} batches with batch size {args.batch}")
    
    for batch_idx, batch_start in enumerate(range(args.begin, args.end, args.batch), start=1):
        batch_end = min(batch_start + args.batch, args.end)
        
        result_dicts = process_attack_batch(
            batch_start=batch_start,
            batch_end=batch_end,
            batch_idx=batch_idx,
            total_batches=total_batches,
            result_dicts=result_dicts,
            harmful_behaviors=harmful_behaviors,
            attack_model=attack_model,
            victim_llm=args.victim_llm,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            checkpoint_file_path=checkpoint_file_path,
            api_provider=args.api_provider,
        )
    
    # Save final results
    output_filename = f"{args.attack_type}-{dataset_name}-{args.begin}_{args.end}.json"
    output_path = os.path.join(args.output_dir, output_filename)
    
    os.makedirs(args.output_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result_dicts, f, ensure_ascii=False, indent=4)
    
    num_completed = sum(1 for item in result_dicts.values() if item.get("output") is not None)
    print(f"\n[INFO] ============ COMPLETION SUMMARY ============")
    print(f"[INFO] Final results saved to: {output_path}")
    print(f"[INFO] Total samples processed: {num_completed}/{len(result_dicts)}")
    print(f"[INFO] Checkpoint file: {checkpoint_file_path}")

