import os
import json
import argparse
import math
import time
from dotenv import load_dotenv


from utils.prompt_utils import (
    GPT_OSS_NO_THINKING_PROMPT,
)


from utils.common import extract_generation, load_dataset, load_checkpoint, save_checkpoint


load_dotenv()
access_token = os.getenv("HF_ACCESS_TOKEN")

os.environ["TORCHINDUCTOR_DISABLE"] = "1"

MODEL_TO_PROMPT_TEMPLATE = {
    "openai/gpt-oss-20b": GPT_OSS_NO_THINKING_PROMPT,
    "openai/gpt-oss-120b": GPT_OSS_NO_THINKING_PROMPT,
}


def get_prompt_template(victim_llm):
    """Select prompt template for the victim model."""
    model_name = victim_llm.lower()
    if model_name in MODEL_TO_PROMPT_TEMPLATE:
        return MODEL_TO_PROMPT_TEMPLATE[model_name]
    raise ValueError(f"Unsupported model: {victim_llm}. Please add prompt template.")


def build_prompt(tokenizer, messages, model_name, prompt_template):
    """
    Format prompt using prompt_template when attack returns (system,user),
    otherwise fall back to tokenizer chat template for multi-turn attacks.
    """
    if (
        len(messages) == 2
        and messages[0].get("role") == "system"
        and messages[1].get("role") == "user"
    ):
        return prompt_template["prompt"].format(
            system_msg=messages[0]["content"],
            user_msg=messages[1]["content"],
        )

    template_kwargs = {"tokenize": False, "add_generation_prompt": True}
    if "qwen3" in model_name.lower():
        template_kwargs["enable_thinking"] = False
    return tokenizer.apply_chat_template(messages, **template_kwargs)


def process_attack_batch(
    batch_start, batch_end, batch_idx, total_batches,
    result_dicts, harmful_behaviors, attack_model,
    llm, tokenizer, sampling_params, lora_request,
    model_name, prompt_template, args, checkpoint_file_path
):
    """Generate attacks and victim LLM responses for a single batch."""
    print(f"\n[INFO] Processing batch {batch_idx}/{total_batches} (samples {batch_start}-{batch_end})")

    prompts = []
    batch_indices = []

    for idx in range(batch_start, batch_end):
        if idx in result_dicts and result_dicts[idx].get("output") is not None:
            print(f"[INFO] Sample {idx} already completed, skipping")
            continue

        harm_prompt = harmful_behaviors[idx]['goal']

        if args.attack_type == "ArtPrompt":
            attack = attack_model.generate(
                harmful_behaviors[idx]['goal'],
                harmful_behaviors[idx]['masked words'],
                harmful_behaviors[idx]['masked instruction'],
            )
        elif args.attack_type == "AutoDAN":
            attack = attack_model.generate(
                harmful_behaviors[idx]['goal'],
                harmful_behaviors[idx]['autodan_prompt'],
            )
        else:
            attack = attack_model.generate(harm_prompt)

        formatted_prompt = build_prompt(tokenizer, attack, model_name, prompt_template)

        if idx not in result_dicts:
            result_dicts[idx] = {
                "id": idx,
                "goal": harm_prompt,
                "attack": [attack[0], attack[-1]] if len(attack) > 3 else attack,
                "formatted_prompt": formatted_prompt,
                "original_output": None,
                "output": None,
            }

        prompts.append(formatted_prompt)
        batch_indices.append(idx)

    if not prompts:
        print(f"[INFO] Batch {batch_idx}/{total_batches}: All samples already completed, skipping")
        return result_dicts

    print(f"[INFO] Generating {len(prompts)} responses via vLLM...")
    try:
        generate_kwargs = {"prompts": prompts, "sampling_params": sampling_params}
        if lora_request is not None:
            generate_kwargs["lora_request"] = lora_request

        responses = llm.generate(**generate_kwargs)

        for i, response in enumerate(responses):
            idx = batch_indices[i]
            output_text = response.outputs[0].text
            finish_reason = response.outputs[0].finish_reason

            result_dicts[idx]["original_output"] = output_text
            result_dicts[idx]["reasoning"] = None
            result_dicts[idx]["finish_reason"] = finish_reason
            result_dicts[idx]["price"] = 0
            result_dicts[idx]["input_tokens"] = len(response.prompt_token_ids) if response.prompt_token_ids else None
            result_dicts[idx]["output_tokens"] = len(response.outputs[0].token_ids)
            result_dicts[idx]["reasoning_tokens"] = 0

            try:
                decoded = extract_generation(output_text, attack_model)
            except Exception as e:
                print(f"[ERROR] Failed to extract generation for sample {idx}: {e}")
                decoded = None

            result_dicts[idx]["output"] = decoded

            if batch_idx == 1 and i == 0:
                print(f"\n[EXAMPLE] Sample {idx}:")
                print(f"Goal: {result_dicts[idx]['goal'][:100]}...")
                print(f"Response: {decoded[:200] if decoded else 'None'}...")

        save_checkpoint(result_dicts, checkpoint_file_path)
        print(f"[INFO] Batch {batch_idx}/{total_batches} completed. Checkpoint saved.")

    except Exception as e:
        print(f"[ERROR] Batch {batch_idx} failed with error: {e}")
        print(f"[INFO] Saving checkpoint before exit...")
        save_checkpoint(result_dicts, checkpoint_file_path)
        print(f"[INFO] Checkpoint saved to: {checkpoint_file_path}")
        raise

    return result_dicts


# ========== Main ==========

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Attack Generation Pipeline (vLLM)")

    # GPU / vLLM configuration
    parser.add_argument('--gpus', type=int, nargs='+', default=[5, 6], help='GPU device IDs')
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.8, help='GPU memory utilization for vLLM')
    parser.add_argument('--max_model_len', type=int, default=None, help='Maximum model sequence length (None = model default)')

    # Model configuration
    parser.add_argument("--victim_llm", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Victim LLM model path")
    parser.add_argument("--temperature", type=float, default=0, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p sampling parameter")
    parser.add_argument("--max_tokens", type=int, default=2048, help="Maximum tokens for response")
    parser.add_argument("--batch", type=int, default=32, help="Batch size for parallel processing")

    parser.add_argument(
        "--attack_type", type=str, default="PosteriorAttack",
        choices=["PosteriorAttack", "PosteriorAttack_No_SystemPrompt", "DeepInception", "ManyShot", "SelfCipher",
                 "CodeChameleon", "ArtPrompt", "PromptAttack", "FlipAttack", "GCG", "GCG_1", "AutoDAN"],
        help="Type of attack model to use",
    )

    # Fine-tuned (PEFT) model
    parser.add_argument('--saved_peft_model', type=str, default='None', help='PEFT model name (set "None" to disable)')
    parser.add_argument('--finetuned_path', type=str, default='finetuned_models', help='Directory containing PEFT models')

    # Dataset
    parser.add_argument("--data_path", type=str, default="data/harmful_behaviors.csv", help="Path to harmful behaviors dataset")
    parser.add_argument("--begin", type=int, default=0, help="Start index for evaluation")
    parser.add_argument("--end", type=int, default=519, help="End index for evaluation")
    parser.add_argument("--output_dir", type=str, default="paper_results", help="Output directory for results")

    # Checkpoint
    parser.add_argument("--checkpoint_file", type=str, default=None, help="Checkpoint filename for resuming")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoint", help="Directory for checkpoint files")

    args = parser.parse_args()

    # Authenticate with Hugging Face
    if access_token:
        from huggingface_hub import login
        login(token=access_token)

    # Configure GPU environment BEFORE importing vllm/torch
    gpu_list = ','.join(map(str, args.gpus))
    os.environ.update({
        'CUDA_DEVICE_ORDER': 'PCI_BUS_ID',
        'CUDA_VISIBLE_DEVICES': gpu_list,
        'WORLD_SIZE': '1',
    })
    print(f"[INFO] Using GPUs: {gpu_list}")

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from transformers import AutoTokenizer

    # Load dataset
    harmful_behaviors = load_dataset(args.data_path)
    args.end = min(args.end, len(harmful_behaviors))
    print(f"[INFO] Loaded {len(harmful_behaviors)} behaviors, evaluating indices {args.begin}-{args.end}")

    # Checkpoint paths
    dataset_name = os.path.splitext(os.path.basename(args.data_path))[0]
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    if args.checkpoint_file:
        checkpoint_file_path = os.path.join(args.checkpoint_dir, args.checkpoint_file)
    else:
        checkpoint_file_path = os.path.join(
            args.checkpoint_dir,
            f"{args.attack_type}-{dataset_name}-{args.begin}_{args.end}.json",
        )

    result_dicts = load_checkpoint(checkpoint_file_path)

    print("[INFO] Using base model without LoRA adapter")
    lora_request = None

    # vLLM sampling params
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    # Initialize vLLM
    llm_config = {
        'model': args.victim_llm,
        'tensor_parallel_size': len(args.gpus),
        'gpu_memory_utilization': args.gpu_memory_utilization,
    }
    if args.max_model_len is not None:
        llm_config['max_model_len'] = args.max_model_len

    llm = LLM(**llm_config)
    tokenizer = AutoTokenizer.from_pretrained(args.victim_llm, trust_remote_code=True)
    print(f"[INFO] Initialized LLM: {args.victim_llm}")

    model_name = args.victim_llm.split('/')[-1]
    prompt_template = get_prompt_template(args.victim_llm)
    print(f"[INFO] Using prompt template for: {args.victim_llm}")

    # Initialize attack model
    print(f"[INFO] Initializing {args.attack_type} model...")

    if args.attack_type == "PosteriorAttack":
        from attack_utils.posteriorattack import PosteriorAttack
        attack_model = PosteriorAttack(victim_llm=args.victim_llm, prompt_style=1)
    elif args.attack_type == "PosteriorAttack_No_SystemPrompt":
        from attack_utils.posteriorattack import PosteriorAttack
        attack_model = PosteriorAttack(victim_llm=args.victim_llm, prompt_style=1)
    elif args.attack_type == "DeepInception":
        from attack_utils.deep_inception import DeepInception
        attack_model = DeepInception(victim_llm=args.victim_llm)
    elif args.attack_type == "ManyShot":
        from attack_utils.many_shot import ManyShot
        attack_model = ManyShot(victim_llm=args.victim_llm)
    elif args.attack_type == "FlipAttack":
        from attack_utils.flip_attack import FlipAttack
        attack_model = FlipAttack(
            victim_llm=args.victim_llm,
            flip_mode="FCW",
            cot=True,
            lang_gpt=True,
            few_shot=False,
        )
    elif args.attack_type == "PromptAttack":
        from attack_utils.prompt_attack import PromptAttack
        attack_model = PromptAttack(victim_llm=args.victim_llm)
    elif args.attack_type == "SelfCipher":
        from attack_utils.cipher_attack import CipherAttack
        attack_model = CipherAttack(
            encode_method="caesar",
            instruction_type="Crimes_And_Illegal_Activities",
            demonstration_toxicity="toxic",
            language="en",
            use_system_role=True,
            use_demonstrations=True,
        )
    elif args.attack_type == "CodeChameleon":
        from attack_utils.code_chameleon import CodeChameleon
        attack_model = CodeChameleon(encrypt_rule='binary_tree', prompt_style='code')
    elif args.attack_type == "ArtPrompt":
        from attack_utils.art_prompt import ArtPrompt
        attack_model = ArtPrompt()
    elif args.attack_type == "GCG":
        from attack_utils.gcg import GCGAttack
        attack_model = GCGAttack()
    elif args.attack_type == "AutoDAN":
        from attack_utils.autodan_attack import AutoDAN
        attack_model = AutoDAN()
    else:
        raise ValueError(f"Unsupported attack type: {args.attack_type}")

    # Process in batches
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
            llm=llm,
            tokenizer=tokenizer,
            sampling_params=sampling_params,
            lora_request=lora_request,
            model_name=model_name,
            prompt_template=prompt_template,
            args=args,
            checkpoint_file_path=checkpoint_file_path,
        )
    del llm
    import gc
    gc.collect()

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
