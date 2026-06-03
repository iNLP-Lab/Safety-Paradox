import os
import json
import pandas
import argparse
from tqdm import tqdm
from dotenv import load_dotenv
from transformers import AutoProcessor
load_dotenv()
access_token = os.getenv("HF_ACCESS_TOKEN")

# Import important libraries
from posterior_attack import PosteriorAttack
os.environ["TORCHINDUCTOR_DISABLE"] = "1"

from utils.prompt_utils import (
    LLAMA2_PROMPT,
    QWEN_PROMPT,
    LLAMA3_PROMPT,
    GEMMA_PROMPT,
    GEMMA4_PROMPT_NO_THINKING,
    QWEN_PROMPT_NO_THINKING,
    MISTRAL_PROMPT,
    FALCON_PROMPT,
)

import triton_kernels_compat  # noqa: F401 — before vLLM; vendor triton_kernels if site-packages is incomplete

from vllm import LLM, SamplingParams


def extract_generation(text: str) -> str:
    """
    Extract the first occurrence of text inside square brackets.
    If no such pair exists, return the original text unchanged.
    """
    if text.find(r'\[') == -1:
        start = text.find('[')
        adjust = 1
    else:
        start = text.find(r'\[')
        adjust = 2
    if text.rfind(r'\]') == -1:
        end = text.rfind(']')
    else:
        end = text.rfind(r'\]')
    if start != -1 and end != -1 and end > start + 1:
        return text[start + adjust:end]
    return text


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PosteriorAttack Generation Pipeline")
    
    # GPU Configuration
    parser.add_argument('--gpus', type=int, nargs='+', default=[5], help='GPU device IDs')
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.8, help='GPU memory utilization for vLLM')
    parser.add_argument('--max_model_len', type=int, default=4096, help='Maximum model sequence length')
    
    # Model Configuration
    parser.add_argument("--victim_llm", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Victim LLM model path")
    parser.add_argument("--victim_name", type=str, default="Qwen2.5-7B-Instruct", help="Victim LLM model name")

    parser.add_argument("--temperature", type=float, default=0, help="Sampling temperature")
    parser.add_argument("--batch", type=int, default=32, help="Batch size for parallel processing")

    # Dataset Configuration
    parser.add_argument("--data_path", type=str, default="data/harmful_behaviors.csv", help="Path to harmful behaviors dataset")
    parser.add_argument("--begin", type=int, default=0, help="Start index for evaluation")
    parser.add_argument("--end", type=int, default=519, help="End index for evaluation")
    parser.add_argument("--output_dict", type=str, default="../reproduce_result", help="Output directory for results")
    parser.add_argument("--attack_type", type=str, default="PosteriorAttack", help="Attack type", choices=["PosteriorAttack", "Direct", "Both"])
    args = parser.parse_args()

    # Authenticate with Hugging Face
    # from huggingface_hub import login
    # login(token=access_token)

    # Configure GPU environment (preserve CUDA_VISIBLE_DEVICES if set by scheduler/parent)
    gpu_list = ','.join(map(str, args.gpus))
    env_update = {'CUDA_DEVICE_ORDER': 'PCI_BUS_ID', 'WORLD_SIZE': '1'}
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        env_update['CUDA_VISIBLE_DEVICES'] = gpu_list
    os.environ.update(env_update)
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', gpu_list)
    print(f"[INFO] Using GPUs: {gpu_list} (CUDA_VISIBLE_DEVICES={visible})")
    print(f"[INFO] Using full finetuned model: {args.victim_llm}")

    # Qwen3.6 / Qwen3_5 MoE: FlashInfer GDN prefill JIT often fails on CUDA 12.4 toolchains
    # (missing cuda::ptx tensormap_* in CCCL). vLLM reads this via EngineArgs / LLM(...),
    # not VLLM_GDN_PREFILL_BACKEND (that env is not wired in vLLM 0.19.x).
    _vl = args.victim_llm.lower()
    _qwen36_gdn_triton = "qwen3.6" in _vl

    # Initialize victim LLM with vLLM
    sampling_params = SamplingParams(
        temperature=args.temperature,
        # top_p=1.0,
        max_tokens=2048
    )

    # Gemma 2 needs tanh softcapping; prebuilt vLLM Flash Attention often doesn't support it.
    # enforce_eager uses a different execution path that may avoid the unsupported kernel.
    llm_kw = dict(
        model=args.victim_llm,
        tensor_parallel_size=len(args.gpus),
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )
    if "gemma-2" in args.victim_llm.lower():
        llm_kw["enforce_eager"] = True
        print("[INFO] Using enforce_eager=True for Gemma 2")
    if _qwen36_gdn_triton:
        llm_kw.setdefault("gdn_prefill_backend", "triton")
        print(
            "[INFO] Qwen3.6: gdn_prefill_backend="
            f"{llm_kw.get('gdn_prefill_backend')} "
            "(avoids FlashInfer GDN nvcc JIT on older CUDA; pass "
            "--gdn-prefill-backend flashinfer via LLM kwargs if you want FlashInfer)"
        )
    llm = LLM(**llm_kw)
    print(f"[INFO] Initialized LLM: {args.victim_llm}")

    # Load harmful behaviors dataset
    import csv
    
    def load_dataset(file_path):
        """Load dataset from CSV file."""
        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            return [row[0] for row in reader]
    
    harmful_behaviors = load_dataset(args.data_path)
    args.end = min(args.end, len(harmful_behaviors))
    print(f"[INFO] Loaded {len(harmful_behaviors)} behaviors, evaluating indices {args.begin}-{args.end}")

    # Select appropriate prompt template based on model
    attack_model = PosteriorAttack(victim_llm=args.victim_llm)
    
    model_name = args.victim_name.lower()
    if 'llama-2' in model_name:
        prompt_template = LLAMA2_PROMPT
    elif 'llama-3' in model_name:
        prompt_template = LLAMA3_PROMPT
    elif 'qwen2' in model_name:
        prompt_template = QWEN_PROMPT
    elif 'qwen3' in model_name:
        prompt_template = QWEN_PROMPT_NO_THINKING
    elif 'gemma-4' in model_name:
        # prompt_template = GEMMA4_PROMPT_NO_THINKING
        tokenizer = AutoProcessor.from_pretrained(args.victim_llm)
    elif 'gemma' in model_name:
        prompt_template = GEMMA_PROMPT
    elif 'mistral' in model_name or 'nemo' in model_name:
        prompt_template = MISTRAL_PROMPT
    elif 'falcon' in model_name:
        prompt_template = FALCON_PROMPT
    else:
        raise ValueError(f"Unsupported model: {args.victim_llm}. Please add prompt template.")
    
    print(f"[INFO] Using prompt template for: {model_name}")
    
    # Determine which attack types to run
    if args.attack_type == "Both":
        attack_types = ["PosteriorAttack", "Direct"]
    else:
        attack_types = [args.attack_type]
    
    for attack_type in attack_types:
        print(f"\n[INFO] Running attack type: {attack_type}")
        
        # Build all prompts for the current attack type (single batch)
        result_dicts = {}
        prompts = []
        indices = []
        
        print(f"[INFO] Building prompts for samples {args.begin}-{args.end}")
        for idx in range(args.begin, args.end):
            harm_prompt = harmful_behaviors[idx]
            if attack_type == "PosteriorAttack":
                posterior_attack = attack_model.generate(harm_prompt)
            else:
                posterior_attack = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": harm_prompt},
                ]
            
            if "gemma-4" in args.victim_llm.lower():
                formatted_prompt = tokenizer.apply_chat_template(
                    posterior_attack,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False
                )
            else:
                formatted_prompt = prompt_template["prompt"].format(
                    system_msg=posterior_attack[0]['content'],
                    user_msg=posterior_attack[1]['content']
                )
            
            result_dicts[idx] = {
                "id": idx,
                "goal": harm_prompt,
                "all_prompt": posterior_attack,
                "formatted_prompt": formatted_prompt,
                "original_output": None,
                "output": None
            }
            
            prompts.append(formatted_prompt)
            indices.append(idx)
        
        # Generate responses using victim LLM (one batch)
        if prompts:
            responses = llm.generate(prompts, sampling_params)
            
            # Store results
            for i, response in enumerate(responses):
                idx = indices[i]
                output_text = response.outputs[0].text
                result_dicts[idx]["original_output"] = output_text
                if attack_type == "PosteriorAttack":
                    output_text = extract_generation(output_text)
                    result_dicts[idx]["output"] = output_text
                else:
                    result_dicts[idx]["output"] = output_text
                
                if i == 0:  # Print first sample as example
                    print(f"\n[EXAMPLE] Sample {idx}:")
                    print(f"Prompt: {result_dicts[idx]['formatted_prompt'][:200]}...")
                    print(f"Response: {output_text[:200]}...")
        
        # Save results for the current attack type
        dataset_name = os.path.splitext(os.path.basename(args.data_path))[0]
        output_filename = f"{attack_type}-{dataset_name}-{args.begin}_{args.end}.json"
        output_path = os.path.join(args.output_dict, output_filename)
        
        os.makedirs(args.output_dict, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result_dicts, f, ensure_ascii=False, indent=4)
        
        print(f"\n[INFO] Results saved to: {output_path}")
        print(f"[INFO] Total samples processed for {attack_type}: {len(result_dicts)}")
