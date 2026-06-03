import importlib.util
import json
import os
import re

_compat_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "Posterior_Attack",
    "triton_kernels_compat.py",
)
if os.path.isfile(_compat_path):
    _spec = importlib.util.spec_from_file_location("triton_kernels_compat", _compat_path)
    if _spec and _spec.loader:
        _tc = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_tc)

from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

import csv
import numpy as np
import string

import argparse
import os

from prompt_utils import (
    LLAMA2_CLS_PROMPT,
    QWEN_CLS_PROMPT,
    LLAMA3_CLS_PROMPT,
    GEMMA_CLS_PROMPT,
    GEMMA4_CLS_PROMPT_NON_THINKING,
    QWEN_CLS_PROMPT_NON_THINKING,
    MISTRAL_CLS_PROMPT,
    FALCON_CLS_PROMPT,
)
os.environ["TORCHINDUCTOR_DISABLE"] = "1"

BOXED_PREFIX = "The final answer (Yes/No) is \\boxed{"
BOXED_TARGETS = ["\\boxed{Yes}", "\\boxed{No}", "\\boxed{yes}", "\\boxed{no}"]


def _has_boxed(text):
    """Strict check: only \\boxed{Yes} or \\boxed{No} count as having a boxed answer."""
    return any(t in text for t in BOXED_TARGETS)


def extract_from_boxed(text):
    """Extract Yes/No from \\boxed{...} in text. Returns (label, found) where label is 1 for Yes, 0 for No, -1 if unclear."""
    match = re.search(r'\\boxed\{([^}]*)\}?', text, re.IGNORECASE)
    if not match:
        return -1, False
    content = match.group(1).strip().lower()
    if content.startswith('yes') or content == 'y':
        return 1, True
    if content.startswith('no') or content == 'n':
        return 0, True
    return -1, True


def evaluate_batch(prompts, answers, start_idx=0):
    """
    Batch evaluation using vLLM for parallel inference.
    Returns per-sample results.
    If response lacks \\boxed{}, runs second pass with prefix "The final answer (Yes/No) is \\boxed{".
    """
    outputs = llm.generate(prompts, sampling_params)

    # Identify indices needing second pass (no explicit \boxed{Yes} or \boxed{No} in response)
    second_pass_indices = []
    for i, output in enumerate(outputs):
        prediction = output.outputs[0].text.strip()
        if not _has_boxed(prediction):
            second_pass_indices.append(i)

    # Second pass for responses without \boxed{}
    if second_pass_indices:
        second_pass_prompts = [prompts[i] + BOXED_PREFIX for i in second_pass_indices]
        second_outputs = llm.generate(second_pass_prompts, sampling_params_second_pass)
        for j, i in enumerate(second_pass_indices):
            # Model output is the completion; full text = prefix + completion
            completion = second_outputs[j].outputs[0].text.strip()
            outputs[i].outputs[0].text = BOXED_PREFIX + completion

    batch_results = []
    for i, output in enumerate(outputs):
        prediction = output.outputs[0].text.strip()
        o, found = extract_from_boxed(prediction)
        if not found:
            # Fallback: original word-based extraction
            prediction_lower = prediction.lower()
            for p in string.punctuation:
                prediction_lower = prediction_lower.replace(p, " ")
            words = prediction_lower.split()
            labels = ["yes", "no"]
            if len(words) == 0:
                o = -1
            elif labels[0] in words:
                o = 1
            elif labels[1] in words[0]:
                o = 0
            else:
                o = -1

        ans_label = answers[i]
        batch_results.append({
            "id": start_idx + i,
            "prompt": prompts[i],
            "prediction": prediction,
            "pred_label": o,
            "ans_label": ans_label,
            "correct": int(o == ans_label)
        })

    return batch_results


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation setting details")
    parser.add_argument('--gpus', type=int, nargs='+', default=[0], help='List of gpus to use')
    # Dataset path
    parser.add_argument('--data_path', type=str, default='data/text_behaviors_val_set.json', help='Dataset path')
    # Model Path
    parser.add_argument('--model', type=str, required=True, help='Full finetuned model path')
    parser.add_argument('--result_dir', type=str, default='results_new', help='Directory to save evaluation results')
    # Parameters for evaluation
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.8, help='GPU memory utilization for vLLM')
    parser.add_argument('--max_model_len', type=int, default=4096, help='Maximum model sequence length')
    parser.add_argument('--enforce_eager', action='store_true', help='Disable CUDA graphs (avoids OOM during graph capture, slower)')

    return parser.parse_args()


if __name__== "__main__":
    args = parse_args()
    
    GPU_list = ','.join(map(str, args.gpus))
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = GPU_list
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', GPU_list)
    print(f"Using GPU: {GPU_list} (CUDA_VISIBLE_DEVICES={visible})")

    model_path = args.model
    path_parts = [p for p in model_path.split('/') if p]
    if model_path.find("grpo") != -1:
        model_name = path_parts[-2]
    else:
        model_name = path_parts[-1]  # e.g. SAD
    base_model_hint = path_parts[-2] if len(path_parts) >= 2 else model_name  # e.g. Qwen2.5-3B-Instruct
    print(f"Using full finetuned model: {model_path}")
    print(f"Model name: {model_name}")
    print(f"Base model hint: {base_model_hint}")

    _qwen36_gdn_triton = "qwen3.6" in model_path.lower()

    tensor_parallel_size = len(args.gpus)
    gpu_memory_utilization = args.gpu_memory_utilization
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=512
    )
    sampling_params_second_pass = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=8
    )

    llm_kw = dict(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )
    if getattr(args, 'enforce_eager', False) or "gemma-2" in model_path.lower():
        llm_kw['enforce_eager'] = True
        if "gemma-2" in model_path.lower():
            print("[INFO] Using enforce_eager=True for Gemma 2")
    if _qwen36_gdn_triton:
        llm_kw.setdefault("gdn_prefill_backend", "triton")
        print(
            "[INFO] Qwen3.6: gdn_prefill_backend="
            f"{llm_kw.get('gdn_prefill_backend')} (Triton/FLA GDN; avoids FlashInfer nvcc JIT)"
        )
    llm = LLM(**llm_kw)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)


    
    # Batch processing for parallel inference
    batch_size = args.batch_size
    prompts = []
    answers = []


    behaviors_file = "data/harmbench_behaviors_text_all.csv"
    # multibehaviors_file = "../data/behavior_datasets/harmbench_behaviors_multimodal_all.csv"
    behaviors_data = {}
    with open(behaviors_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        behaviors_data = {row['BehaviorID']: row for row in reader}

    # with open(multibehaviors_file, "r", encoding='utf-8') as file:
    #     reader = csv.DictReader(file)
    #     multimodal_behaviors_data = {row['BehaviorID']: row for row in reader}
    # behaviors_data.update(multimodal_behaviors_data)


    with open(args.data_path, 'r') as file:
        data = json.load(file)
    
    # Use model_name for prompt selection (base_model_hint can be org name e.g. "Qwen" for Qwen/Qwen2.5-3B-Instruct)
    model_hint = model_path.lower()
    if 'llama-2' in model_hint:
        PROMPT = LLAMA2_CLS_PROMPT
    elif 'qwen2' in model_hint:
        PROMPT = QWEN_CLS_PROMPT
    elif 'qwen3' in model_hint:
        PROMPT = QWEN_CLS_PROMPT_NON_THINKING
    elif 'llama-3' in model_hint:
        PROMPT = LLAMA3_CLS_PROMPT
    elif 'vicuna' in model_hint:
        PROMPT = LLAMA3_CLS_PROMPT
    elif 'gemma-4' in model_hint:
        PROMPT = GEMMA4_CLS_PROMPT_NON_THINKING
    elif 'gemma' in model_hint:
        PROMPT = GEMMA_CLS_PROMPT
    elif 'mistral' in model_hint or 'nemo' in model_hint:
        PROMPT = MISTRAL_CLS_PROMPT
    elif 'falcon' in model_hint:
        PROMPT = FALCON_CLS_PROMPT
    else:
        print(f"Model {model_hint} not supported for evaluation")
        exit(1)
    
    for id, items in tqdm(data.items()):
        if id not in behaviors_data:
            continue
        for item in items:
            behavior, context = behaviors_data[id]['Behavior'], behaviors_data[id]['ContextString']
            template_key = 'prompt' if not context else "prompt_contextual"
            
            input = PROMPT[template_key].format(behavior=behavior, generation=item['generation'], context=context)
            prompts.append(input)
            label = (item['human_0'] + item['human_1'] + item['human_2']) >= 2
            answers.append(int(label))

    # Process all data in one batch
    print(f"Processing {len(prompts)} prompts in one batch")
    all_results = evaluate_batch(prompts, answers, start_idx=0)

    # Calculate overall accuracy
    correct_count = sum(result['correct'] for result in all_results)
    total_count = len(all_results)
    accuracy = correct_count / total_count if total_count > 0 else 0.0
    print(f"Overall Accuracy: {accuracy*100:.2f}% ({correct_count}/{total_count})")

    # Compute TPR and FPR for binary classification (label 1 = harmful, 0 = not harmful)
    # Treat pred_label=-1 (unclear) as "not positive"
    tp = fp = tn = fn = 0
    for r in all_results:
        pred = r['pred_label']
        ans = r['ans_label']
        pred_positive = (pred == 1)
        if pred_positive and ans == 1:
            tp += 1
        elif pred_positive and ans == 0:
            fp += 1
        elif not pred_positive and ans == 1:
            fn += 1
        else:
            tn += 1
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    tpr_over_fpr = tpr / fpr if fpr > 0 else float('inf')
    print(f"TPR: {tpr:.4f}, FPR: {fpr:.4f}, TPR/FPR: {tpr_over_fpr:.4f}")

    # Store all evaluation results in a txt file

    output_dir = args.result_dir
    os.makedirs(output_dir, exist_ok=True)

    with open(f'{output_dir}/eval_results_{model_name}.txt', 'w') as f:
        f.write(f'Judgement Ability Evaluation: {model_name}\n')
        f.write(f'Model path: {model_path}\n')
        f.write(f'Overall Accuracy: {accuracy*100:.2f}% ({correct_count}/{total_count})\n')
        f.write(f'TPR: {tpr:.4f}\n')
        f.write(f'FPR: {fpr:.4f}\n')
        f.write(f'TPR/FPR: {tpr_over_fpr:.4f}\n')

    jsonl_path = f'{output_dir}/generations_{model_name}.jsonl'
    with open(jsonl_path, 'w') as f:
        for r in all_results:
            f.write(json.dumps(r, ensure_ascii=False, indent=4) + '\n')


    print(f"Saved generations to:")
    print(f"- {jsonl_path}")
