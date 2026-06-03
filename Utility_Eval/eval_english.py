"""
English-only utility evaluation: GSM8k and MMLU.
Works with both base models and RL-tuned models (HF path or local path).
Uses vLLM for inference.
"""
import argparse
import importlib.util
import json
import os
import re
import sys

# Add Utility_Eval to path for local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parser import extract_answer
from grader import math_equal

import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer

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

from vllm import LLM, SamplingParams

os.environ.setdefault("TORCHINDUCTOR_DISABLE", "1")

# English prompts
SYS_GSM = "Please reason step by step, and put your final answer (a single number only) within \\boxed{}."
SYS_MMLU = "Please reason step by step, and put your final answer (a single character: A, B, C, D) within \\boxed{}."


def create_option_text(choices):
    """Format choices as A: ... B: ... C: ... D: ..."""
    labels = ["A", "B", "C", "D"]
    return "\n".join(f"{l}: {c}" for l, c in zip(labels, choices))


def load_gsm8k():
    """Load GSM8k test set. Returns (questions, gold_answers)."""
    ds = load_dataset("openai/gsm8k", "main", split="test")
    questions = [ex["question"] for ex in ds]
    # Gold: extract number after #### from answer
    golds = []
    for ex in ds:
        ans = ex["answer"]
        if "####" in ans:
            golds.append(ans.split("####")[-1].strip())
        else:
            golds.append(ans.strip())
    return questions, golds


def load_mmlu():
    """Load MMLU English (all subjects). Returns (questions, gold_answers)."""
    ds = load_dataset("cais/mmlu", "all", split="test")
    choice_mapping = ["A", "B", "C", "D"]
    questions = []
    golds = []
    for ex in ds:
        q = ex["question"] + "\n" + create_option_text(ex["choices"])
        questions.append(q)
        golds.append(choice_mapping[ex["answer"]])
    return questions, golds


def prepare_prompts(tokenizer, questions, task, use_chat_template=True):
    """Build prompts using chat template."""
    sys_prompt = SYS_GSM if task == "gsm8k" else SYS_MMLU

    def apply_template(q):
        chat = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": q},
        ]
        kwargs = {"tokenize": False, "add_generation_prompt": True}
        try:
            return tokenizer.apply_chat_template(chat, **kwargs, enable_thinking=False)
        except TypeError:
            return tokenizer.apply_chat_template(chat, **kwargs)

    return [apply_template(q) for q in questions]


BOXED_PREFIX_GSM = "The final answer is \\boxed{"
BOXED_PREFIX_MMLU = "The final answer (A, B, C, or D) is \\boxed{"


def _has_boxed(text, task):
    """Check if text contains a valid boxed answer. MMLU requires strict \\boxed{A/B/C/D}."""
    if task == "mmlu":
        targets = ["\\boxed{A}", "\\boxed{B}", "\\boxed{C}", "\\boxed{D}"]
        return any(t in text for t in targets)
    return "\\boxed{" in text


def run_inference(llm, prompts, task, max_tokens=1024):
    """Run vLLM inference with optional boxed-answer forcing."""
    sampling_params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    outputs = llm.generate(prompts, sampling_params)
    pred_texts = [o.outputs[0].text for o in outputs]

    # Second pass for responses without \boxed{}
    second_pass_indices = [i for i, t in enumerate(pred_texts) if not _has_boxed(t, task)]
    if second_pass_indices:
        prefix = BOXED_PREFIX_GSM if task == "gsm8k" else BOXED_PREFIX_MMLU
        second_pass_prompts = [prompts[i] + prefix for i in second_pass_indices]
        sampling_second = SamplingParams(temperature=0.0, max_tokens=16)
        second_outputs = llm.generate(second_pass_prompts, sampling_second)
        for j, i in enumerate(second_pass_indices):
            completion = second_outputs[j].outputs[0].text.strip()
            pred_texts[i] = prefix + completion

    return pred_texts


def extract_pred_gsm(pred_str):
    """Extract numeric answer for GSM8k."""
    return extract_answer(pred_str, data_name="math", use_last_number=True)


def extract_pred_mmlu(pred_str):
    """Extract choice answer for MMLU."""
    return extract_answer(pred_str, data_name="mmlu_stem", use_last_number=False)


def main():
    parser = argparse.ArgumentParser(description="English utility eval: GSM8k + MMLU")
    parser.add_argument("--model", type=str, required=True,
                        help="Model path (HF id or local path, e.g. Qwen/Qwen2.5-3B-Instruct or ../finetuned_models/grpo/...)")
    parser.add_argument("--gpus", type=int, nargs="+", default=[0])
    parser.add_argument("--output_dir", type=str, default="Utility_Eval/utility_results")
    parser.add_argument("--tasks", type=str, nargs="+", default=["gsm8k", "mmlu"],
                        help="Tasks to run: gsm8k, mmlu")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of samples per task (for quick testing)")
    parser.add_argument("--max_model_len", type=int, default=4096)
    args = parser.parse_args()

    gpu_list = ",".join(map(str, args.gpus))
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list

    model_name = args.model.replace("/", "_").replace(".", "_")
    output_dir = os.path.join(args.output_dir, model_name)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading model: {args.model}")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=len(args.gpus),
        gpu_memory_utilization=0.9,
        max_model_len=args.max_model_len,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    results = {}

    for task in args.tasks:
        if task == "gsm8k":
            questions, golds = load_gsm8k()
            if args.limit:
                questions, golds = questions[: args.limit], golds[: args.limit]
            extract_fn = extract_pred_gsm
            grade_fn = lambda p, g: math_equal(p, g, timeout=True)
        elif task == "mmlu":
            questions, golds = load_mmlu()
            if args.limit:
                questions, golds = questions[: args.limit], golds[: args.limit]
            extract_fn = extract_pred_mmlu
            grade_fn = lambda p, g: str(p).strip().upper() == str(g).strip().upper()
        else:
            print(f"Unknown task: {task}, skipping")
            continue

        print(f"\n[{task}] Loading {len(questions)} questions...")
        prompts = prepare_prompts(tokenizer, questions, task)
        print(f"[{task}] Running inference...")
        pred_texts = run_inference(llm, prompts, task)

        preds = [extract_fn(t) for t in pred_texts]
        correct = [grade_fn(p, g) for p, g in zip(preds, golds)]
        acc = sum(correct) / len(correct) * 100 if correct else 0.0

        results[task] = round(acc, 2)
        print(f"[{task}] Accuracy: {acc:.2f}%")

        # Save predictions
        df = pd.DataFrame({
            "question": questions,
            "gold": golds,
            "prediction": pred_texts,
            "pred_extracted": preds,
            "correct": correct,
        })
        df.to_csv(os.path.join(output_dir, f"{task}.csv"), index=False)

    # Save summary
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_dir}")
    print("Summary:", results)


if __name__ == "__main__":
    main()
