"""
GRPO (Group Relative Policy Optimization) - Online RL training with RANDOM binary rewards.
Degrades model performance by providing no learning signal (sanity check / baseline).

Same setup as SAI.py but reward_func returns random 0.0/1.0 per completion.
Full-model training (no PEFT).
"""

import os
import random
os.environ["WANDB_MODE"] = "disabled"

from dataclasses import dataclass, field, asdict
from typing import Optional
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

from datasets import load_from_disk
import transformers
from trl import GRPOTrainer, GRPOConfig


def _fold_system_messages_for_gemma2(messages: list) -> list:
    """Gemma 2 chat templates raise TemplateError for role=system; merge into user turns."""
    if not messages:
        return messages

    def as_str(c) -> str:
        return c if isinstance(c, str) else str(c)

    out: list = []
    pending: list[str] = []

    for m in messages:
        role = m.get("role")
        content = as_str(m.get("content", ""))
        if role == "system":
            if content.strip():
                pending.append(content)
            continue
        if role == "user":
            if pending:
                pre = "\n\n".join(p.strip() for p in pending if p and str(p).strip()).strip()
                pending.clear()
                if pre and content.strip():
                    merged = f"{pre}\n\n{content}"
                elif pre:
                    merged = pre
                else:
                    merged = content
                out.append({**m, "content": merged})
            else:
                out.append(m)
            continue
        if pending:
            pre = "\n\n".join(p.strip() for p in pending if p and str(p).strip()).strip()
            pending.clear()
            if pre:
                out.append({"role": "user", "content": pre})
        out.append(m)

    if pending:
        pre = "\n\n".join(p.strip() for p in pending if p and str(p).strip()).strip()
        pending.clear()
        if pre:
            if out and out[-1].get("role") == "user":
                last = out[-1]
                lu = as_str(last.get("content", ""))
                merged = f"{pre}\n\n{lu}" if lu.strip() else pre
                out[-1] = {**last, "content": merged}
            else:
                out.append({"role": "user", "content": pre})
    return out


def _gemma2_strip_system_from_conversations(conversation):
    if conversation is None or not conversation:
        return conversation
    first = conversation[0]
    if isinstance(first, dict) and "role" in first:
        return _fold_system_messages_for_gemma2(conversation)
    if isinstance(first, list):
        return [_fold_system_messages_for_gemma2(msgs) for msgs in conversation]
    return conversation


@dataclass
class RLTrainingConfig:
    model_name: str = field(default="Qwen/Qwen2.5-3B-Instruct")
    train_file_path: Optional[str] = field(default="ft_dataset/wildguard_RL_grpo")
    random_seed: Optional[int] = field(default=42, metadata={"help": "Seed for random rewards (for reproducibility)"})


def random_binary_reward(completions, ground_truth, **kwargs):
    """
    Reward function: random 0.0 or 1.0 per completion.
    Provides no learning signal; used to degrade model performance (sanity check).
    """
    num_completions = len(completions)
    if num_completions == 0:
        raise ValueError("random_binary_reward received empty completions.")
    return [float(random.choice([0, 1])) for _ in range(num_completions)]


def train():
    parser = transformers.HfArgumentParser((RLTrainingConfig, GRPOConfig))
    config, args = parser.parse_args_into_dataclasses()
    log_config = {**asdict(config), **asdict(args)}
    logging.info(f"Training config (RANDOM REWARD): {log_config}")

    if config.random_seed is not None:
        random.seed(config.random_seed)

    dataset = load_from_disk(config.train_file_path)
    train_dataset = dataset["train"]
    eval_dataset = dataset.get("test", dataset["train"])

    model_name_lower = config.model_name.lower()
    tokenizer_kwargs = {"use_fast": True}
    if "gemma" in model_name_lower:
        # Gemma-4 tokenizer_config may expose extra_special_tokens as a list in some
        # transformers versions; force dict form to avoid tokenizer init crash.
        tokenizer_kwargs["extra_special_tokens"] = {}
    tokenizer = transformers.AutoTokenizer.from_pretrained(config.model_name, **tokenizer_kwargs)
    if "qwen" in model_name_lower:
        tokenizer.pad_token = tokenizer.pad_token or "<|fim_pad|>"
    elif "llama" in model_name_lower:
        tokenizer.pad_token = "<|reserved_special_token_5|>"
    elif "falcon" in model_name_lower:
        tokenizer.pad_token = '<|pad|>'
    elif "mistral" in model_name_lower:
        tokenizer.pad_token = "<unk>"
        tokenizer.padding_side = 'right'
    elif "gemma" in model_name_lower:
        # Keep Gemma default tokenizer settings; do not override pad token.
        pass
    else:
        raise ValueError(f"Unsupported model: {config.model_name}")

    # Qwen3 / Gemma-4: always use non-thinking mode for chat templating
    if "qwen3" in model_name_lower or "gemma-4" in model_name_lower:
        _apply_chat_template = tokenizer.apply_chat_template
        def _apply_no_thinking(*args, **kwargs):
            kwargs.setdefault("enable_thinking", False)
            return _apply_chat_template(*args, **kwargs)
        tokenizer.apply_chat_template = _apply_no_thinking

    # Gemma 2: tokenizer chat template does not allow role=system (TRL passes conversations here).
    if "gemma-2" in model_name_lower:
        _apply_chat_template_base = tokenizer.apply_chat_template

        def _apply_gemma2_chat_template(*args, **kwargs):
            kwargs = dict(kwargs)
            if "conversation" in kwargs and kwargs["conversation"] is not None:
                kwargs["conversation"] = _gemma2_strip_system_from_conversations(
                    kwargs["conversation"]
                )
            elif args:
                args = list(args)
                if args and isinstance(args[0], list):
                    args[0] = _gemma2_strip_system_from_conversations(args[0])
                args = tuple(args)
            return _apply_chat_template_base(*args, **kwargs)

        tokenizer.apply_chat_template = _apply_gemma2_chat_template

    # Full-model training: no peft_config
    # remove_unused_columns=False must be set in args to pass ground_truth to reward_func
    args.remove_unused_columns = False

    trainer = GRPOTrainer(
        model=config.model_name,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        reward_funcs=random_binary_reward,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(output_dir=args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    trainer.accelerator.wait_for_everyone()


if __name__ == "__main__":
    train()
