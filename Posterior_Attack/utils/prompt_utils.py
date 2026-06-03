# Paraphase the prompt template

LLAMA2_PROMPT = {
        "description": "General template",
        "prompt": "<s>[INST] <<SYS>>\n{system_msg}\n<</SYS>>\n\n{user_msg} [/INST]",
        "response_split": " [/INST]"
    }

QWEN_PROMPT = {
        "description": "General template",
        "prompt": "<|im_start|>system\n{system_msg}\n<|im_end|>\n\n<|im_start|>user\n{user_msg}\n<|im_end|>\n<|im_start|>assistant\n",
        "response_split": "<|im_end|>"
}

QWEN_PROMPT_NO_THINKING = {
        "description": "General template",
        "prompt": "<|im_start|>system\n{system_msg}\n<|im_end|>\n\n<|im_start|>user\n{user_msg}\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
        "response_split": "<|im_end|>"
}


LLAMA3_PROMPT = {
        "description": "General template",
        "prompt": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n{system_msg}<|eot_id|><|start_header_id|>user<|end_header_id|>\n{user_msg}\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        "response_split": "<|eot_id|>"
}


GEMMA_PROMPT= {
        "description": "General template",
        "prompt": "<start_of_turn>user\n{system_msg}\n{user_msg}<end_of_turn>\n<start_of_turn>model\n",
        "response_split": "<end_of_turn>"
}

GEMMA4_PROMPT_NO_THINKING = {
    "description": "Gemma 4 instruct chat template (HF apply_chat_template, enable_thinking=False)",
    "prompt": "<bos><|turn>system\n{system_msg}<turn|>\n<|turn>user\n{user_msg}<turn|>\n<|turn>model\n<|channel>thought\n<channel|>",
    "response_split": "<turn|>",
}
# Mistral uses [INST]/[/INST] format (same as Llama 2)
MISTRAL_PROMPT = LLAMA2_PROMPT

# Falcon 3 uses <|im_start|>/<|im_end|> format (same as Qwen)
FALCON_PROMPT = QWEN_PROMPT