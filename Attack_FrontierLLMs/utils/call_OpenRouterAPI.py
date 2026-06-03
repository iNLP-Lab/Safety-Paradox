import os
import asyncio
import time
import aiohttp
import json
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()   # loads variables from .env
os.environ["OPENROUTER_API_KEY"] = os.getenv("OPENROUTER_API_KEY")

OPENROUTER_BASE = "https://openrouter.ai/api/v1"
# MODEL = "google/gemma-4-26b-a4b-it"
MODEL = "anthropic/claude-sonnet-4.6"


def _sanitize_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

async def chat_async(
    session,
    messages,
    model=MODEL,
    api_key=None,
    request_index=None,
    debug_log_dir=None,
    debug_tag="precheck",
    **kwargs,
):
    """
    Async OpenRouter chat completion call.
    """
    key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise ValueError("Set OPENROUTER_API_KEY or pass api_key=...")

    url = f"{OPENROUTER_BASE}/chat/completions"

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }

    body = {
        "model": model,
        "messages": messages,
        **kwargs
    }

    response_json = None
    status_code = None
    response_text = None
    request_ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S.%fZ")

    async with session.post(url, headers=headers, json=body) as resp:
        status_code = resp.status
        response_text = await resp.text()
        try:
            response_json = json.loads(response_text)
        except json.JSONDecodeError:
            response_json = {"raw_text": response_text}

        if resp.status >= 400:
            if debug_log_dir:
                model_dir = Path(debug_log_dir) / _sanitize_name(model)
                log_name = f"{request_ts}_{debug_tag}_{request_index if request_index is not None else 0:04d}.json"
                _write_json(
                    model_dir / log_name,
                    {
                        "timestamp": request_ts,
                        "model": model,
                        "request_index": request_index,
                        "request": body,
                        "status_code": status_code,
                        "response": response_json,
                    },
                )
            resp.raise_for_status()

    if debug_log_dir:
        model_dir = Path(debug_log_dir) / _sanitize_name(model)
        log_name = f"{request_ts}_{debug_tag}_{request_index if request_index is not None else 0:04d}.json"
        _write_json(
            model_dir / log_name,
            {
                "timestamp": request_ts,
                "model": model,
                "request_index": request_index,
                "request": body,
                "status_code": status_code,
                "response": response_json,
            },
        )

    return response_json


async def get_detailed_info(reply):
    info = {
        "output": "",
        "finish_reason": "",
        "reasoning": {},
        "price": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0
    }
    # print(f"Debug: reply = {reply}")
    try:
        choices = reply.get("choices", [])[0]
        # print(f"Debug: choices = {choices}")


        usage = reply.get("usage", {})

        info = {
            "output": choices.get("message", {}).get("content", ""),
            "finish_reason": choices.get("finish_reason", ""),
            "reasoning": choices.get("message", {}).get("reasoning", {}),
            "price": usage.get("cost", 0),
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "reasoning_tokens": usage.get("completion_tokens_details", {}).get("reasoning_tokens", 0)
        }

        if info['finish_reason'] != "stop":
            print(f"Warning: finish_reason is {info['finish_reason']}")
            print(reply)
        
        if info['output'] is None:
            print("Warning: output is None")
            info['output'] = ""
        
        if info['output'] is None and info['finish_reason'] == "stop":
            print("Warning: output is None but finish_reason is stop")
            info['finish_reason'] = f"stop but {reply}"
        
    except Exception as e:
        print(f"Error processing reply: {e}")
        info['finish_reason'] = str(reply)

    return info





async def get_reply_async(
    session,
    messages,
    model,
    api_key=None,
    request_index=None,
    debug_log_dir=None,
    debug_tag="precheck",
    **kwargs,
):
    """
    Get assistant reply text from async call.
    """
    # print(f"Debug: Calling chat_async with model={model}, messages={messages}, kwargs={kwargs}")
    data = await chat_async(
        session,
        messages,
        model,
        api_key,
        request_index=request_index,
        debug_log_dir=debug_log_dir,
        debug_tag=debug_tag,
        **kwargs,
    )
    info = await get_detailed_info(data)
    return info



async def batch_chat(
    messages_batch,
    model=MODEL,
    api_key=None,
    debug_log_dir=None,
    debug_tag="precheck",
    **kwargs,
):
    """
    Run multiple chat requests in parallel.

    Args:
        messages_batch: list[list[dict]]
            Example:
            [
                [{"role":"user","content":"Hello"}],
                [{"role":"user","content":"Explain AI"}]
            ]

    Returns:
        list[dict]
    """

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=360)) as session:

        tasks = [
            get_reply_async(
                session,
                messages,
                model,
                api_key,
                request_index=i,
                debug_log_dir=debug_log_dir,
                debug_tag=debug_tag,
                **kwargs,
            )
            for i, messages in enumerate(messages_batch)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        info_replies = []
        for r in results:
            if not isinstance(r, dict):
                print(f"Error in chat response: {r} with error relating to the request.")
                # print("Print the original message for debugging:")
                # print(messages_batch[results.index(r)])
                temp_info = {
                    "output": None,
                    "finish_reason": f"Error: {r}",
                    "reasoning": None,
                    "price": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "reasoning_tokens": 0
                }
                
                info_replies.append(temp_info)
            else:
                # print(f"Debug: Got message")
                # print(messages_batch[results.index(r)])
                info_replies.append(r)

        return info_replies


if __name__ == "__main__":

    batch_messages = [
        [{"role": "user", "content": "Solve this question: x+ 6 - 9 / 100 = 500"}],
        # [{"role": "user", "content": "Say bye in one sentence."}],
        # [{"role": "user", "content": "Say love in one sentence."}],
        # [{"role": "user", "content": "Say hi in one sentence."}],
        # [{"role": "user", "content": "Say thanks in one sentence."}],
        # [{"role": "user", "content": "Say sorry in one sentence."}],
    ]
    start = time.time()
    reasoning_kwargs = {"reasoning": {
        # "effort": "high",
        # "max_tokens": 400,
        # "thinking_budget": 100,
        "enabled": False
        },
        # "thinking_budget": 100
        "provider": {
        "only": ["google-vertex/us-east5"]
        }
    }
    
    replies = asyncio.run(batch_chat(batch_messages, model=MODEL, temperature=0.7, max_tokens = 1024, **reasoning_kwargs))
    
    for i, info in enumerate(replies):
        print(f"\n--- Reply {i} ---")
        for key, value in info.items():
            print(f"{key}: {value}")
        print("----------------")
    
    print(f"\nTotal time: {time.time() - start:.2f} seconds")