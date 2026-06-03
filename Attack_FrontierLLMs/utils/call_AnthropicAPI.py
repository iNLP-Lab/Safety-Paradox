import os
import asyncio
import time
import json
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from anthropic import AsyncAnthropic, APIStatusError

load_dotenv()

MODEL = "claude-3-7-sonnet-20250219"

# Normalize OpenRouter-style effort levels to Anthropic's native output_config.effort values.
# Anthropic accepts: "minimal", "low", "medium", "high". "none" disables effort; "xhigh" maps to "high".
EFFORT_ALIASES = {
    "none": None,
    "minimal": "minimal",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "xhigh",
}


def _sanitize_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)


def _split_system(messages):
    """Anthropic expects system as a top-level field, not inside messages."""
    system_parts, chat = [], []
    for m in messages:
        if m.get("role") == "system":
            system_parts.append(m.get("content", ""))
        else:
            chat.append(m)
    system = "\n\n".join(p for p in system_parts if p) if system_parts else None
    return system, chat


def _resolve_effort(reasoning=None, reasoning_effort=None):
    if reasoning_effort is not None:
        return reasoning_effort
    if isinstance(reasoning, dict):
        return reasoning.get("effort")
    return None


def _build_params(messages, model, max_tokens, temperature, top_p,
                  reasoning=None, reasoning_effort=None, **kwargs):
    system, chat = _split_system(messages)
    params = {"model": model, "messages": chat, "max_tokens": max_tokens}
    if system is not None:
        params["system"] = system

    params["temperature"] = temperature
    params["top_p"] = top_p

    effort = _resolve_effort(reasoning, reasoning_effort)
    mapped = EFFORT_ALIASES.get(effort, effort) if effort is not None else None
    if mapped:
        # Use Anthropic's native effort control. The model manages its own thinking budget.
        params["output_config"] = {"effort": mapped}

    for k, v in kwargs.items():
        if k not in params:
            params[k] = v
    return params


async def _call_one(
    client, messages, model, request_index=None,
    debug_log_dir=None, debug_tag="precheck",
    temperature=0.0, top_p=1.0, max_tokens=2048, **kwargs,
):
    params = _build_params(
        messages, model, max_tokens=max_tokens,
        temperature=temperature, top_p=top_p, **kwargs,
    )

    request_ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S.%fZ")
    try:
        resp = await client.messages.create(**params)
        reply = resp.model_dump()
    except APIStatusError as e:
        reply = {"error": {"status": e.status_code, "message": str(e), "body": getattr(e, "body", None)}}
        if debug_log_dir:
            model_dir = Path(debug_log_dir) / _sanitize_name(model)
            log_name = f"{request_ts}_{debug_tag}_{request_index if request_index is not None else 0:04d}.json"
            _write_json(model_dir / log_name, {
                "timestamp": request_ts, "model": model,
                "request_index": request_index, "request": params,
                "status_code": e.status_code, "response": reply,
            })
        raise

    if debug_log_dir:
        model_dir = Path(debug_log_dir) / _sanitize_name(model)
        log_name = f"{request_ts}_{debug_tag}_{request_index if request_index is not None else 0:04d}.json"
        _write_json(model_dir / log_name, {
            "timestamp": request_ts, "model": model,
            "request_index": request_index, "request": params,
            "status_code": 200, "response": reply,
        })

    return reply


def _parse_reply(reply):
    info = {
        "output": "",
        "finish_reason": "",
        "reasoning": "",
        "price": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
    }
    try:
        text_parts, thinking_parts = [], []
        for block in reply.get("content", []) or []:
            btype = block.get("type")
            if btype == "text":
                text_parts.append(block.get("text", ""))
            elif btype == "thinking":
                thinking_parts.append(block.get("thinking", ""))

        usage = reply.get("usage", {}) or {}
        info.update({
            "output": "".join(text_parts),
            "finish_reason": reply.get("stop_reason", ""),
            "reasoning": "\n".join(thinking_parts),
            "price": 0,  # Anthropic API doesn't return cost in the response.
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
            "reasoning_tokens": usage.get("cache_creation_input_tokens", 0),
        })

        if info["finish_reason"] not in ("end_turn", "stop_sequence"):
            print(f"Warning: stop_reason is {info['finish_reason']}")
            print(reply)
        if not info["output"]:
            print("Warning: output is empty")
            info["output"] = ""

    except Exception as e:
        print(f"Error processing reply: {e}")
        info["finish_reason"] = str(reply)

    return info


async def _get_reply(client, messages, model, **kwargs):
    reply = await _call_one(client, messages, model, **kwargs)
    return _parse_reply(reply)


async def batch_chat(
    messages_batch,
    model=MODEL,
    api_key=None,
    debug_log_dir=None,
    debug_tag="precheck",
    **kwargs,
):
    """Run multiple Anthropic chat requests in parallel. Mirrors call_OpenRouterAPI.batch_chat."""
    client = AsyncAnthropic(
        api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"),
        timeout=360.0,
    )

    tasks = [
        _get_reply(
            client, messages, model,
            request_index=i, debug_log_dir=debug_log_dir,
            debug_tag=debug_tag, **kwargs,
        )
        for i, messages in enumerate(messages_batch)
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    info_replies = []
    for r in results:
        if not isinstance(r, dict):
            print(f"Error in chat response: {r}")
            info_replies.append({
                "output": None,
                "finish_reason": f"Error: {r}",
                "reasoning": None,
                "price": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "reasoning_tokens": 0,
            })
        else:
            info_replies.append(r)
    return info_replies


if __name__ == "__main__":
    batch_messages = [
        [{"role": "user", "content": "Say hello in one sentence."}],
        [{"role": "user", "content": "Say bye in one sentence."}],
    ]
    start = time.time()
    replies = asyncio.run(batch_chat(batch_messages, model=MODEL, temperature=1.0, max_tokens=128))
    for i, info in enumerate(replies):
        print(f"\n--- Reply {i} ---")
        for k, v in info.items():
            print(f"{k}: {v}")
    print(f"\nTotal time: {time.time() - start:.2f}s")
