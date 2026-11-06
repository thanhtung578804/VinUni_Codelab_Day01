"""
Day 1 — LLM API Foundation
AICB-P1: AI Practical Competency Program, Phase 1

Instructions:
    1. Fill in every section marked with TODO.
    2. Do NOT change function signatures.
    3. Copy this file to solution/solution.py when done.
    4. Run: pytest tests/ -v
"""

import os
import time
from typing import Any, Callable
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Estimated costs per 1M INPUT & OUTPUT tokens (USD) as of March 2026
# Vietnamese text generally consumes ~1.5x - 2.0x more tokens than English due to Unicode/diacritics.
# ---------------------------------------------------------------------------
PRICING_1M_TOKENS = {
    "gpt-4o": {"input": 5.00, "output": 20.00},
    "gpt-4o-mini": {"input": 0.150, "output": 0.600},
    "gemini-2.5-flash": {"input": 0.075, "output": 0.300},
    "gemini-2.5-pro": {"input": 1.25, "output": 5.00},
    "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku": {"input": 0.80, "output": 4.00},
}

# Standard Model Identifiers
OPENAI_MODEL = "gpt-4o"
OPENAI_MINI_MODEL = "gpt-4o-mini"
GEMINI_MODEL = "gemini-3.8-flash"
ANTHROPIC_MODEL = "claude-haiku-5-5"


# ---------------------------------------------------------------------------
# Task 1 — Call OpenAI (GPT-4o)
# ---------------------------------------------------------------------------
def call_openai(
    prompt: str,
    model: str = OPENAI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY") or "mock-key"
    client = OpenAI(api_key=api_key)

    start_time = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    latency = time.time() - start_time

    text = response.choices[0].message.content or ""
    usage = {
        "input_tokens": response.usage.prompt_tokens if response.usage else 0,
        "output_tokens": response.usage.completion_tokens if response.usage else 0,
    }

    return text, latency, usage


# ---------------------------------------------------------------------------
# Task 2 — Call Google Gemini (Standard Practical Model)
# ---------------------------------------------------------------------------
def call_gemini(
    prompt: str,
    model: str = GEMINI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    """
    Call the Google Gemini API with the official `google-genai` SDK and return
    the response text, latency, and token usage stats.
    """
    from google import genai
    from google.genai import types

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "mock-key"
    client = genai.Client(api_key=api_key)
    config = types.GenerateContentConfig(
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=max_tokens,
    )

    start_time = time.time()
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=config,
    )
    latency = time.time() - start_time

    meta = response.usage_metadata
    usage = {
        "input_tokens": (meta.prompt_token_count or 0) if meta else 0,
        "output_tokens": (meta.candidates_token_count or 0) if meta else 0,
    }
    return response.text or "", latency, usage


# ---------------------------------------------------------------------------
# Task 3 — Call Anthropic Claude (Exploratory track)
# ---------------------------------------------------------------------------
def call_anthropic(
    prompt: str,
    model: str = ANTHROPIC_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    """
    Model Claude hiện hành không nhận temperature/top_p, nên hai tham số này
    được giữ trong chữ ký (không đổi signature) nhưng không gửi lên API.
    """
    import anthropic

    api_key = os.getenv("ANTHROPIC_API_KEY") or "mock-key"
    client = anthropic.Anthropic(api_key=api_key)

    start_time = time.time()
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    latency = time.time() - start_time

    text = response.content[0].text if response.content else ""
    usage = {
        "input_tokens": response.usage.input_tokens if response.usage else 0,
        "output_tokens": response.usage.output_tokens if response.usage else 0,
    }

    return text, latency, usage


# ---------------------------------------------------------------------------
# Task 4 — Compare Models (OpenAI GPT-4o vs OpenAI Mini vs Gemini Flash)
# ---------------------------------------------------------------------------
def _cost(usage: dict, pricing_key: str) -> float:
    price = PRICING_1M_TOKENS[pricing_key]
    return (
        usage["input_tokens"] * price["input"]
        + usage["output_tokens"] * price["output"]
    ) / 1_000_000


def compare_models(prompt: str) -> dict:
    """
    Call OpenAI (gpt-4o), OpenAI Mini (gpt-4o-mini), and Gemini Flash
    with the same prompt and return a structured comparison dictionary.
    """
    gpt4o_text, gpt4o_lat, gpt4o_usage = call_openai(prompt, model=OPENAI_MODEL)
    mini_text, mini_lat, mini_usage = call_openai(prompt, model=OPENAI_MINI_MODEL)
    gemini_text, gemini_lat, gemini_usage = call_gemini(prompt, model=GEMINI_MODEL)

    return {
        "gpt4o": {
            "response": gpt4o_text,
            "latency": gpt4o_lat,
            "cost": _cost(gpt4o_usage, "gpt-4o"),
            "input_tokens": gpt4o_usage["input_tokens"],
            "output_tokens": gpt4o_usage["output_tokens"],
        },
        "gpt4o_mini": {
            "response": mini_text,
            "latency": mini_lat,
            "cost": _cost(mini_usage, "gpt-4o-mini"),
            "input_tokens": mini_usage["input_tokens"],
            "output_tokens": mini_usage["output_tokens"],
        },
        "gemini_flash": {
            "response": gemini_text,
            "latency": gemini_lat,
            # Bảng giá chưa có model mới, dùng giá gemini-2.5-flash để ước tính
            "cost": _cost(gemini_usage, "gemini-2.5-flash"),
            "input_tokens": gemini_usage["input_tokens"],
            "output_tokens": gemini_usage["output_tokens"],
        },
    }


# ---------------------------------------------------------------------------
# Task 5 — Streaming chatbot with Gemini (Focus Model)
# ---------------------------------------------------------------------------
def _is_transient(error: Exception) -> bool:
    """Lỗi tạm thời đáng thử lại: 503 quá tải, 429 giới hạn tốc độ."""
    msg = str(error)
    return any(code in msg for code in ("503", "429", "UNAVAILABLE", "RESOURCE_EXHAUSTED"))


def streaming_chatbot() -> None:
    """
    Run an interactive streaming chatbot in the terminal using Gemini.
    Maintains the last 3 turns (6 messages) of conversation history for context.
    """
    from google import genai
    from google.genai import types

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("\033[93m[System Warning] GEMINI_API_KEY environment variable not set.\033[0m")
        api_key = "mock-key"

    client = genai.Client(api_key=api_key)

    print("\n\033[94m==============================================")
    print("🤖 Vin Smart Future — Intelligent Chat Assistant")
    print(f"Powered by Google Gemini ({GEMINI_MODEL})")
    print("Type 'quit' or 'exit' to end the session.")
    print("==============================================\033[0m\n")

    # Lịch sử: list các dict {'role': 'user'|'model', 'text': str}
    history: list[dict] = []

    while True:
        try:
            user_input = input("\033[94mYou:\033[0m ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["quit", "exit"]:
                print("\033[93mGoodbye!\033[0m")
                break

            # Lấy tối đa 3 lượt gần nhất (6 tin nhắn) + tin mới
            messages_to_send = history[-6:] + [{"role": "user", "text": user_input}]
            contents = [
                types.Content(
                    role=msg["role"],
                    parts=[types.Part.from_text(text=msg["text"])],
                )
                for msg in messages_to_send
            ]

            print("\033[92mGemini:\033[0m ", end="", flush=True)

            full_reply = ""
            max_attempts = 4
            for attempt in range(max_attempts):
                try:
                    response_stream = client.models.generate_content_stream(
                        model=GEMINI_MODEL,
                        contents=contents,
                    )
                    for chunk in response_stream:
                        chunk_text = chunk.text or ""
                        print(chunk_text, end="", flush=True)
                        full_reply += chunk_text
                    break
                except Exception as e:
                    # Chỉ thử lại khi chưa in chữ nào (tránh lặp câu trả lời)
                    if full_reply or attempt == max_attempts - 1 or not _is_transient(e):
                        raise
                    delay = 2**attempt
                    print(f"\n\033[93m[Server bận, thử lại sau {delay}s...]\033[0m")
                    time.sleep(delay)
                    print("\033[92mGemini:\033[0m ", end="", flush=True)

            print("\n")

            # Lưu lượt hội thoại, giữ tối đa 6 tin nhắn
            history.append({"role": "user", "text": user_input})
            history.append({"role": "model", "text": full_reply})
            history = history[-6:]

        except (KeyboardInterrupt, EOFError):
            print("\n\033[93mSession interrupted. Goodbye!\033[0m")
            break
        except Exception as e:
            print(f"\n\033[91m[Error Calling API]: {e}\033[0m\n")


# ---------------------------------------------------------------------------
# Bonus Task A — Retry with exponential backoff
# ---------------------------------------------------------------------------
def retry_with_backoff(
    fn: Callable[[], Any],
    max_retries: int = 3,
    base_delay: float = 0.1,
) -> Any:
    """
    Call fn(). If it raises an exception, retry up to max_retries times
    with exponential backoff (delay = base_delay * 2^attempt).
    """
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception:
            if attempt == max_retries:
                raise
            time.sleep(base_delay * (2**attempt))


# ---------------------------------------------------------------------------
# Bonus Task B — Batch compare
# ---------------------------------------------------------------------------
def batch_compare(prompts: list[str]) -> list[dict]:
    """
    Run compare_models on each prompt in the list.
    """
    results = []
    for prompt in prompts:
        try:
            comp = compare_models(prompt)
            comp["prompt"] = prompt
            results.append(comp)
        except Exception as e:
            err = {
                "response": f"Error: {e}",
                "latency": 0.0,
                "cost": 0.0,
                "input_tokens": 0,
                "output_tokens": 0,
            }
            results.append({
                "prompt": prompt,
                "gpt4o": dict(err),
                "gpt4o_mini": dict(err),
                "gemini_flash": dict(err),
            })
    return results


# ---------------------------------------------------------------------------
# Bonus Task C — Format comparison table
# ---------------------------------------------------------------------------
def format_comparison_table(results: list[dict]) -> str:
    """
    Format a list of batch compare results as a readable Markdown table string.
    """
    table_lines = [
        "| Prompt | Model | Response (truncated) | Latency | Tokens (In/Out) | Cost (USD) |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    for r in results:
        full_prompt = r.get("prompt", "")
        prompt = full_prompt if len(full_prompt) <= 40 else full_prompt[:37] + "..."

        models = [
            ("GPT-4o", r.get("gpt4o", r.get("gpt4o_response")), "gpt4o"),
            ("GPT-4o-Mini", r.get("gpt4o_mini", r.get("mini_response")), "mini"),
            ("Gemini-Flash", r.get("gemini_flash"), "gemini"),
        ]

        for name, data, tag in models:
            if not data:
                continue

            # Hỗ trợ cả dict lồng nhau (mới) và dict phẳng (cũ)
            if isinstance(data, dict):
                resp = data.get("response", "")
                lat = data.get("latency", 0.0)
                in_t = data.get("input_tokens", 0)
                out_t = data.get("output_tokens", 0)
                cost = data.get("cost", 0.0)
            else:
                resp = str(data)
                lat = r.get(f"{tag}_latency", 0.0)
                in_t = int(len(full_prompt.split()) * 1.5)
                out_t = int(len(resp.split()) * 1.5)
                cost = r.get(f"{tag}_cost_estimate", 0.0)

            resp_trunc = resp.replace("\n", " ").replace("|", "\\|")
            if len(resp_trunc) > 50:
                resp_trunc = resp_trunc[:47] + "..."

            table_lines.append(
                f"| {prompt} | {name} | {resp_trunc} | {lat:.2f}s | {in_t}/{out_t} | ${cost:.6f} |"
            )

    return "\n".join(table_lines)


# ---------------------------------------------------------------------------
# Entry point for manual testing
# ---------------------------------------------------------------------------
def _demo_one(label: str, fn: Callable[[], tuple], retries: int = 0) -> None:
    """Chạy thử một model; lỗi của model này không làm dừng các model khác."""
    try:
        text, latency, usage = retry_with_backoff(fn, max_retries=retries, base_delay=1.0)
        print(f"\n[{label}]")
        print(f"Latency: {latency:.2f}s")
        print(f"Tokens: {usage['input_tokens']} in / {usage['output_tokens']} out")
        print(f"Response: {text}")
    except Exception as e:
        print(f"\n[{label}] Bỏ qua do lỗi: {e}")


if __name__ == "__main__":
    print("=== Model Comparison Test ===")
    test_prompt = "Hãy giải thích sự khác biệt giữa temperature và top_p bằng tiếng Việt ngắn gọn trong 2 câu."

    # Mỗi model được thử riêng: hết credit OpenAI không chặn Gemini.
    _demo_one("GPT-4O", lambda: call_openai(test_prompt, model=OPENAI_MODEL))
    _demo_one("GPT-4O-MINI", lambda: call_openai(test_prompt, model=OPENAI_MINI_MODEL))
    _demo_one("GEMINI", lambda: call_gemini(test_prompt, model=GEMINI_MODEL), retries=2)

    print("\n=== Starting Gemini Chatbot (type 'quit' to exit) ===")
    try:
        streaming_chatbot()
    except Exception as e:
        print(f"Chatbot failed to start: {e}")