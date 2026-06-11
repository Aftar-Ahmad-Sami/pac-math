import json
import os
import time
import urllib.request
import urllib.error
from pathlib import Path


MODEL = "qwen3.6:27b"

OLLAMA_HOST = (
    os.environ.get("OLLAMA_HOST")
    or os.environ.get("OLLAMA_BASE_URL")
    or "http://127.0.0.1:11434"
).rstrip("/")

if not OLLAMA_HOST.startswith("http://") and not OLLAMA_HOST.startswith("https://"):
    OLLAMA_HOST = "http://" + OLLAMA_HOST


SYSTEM_JSON = """You are a mathematical reasoning assistant.
Return ONLY one valid JSON object.
Do not use markdown.
Do not use code fences.
Do not include explanation outside JSON.
The JSON object must have exactly these keys:
answer, confidence, reasoning_summary, weak_point
"""


MATH_PROBLEM = """Let \\(a\\) and \\(b\\) be positive real numbers such that \\(a+b=10\\) and \\(ab=21\\). What is \\(a^2+b^2\\)?"""


USER_PROMPT = f"""Solve the problem.

Problem:
{MATH_PROBLEM}

Return only this JSON schema:
{{
  "answer": "final answer only",
  "confidence": 0,
  "reasoning_summary": "brief summary",
  "weak_point": "possible uncertainty or none"
}}
"""


REVISION_PROMPT = f"""You are Agent A. You previously solved this problem.

Problem:
{MATH_PROBLEM}

Your previous answer:
{{"answer": "58", "confidence": 70, "reasoning_summary": "Used a^2+b^2=(a+b)^2-2ab.", "weak_point": "none"}}

Partner critique:
Your formula is correct. Since a+b=10 and ab=21, a^2+b^2=100-42=58.

Revise your answer if needed.

Return only this JSON schema:
{{
  "answer": "final answer only",
  "confidence": 0,
  "reasoning_summary": "brief summary",
  "weak_point": "possible uncertainty or none"
}}
"""


def post_json(url, payload, timeout=240):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    started = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            text = resp.read().decode("utf-8", errors="replace")
            elapsed = time.time() - started
            return {
                "ok": True,
                "status": resp.status,
                "elapsed_sec": elapsed,
                "text": text,
            }
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        elapsed = time.time() - started
        return {
            "ok": False,
            "status": e.code,
            "elapsed_sec": elapsed,
            "text": body,
            "error": repr(e),
        }
    except Exception as e:
        elapsed = time.time() - started
        return {
            "ok": False,
            "status": None,
            "elapsed_sec": elapsed,
            "text": "",
            "error": repr(e),
        }


def try_json_loads(text):
    try:
        return json.loads(text), None
    except Exception as e:
        return None, repr(e)


def extract_chat_fields(obj):
    if not isinstance(obj, dict):
        return {}

    msg = obj.get("message")
    if not isinstance(msg, dict):
        msg = {}

    return {
        "top_level_keys": list(obj.keys()),
        "message_keys": list(msg.keys()),
        "message_content": msg.get("content", ""),
        "message_thinking": msg.get("thinking", ""),
        "response": obj.get("response", ""),
        "error": obj.get("error", ""),
        "done": obj.get("done"),
        "done_reason": obj.get("done_reason"),
        "total_duration": obj.get("total_duration"),
        "load_duration": obj.get("load_duration"),
        "prompt_eval_count": obj.get("prompt_eval_count"),
        "eval_count": obj.get("eval_count"),
    }


def simple_parse_candidate(raw):
    """
    This is not your full project parser.
    This only tells us whether the raw output contains a usable JSON object.
    """
    if raw is None:
        raw = ""

    raw = str(raw).strip()

    result = {
        "raw_len": len(raw),
        "strict_json_ok": False,
        "brace_json_ok": False,
        "parsed_keys": [],
        "answer": None,
        "confidence": None,
        "parse_error": None,
    }

    if not raw:
        result["parse_error"] = "empty_raw"
        return result

    obj, err = try_json_loads(raw)
    if isinstance(obj, dict):
        result["strict_json_ok"] = True
        result["parsed_keys"] = list(obj.keys())
        result["answer"] = obj.get("answer")
        result["confidence"] = obj.get("confidence")
        return result

    # Try first {...} block.
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        block = raw[start : end + 1]
        obj, err2 = try_json_loads(block)
        if isinstance(obj, dict):
            result["brace_json_ok"] = True
            result["parsed_keys"] = list(obj.keys())
            result["answer"] = obj.get("answer")
            result["confidence"] = obj.get("confidence")
            return result
        result["parse_error"] = err2
    else:
        result["parse_error"] = err

    return result


def print_case(title, endpoint, payload):
    print("\n" + "=" * 120)
    print(title)
    print("=" * 120)
    print("URL:", OLLAMA_HOST + endpoint)
    print("Payload keys:", list(payload.keys()))
    print("Payload format:", payload.get("format"))
    print("Payload options:", payload.get("options"))
    print("Payload think:", payload.get("think"))

    res = post_json(OLLAMA_HOST + endpoint, payload)

    print("\nHTTP OK:", res["ok"])
    print("HTTP status:", res["status"])
    print("elapsed_sec:", round(res["elapsed_sec"], 3))

    raw_http = res["text"]
    print("raw_http_len:", len(raw_http))
    print("\nRAW HTTP FIRST 3000 CHARS:")
    print(raw_http[:3000])

    obj, err = try_json_loads(raw_http)
    print("\nHTTP JSON parse error:", err)

    fields = extract_chat_fields(obj)
    print("\nExtracted response metadata:")
    for k, v in fields.items():
        if isinstance(v, str):
            print(f"{k}: len={len(v)} repr={repr(v[:500])}")
        else:
            print(f"{k}: {v}")

    # Candidate raw text candidates.
    possible_texts = {
        "message.content": fields.get("message_content", ""),
        "message.thinking": fields.get("message_thinking", ""),
        "response": fields.get("response", ""),
        "raw_http": raw_http,
    }

    print("\nParser test on possible text fields:")
    for name, text in possible_texts.items():
        parsed = simple_parse_candidate(text)
        print(f"\n--- {name} ---")
        print(json.dumps(parsed, indent=2, ensure_ascii=False))
        if text:
            print("text_preview:")
            print(str(text)[:1000])


def main():
    print("OLLAMA_HOST:", OLLAMA_HOST)
    print("MODEL:", MODEL)

    # Check tags endpoint.
    tags = post_json(OLLAMA_HOST + "/api/tags", payload={})
    print("\n/api/tags status:", tags["status"], "ok:", tags["ok"])
    print(tags["text"][:1000])

    common_options = {
        "temperature": 0.2,
        "top_p": 0.9,
        "num_ctx": 4096,
        "num_predict": 512,
    }

    tests = []

    # 1. Minimal plain chat. This tests whether model returns anything at all.
    tests.append((
        "CHAT plain tiny prompt, no JSON mode",
        "/api/chat",
        {
            "model": MODEL,
            "stream": False,
            "messages": [
                {"role": "user", "content": "Reply with exactly: hello"}
            ],
            "options": common_options,
        },
    ))

    # 2. Chat JSON mode. This may be the failing mode.
    tests.append((
        "CHAT with format=json, independent solve",
        "/api/chat",
        {
            "model": MODEL,
            "stream": False,
            "format": "json",
            "messages": [
                {"role": "system", "content": SYSTEM_JSON},
                {"role": "user", "content": USER_PROMPT},
            ],
            "options": common_options,
        },
    ))

    # 3. Chat no JSON mode, but strict JSON prompt.
    tests.append((
        "CHAT no format=json, independent solve",
        "/api/chat",
        {
            "model": MODEL,
            "stream": False,
            "messages": [
                {"role": "system", "content": SYSTEM_JSON},
                {"role": "user", "content": USER_PROMPT},
            ],
            "options": common_options,
        },
    ))

    # 4. Chat no JSON mode plus think false.
    tests.append((
        "CHAT no format=json, think=false, independent solve",
        "/api/chat",
        {
            "model": MODEL,
            "stream": False,
            "think": False,
            "messages": [
                {"role": "system", "content": SYSTEM_JSON},
                {"role": "user", "content": USER_PROMPT},
            ],
            "options": common_options,
        },
    ))

    # 5. Chat JSON mode plus think false.
    tests.append((
        "CHAT format=json, think=false, independent solve",
        "/api/chat",
        {
            "model": MODEL,
            "stream": False,
            "format": "json",
            "think": False,
            "messages": [
                {"role": "system", "content": SYSTEM_JSON},
                {"role": "user", "content": USER_PROMPT},
            ],
            "options": common_options,
        },
    ))

    # 6. Generate endpoint, no JSON mode.
    tests.append((
        "GENERATE no format=json, independent solve",
        "/api/generate",
        {
            "model": MODEL,
            "stream": False,
            "prompt": SYSTEM_JSON + "\n\n" + USER_PROMPT,
            "options": common_options,
        },
    ))

    # 7. Generate endpoint, format=json.
    tests.append((
        "GENERATE format=json, independent solve",
        "/api/generate",
        {
            "model": MODEL,
            "stream": False,
            "format": "json",
            "prompt": SYSTEM_JSON + "\n\n" + USER_PROMPT,
            "options": common_options,
        },
    ))

    # 8. Revision-like prompt. This tests A1-like behavior.
    tests.append((
        "CHAT no format=json, think=false, revision prompt",
        "/api/chat",
        {
            "model": MODEL,
            "stream": False,
            "think": False,
            "messages": [
                {"role": "system", "content": SYSTEM_JSON},
                {"role": "user", "content": REVISION_PROMPT},
            ],
            "options": common_options,
        },
    ))

    for title, endpoint, payload in tests:
        print_case(title, endpoint, payload)


if __name__ == "__main__":
    main()