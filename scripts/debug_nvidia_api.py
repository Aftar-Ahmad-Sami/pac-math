from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from pacmath.ollama_client import build_model_client, safe_json_loads


def main():
    model = "nvidia/nemotron-3-ultra-550b-a55b"
    client = build_model_client(config)
    prompt = (
        "Return exactly one JSON object with these keys: "
        "answer, confidence, reasoning_summary, weak_point. "
        "Problem: If x=2, what is x+3?"
    )
    resp = client.chat(
        model=model,
        prompt=prompt,
        system="You are a precise math assistant. Return JSON only.",
        options={"temperature": 0.0, "top_p": 0.9, "num_predict": 300},
        json_mode=True,
        think=False,
    )
    print("provider:", (resp.raw or {}).get("provider"))
    print("text_len:", len(resp.text or ""))
    print("text:", resp.text)
    print("tokens:", resp.total_tokens)
    raw = resp.raw or {}
    msg = raw.get("message", {}) if isinstance(raw.get("message"), dict) else {}
    print("reasoning_content_len:", msg.get("reasoning_content_len"))
    print("done_reason:", raw.get("done_reason"))
    parsed = safe_json_loads(resp.text)
    print("parsed:", parsed)
    if not parsed:
        raise SystemExit("NVIDIA test failed: response was not parseable JSON")
    print("NVIDIA API test OK")


if __name__ == "__main__":
    main()
