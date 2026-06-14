from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from pacmath.ollama_client import build_model_client, safe_json_loads


def test_model(model: str):
    client = build_model_client(config)
    prompt = (
        "Return exactly one JSON object with these keys: "
        "answer, confidence, reasoning_summary, weak_point. "
        "Problem: If x=2, what is x+3?"
    )
    print("model:", model)
    print("provider:", client.provider_for_model(model))
    print("routed_model:", client.routed_model_name(model))
    resp = client.chat(
        model=model,
        prompt=prompt,
        system="You are a precise math assistant. Return JSON only.",
        options={"temperature": 0.0, "top_p": 0.9, "num_predict": 500},
        json_mode=True,
        think=False,
    )
    print("raw_provider:", (resp.raw or {}).get("provider"))
    print("done_reason:", (resp.raw or {}).get("done_reason"))
    print("text_len:", len(resp.text or ""))
    print("text:", resp.text)
    parsed = safe_json_loads(resp.text)
    print("parsed:", parsed)
    if not parsed:
        raise SystemExit("Test failed: response was not parseable JSON")


if __name__ == "__main__":
    # Edit this line for any NVIDIA-hosted model listed on build.nvidia.com.
    test_model("google/gemma-4-31b-it")
