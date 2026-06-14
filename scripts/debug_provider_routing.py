from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from pacmath.ollama_client import build_model_client


def main():
    client = build_model_client(config)
    models = [
        "gemma4:31b",
        "phi4:14b",
        "qwen3.6:27b",
        "nvidia/nemotron-3-ultra-550b-a55b",
        "google/gemma-4-31b-it",
        "nvidia:google/gemma-4-31b-it",
        "ollama:gemma4:31b",
    ]
    print("Provider routing check")
    for model in models:
        print(f"{model:45s} -> provider={client.provider_for_model(model):7s} routed_model={client.routed_model_name(model)}")

    print("\nConfigured MODEL_PAIRS:")
    for pair in config.MODEL_PAIRS:
        a = pair["agent_a_model"]
        b = pair["agent_b_model"]
        print(f"{pair['pair_id']}: A={a} [{client.provider_for_model(a)}], B={b} [{client.provider_for_model(b)}]")


if __name__ == "__main__":
    main()
