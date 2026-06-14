# PAC-Math v21 Provider Routing + Length Retry

This version fixes two issues from v20:

1. **Agent A and Agent B can each be either Ollama or NVIDIA.**
   - Ollama model names normally use colon tags, for example `gemma4:31b`, `phi4:14b`, `qwen3.6:27b`.
   - NVIDIA-hosted model names normally use slash names, for example `nvidia/nemotron-3-ultra-550b-a55b` or `google/gemma-4-31b-it`.
   - Slash-style names are routed to NVIDIA by default, so `google/gemma-4-31b-it` will no longer be sent to Ollama.
   - Explicit prefixes are also supported:
     - `nvidia:google/gemma-4-31b-it`
     - `ollama:gemma4:31b`

2. **Automatic retry after generation length cutoff.**
   - The default `num_predict` was increased from 700 to 1400.
   - If a response still ends with `done_reason=length` or `eval_count == num_predict`, the same JSON request is retried once with `LENGTH_RETRY_NUM_PREDICT = 2200`.

## Important config fields

```python
NVIDIA_ROUTE_SLASH_MODELS = True
MODEL_PROVIDER_OVERRIDES = {
    "nvidia/nemotron-3-ultra-550b-a55b": "nvidia",
    "google/gemma-4-31b-it": "nvidia",
}

GENERATION_OPTIONS = {
    "temperature": 0.2,
    "top_p": 0.9,
    "num_ctx": 8192,
    "num_predict": 1400,
    "seed": 42,
}

LENGTH_RETRY_ENABLED = True
LENGTH_RETRY_NUM_PREDICT = 2200
```

## Debug commands

Check routing without making model calls:

```bash
python scripts/debug_provider_routing.py
```

Test a specific NVIDIA model endpoint:

```bash
export NVIDIA_API_KEY="your_key_here"
python scripts/debug_nvidia_model_api.py
```

Then run pilot:

```bash
python scripts/run_pilot.py
python scripts/summarize_results.py
python scripts/diagnose_experiment.py
python scripts/audit_experiment.py
python scripts/export_main_tables.py
```

Do not run full until pilot parse rates and C2W behavior are acceptable.
