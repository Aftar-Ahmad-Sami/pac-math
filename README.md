# PAC-Math v20: NVIDIA API Endpoint Support

This version adds a mixed-provider model client so PAC-Math can run local Ollama
models and NVIDIA OpenAI-compatible API models in the same debate pair.

Main new pair in `config.py`:

```python
MODEL_PAIRS = [
    {
        "pair_id": "gemma4_31b__nemotron3_ultra_550b_standard_v20",
        "agent_a_model": "gemma4:31b",
        "agent_b_model": "nvidia/nemotron-3-ultra-550b-a55b",
    },
    {
        "pair_id": "nemotron3_ultra_550b__gemma4_31b_standard_v20",
        "agent_a_model": "nvidia/nemotron-3-ultra-550b-a55b",
        "agent_b_model": "gemma4:31b",
    },
]
```

## Setup

```bash
pip install -r requirements.txt
export NVIDIA_API_KEY="your_key_here"
```

or create a local `.env` file:

```bash
NVIDIA_API_KEY=your_key_here
```

## NVIDIA controls

`config.py` contains:

```python
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_RATE_LIMIT_RPM = 38.0
NVIDIA_STREAM = True
NVIDIA_ENABLE_THINKING = False
NVIDIA_REASONING_BUDGET = 0
PRIMARY_METHOD = "pac_math_pair_topic_stage"
```

The main controlled experiment keeps native hidden thinking disabled. This keeps
Ollama and NVIDIA models under the same observable JSON-output protocol. If you
want a native-thinking ablation later, set `NVIDIA_ENABLE_THINKING = True` and
use a separate protocol version.

## Test the NVIDIA endpoint first

```bash
python scripts/debug_nvidia_api.py
```

Expected: parseable JSON and `NVIDIA API test OK`.

## Recommended run order

```bash
python scripts/run_pilot.py
python scripts/summarize_results.py
python scripts/diagnose_experiment.py
python scripts/audit_experiment.py
python scripts/export_main_tables.py
```

Run full only if pilot passes:

- all A0/B0/A1/B1 parse rates are near 1.0;
- candidate oracle possible is high enough;
- `pac_math_pair_topic_stage` beats or matches `stateless_debate` accuracy;
- `pac_math_pair_topic_stage` reduces C2W relative to `stateless_debate`.

## Main changes

1. Added `NvidiaClient` using the OpenAI SDK and NVIDIA base URL.
2. Added process-local rate limiting for hosted API calls.
3. Added `MultiProviderClient` to route `nvidia/...` models to NVIDIA and all other models to Ollama.
4. Kept PAC-Math parsing over visible `.content`, not hidden `reasoning_content`.
5. Added `scripts/debug_nvidia_api.py`.
6. Made CSV writing one-physical-line-per-row to avoid stale/mixed output confusion.
7. Set `PRIMARY_METHOD = "pac_math_pair_topic_stage"` for safer paper comparisons.
