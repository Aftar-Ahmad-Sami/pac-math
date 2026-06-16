from pathlib import Path

# ============================================================
# PAC-Math configuration
# No command-line arguments are used anywhere in this project.
# Edit this file when you want to change paths, models, sizes,
# or experiment settings.
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
SPLIT_DIR = DATA_DIR / "splits"
OUT_DIR = PROJECT_ROOT / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CALIBRATION_FILE = SPLIT_DIR / "math_calibration_400_seed42.jsonl"
TEST_FILE = SPLIT_DIR / "math500_test.jsonl"

# Ollama server
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_TIMEOUT_SECONDS = 900

# NVIDIA API endpoint for hosted OpenAI-compatible models.
# Put NVIDIA_API_KEY=... in your shell or in a local .env file.
# The free endpoint can be limited to 40 RPM, so use a conservative throttle.
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_API_KEY_ENV = "NVIDIA_API_KEY"
NVIDIA_TIMEOUT_SECONDS = 900
NVIDIA_RATE_LIMIT_RPM = 30.0  # keep below free-tier 40 RPM; retries/fallbacks also consume requests
NVIDIA_MAX_API_RETRIES = 6
NVIDIA_RATE_LIMIT_BACKOFF_SECONDS = 20.0
NVIDIA_STREAM = False

# Main controlled PAC-Math protocol disables model-specific hidden thinking.
# Keep native thinking as an appendix ablation only.
NVIDIA_ENABLE_THINKING = False
NVIDIA_REASONING_BUDGET = 0
NVIDIA_EXTRA_BODY = {}
# Provider routing rules.
# Any model name can be used in either agent_a_model or agent_b_model.
# - Ollama models usually look like: qwen3.6:27b, phi4:14b, gemma4:31b
# - NVIDIA-hosted models usually look like: vendor/model, e.g.
#   nvidia/nemotron-3-ultra-550b-a55b or google/gemma-4-31b-it
# Slash-style model names are routed to NVIDIA by default. Use
# MODEL_PROVIDER_OVERRIDES to force exceptions.
NVIDIA_MODEL_PREFIXES = [
    "nvidia/",
    "google/",
    "meta/",
    "mistralai/",
    "microsoft/",
    "deepseek-ai/",
    "qwen/",
    "moonshotai/",
    "openai/",
]
NVIDIA_ROUTE_SLASH_MODELS = True
MODEL_PROVIDER_OVERRIDES = {
    "nvidia/nemotron-3-ultra-550b-a55b": "nvidia",
    "google/gemma-4-31b-it": "nvidia",
}

# Keep the first run small. After smoke/pilot works, run scripts/run_full.py.
SMOKE_CALIBRATION_N = 8
SMOKE_TEST_N = 8
PILOT_CALIBRATION_N = 200
PILOT_TEST_N = 120
FULL_CALIBRATION_N = 400
FULL_TEST_N = None  # None means all rows in TEST_FILE.

# Models available from your ollama list:
# llama3.1:8b, qwen3:8b, gemma4:26b, gemma4:31b
# Your RTX 2000 Ada has 16 GB VRAM. gemma4:26b/31b may be slow or may spill to CPU.
# Start with qwen3:8b and llama3.1:8b. Add gemma only after the pipeline works.
MODEL_PAIRS = [
    # v22 mixed local/API pilot pair. Explicit prefixes make the provider
    # unambiguous and the pair_id now matches the actual model source.
    {
        "pair_id": "ollama_gemma4_31b__nvidia_nemotron3_ultra_550b_standard_v23",
        "agent_a_model": "ollama:gemma4:31b",
        "agent_b_model": "nvidia:nvidia/nemotron-3-ultra-550b-a55b",
    },
    {
        "pair_id": "nvidia_nemotron3_ultra_550b__ollama_gemma4_31b_standard_v23",
        "agent_a_model": "nvidia:nvidia/nemotron-3-ultra-550b-a55b",
        "agent_b_model": "ollama:gemma4:31b",
    },

    # Hosted/API-only version. Enable this only if you want NVIDIA-served Gemma.
    # {"pair_id": "nvidia_google_gemma4_31b_it__nvidia_nemotron3_ultra_550b_standard_v23", "agent_a_model": "nvidia:google/gemma-4-31b-it", "agent_b_model": "nvidia:nvidia/nemotron-3-ultra-550b-a55b"},
    # {"pair_id": "nvidia_nemotron3_ultra_550b__nvidia_google_gemma4_31b_it_standard_v23", "agent_a_model": "nvidia:nvidia/nemotron-3-ultra-550b-a55b", "agent_b_model": "nvidia:google/gemma-4-31b-it"},

    # Completed local baselines. Keep disabled unless intentionally recomputing.
    # {"pair_id": "qwen36_27b__phi4_14b_standard_v19", "agent_a_model": "qwen3.6:27b", "agent_b_model": "phi4:14b"},
    # {"pair_id": "phi4_14b__qwen36_27b_standard_v19", "agent_a_model": "phi4:14b", "agent_b_model": "qwen3.6:27b"},
    # {"pair_id": "qwen3_8b__phi4_14b_standard_v16", "agent_a_model": "qwen3:8b", "agent_b_model": "phi4:14b"},
    # {"pair_id": "phi4_14b__qwen3_8b_standard_v16", "agent_a_model": "phi4:14b", "agent_b_model": "qwen3:8b"},
]

# Generation settings
# Generation settings. Keep temperature low for reproducibility.
GENERATION_OPTIONS = {
    "temperature": 0.2,
    "top_p": 0.9,
    "num_ctx": 8192,
    # 700 was too small for NVIDIA/Gemma style JSON responses and caused
    # done_reason=length / eval_count=700 truncation. Keep this large enough
    # for complete JSON but still bounded for cost and latency.
    "num_predict": 2600,
    "seed": 42,
}

# If a candidate hits the generation length limit, PAC-Math automatically
# retries that JSON request once with this larger cap before marking it failed.
LENGTH_RETRY_ENABLED = True
LENGTH_RETRY_NUM_PREDICT = 5000


# v18 Ollama compatibility controls. Qwen-style reasoning models can return
# empty final content through /api/generate. Disable thinking when supported and
# fall back to /api/chat before marking a candidate as PARSE_ERROR.
OLLAMA_DISABLE_THINKING = True
OLLAMA_CHAT_FALLBACK = True
OLLAMA_PREFER_CHAT = True

# Debate and parsing
DEBATE_ROUNDS = 2  # implemented as critique round + revision round
MAX_JSON_RETRIES = 4

# Reliability smoothing
ALPHA = 10.0
MIN_SPECIFIC_COUNT = 0  # hierarchical formula handles sparse counts; no hard threshold needed.

# Topic handling
# Main MATH topics are kept as provided. Unknown/missing topics are mapped to "unknown".
TOPIC_COL_CANDIDATES = ["topic", "type", "subject", "category"]
PROBLEM_COL_CANDIDATES = ["problem", "question", "Problem"]
ANSWER_COL_CANDIDATES = ["answer", "solution", "Answer", "final_answer"]


# Cache/protocol versioning
# Any change that affects generated A0/B0/A1/B1 records must bump this string.
# The runner will automatically ignore and regenerate cached records whose
# protocol_version does not match this value. This prevents silently reusing
# stale records from older debate protocols.
PROTOCOL_VERSION = "standard_debate_v23_nvidia_rate_norm_fix"

# Experiment behavior
SAVE_EVERY_N_PROBLEMS = 5
STOP_ON_OLLAMA_ERROR = False
PYTHONHASHSEED = 42

# Cache/retry behavior. A record containing A0/B0/A1/B1 but with PARSE_ERROR
# should not be treated as complete; otherwise bad NVIDIA/Ollama responses are
# cached forever and calibration/test summaries silently include failures.
REQUIRE_ALL_CANDIDATES_PARSE_OK = True
RECORD_PARSE_RETRY_ATTEMPTS = 2
INSPECT_ONLY_CURRENT_PROTOCOL = True


# Adaptive PAC selector
# This is the non-static, calibration-trained selector. It learns candidate
# correctness and C2W-risk models on the calibration split only, then freezes
# them for pilot/full test evaluation.
ADAPTIVE_SELECTOR_ENABLED = True

# Main paper method. Keep this fixed unless the research claim changes.
PRIMARY_METHOD = "pac_math_pair_topic_stage"

ADAPTIVE_LAMBDA_GRID = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
ADAPTIVE_SUPPORT_GRID = [0.0, 0.01, 0.03, 0.05]
ADAPTIVE_CROSS_STAGE_GRID = [0.0, 0.02, 0.05]
ADAPTIVE_INDEPENDENT_GRID = [0.0, 0.01, 0.03]

# Verifier-enhanced PAC selector
# This adds an extra no-gold LLM verification pass over A0/B0/A1/B1.
# It is slower, but it makes the core method dynamic rather than a static table lookup.
VERIFIER_ENABLED = False
VERIFIER_MODEL = "phi4:14b"  # use the stronger partner as the verifier
VERIFIER_OPTIONS = {
    "temperature": 0.0,
    "top_p": 0.9,
    "num_ctx": 8192,
    "num_predict": 700,
    "seed": 42,
}
MAX_VERIFIER_RETRIES = 1

# Debate protocol
# v16 deliberately disables evidence gating. The evidence gate reduced useful post-debate corrections
# in the full run, so the next valid experiment is standard debate with v15 cache protection.
EVIDENCE_GATED_DEBATE = False
MIN_CHANGE_JUSTIFICATION_SCORE = 70.0
MIN_REVISED_VALIDITY_SCORE = 60.0
MAX_INITIAL_VALIDITY_FOR_CHANGE = 65.0
