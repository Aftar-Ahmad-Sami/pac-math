# PAC-Math v22: Provider-Explicit Routing + Parse-Clean Cache

This version fixes two issues seen in the mixed Ollama/NVIDIA pilot:

1. **Provider ambiguity**: the model pair now uses explicit provider prefixes.
   - `ollama:gemma4:31b` routes to local Ollama.
   - `nvidia:nvidia/nemotron-3-ultra-550b-a55b` routes to NVIDIA API.

2. **PARSE_ERROR caching**: records with A0/B0/A1/B1 are no longer considered complete if any candidate has `parse_ok=false` when `REQUIRE_ALL_CANDIDATES_PARSE_OK=True`. This prevents bad NVIDIA/Ollama outputs from being reused silently.

Key settings in `config.py`:

```python
PROTOCOL_VERSION = "standard_debate_v22_provider_parse_retry"
NVIDIA_STREAM = False
MAX_JSON_RETRIES = 3
REQUIRE_ALL_CANDIDATES_PARSE_OK = True
RECORD_PARSE_RETRY_ATTEMPTS = 1
```

Default model pairs:

```python
ollama:gemma4:31b  x  nvidia:nvidia/nemotron-3-ultra-550b-a55b
nvidia:nvidia/nemotron-3-ultra-550b-a55b  x  ollama:gemma4:31b
```

Before running:

```bash
export NVIDIA_API_KEY="your_key_here"
python scripts/debug_provider_routing.py
python scripts/debug_nvidia_model_api.py
```

Clean v22 only:

```bash
rm -f outputs/pilot/records/*standard_v22*.jsonl
rm -f outputs/pilot/records/*ollama_gemma4_31b__nvidia_nemotron3_ultra_550b*.jsonl
rm -f outputs/pilot/records/*nvidia_nemotron3_ultra_550b__ollama_gemma4_31b*.jsonl
rm -f outputs/pilot/method_rows/*standard_v22*.jsonl
rm -f outputs/pilot/method_rows/*ollama_gemma4_31b__nvidia_nemotron3_ultra_550b*.jsonl
rm -f outputs/pilot/method_rows/*nvidia_nemotron3_ultra_550b__ollama_gemma4_31b*.jsonl
rm -f outputs/pilot/combined_test_method_rows.jsonl outputs/pilot/combined_test_method_rows.csv
rm -f outputs/pilot/summary_methods.csv outputs/pilot/mcnemar_primary_comparisons.csv outputs/pilot/summary_by_topic.csv
rm -f outputs/pilot/audit_segments.csv outputs/pilot/diagnostic_*.csv
rm -f outputs/pilot/main_table_methods.csv outputs/pilot/main_table_methods.tex
rm -f outputs/pilot/appendix_table_methods.csv outputs/pilot/appendix_table_methods.tex
find outputs/pilot/memory -type f -name '*standard_v22*' -delete 2>/dev/null
find outputs/pilot/selectors -type f -name '*standard_v22*' -delete 2>/dev/null
```

Run pilot:

```bash
python scripts/run_pilot.py
python scripts/summarize_results.py
python scripts/diagnose_experiment.py
python scripts/audit_experiment.py
python scripts/export_main_tables.py
python scripts/inspect_candidate_parse_failures.py
```

Do not run full until every candidate parse rate is at least 0.98 for both role orders.
