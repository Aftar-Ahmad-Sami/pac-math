# PAC-Math v23 NVIDIA rate-limit + normalizer fix

This build fixes the NVIDIA/Gemma pilot issues observed after v21/v22:

- Provider routing still supports either Agent A or Agent B from Ollama or NVIDIA.
- NVIDIA API key handling no longer prints the configured env var value when it is accidentally set to the key.
- NVIDIA 429 rate-limit errors are retried with conservative backoff.
- NVIDIA request rate is lowered to 30 RPM by default.
- JSON generation budget is increased (`num_predict=2600`, length retry `5000`).
- Prompts now explicitly require escaped JSON backslashes for LaTeX and prefer plain ASCII final answers.
- Answer normalization handles common MATH equivalences:
  - `90` vs `90^\circ`
  - `80%` vs `80\%`
  - `312` vs `312 dollars`
  - `20` vs `20 text{sq units}`
  - `sqrt(85)` vs `√85`
  - `4sqrt(65)` vs `4√65`
  - `.039` vs `0.039`
  - `\dfrac23` vs `2/3`
  - `(4.2,4.6)` vs `(21/5,23/5)`
  - unordered short answer lists like `E,H` vs `H,E`
- `inspect_candidate_parse_failures.py` reports only the current protocol by default, so old v16/v17 pilot cache files no longer dominate the output.

## Important

If your console still shows pair IDs ending in `standard_v21`, you are not running this v23 code/config.
The active v23 pair IDs are:

- `ollama_gemma4_31b__nvidia_nemotron3_ultra_550b_standard_v23`
- `nvidia_nemotron3_ultra_550b__ollama_gemma4_31b_standard_v23`

The protocol version is:

- `standard_debate_v23_nvidia_rate_norm_fix`

You can rename the pair IDs to v23 if desired; the protocol version is what controls cache reuse.

## Run

```bash
unset NVIDIA_API_KEY_ENV
export NVIDIA_API_KEY="your_key_here"

python scripts/debug_provider_routing.py
python scripts/debug_nvidia_model_api.py
python scripts/run_pilot.py
python scripts/summarize_results.py
python scripts/diagnose_experiment.py
python scripts/audit_experiment.py
python scripts/export_main_tables.py
python scripts/inspect_candidate_parse_failures.py
```
