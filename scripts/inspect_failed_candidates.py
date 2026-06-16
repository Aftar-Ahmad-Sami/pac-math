from pathlib import Path
import json
import config

SPLIT = "pilot"
PAIR_ID = config.MODEL_PAIRS[0]["pair_id"] if config.MODEL_PAIRS else ""
PATHS = [
    config.OUT_DIR / SPLIT / "records" / f"test_{PAIR_ID}.jsonl",
    config.OUT_DIR / SPLIT / "cache" / f"test_{PAIR_ID}.jsonl",
]

path = next((p for p in PATHS if p.exists()), None)
if path is None:
    raise FileNotFoundError("Could not find candidate file in: " + ", ".join(str(p) for p in PATHS))

print(f"Reading: {path}")
shown = 0
with path.open("r", encoding="utf-8") as f:
    for line_num, line in enumerate(f, 1):
        if not line.strip():
            continue
        r = json.loads(line)
        candidates = r.get("candidates", [])
        if not isinstance(candidates, list):
            continue
        for c in candidates:
            if not isinstance(c, dict):
                continue
            if c.get("parse_ok") is False or c.get("answer") in [None, "", "PARSE_ERROR"]:
                print("\n" + "=" * 100)
                print("line_num:", line_num)
                print("run_id:", r.get("run_id"))
                print("candidate_id:", c.get("candidate_id"))
                print("model:", c.get("model"))
                print("stage:", c.get("stage"))
                print("parse_ok:", c.get("parse_ok"))
                print("answer:", c.get("answer"))
                print("attempts:", c.get("attempts"))
                raw = c.get("raw_response") or ""
                print("raw_response_len:", len(raw))
                print("raw_response_head:", raw[:1000])
                trace = c.get("request_trace", [])
                print("request_trace_len:", len(trace))
                for i, item in enumerate(trace[:8], 1):
                    print(f"  trace[{i}]:", item)
                shown += 1
        if shown >= 10:
            break
print("\nShown failed candidates:", shown)
