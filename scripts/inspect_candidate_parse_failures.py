from __future__ import annotations

import json
from pathlib import Path
from collections import Counter, defaultdict

import config


def iter_records(split: str):
    rec_dir = config.OUT_DIR / split / "records"
    for path in sorted(rec_dir.glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield path, json.loads(line)


def summarize(split: str = "pilot") -> None:
    failures = []
    candidate_counts = Counter()
    pair_counts = Counter()
    model_counts = Counter()

    current_only = bool(getattr(config, "INSPECT_ONLY_CURRENT_PROTOCOL", True))
    current_protocol = str(getattr(config, "PROTOCOL_VERSION", ""))
    skipped_old = 0

    for path, rec in iter_records(split):
        if current_only and str(rec.get("protocol_version", "")) != current_protocol:
            skipped_old += 1
            continue
        for cand in rec.get("candidates", []) or []:
            cid = str(cand.get("candidate_id", ""))
            candidate_counts[(rec.get("pair_id"), cid)] += 1
            if bool(cand.get("parse_ok", True)):
                continue
            trace = cand.get("request_trace") or []
            last = trace[-1] if isinstance(trace, list) and trace else {}
            item = {
                "file": path.name,
                "pair_id": rec.get("pair_id"),
                "problem_index": rec.get("problem_index"),
                "candidate_id": cid,
                "agent_id": cand.get("agent_id"),
                "model": cand.get("model"),
                "stage": cand.get("stage"),
                "answer": cand.get("answer"),
                "attempts": cand.get("attempts"),
                "raw_response_head": str(cand.get("raw_response", ""))[:300],
                "last_trace_kind": last.get("kind") if isinstance(last, dict) else None,
                "last_trace_json_mode": last.get("json_mode") if isinstance(last, dict) else None,
                "last_trace_error": last.get("error") if isinstance(last, dict) else None,
                "last_trace_text_head": last.get("text_head") if isinstance(last, dict) else None,
                "last_trace_done_reason": last.get("done_reason") if isinstance(last, dict) else None,
                "last_trace_length_limited": last.get("length_limited") if isinstance(last, dict) else None,
            }
            failures.append(item)
            pair_counts[item["pair_id"]] += 1
            model_counts[str(item["model"])] += 1

    print(f"Split: {split}")
    if current_only:
        print(f"Current protocol only: {current_protocol}")
        print(f"Skipped old-protocol records: {skipped_old}")
    print(f"Parse failures: {len(failures)}")
    print("\nFailures by pair:")
    for k, v in pair_counts.most_common():
        print(f"  {k}: {v}")
    print("\nFailures by model:")
    for k, v in model_counts.most_common():
        print(f"  {k}: {v}")
    print("\nFirst 30 failures:")
    for item in failures[:30]:
        print(json.dumps(item, ensure_ascii=False))

    out = config.OUT_DIR / split / "candidate_parse_failures.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for item in failures:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    for split in ["smoke", "pilot", "full"]:
        if (config.OUT_DIR / split / "records").exists():
            summarize(split)
            print("\n" + "=" * 100 + "\n")
