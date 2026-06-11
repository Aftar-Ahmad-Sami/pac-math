import json
from pathlib import Path


path = Path("outputs/pilot/records/test_qwen36_27b__phi4_14b_standard_v17.jsonl")
if not path.exists():
    path = Path("outputs/pilot/cache/test_qwen36_27b__phi4_14b_standard_v17.jsonl")

print("Reading:", path)


def get_candidate(r, cid):
    cands = r.get("candidates")

    if isinstance(cands, dict):
        return cands.get(cid, {}) or r.get(cid, {})

    if isinstance(cands, list):
        for c in cands:
            if not isinstance(c, dict):
                continue

            if (
                c.get("candidate_id") == cid
                or c.get("id") == cid
                or c.get("cid") == cid
                or c.get("name") == cid
            ):
                return c

    return r.get(cid, {}) if isinstance(r.get(cid, {}), dict) else {}


def get_raw_text(c):
    return (
        c.get("raw_response")
        or c.get("raw")
        or c.get("text")
        or c.get("content")
        or ""
    )


# First, print structure of the first record.
with path.open("r", encoding="utf-8") as f:
    first = json.loads(next(f))

print("\nTop-level keys:")
print(list(first.keys()))

print("\nCandidate structure:")
cands = first.get("candidates", [])
print(type(cands), "len =", len(cands) if isinstance(cands, list) else "NA")
if isinstance(cands, list) and cands:
    print("candidate keys:", list(cands[0].keys()))
    print("candidate ids:", [c.get("candidate_id") for c in cands if isinstance(c, dict)])


# Now show failed/low-parse qwen3.6 candidates.
shown = 0

with path.open("r", encoding="utf-8") as f:
    for line_num, line in enumerate(f, 1):
        line = line.strip()
        if not line:
            continue

        r = json.loads(line)

        for cid in ["A0", "A1"]:
            c = get_candidate(r, cid)

            parse_ok = c.get("parse_ok")
            answer = c.get("answer")
            raw = get_raw_text(c)

            if parse_ok is False or answer in [None, "", "PARSE_ERROR"]:
                print("\n" + "=" * 100)
                print("line_num:", line_num)
                print("run_id:", r.get("run_id"))
                print("problem_index:", r.get("problem_index"))
                print("problem_hash:", r.get("problem_hash"))
                print("candidate:", cid)
                print("model:", c.get("model"))
                print("parse_ok:", parse_ok)
                print("answer:", answer)
                print("confidence:", c.get("confidence"))
                print("attempts:", c.get("attempts"))
                print("normalized_answer:", c.get("normalized_answer"))
                print("normalization_status:", c.get("normalization_status"))
                print("is_correct:", c.get("is_correct"))
                print("raw_response_len:", len(str(raw)))
                print("\nRAW RESPONSE START")
                print(str(raw)[:5000])
                print("RAW RESPONSE END")

                shown += 1

        if shown >= 8:
            break

print("\nShown failed candidates:", shown)