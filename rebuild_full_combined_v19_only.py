import json
from pathlib import Path
from collections import Counter

FULL = Path("outputs/full")
METHOD_ROWS = FULL / "method_rows"
OUT = FULL / "combined_test_method_rows.jsonl"

wanted_pairs = {
    "qwen36_27b__phi4_14b_standard_v19",
    "phi4_14b__qwen36_27b_standard_v19",
}

files = sorted(METHOD_ROWS.glob("test_*standard_v19*_methods.jsonl"))

print("Using TEST method row files only:")
for f in files:
    print(" ", f.name)

rows = []
bad_rows = 0

for f in files:
    with f.open("r", encoding="utf-8") as r:
        for line_num, line in enumerate(r, 1):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except Exception as e:
                bad_rows += 1
                print(f"BAD JSON: {f.name}:{line_num}: {e}")
                continue

            pair_id = obj.get("pair_id")
            if pair_id not in wanted_pairs:
                print(f"SKIP unexpected pair_id in {f.name}:{line_num}: {pair_id}")
                continue

            rows.append(obj)

backup = FULL / "combined_test_method_rows.BAD_45000_BACKUP.jsonl"
if OUT.exists():
    OUT.rename(backup)
    print("Backed up old combined file to:", backup)

with OUT.open("w", encoding="utf-8") as w:
    for obj in rows:
        w.write(json.dumps(obj, ensure_ascii=False) + "\n")

print("\nWrote:", OUT)
print("Rows:", len(rows))
print("Bad rows:", bad_rows)

pair_counts = Counter(r.get("pair_id") for r in rows)
method_counts = Counter(r.get("method") for r in rows)

print("\nPairs:")
for k, v in sorted(pair_counts.items()):
    print(k, v)

print("\nMethods:", len(method_counts))
for k, v in sorted(method_counts.items()):
    print(k, v)

print("\nExpected:")
print("rows = 25000")
print("methods = 25")
print("pairs = 2")
print("each method count = 1000")