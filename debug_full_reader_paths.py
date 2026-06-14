from pathlib import Path
import json
from collections import Counter


ROOT = Path(".")
FULL = ROOT / "outputs" / "full"
COMBINED = FULL / "combined_test_method_rows.jsonl"


def count_lines(path):
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


print("=" * 100)
print("FULL DIRECTORY FILES")
print("=" * 100)
for p in sorted(FULL.glob("*")):
    if p.is_file():
        print(f"{p} | lines={count_lines(p)}")
    else:
        print(f"{p}/")


print("\n" + "=" * 100)
print("METHOD ROW FILES")
print("=" * 100)
for p in sorted((FULL / "method_rows").glob("*.jsonl")):
    print(f"{p.name} | lines={count_lines(p)}")


print("\n" + "=" * 100)
print("RECORD FILES")
print("=" * 100)
for p in sorted((FULL / "records").glob("*.jsonl")):
    print(f"{p.name} | lines={count_lines(p)}")


print("\n" + "=" * 100)
print("COMBINED FILE CHECK")
print("=" * 100)

if not COMBINED.exists():
    print("MISSING:", COMBINED)
else:
    rows = []
    with COMBINED.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    print("combined path:", COMBINED)
    print("rows:", len(rows))

    pair_counts = Counter(r.get("pair_id") for r in rows)
    method_counts = Counter(r.get("method") for r in rows)

    print("\npairs:")
    for k, v in sorted(pair_counts.items()):
        print(k, v)

    print("\nmethods:", len(method_counts))
    for k, v in sorted(method_counts.items()):
        print(k, v)

    print("\nfirst row keys:")
    print(sorted(rows[0].keys()))

    # Possible problem id fields
    for key in ["problem_index", "problem_id", "problem_hash", "run_id"]:
        vals = {r.get(key) for r in rows if key in r}
        print(f"unique {key}:", len(vals))


print("\n" + "=" * 100)
print("SCRIPT PATH REFERENCES")
print("=" * 100)

keywords = [
    "combined_test_method_rows",
    "method_rows",
    "summary_methods",
    "outputs/full",
    "full",
]

for script in [
    "scripts/summarize_results.py",
    "scripts/diagnose_experiment.py",
    "scripts/audit_experiment.py",
    "scripts/export_main_tables.py",
]:
    path = ROOT / script
    print("\n---", script, "---")
    if not path.exists():
        print("MISSING")
        continue

    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for i, line in enumerate(lines, 1):
        if any(k in line for k in keywords):
            print(f"{i:04d}: {line}")