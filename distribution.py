from pathlib import Path
import pandas as pd

split_dir = Path("data/splits")

calib = pd.read_json(split_dir / "math_calibration_400_seed42.jsonl", lines=True)
test = pd.read_json(split_dir / "math500_test.jsonl", lines=True)

print("Calibration size:", len(calib))
print("Test size:", len(test))

print("\nCalibration columns:")
print(calib.columns.tolist())

print("\nTest columns:")
print(test.columns.tolist())

print("\nCalibration topic distribution:")
if "topic" in calib.columns:
    print(calib["topic"].value_counts())
elif "type" in calib.columns:
    print(calib["type"].value_counts())
else:
    print("No topic/type column found.")

overlap = set(calib["problem_hash"]) & set(test["problem_hash"])
print("\nOverlap between calibration and MATH-500:", len(overlap))

assert len(calib) == 400
assert len(test) == 500
assert len(overlap) == 0

print("\nAll checks passed.")