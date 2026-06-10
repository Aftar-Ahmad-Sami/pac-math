from pathlib import Path
import hashlib
import re
import pandas as pd
from datasets import load_dataset

OUT_DIR = Path("data")
RAW_DIR = OUT_DIR / "raw"
SPLIT_DIR = OUT_DIR / "splits"

RAW_DIR.mkdir(parents=True, exist_ok=True)
SPLIT_DIR.mkdir(parents=True, exist_ok=True)


def normalize_text(text: str) -> str:
    text = str(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def problem_hash(problem: str) -> str:
    return hashlib.sha256(normalize_text(problem).encode("utf-8")).hexdigest()


def find_problem_col(df: pd.DataFrame) -> str:
    for col in ["problem", "question", "Problem"]:
        if col in df.columns:
            return col
    raise ValueError(f"Could not find problem/question column. Columns: {list(df.columns)}")


def save_dataset(ds, path: Path) -> pd.DataFrame:
    df = pd.DataFrame(ds)

    problem_col = find_problem_col(df)
    df["problem_hash"] = df[problem_col].apply(problem_hash)

    df.to_json(path.with_suffix(".jsonl"), orient="records", lines=True, force_ascii=False)
    df.to_csv(path.with_suffix(".csv"), index=False)

    print(f"Saved {len(df):,} rows to {path.with_suffix('.jsonl')}")
    return df


print("Downloading MATH-500...")
math500 = load_dataset("HuggingFaceH4/MATH-500", split="test")
math500_df = save_dataset(math500, RAW_DIR / "math500")

print("Downloading full MATH training dataset...")
math_full = load_dataset("qwedsacf/competition_math")

print("Available splits:", list(math_full.keys()))

if "train" not in math_full:
    raise KeyError(f"No train split found. Available splits: {list(math_full.keys())}")

math_train_df = save_dataset(math_full["train"], RAW_DIR / "math_train")

# Some mirrors do not provide a test split. That is okay because MATH-500 is the main test set.
if "test" in math_full:
    _ = save_dataset(math_full["test"], RAW_DIR / "math_test")
else:
    print("No test split found in qwedsacf/competition_math. Skipping math_test because MATH-500 is used as test.")

print("Removing MATH-500 overlap from MATH train...")

math500_problem_col = find_problem_col(math500_df)
math_train_problem_col = find_problem_col(math_train_df)

math500_hashes = set(math500_df["problem_hash"])

calib_pool_df = math_train_df[~math_train_df["problem_hash"].isin(math500_hashes)].copy()

# Normalize topic column name
if "type" in calib_pool_df.columns and "topic" not in calib_pool_df.columns:
    calib_pool_df["topic"] = calib_pool_df["type"]
elif "subject" in calib_pool_df.columns and "topic" not in calib_pool_df.columns:
    calib_pool_df["topic"] = calib_pool_df["subject"]
elif "category" in calib_pool_df.columns and "topic" not in calib_pool_df.columns:
    calib_pool_df["topic"] = calib_pool_df["category"]
else:
    calib_pool_df["topic"] = "unknown"

calib_pool_df.to_json(
    RAW_DIR / "math_train_minus_math500.jsonl",
    orient="records",
    lines=True,
    force_ascii=False,
)
calib_pool_df.to_csv(RAW_DIR / "math_train_minus_math500.csv", index=False)

print(f"Calibration pool size after removing MATH-500 overlap: {len(calib_pool_df):,}")

print("Creating one 400-problem stratified calibration split...")

target_n = 400
groups = []

for topic, group in calib_pool_df.groupby("topic"):
    n_topic = round(target_n * len(group) / len(calib_pool_df))
    n_topic = max(1, min(len(group), n_topic))
    groups.append(group.sample(n=n_topic, random_state=42))

calib_400 = pd.concat(groups, ignore_index=True)

# Adjust exactly to 400 after rounding
if len(calib_400) > target_n:
    calib_400 = calib_400.sample(n=target_n, random_state=42)
elif len(calib_400) < target_n:
    selected_hashes = set(calib_400["problem_hash"])
    remaining = calib_pool_df[~calib_pool_df["problem_hash"].isin(selected_hashes)]
    add = remaining.sample(n=target_n - len(calib_400), random_state=42)
    calib_400 = pd.concat([calib_400, add], ignore_index=True)

calib_400 = calib_400.sample(frac=1, random_state=42).reset_index(drop=True)

calib_400.to_csv(SPLIT_DIR / "math_calibration_400_seed42.csv", index=False)
calib_400.to_json(
    SPLIT_DIR / "math_calibration_400_seed42.jsonl",
    orient="records",
    lines=True,
    force_ascii=False,
)

math500_df.to_csv(SPLIT_DIR / "math500_test.csv", index=False)
math500_df.to_json(
    SPLIT_DIR / "math500_test.jsonl",
    orient="records",
    lines=True,
    force_ascii=False,
)

print("Done.")
print(f"MATH-500 test: {len(math500_df):,}")
print(f"Calibration pool: {len(calib_pool_df):,}")
print(f"Calibration split: {len(calib_400):,}")
print("Files saved under data/raw and data/splits.")