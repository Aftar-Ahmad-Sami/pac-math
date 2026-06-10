from pathlib import Path
import re
import sys
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

OUTPUT_ROOT = PROJECT_ROOT / "outputs"


KEEP_METHODS = [
    "stateless_debate",
    "4cand_majority",
    "pac_math_pair_topic_stage",
    "pac_math_adaptive_learned",
    "pac_math_router_safety",
    "oracle_candidate",
]


def canon(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    canon_map = {canon(col): col for col in df.columns}
    for candidate in candidates:
        key = canon(candidate)
        if key in canon_map:
            return canon_map[key]
    return None


def to_bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(float) > 0

    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin(["1", "true", "yes", "y", "correct"])
    )


def load_diagnostic_by_pair() -> pd.DataFrame | None:
    path = OUTPUT_ROOT / "full" / "diagnostic_by_pair.csv"

    if not path.exists():
        return None

    df = pd.read_csv(path)

    pair_col = find_col(df, ["pair_id", "pair"])
    method_col = find_col(df, ["method"])
    acc_col = find_col(df, ["acc", "accuracy"])
    c2w_count_col = find_col(df, ["c2w_count", "c2w_flip_count"])
    c2w_rate_col = find_col(df, ["c2w_rate", "c2w_flip_rate"])
    preservation_col = find_col(df, ["preservation", "preservation_rate"])
    oracle_col = find_col(df, ["candidate_oracle_possible", "oracle_possible"])

    required = {
        "pair_id": pair_col,
        "method": method_col,
        "accuracy": acc_col,
        "c2w_count": c2w_count_col,
        "c2w_rate": c2w_rate_col,
    }

    missing = [name for name, col in required.items() if col is None]
    if missing:
        print(f"{path} exists but is missing expected columns: {missing}")
        print("Available columns:")
        print(list(df.columns))
        return None

    out = pd.DataFrame(
        {
            "pair_id": df[pair_col],
            "method": df[method_col],
            "accuracy": df[acc_col],
            "c2w_count": df[c2w_count_col],
            "c2w_rate": df[c2w_rate_col],
        }
    )

    if preservation_col is not None:
        out["preservation_rate"] = df[preservation_col]

    if oracle_col is not None:
        out["candidate_oracle_possible"] = df[oracle_col]

    return out


def load_combined_rows() -> pd.DataFrame:
    csv_path = OUTPUT_ROOT / "full" / "combined_test_method_rows.csv"
    jsonl_path = OUTPUT_ROOT / "full" / "combined_test_method_rows.jsonl"

    if csv_path.exists():
        return pd.read_csv(csv_path)

    if jsonl_path.exists():
        return pd.read_json(jsonl_path, lines=True)

    raise FileNotFoundError(
        "Missing outputs/full/combined_test_method_rows.csv or .jsonl. "
        "Run scripts/run_full.py first."
    )


def find_candidate_correct_col(df: pd.DataFrame, candidate_id: str) -> str | None:
    target = canon(candidate_id)

    exact_candidates = [
        f"{candidate_id}_correct",
        f"{candidate_id}_is_correct",
        f"{candidate_id.lower()}_correct",
        f"{candidate_id.lower()}_is_correct",
        f"candidate_{candidate_id}_correct",
        f"candidate_{candidate_id}_is_correct",
    ]

    exact = find_col(df, exact_candidates)
    if exact is not None:
        return exact

    for col in df.columns:
        key = canon(col)
        if target in key and "correct" in key and "selected" not in key:
            return col

    return None


def derive_from_combined_rows(df: pd.DataFrame) -> pd.DataFrame:
    pair_col = find_col(df, ["pair_id", "pair"])
    method_col = find_col(df, ["method"])

    if pair_col is None or method_col is None:
        raise ValueError(
            f"Could not find pair_id/method columns. Available columns: {list(df.columns)}"
        )

    selected_correct_col = find_col(
        df,
        [
            "selected_correct",
            "selected_is_correct",
            "selected_candidate_correct",
            "is_correct",
            "correct",
            "answer_correct",
            "final_correct",
        ],
    )

    c2w_col = find_col(
        df,
        [
            "c2w_flip",
            "c2w",
            "is_c2w_flip",
            "correct_to_wrong_flip",
            "selected_c2w_flip",
        ],
    )

    w2c_col = find_col(
        df,
        [
            "w2c_recovery",
            "w2c",
            "is_w2c_recovery",
            "wrong_to_correct_recovery",
            "selected_w2c_recovery",
        ],
    )

    oracle_col = find_col(
        df,
        [
            "candidate_oracle_possible",
            "oracle_possible",
            "any_candidate_correct",
            "candidate_oracle",
        ],
    )

    candidate_cols = {
        cand: find_candidate_correct_col(df, cand)
        for cand in ["A0", "B0", "A1", "B1"]
    }

    if selected_correct_col is not None:
        selected_correct = to_bool_series(df[selected_correct_col])
    else:
        if not all(candidate_cols.values()):
            raise ValueError(
                "Could not find selected correctness column or candidate correctness columns.\n"
                f"Available columns: {list(df.columns)}\n"
                f"Candidate correctness detected: {candidate_cols}"
            )

        selected_id_col = find_col(
            df,
            [
                "selected_candidate_id",
                "selected_candidate",
                "chosen_candidate_id",
                "chosen_candidate",
            ],
        )

        if selected_id_col is None:
            raise ValueError(
                "Could not derive selected correctness because selected_candidate_id is missing.\n"
                f"Available columns: {list(df.columns)}"
            )

        selected_correct_values = []
        for _, row in df.iterrows():
            selected_id = str(row[selected_id_col]).strip().upper()
            col = candidate_cols.get(selected_id)
            if col is None:
                selected_correct_values.append(False)
            else:
                selected_correct_values.append(bool(to_bool_series(pd.Series([row[col]])).iloc[0]))

        selected_correct = pd.Series(selected_correct_values, index=df.index)

    if c2w_col is not None:
        c2w = to_bool_series(df[c2w_col])
    else:
        a0_col = candidate_cols.get("A0")
        b0_col = candidate_cols.get("B0")
        if a0_col is None or b0_col is None:
            raise ValueError(
                "Could not derive C2W because A0/B0 correctness columns are missing.\n"
                f"Available columns: {list(df.columns)}"
            )

        a0_correct = to_bool_series(df[a0_col])
        b0_correct = to_bool_series(df[b0_col])
        initial_correct_available = a0_correct | b0_correct
        c2w = initial_correct_available & (~selected_correct)

    if w2c_col is not None:
        w2c = to_bool_series(df[w2c_col])
    else:
        a0_col = candidate_cols.get("A0")
        b0_col = candidate_cols.get("B0")
        if a0_col is None or b0_col is None:
            raise ValueError(
                "Could not derive W2C because A0/B0 correctness columns are missing.\n"
                f"Available columns: {list(df.columns)}"
            )

        a0_correct = to_bool_series(df[a0_col])
        b0_correct = to_bool_series(df[b0_col])
        both_initial_wrong = (~a0_correct) & (~b0_correct)
        w2c = both_initial_wrong & selected_correct

    if oracle_col is not None:
        oracle_possible = to_bool_series(df[oracle_col])
    else:
        if all(candidate_cols.values()):
            oracle_possible = (
                to_bool_series(df[candidate_cols["A0"]])
                | to_bool_series(df[candidate_cols["B0"]])
                | to_bool_series(df[candidate_cols["A1"]])
                | to_bool_series(df[candidate_cols["B1"]])
            )
        else:
            oracle_possible = pd.Series([False] * len(df), index=df.index)

    tmp = df[[pair_col, method_col]].copy()
    tmp = tmp.rename(columns={pair_col: "pair_id", method_col: "method"})
    tmp["selected_correct"] = selected_correct.astype(bool)
    tmp["c2w"] = c2w.astype(bool)
    tmp["w2c"] = w2c.astype(bool)
    tmp["oracle_possible"] = oracle_possible.astype(bool)

    rows = []
    for (pair_id, method), g in tmp.groupby(["pair_id", "method"]):
        rows.append(
            {
                "pair_id": pair_id,
                "method": method,
                "n": len(g),
                "accuracy": g["selected_correct"].mean(),
                "c2w_count": int(g["c2w"].sum()),
                "c2w_rate": g["c2w"].mean(),
                "w2c_count": int(g["w2c"].sum()),
                "w2c_rate": g["w2c"].mean(),
                "candidate_oracle_possible": g["oracle_possible"].mean(),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    out = load_diagnostic_by_pair()

    if out is None:
        print("Falling back to combined test method rows.")
        combined = load_combined_rows()
        out = derive_from_combined_rows(combined)

    out = out[out["method"].isin(KEEP_METHODS)].copy()
    out = out.sort_values(["method", "pair_id"]).reset_index(drop=True)

    out_path = OUTPUT_ROOT / "full" / "role_order_comparison.csv"
    out.to_csv(out_path, index=False)

    print(out.to_string(index=False))
    print(f"Saved {out_path}")

    n_pairs = out["pair_id"].nunique()
    if n_pairs < 2:
        print()
        print("Warning: only one pair_id found.")
        print("Run the reverse role-order pair before interpreting role bias:")
        print("  phi4_14b__qwen3_8b_standard_v16")


if __name__ == "__main__":
    main()