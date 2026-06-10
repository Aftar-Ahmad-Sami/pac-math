from pathlib import Path
import sys
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402


OUTPUT_ROOT = PROJECT_ROOT / "outputs"


DEFAULT_MAIN_METHODS = [
    "single_A",
    "single_B",
    "stateless_debate",
    "4cand_majority",
    "pac_math_pair_topic_stage",
    "pac_math_adaptive_learned",
    "pac_math_router_safety",
    "oracle_candidate",
]

DEFAULT_APPENDIX_METHODS = [
    "4cand_confidence",
    "overall_reliability",
    "agent_topic_reliability",
    "pac_math_anchor_gate",
    "pac_math_cross_stage_support",
    "pac_math_cross_agent_anchor_gate",
    "pac_math_pair_topic_stage_support_sum",
    "pac_math_router_balanced",
    "pac_math_router_accuracy",
    "pac_math_safety_first",
]


def _method_order(methods):
    return {method: i for i, method in enumerate(methods)}


def _format_for_latex(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    integer_cols = {
        "n",
        "c2w_risk_n",
        "c2w_flip_count",
        "both_wrong_n",
        "w2c_count",
    }

    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            if col in integer_cols:
                out[col] = out[col].map(lambda x: "" if pd.isna(x) else f"{int(x)}")
            else:
                out[col] = out[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.4f}")

    return out


def export_method_table(split_name: str, methods: list[str], table_name: str) -> None:
    split_dir = OUTPUT_ROOT / split_name
    summary_path = split_dir / "summary_methods.csv"

    if not summary_path.exists():
        print(f"Missing {summary_path}. Skipping {split_name}.")
        return

    df = pd.read_csv(summary_path)

    if "method" not in df.columns:
        raise ValueError(f"{summary_path} does not contain a 'method' column.")

    table_df = df[df["method"].isin(methods)].copy()

    order = _method_order(methods)
    table_df["method_order"] = table_df["method"].map(order)
    table_df = table_df.sort_values("method_order").drop(columns=["method_order"])

    keep_cols = [
        "method",
        "n",
        "accuracy",
        "c2w_risk_n",
        "c2w_flip_count",
        "c2w_flip_rate",
        "both_wrong_n",
        "w2c_count",
        "w2c_recovery_rate",
        "preservation_rate",
        "candidate_oracle_possible",
        "oracle_gap",
        "mean_method_tokens",
        "mean_method_duration_seconds",
    ]

    keep_cols = [col for col in keep_cols if col in table_df.columns]
    table_df = table_df[keep_cols]

    out_csv = split_dir / f"{table_name}.csv"
    out_tex = split_dir / f"{table_name}.tex"

    table_df.to_csv(out_csv, index=False)
    _format_for_latex(table_df).to_latex(out_tex, index=False, escape=False)

    print(f"Saved {out_csv}")
    print(f"Saved {out_tex}")


def main() -> None:
    main_methods = getattr(config, "MAIN_METHODS", DEFAULT_MAIN_METHODS)
    appendix_methods = getattr(config, "APPENDIX_METHODS", DEFAULT_APPENDIX_METHODS)

    for split_name in ["pilot", "full"]:
        export_method_table(split_name, main_methods, "main_table_methods")
        export_method_table(split_name, appendix_methods, "appendix_table_methods")


if __name__ == "__main__":
    main()