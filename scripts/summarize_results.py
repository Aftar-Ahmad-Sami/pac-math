from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pathlib import Path

import pandas as pd

import config
from pacmath.stats import add_holm_correction, mcnemar_pair, number_needed_to_treat, summarize_methods


def summarize_one(experiment_name: str) -> None:
    base = config.OUT_DIR / experiment_name
    path = base / "combined_test_method_rows.csv"
    if not path.exists():
        print(f"Missing {path}. Run the experiment first.")
        return

    df = pd.read_csv(path)
    methods = sorted(df["method"].unique()) if "method" in df.columns else []
    pairs = sorted(df["pair_id"].unique()) if "pair_id" in df.columns else []
    if methods:
        counts = df.groupby("method").size()
        if counts.nunique() > 1:
            print(f"WARNING: Uneven method row counts in {experiment_name}. Some records may be incomplete:")
            print(counts.to_string())
        expected_n = None
        if experiment_name == "smoke":
            expected_n = config.SMOKE_TEST_N * max(1, len(pairs))
        elif experiment_name == "pilot":
            expected_n = config.PILOT_TEST_N * max(1, len(pairs))
        elif experiment_name == "full" and config.FULL_TEST_N is not None:
            expected_n = config.FULL_TEST_N * max(1, len(pairs))
        if expected_n is not None and len(counts) and int(counts.max()) < expected_n:
            print(f"WARNING: {experiment_name} has only {int(counts.max())}/{expected_n} rows per method. Rerun to retry incomplete records.")
    summary = summarize_methods(df)
    out_summary = base / "summary_methods.csv"
    summary.to_csv(out_summary, index=False)
    print(f"\n=== {experiment_name}: method summary ===")
    print(summary.to_string(index=False))
    print(f"Saved {out_summary}")

    # Primary comparisons.
    comparisons = []
    method_set = set(df["method"])
    if "pac_math_verifier_balanced" in method_set:
        primary = "pac_math_verifier_balanced"
    elif "pac_math_router_balanced" in method_set:
        primary = "pac_math_router_balanced"
    elif "pac_math_adaptive_learned" in method_set:
        primary = "pac_math_adaptive_learned"
    else:
        primary = "pac_math_pair_topic_stage"
    for baseline in ["stateless_debate", "4cand_majority", "4cand_confidence", "overall_reliability", "agent_topic_reliability", "pac_math_pair_topic_stage_support_sum", "pac_math_pair_topic_stage_guard", "pac_math_independent_only", "pac_math_safety_first", "pac_math_utility", "pac_math_accuracy_first", "pac_math_anchor_gate", "pac_math_cross_stage_support", "pac_math_stage_gate", "pac_math_adaptive_learned", "pac_math_adaptive_accuracy", "pac_math_adaptive_safety", "pac_math_router_accuracy", "pac_math_router_safety", "pac_math_verifier_best", "pac_math_verifier_safety", "pac_math_pair_topic_stage"]:
        if baseline in set(df["method"]) and primary in set(df["method"]):
            comparisons.append(mcnemar_pair(df, primary, baseline))
    comp_df = add_holm_correction(comparisons)
    out_comp = base / "mcnemar_primary_comparisons.csv"
    comp_df.to_csv(out_comp, index=False)
    print(f"Saved {out_comp}")

    # Number needed to treat for C2W vs stateless debate.
    try:
        c2w = summary.set_index("method")["c2w_flip_rate"].to_dict()
        if "stateless_debate" in c2w and primary in c2w:
            nnt = number_needed_to_treat(c2w["stateless_debate"], c2w[primary])
            if nnt == float("inf"):
                print("NNT vs stateless debate: inf/no benefit because PAC-Math did not reduce C2W in this experiment.")
            else:
                print(f"NNT vs stateless debate for preventing one C2W flip: {nnt:.2f}")
    except Exception as e:
        print(f"Could not compute NNT: {e}")

    # Per-topic summary.
    topic_rows = []
    for (method, topic), g in df.groupby(["method", "topic"]):
        risk = g[g["initial_any_correct"]]
        topic_rows.append({
            "method": method,
            "topic": topic,
            "n": len(g),
            "accuracy": g["is_correct"].mean(),
            "c2w_flip_rate": risk["correct_to_wrong"].mean() if len(risk) else float("nan"),
        })
    topic_df = pd.DataFrame(topic_rows)
    out_topic = base / "summary_by_topic.csv"
    topic_df.to_csv(out_topic, index=False)
    print(f"Saved {out_topic}")


if __name__ == "__main__":
    for exp in ["smoke", "pilot", "full"]:
        summarize_one(exp)
