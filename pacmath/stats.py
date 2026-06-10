from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests


def summarize_methods(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    # Candidate oracle is problem-level. It is repeated across methods, so computing
    # it per method is fine as long as each method has one row per problem.
    for method, g in df.groupby("method"):
        n = len(g)
        acc = float(g["is_correct"].mean()) if n else float("nan")
        risk = g[g["initial_any_correct"]]
        c2w = float(risk["correct_to_wrong"].mean()) if len(risk) else float("nan")
        both_wrong = g[g["initial_both_wrong"]]
        w2c = float(both_wrong["wrong_to_correct"].mean()) if len(both_wrong) else float("nan")
        preservation = float(risk["preserved_correct"].mean()) if len(risk) else float("nan")
        oracle_possible = float(g["candidate_oracle_possible"].mean()) if n else float("nan")
        oracle_gap = float(oracle_possible - acc) if n else float("nan")
        rows.append({
            "method": method,
            "n": n,
            "accuracy": acc,
            "c2w_risk_n": int(len(risk)),
            "c2w_flip_count": int(risk["correct_to_wrong"].sum()) if len(risk) else 0,
            "c2w_flip_rate": c2w,
            "both_wrong_n": int(len(both_wrong)),
            "w2c_count": int(both_wrong["wrong_to_correct"].sum()) if len(both_wrong) else 0,
            "w2c_recovery_rate": w2c,
            "preservation_rate": preservation,
            "candidate_oracle_possible": oracle_possible,
            "oracle_gap": oracle_gap,
            "mean_method_tokens": float(g["method_total_tokens"].mean()) if "method_total_tokens" in g else float("nan"),
            "mean_method_duration_seconds": float(g["method_duration_seconds"].mean()) if "method_duration_seconds" in g else float("nan"),
            "mean_full_run_tokens": float(g["full_run_total_tokens"].mean()) if "full_run_total_tokens" in g else (float(g["total_tokens"].mean()) if "total_tokens" in g else float("nan")),
            "mean_full_run_wall_time_seconds": float(g["full_run_wall_time_seconds"].mean()) if "full_run_wall_time_seconds" in g else (float(g["wall_time_seconds"].mean()) if "wall_time_seconds" in g else float("nan")),
        })
    return pd.DataFrame(rows).sort_values("method")

def paired_bootstrap_diff(df: pd.DataFrame, method_a: str, method_b: str, metric_col: str, n_boot: int = 2000, seed: int = 42) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    pivot = df.pivot_table(index=["pair_id", "problem_index"], columns="method", values=metric_col, aggfunc="first").dropna(subset=[method_a, method_b])
    diffs = pivot[method_a].to_numpy(dtype=float) - pivot[method_b].to_numpy(dtype=float)
    n = len(diffs)
    if n == 0:
        return {"mean_diff": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0}
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots.append(float(np.mean(diffs[idx])))
    return {
        "mean_diff": float(np.mean(diffs)),
        "ci_low": float(np.percentile(boots, 2.5)),
        "ci_high": float(np.percentile(boots, 97.5)),
        "n": int(n),
    }


def mcnemar_pair(df: pd.DataFrame, method_a: str, method_b: str) -> Dict[str, float]:
    pivot = df.pivot_table(index=["pair_id", "problem_index"], columns="method", values="is_correct", aggfunc="first").dropna(subset=[method_a, method_b])
    a = pivot[method_a].astype(bool).to_numpy()
    b = pivot[method_b].astype(bool).to_numpy()
    both_correct = int(np.sum(a & b))
    a_correct_b_wrong = int(np.sum(a & ~b))
    a_wrong_b_correct = int(np.sum(~a & b))
    both_wrong = int(np.sum(~a & ~b))
    table = [[both_correct, a_correct_b_wrong], [a_wrong_b_correct, both_wrong]]
    discordant = a_correct_b_wrong + a_wrong_b_correct
    if discordant == 0:
        statistic = 0.0
        pvalue = 1.0
        note = "no_discordant_pairs"
    else:
        result = mcnemar(table, exact=False, correction=True)
        statistic = float(result.statistic)
        pvalue = float(result.pvalue)
        note = "ok"
    return {
        "method_a": method_a,
        "method_b": method_b,
        "both_correct": both_correct,
        "a_correct_b_wrong": a_correct_b_wrong,
        "a_wrong_b_correct": a_wrong_b_correct,
        "both_wrong": both_wrong,
        "discordant_pairs": discordant,
        "statistic": statistic,
        "pvalue": pvalue,
        "note": note,
    }


def add_holm_correction(rows: List[Dict[str, float]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty or "pvalue" not in df:
        return df
    reject, p_adj, _, _ = multipletests(df["pvalue"].to_numpy(), method="holm")
    df["pvalue_holm"] = p_adj
    df["reject_holm_0p05"] = reject
    return df


def number_needed_to_treat(c2w_baseline: float, c2w_method: float) -> float:
    diff = c2w_baseline - c2w_method
    if diff <= 0:
        return float("inf")
    return 1.0 / diff
