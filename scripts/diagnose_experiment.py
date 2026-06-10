from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
from collections import Counter, defaultdict

import pandas as pd

import config


def _load_jsonl(path: Path):
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _safe_pct(x):
    if pd.isna(x):
        return "nan"
    return f"{100*x:.2f}%"


def diagnose_one(exp: str) -> None:
    base = config.OUT_DIR / exp
    combined = base / "combined_test_method_rows.csv"
    if not combined.exists():
        print(f"\n=== {exp}: missing combined_test_method_rows.csv ===")
        return

    df = pd.read_csv(combined)
    active_pairs = set(str(x) for x in df.get("pair_id", pd.Series(dtype=str)).dropna().unique())
    print(f"\n=== {exp}: diagnostics ===")
    print(f"Rows: {len(df):,} | methods: {df['method'].nunique()} | pairs: {df['pair_id'].nunique()} | problems per method: {len(df) // max(1, df['method'].nunique())}")

    # Problem-level candidate availability.
    prob = df[df["method"] == "oracle_candidate"].copy()
    risk_n = int(prob["initial_any_correct"].sum())
    both_wrong_n = int(prob["initial_both_wrong"].sum())
    oracle_n = int(prob["candidate_oracle_possible"].sum())
    print(f"Problem-level C2W-risk cases: {risk_n}/{len(prob)}")
    print(f"Problem-level both-initially-wrong cases: {both_wrong_n}/{len(prob)}")
    print(f"Candidate oracle possible: {oracle_n}/{len(prob)} ({oracle_n/max(1,len(prob)):.3f})")

    # Main methods compact view.
    main_methods = [
        "stateless_debate",
        "4cand_majority",
        "4cand_confidence",
        "overall_reliability",
        "agent_topic_reliability",
        "pac_math_pair_topic_stage",
        "pac_math_pair_topic_stage_support_sum",
        "pac_math_pair_topic_stage_guard",
        "pac_math_independent_only",
        "pac_math_safety_first",
        "pac_math_utility",
        "pac_math_accuracy_first",
        "pac_math_anchor_gate",
        "pac_math_cross_stage_support",
        "pac_math_stage_gate",
        "pac_math_adaptive_learned",
        "pac_math_adaptive_accuracy",
        "pac_math_adaptive_safety",
        "pac_math_router_balanced",
        "pac_math_router_accuracy",
        "pac_math_router_safety",
        "pac_math_verifier_best",
        "pac_math_verifier_balanced",
        "pac_math_verifier_safety",
        "oracle_candidate",
    ]
    rows = []
    for m in main_methods:
        g = df[df["method"] == m]
        if g.empty:
            continue
        risk = g[g["initial_any_correct"]]
        rows.append({
            "method": m,
            "acc": g["is_correct"].mean(),
            "c2w_count": int(risk["correct_to_wrong"].sum()) if len(risk) else 0,
            "c2w_rate": risk["correct_to_wrong"].mean() if len(risk) else float("nan"),
            "preservation": risk["preserved_correct"].mean() if len(risk) else float("nan"),
            "selected_A0": int((g["selected_candidate_id"] == "A0").sum()),
            "selected_B0": int((g["selected_candidate_id"] == "B0").sum()),
            "selected_A1": int((g["selected_candidate_id"] == "A1").sum()),
            "selected_B1": int((g["selected_candidate_id"] == "B1").sum()),
        })
    diag_df = pd.DataFrame(rows)
    out = base / "diagnostic_method_selection.csv"
    diag_df.to_csv(out, index=False)
    print("\nMain method diagnostics:")
    print(diag_df.to_string(index=False))
    print(f"Saved {out}")

    # Pair-level breakdown.
    pair_rows = []
    for (pair_id, method), g in df[df["method"].isin(main_methods)].groupby(["pair_id", "method"]):
        risk = g[g["initial_any_correct"]]
        pair_rows.append({
            "pair_id": pair_id,
            "method": method,
            "n": len(g),
            "acc": g["is_correct"].mean(),
            "risk_n": len(risk),
            "c2w_count": int(risk["correct_to_wrong"].sum()) if len(risk) else 0,
            "c2w_rate": risk["correct_to_wrong"].mean() if len(risk) else float("nan"),
        })
    pair_df = pd.DataFrame(pair_rows).sort_values(["pair_id", "method"])
    out_pair = base / "diagnostic_by_pair.csv"
    pair_df.to_csv(out_pair, index=False)
    print(f"Saved {out_pair}")

    # Parse and normalization status from raw test records.
    rec_dir = base / "records"
    parse_rows = []
    for rec_path in rec_dir.glob("test_*.jsonl"):
        records = _load_jsonl(rec_path)
        for rec in records:
            rec_pair = str(rec.get("pair_id", rec_path.stem))
            if active_pairs and rec_pair not in active_pairs:
                continue
            if not rec.get("candidates"):
                parse_rows.append({
                    "pair_id": rec_pair,
                    "candidate_id": "NO_CANDIDATES",
                    "parse_ok": False,
                    "normalization_status": "NO_CANDIDATES",
                    "is_correct": False,
                })
                continue
            for c in rec["candidates"]:
                parse_rows.append({
                    "pair_id": rec_pair,
                    "candidate_id": c.get("candidate_id"),
                    "parse_ok": bool(c.get("parse_ok", False)),
                    "normalization_status": c.get("normalization_status", ""),
                    "is_correct": bool(c.get("is_correct", False)),
                })
    parse_df = pd.DataFrame(parse_rows)
    if not parse_df.empty:
        parse_summary = parse_df.groupby(["pair_id", "candidate_id"]).agg(
            n=("candidate_id", "size"),
            parse_ok_rate=("parse_ok", "mean"),
            correct_rate=("is_correct", "mean"),
        ).reset_index()
        out_parse = base / "diagnostic_parse_candidate_summary.csv"
        parse_summary.to_csv(out_parse, index=False)
        print("\nParse/candidate summary:")
        print(parse_summary.to_string(index=False))
        print(f"Saved {out_parse}")

    # Memory counts and reliability spread.
    mem_dir = base / "memory"
    mem_rows = []
    for mem_path in mem_dir.glob("*_memory.json"):
        data = json.loads(mem_path.read_text(encoding="utf-8"))
        pair_file = mem_path.stem.replace("_memory", "")
        for key_str, val in data.get("counts", {}).items():
            parts = key_str.split("|")
            level = parts[0]
            if level == "pair_agent_topic_stage":
                _, pair_id, agent_id, topic, stage = parts
                total = int(val.get("total", 0))
                correct = int(val.get("correct", 0))
                rate = (correct + 1) / (total + 2) if total >= 0 else 0.5
                if active_pairs and pair_id not in active_pairs:
                    continue
                mem_rows.append({
                    "pair_id": pair_id,
                    "agent_id": agent_id,
                    "topic": topic,
                    "stage": stage,
                    "correct": correct,
                    "total": total,
                    "rate_smoothed_basic": rate,
                })
    mem_df = pd.DataFrame(mem_rows)
    if not mem_df.empty:
        out_mem = base / "diagnostic_memory_pair_agent_topic_stage.csv"
        mem_df.to_csv(out_mem, index=False)
        print(f"Saved {out_mem}")
        print("\nMemory total count summary:")
        print(mem_df.groupby(["pair_id", "stage"])["total"].describe().to_string())


def main():
    for exp in ["smoke", "pilot", "full"]:
        diagnose_one(exp)


if __name__ == "__main__":
    main()
