from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import config
from pacmath.io_utils import read_jsonl, rows_to_csv


def summarize(df: pd.DataFrame, name: str) -> pd.DataFrame:
    rows = []
    for method, g in df.groupby('method'):
        n = len(g)
        risk = int(g['initial_any_correct'].sum()) if 'initial_any_correct' in g else 0
        c2w = int(g['correct_to_wrong'].sum()) if 'correct_to_wrong' in g else 0
        both_wrong = int(g['initial_both_wrong'].sum()) if 'initial_both_wrong' in g else 0
        w2c = int(g['wrong_to_correct'].sum()) if 'wrong_to_correct' in g else 0
        rows.append({
            'segment': name,
            'method': method,
            'n': n,
            'accuracy': g['is_correct'].mean(),
            'c2w_risk_n': risk,
            'c2w_count': c2w,
            'c2w_rate': c2w / risk if risk else 0.0,
            'both_wrong_n': both_wrong,
            'w2c_count': w2c,
            'w2c_rate': w2c / both_wrong if both_wrong else 0.0,
            'oracle_possible': g['candidate_oracle_possible'].mean() if 'candidate_oracle_possible' in g else None,
        })
    return pd.DataFrame(rows).sort_values(['segment','method'])


def audit_one(exp_name: str):
    out_dir = config.OUT_DIR / exp_name
    rows_path = out_dir / 'combined_test_method_rows.csv'
    if not rows_path.exists():
        print(f'Missing {rows_path}')
        return
    df = pd.read_csv(rows_path)
    print(f'\n=== {exp_name}: rows={len(df):,}, methods={df.method.nunique()}, problems={df.run_id.nunique()} ===')
    summaries = [summarize(df, 'all')]
    # Segment summaries are useful because pilot uses the first 120 MATH-500 rows.
    if 'problem_index' in df.columns and df['problem_index'].max() >= 120:
        summaries.append(summarize(df[df['problem_index'] < 120], 'first_120'))
        summaries.append(summarize(df[df['problem_index'] >= 120], 'after_120'))
    all_summ = pd.concat(summaries, ignore_index=True)
    keep_methods = [
        'single_A','single_B','stateless_debate','4cand_majority','4cand_confidence',
        'pac_math_anchor_gate','pac_math_cross_agent_anchor_gate','pac_math_adaptive_learned',
        'pac_math_pair_topic_stage','oracle_candidate'
    ]
    print(all_summ[all_summ.method.isin(keep_methods)].to_string(index=False))
    out = out_dir / 'audit_segments.csv'
    all_summ.to_csv(out, index=False)
    print(f'Saved {out}')

    # Candidate record audit.
    rec_dir = out_dir / 'records'
    for rec_path in sorted(rec_dir.glob('test_*.jsonl')):
        recs = read_jsonl(rec_path)
        n = len(recs)
        complete = 0
        protocol_versions = sorted({str(r.get('protocol_version', 'MISSING')) for r in recs})
        evidence_flags = sorted({str((r.get('protocol_metadata') or {}).get('evidence_gated_debate', 'MISSING')) for r in recs})
        parse_ok = {'A0':0,'B0':0,'A1':0,'B1':0}
        correct = {'A0':0,'B0':0,'A1':0,'B1':0}
        changed = {'A1':0,'B1':0}
        third_answer = 0
        for r in recs:
            cs = {c.get('candidate_id'): c for c in r.get('candidates', [])}
            if {'A0','B0','A1','B1'}.issubset(cs):
                complete += 1
            for cid in parse_ok:
                c = cs.get(cid, {})
                if c.get('parse_ok'):
                    parse_ok[cid]+=1
                if c.get('is_correct'):
                    correct[cid]+=1
            for cid, base in [('A1','A0'),('B1','B0')]:
                if cs.get(cid,{}).get('normalized_answer') != cs.get(base,{}).get('normalized_answer'):
                    changed[cid]+=1
        print(f'Candidate audit {rec_path.name}: n={n}, complete={complete}')
        print('  protocol_versions:', protocol_versions)
        print('  evidence_gated_debate:', evidence_flags)
        print('  parse_ok_rate:', {k: round(v/max(1,n),4) for k,v in parse_ok.items()})
        print('  correct_rate :', {k: round(v/max(1,n),4) for k,v in correct.items()})
        print('  changed_rate :', {k: round(v/max(1,n),4) for k,v in changed.items()})


def main():
    for exp in ['pilot','full']:
        audit_one(exp)

if __name__ == '__main__':
    main()
