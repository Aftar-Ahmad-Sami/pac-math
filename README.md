# PAC-Math v16 Standard Debate Recovery Run

This version is the next required experiment after v15.

Why this change is needed:
- v15 evidence gating reduced useful post-debate candidate diversity.
- In the v15 full run, A1/B1 were barely better than A0/B0, so selectors had little room to work.
- Earlier standard Phi-4 pilot had much stronger post-debate candidates, so the next valid step is to rerun standard debate with protocol-version cache protection.

Important code fixes:
1. `EVIDENCE_GATED_DEBATE = False` and `PROTOCOL_VERSION = "standard_debate_v16"`.
2. New pair id: `qwen3_8b__phi4_14b_standard_v16`, so old evidence-gated cache is not reused.
3. Fixed standard-debate finalization. In v15, standard mode still called the evidence-gate guard and silently rejected normal revisions. v16 accepts parsed revised answers directly when the evidence gate is disabled.
4. Added protocol/evidence-gate audit output.
5. Suppressed noisy SymPy SyntaxWarnings during answer normalization.

Run:

```bash
conda activate pac-math
python scripts/run_full.py
python scripts/summarize_results.py
python scripts/diagnose_experiment.py
python scripts/audit_experiment.py
```

What to check:
- In `audit_experiment.py`, the full record audit should show `protocol_versions: ['standard_debate_v16']` and `evidence_gated_debate: ['False']`.
- Candidate quality should recover compared with evidence-gated v15, especially A1/B1 correct rates.
- Main comparison: `stateless_debate`, `4cand_majority`, `pac_math_cross_agent_anchor_gate`, `pac_math_adaptive_learned`, `oracle_candidate`.
