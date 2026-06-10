from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
except Exception:  # pragma: no cover
    LogisticRegression = None
    Pipeline = None
    StandardScaler = None

from .reliability import ReliabilityMemory


class ConstantProbabilityModel:
    """Tiny predict_proba-compatible fallback for one-class calibration labels."""

    def __init__(self, p: float):
        self.p = float(max(0.0, min(1.0, p)))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        n = len(X)
        return np.column_stack([np.full(n, 1.0 - self.p), np.full(n, self.p)])


def _norm(c: Dict[str, Any]) -> str:
    return str(c.get("normalized_answer", "") or "")


def _confidence(c: Dict[str, Any]) -> float:
    try:
        return max(0.0, min(100.0, float(c.get("confidence", 0) or 0)))
    except Exception:
        return 0.0


def _cand_by_id(candidates: List[Dict[str, Any]], cid: str) -> Optional[Dict[str, Any]]:
    for c in candidates:
        if c.get("candidate_id") == cid:
            return c
    return None


def _stateless_choice(candidates: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    a1 = _cand_by_id(candidates, "A1")
    b1 = _cand_by_id(candidates, "B1")
    if not a1 or not b1:
        return None
    if _norm(a1) and _norm(a1) == _norm(b1):
        return a1
    return a1 if _confidence(a1) >= _confidence(b1) else b1


def _initial_any_correct(candidates: List[Dict[str, Any]]) -> bool:
    a0 = _cand_by_id(candidates, "A0") or {}
    b0 = _cand_by_id(candidates, "B0") or {}
    return bool(a0.get("is_correct", False)) or bool(b0.get("is_correct", False))


@dataclass
class SelectorParams:
    lambda_risk: float
    support_weight: float
    cross_stage_weight: float
    independent_weight: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "lambda_risk": float(self.lambda_risk),
            "support_weight": float(self.support_weight),
            "cross_stage_weight": float(self.cross_stage_weight),
            "independent_weight": float(self.independent_weight),
        }


class AdaptiveCandidateSelector:
    """Calibration-trained candidate selector for PAC-Math.

    This is intentionally not a hand-set deterministic rule. It learns two
    candidate-level models from the calibration split only:

    1. P(candidate is correct)
    2. P(selecting this candidate causes a correct-to-wrong flip)

    At test time the selector is frozen. It never uses test labels. It selects at
    the answer level, not only at the candidate level, so repeated independent
    and post-debate support can help when calibrated evidence says it should.
    """

    def __init__(
        self,
        lambda_grid: Optional[List[float]] = None,
        support_grid: Optional[List[float]] = None,
        cross_stage_grid: Optional[List[float]] = None,
        independent_grid: Optional[List[float]] = None,
        random_state: int = 42,
    ):
        self.lambda_grid = lambda_grid or [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
        self.support_grid = support_grid or [0.0, 0.01, 0.03, 0.05]
        self.cross_stage_grid = cross_stage_grid or [0.0, 0.02, 0.05]
        self.independent_grid = independent_grid or [0.0, 0.01, 0.03]
        self.random_state = int(random_state)

        self.topic_vocab: List[str] = []
        self.feature_names: List[str] = []
        self.correct_model: Any = None
        self.risk_model: Any = None
        self.best_params: Dict[str, SelectorParams] = {}
        self.policy_names = [
            "stateless",
            "4cand_majority",
            "4cand_confidence",
            "pair_topic_stage",
            "pair_topic_stage_support_sum",
            "anchor_gate",
        ]
        self.policy_tables: Dict[str, Dict[str, str]] = {}
        self.calibration_summary: Dict[str, Any] = {}
        self.is_fit = False

    def _fit_binary(self, X: np.ndarray, y: np.ndarray) -> Any:
        y = np.asarray(y).astype(int)
        if len(set(y.tolist())) < 2 or LogisticRegression is None:
            return ConstantProbabilityModel(float(y.mean()) if len(y) else 0.5)
        return Pipeline([
            ("scale", StandardScaler()),
            ("lr", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=self.random_state, solver="liblinear")),
        ]).fit(X, y)

    def fit(self, records: List[Dict[str, Any]], memory: ReliabilityMemory) -> "AdaptiveCandidateSelector":
        complete = [r for r in records if {c.get("candidate_id") for c in r.get("candidates", [])} >= {"A0", "B0", "A1", "B1"}]
        self.topic_vocab = sorted({str(r.get("topic", "unknown")) for r in complete})

        rows: List[Dict[str, Any]] = []
        X_list: List[List[float]] = []
        y_correct: List[int] = []
        y_risk: List[int] = []

        # Build feature rows. The first call fixes feature_names.
        for rec in complete:
            pair_id = str(rec.get("pair_id", ""))
            topic = str(rec.get("topic", "unknown"))
            candidates = rec.get("candidates", [])
            initial_correct = _initial_any_correct(candidates)
            for c in candidates:
                feats = self._feature_dict(c, candidates, memory, pair_id, topic)
                if not self.feature_names:
                    self.feature_names = list(feats.keys())
                X_list.append([float(feats[name]) for name in self.feature_names])
                correct = bool(c.get("is_correct", False))
                y_correct.append(1 if correct else 0)
                y_risk.append(1 if (initial_correct and not correct) else 0)
                rows.append({"record": rec, "candidate": c, "features": feats, "correct": correct, "risk": bool(initial_correct and not correct)})

        if not X_list:
            # Degenerate but safe fallback.
            self.correct_model = ConstantProbabilityModel(0.5)
            self.risk_model = ConstantProbabilityModel(0.5)
            self.best_params = {
                "adaptive_learned": SelectorParams(1.0, 0.0, 0.0, 0.0),
                "adaptive_accuracy": SelectorParams(0.0, 0.0, 0.0, 0.0),
                "adaptive_safety": SelectorParams(3.0, 0.0, 0.0, 0.0),
            }
            self.policy_tables = {"router_balanced": {}, "router_accuracy": {}, "router_safety": {}}
            self.calibration_summary = {"n_records": 0, "n_candidates": 0, "warning": "no calibration candidates"}
            self.is_fit = True
            return self

        X = np.asarray(X_list, dtype=float)
        self.correct_model = self._fit_binary(X, np.asarray(y_correct))
        self.risk_model = self._fit_binary(X, np.asarray(y_risk))

        param_grid = [
            SelectorParams(lam, sup, cross, indep)
            for lam in self.lambda_grid
            for sup in self.support_grid
            for cross in self.cross_stage_grid
            for indep in self.independent_grid
        ]

        stateless_acc, stateless_c2w = self._eval_stateless(complete)
        metrics = []
        for params in param_grid:
            metrics.append((params, self._eval_params(complete, memory, params)))

        # Objective 1: learned constrained selector.
        # Prefer settings that do no worse than stateless C2W; among those,
        # maximize accuracy. If no setting meets the constraint, minimize C2W.
        feasible = [(p, m) for p, m in metrics if m["c2w_rate"] <= stateless_c2w]
        if feasible:
            p_learned, m_learned = sorted(feasible, key=lambda pm: (pm[1]["accuracy"], -pm[1]["c2w_rate"], pm[1]["w2c_rate"]), reverse=True)[0]
        else:
            p_learned, m_learned = sorted(metrics, key=lambda pm: (-pm[1]["c2w_rate"], pm[1]["accuracy"], pm[1]["w2c_rate"]), reverse=True)[0]

        # Objective 2: pure accuracy ablation.
        p_acc, m_acc = sorted(metrics, key=lambda pm: (pm[1]["accuracy"], -pm[1]["c2w_rate"], pm[1]["w2c_rate"]), reverse=True)[0]

        # Objective 3: pure safety ablation.
        p_safe, m_safe = sorted(metrics, key=lambda pm: (-pm[1]["c2w_rate"], pm[1]["accuracy"], pm[1]["w2c_rate"]), reverse=True)[0]

        self.best_params = {
            "adaptive_learned": p_learned,
            "adaptive_accuracy": p_acc,
            "adaptive_safety": p_safe,
        }

        # Fit a dynamic policy router from calibration records only.
        # Unlike the candidate-level selector above, this chooses a base policy
        # per problem state bucket, for example when post-debate answers agree
        # or when post-debate answers are anchored to an independent answer.
        self.policy_tables, router_calib_summary = self._fit_policy_router(complete, memory, stateless_c2w)

        self.calibration_summary = {
            "n_records": len(complete),
            "n_candidates": len(X_list),
            "stateless_accuracy": float(stateless_acc),
            "stateless_c2w_rate": float(stateless_c2w),
            "correct_label_rate": float(np.mean(y_correct)),
            "risk_label_rate": float(np.mean(y_risk)),
            "best_params": {k: v.to_dict() for k, v in self.best_params.items()},
            "best_metrics": {
                "adaptive_learned": m_learned,
                "adaptive_accuracy": m_acc,
                "adaptive_safety": m_safe,
            },
            "policy_router": router_calib_summary,
            "feature_names": list(self.feature_names),
        }
        self.is_fit = True
        return self

    def _eval_stateless(self, records: List[Dict[str, Any]]) -> Tuple[float, float]:
        corrects = []
        c2w = []
        for rec in records:
            candidates = rec.get("candidates", [])
            chosen = _stateless_choice(candidates) or candidates[0]
            is_correct = bool(chosen.get("is_correct", False))
            initial_correct = _initial_any_correct(candidates)
            corrects.append(1 if is_correct else 0)
            if initial_correct:
                c2w.append(1 if not is_correct else 0)
        return float(np.mean(corrects)) if corrects else 0.0, float(np.mean(c2w)) if c2w else 0.0

    def _eval_params(self, records: List[Dict[str, Any]], memory: ReliabilityMemory, params: SelectorParams) -> Dict[str, float]:
        corrects = []
        c2w = []
        w2c = []
        for rec in records:
            selected, _ = self.select(rec.get("candidates", []), memory, str(rec.get("pair_id", "")), str(rec.get("topic", "unknown")), params=params)
            is_correct = bool(selected.get("is_correct", False))
            candidates = rec.get("candidates", [])
            initial_correct = _initial_any_correct(candidates)
            initial_both_wrong = not initial_correct
            corrects.append(1 if is_correct else 0)
            if initial_correct:
                c2w.append(1 if not is_correct else 0)
            if initial_both_wrong:
                w2c.append(1 if is_correct else 0)
        return {
            "accuracy": float(np.mean(corrects)) if corrects else 0.0,
            "c2w_rate": float(np.mean(c2w)) if c2w else 0.0,
            "w2c_rate": float(np.mean(w2c)) if w2c else 0.0,
        }

    def _feature_dict(
        self,
        c: Dict[str, Any],
        candidates: List[Dict[str, Any]],
        memory: ReliabilityMemory,
        pair_id: str,
        topic: str,
    ) -> Dict[str, float]:
        cid = str(c.get("candidate_id", ""))
        stage = str(c.get("stage", ""))
        agent_id = str(c.get("agent_id", ""))
        model = str(c.get("model", ""))
        norm = _norm(c)
        conf = _confidence(c) / 100.0

        same_answer = [x for x in candidates if _norm(x) and _norm(x) == norm]
        support_count = len(same_answer)
        independent_support = sum(1 for x in same_answer if x.get("stage") == "independent")
        post_support = sum(1 for x in same_answer if x.get("stage") == "post_debate")
        cross_stage = 1.0 if independent_support > 0 and post_support > 0 else 0.0

        a0 = _cand_by_id(candidates, "A0") or {}
        b0 = _cand_by_id(candidates, "B0") or {}
        a1 = _cand_by_id(candidates, "A1") or {}
        b1 = _cand_by_id(candidates, "B1") or {}
        own0 = a0 if cid.startswith("A") else b0
        partner0 = b0 if cid.startswith("A") else a0
        partner1 = b1 if cid.startswith("A") else a1
        stateless = _stateless_choice(candidates) or {}

        rel = memory.score_pair_topic_stage(pair_id, agent_id, model, topic, stage)
        saf = memory.score_candidate_safety(pair_id, agent_id, model, topic, stage, cid)

        feats: Dict[str, float] = {
            "confidence": conf,
            "is_A": 1.0 if cid.startswith("A") else 0.0,
            "is_B": 1.0 if cid.startswith("B") else 0.0,
            "is_independent": 1.0 if stage == "independent" else 0.0,
            "is_post_debate": 1.0 if stage == "post_debate" else 0.0,
            "is_A0": 1.0 if cid == "A0" else 0.0,
            "is_B0": 1.0 if cid == "B0" else 0.0,
            "is_A1": 1.0 if cid == "A1" else 0.0,
            "is_B1": 1.0 if cid == "B1" else 0.0,
            "support_count": float(support_count) / 4.0,
            "independent_support": float(independent_support) / 2.0,
            "post_support": float(post_support) / 2.0,
            "cross_stage_support": cross_stage,
            "same_as_own_initial": 1.0 if norm and norm == _norm(own0) else 0.0,
            "changed_from_own_initial": 1.0 if stage == "post_debate" and norm and norm != _norm(own0) else 0.0,
            "same_as_partner_initial": 1.0 if norm and norm == _norm(partner0) else 0.0,
            "same_as_partner_post": 1.0 if norm and norm == _norm(partner1) else 0.0,
            "same_as_stateless": 1.0 if norm and norm == _norm(stateless) else 0.0,
            "reliability_score": float(rel.get("score", 0.5)),
            "reliability_topic_total_log": math.log1p(float(rel.get("topic_total", 0))),
            "candidate_accuracy_score": float(saf.get("accuracy_score", 0.5)),
            "candidate_c2w_score": float(saf.get("c2w_score", 0.5)),
            "candidate_utility_score": float(saf.get("utility_score", 0.0)),
            "candidate_topic_total_log": math.log1p(float(saf.get("topic_total", 0))),
            "candidate_c2w_risk_log": math.log1p(float(saf.get("topic_c2w_risk", 0))),
        }
        for t in self.topic_vocab:
            feats[f"topic__{t}"] = 1.0 if topic == t else 0.0
        return feats

    def _predict_candidate_scores(
        self,
        candidates: List[Dict[str, Any]],
        memory: ReliabilityMemory,
        pair_id: str,
        topic: str,
    ) -> List[Dict[str, Any]]:
        rows = []
        X_list = []
        for c in candidates:
            feats = self._feature_dict(c, candidates, memory, pair_id, topic)
            # New unseen topics simply have all-zero topic one-hot features.
            X_list.append([float(feats.get(name, 0.0)) for name in self.feature_names])
            rows.append({"candidate": c, "features": feats})
        if not rows:
            return []
        X = np.asarray(X_list, dtype=float)
        p_correct = self.correct_model.predict_proba(X)[:, 1]
        p_risk = self.risk_model.predict_proba(X)[:, 1]
        for row, pc, pr in zip(rows, p_correct, p_risk):
            row["p_correct"] = float(max(0.0, min(1.0, pc)))
            row["p_c2w"] = float(max(0.0, min(1.0, pr)))
        return rows


    def _policy_bucket_keys(self, candidates: List[Dict[str, Any]], topic: str) -> List[str]:
        """Return most-specific to least-specific router buckets for a problem.

        Buckets use only observable candidate structure, not correctness labels.
        This makes the router dynamic without using test leakage.
        """
        a0 = _cand_by_id(candidates, "A0") or {}
        b0 = _cand_by_id(candidates, "B0") or {}
        a1 = _cand_by_id(candidates, "A1") or {}
        b1 = _cand_by_id(candidates, "B1") or {}
        n_a0, n_b0, n_a1, n_b1 = _norm(a0), _norm(b0), _norm(a1), _norm(b1)
        post_agree = bool(n_a1 and n_a1 == n_b1)
        init_agree = bool(n_a0 and n_a0 == n_b0)
        a_changed = bool(n_a0 and n_a1 and n_a0 != n_a1)
        b_changed = bool(n_b0 and n_b1 and n_b0 != n_b1)
        post_anchor = bool((n_a1 and (n_a1 == n_a0 or n_a1 == n_b0)) or (n_b1 and (n_b1 == n_a0 or n_b1 == n_b0)))
        cross_support = False
        for n in {n_a0, n_b0}:
            if n and (n == n_a1 or n == n_b1):
                cross_support = True
                break
        flags = f"pa{int(post_agree)}_ia{int(init_agree)}_ach{int(a_changed)}_bch{int(b_changed)}_anch{int(post_anchor)}_cross{int(cross_support)}"
        return [
            f"topic={topic}|{flags}",
            f"topic={topic}|pa{int(post_agree)}_anch{int(post_anchor)}_cross{int(cross_support)}",
            f"global|{flags}",
            f"global|pa{int(post_agree)}_anch{int(post_anchor)}_cross{int(cross_support)}",
            "global",
        ]

    def _choose_by_policy(self, policy: str, candidates: List[Dict[str, Any]], memory: ReliabilityMemory, pair_id: str, topic: str) -> Dict[str, Any]:
        if not candidates:
            return {}
        a0 = _cand_by_id(candidates, "A0")
        b0 = _cand_by_id(candidates, "B0")
        a1 = _cand_by_id(candidates, "A1")
        b1 = _cand_by_id(candidates, "B1")

        if policy == "stateless":
            return _stateless_choice(candidates) or a1

        if policy == "4cand_confidence":
            return sorted(candidates, key=lambda c: (_confidence(c), 1 if c.get("stage") == "independent" else 0), reverse=True)[0]

        if policy == "4cand_majority":
            counts: Dict[str, int] = defaultdict(int)
            for c in candidates:
                n = _norm(c)
                if n:
                    counts[n] += 1
            if not counts:
                return a1
            best_norm = sorted(counts.keys(), key=lambda n: (counts[n], max(_confidence(c) for c in candidates if _norm(c) == n)), reverse=True)[0]
            reps = [c for c in candidates if _norm(c) == best_norm]
            return sorted(reps, key=lambda c: (_confidence(c), 1 if c.get("stage") == "independent" else 0), reverse=True)[0]

        def rel_score(c: Dict[str, Any]) -> float:
            return float(memory.score_pair_topic_stage(pair_id, str(c.get("agent_id", "")), str(c.get("model", "")), topic, str(c.get("stage", ""))).get("score", 0.5))

        if policy == "pair_topic_stage":
            return sorted(candidates, key=lambda c: (rel_score(c), _confidence(c), 1 if c.get("stage") == "independent" else 0), reverse=True)[0]

        if policy == "pair_topic_stage_support_sum":
            answer_scores: Dict[str, float] = defaultdict(float)
            for c in candidates:
                n = _norm(c)
                if n:
                    answer_scores[n] += rel_score(c)
            if not answer_scores:
                return a1
            best_norm = sorted(answer_scores.keys(), key=lambda n: (answer_scores[n], sum(1 for c in candidates if _norm(c) == n)), reverse=True)[0]
            reps = [c for c in candidates if _norm(c) == best_norm]
            return sorted(reps, key=lambda c: (_confidence(c), 1 if c.get("stage") == "independent" else 0), reverse=True)[0]

        if policy == "anchor_gate":
            st = _stateless_choice(candidates) or a1
            st_norm = _norm(st)
            if st_norm and (st_norm == _norm(a0) or st_norm == _norm(b0)):
                return st
            return sorted([a0, b0], key=lambda c: (rel_score(c), _confidence(c)), reverse=True)[0]

        # Safe fallback.
        return _stateless_choice(candidates) or a1

    def _policy_outcome(self, selected: Dict[str, Any], candidates: List[Dict[str, Any]]) -> Dict[str, float]:
        initial_correct = _initial_any_correct(candidates)
        both_wrong = not initial_correct
        correct = bool(selected.get("is_correct", False))
        return {
            "correct": 1.0 if correct else 0.0,
            "c2w": 1.0 if initial_correct and not correct else 0.0,
            "c2w_risk": 1.0 if initial_correct else 0.0,
            "w2c": 1.0 if both_wrong and correct else 0.0,
            "both_wrong": 1.0 if both_wrong else 0.0,
        }

    def _summarize_policy_rows(self, rows: List[Dict[str, float]]) -> Dict[str, float]:
        n = max(1, len(rows))
        c2w_den = sum(r["c2w_risk"] for r in rows)
        w2c_den = sum(r["both_wrong"] for r in rows)
        return {
            "n": float(len(rows)),
            "accuracy": sum(r["correct"] for r in rows) / n,
            "c2w_rate": (sum(r["c2w"] for r in rows) / c2w_den) if c2w_den else 0.0,
            "w2c_rate": (sum(r["w2c"] for r in rows) / w2c_den) if w2c_den else 0.0,
            "c2w_risk_n": float(c2w_den),
        }

    def _fit_policy_router(self, records: List[Dict[str, Any]], memory: ReliabilityMemory, stateless_c2w: float) -> Tuple[Dict[str, Dict[str, str]], Dict[str, Any]]:
        # Collect outcomes by bucket and policy.
        bucket_policy_rows: Dict[str, Dict[str, List[Dict[str, float]]]] = defaultdict(lambda: defaultdict(list))
        global_rows: Dict[str, List[Dict[str, float]]] = defaultdict(list)
        for rec in records:
            candidates = rec.get("candidates", [])
            if {c.get("candidate_id") for c in candidates} < {"A0", "B0", "A1", "B1"}:
                continue
            pair_id = str(rec.get("pair_id", ""))
            topic = str(rec.get("topic", "unknown"))
            keys = self._policy_bucket_keys(candidates, topic)
            for policy in self.policy_names:
                selected = self._choose_by_policy(policy, candidates, memory, pair_id, topic)
                out = self._policy_outcome(selected, candidates)
                global_rows[policy].append(out)
                for key in keys:
                    bucket_policy_rows[key][policy].append(out)

        def choose_policy(policy_rows: Dict[str, List[Dict[str, float]]], objective: str, min_n: int = 8) -> Tuple[str, Dict[str, float]]:
            summaries = {p: self._summarize_policy_rows(rows) for p, rows in policy_rows.items() if len(rows) >= min_n}
            if not summaries:
                summaries = {p: self._summarize_policy_rows(rows) for p, rows in global_rows.items()}
            stateless_summary = summaries.get("stateless") or self._summarize_policy_rows(global_rows.get("stateless", []))
            if objective == "router_accuracy":
                best_policy = sorted(summaries.keys(), key=lambda p: (summaries[p]["accuracy"], -summaries[p]["c2w_rate"], summaries[p]["w2c_rate"]), reverse=True)[0]
            elif objective == "router_safety":
                best_policy = sorted(summaries.keys(), key=lambda p: (-summaries[p]["c2w_rate"], summaries[p]["accuracy"], summaries[p]["w2c_rate"]), reverse=True)[0]
            else:
                # Balanced: first require no worse than stateless C2W in this bucket if possible,
                # then maximize accuracy. If impossible, minimize C2W and then maximize accuracy.
                feasible = [p for p in summaries if summaries[p]["c2w_rate"] <= stateless_summary["c2w_rate"]]
                if feasible:
                    best_policy = sorted(feasible, key=lambda p: (summaries[p]["accuracy"], -summaries[p]["c2w_rate"], summaries[p]["w2c_rate"]), reverse=True)[0]
                else:
                    best_policy = sorted(summaries.keys(), key=lambda p: (-summaries[p]["c2w_rate"], summaries[p]["accuracy"], summaries[p]["w2c_rate"]), reverse=True)[0]
            return best_policy, summaries[best_policy]

        tables: Dict[str, Dict[str, str]] = {"router_balanced": {}, "router_accuracy": {}, "router_safety": {}}
        table_metrics: Dict[str, Dict[str, Any]] = {"router_balanced": {}, "router_accuracy": {}, "router_safety": {}}
        for key, policy_rows in bucket_policy_rows.items():
            for obj in tables:
                policy, summ = choose_policy(policy_rows, obj)
                tables[obj][key] = policy
                table_metrics[obj][key] = {"policy": policy, **summ}

        # Always include a global fallback.
        for obj in tables:
            policy, summ = choose_policy(global_rows, obj, min_n=1)
            tables[obj]["global"] = policy
            table_metrics[obj]["global"] = {"policy": policy, **summ}

        summary = {
            "policies": list(self.policy_names),
            "n_buckets": {obj: len(tables[obj]) for obj in tables},
            "global_policy": {obj: tables[obj].get("global") for obj in tables},
            "global_metrics": {obj: table_metrics[obj].get("global", {}) for obj in tables},
            "note": "Router buckets and policy choices are learned only from calibration records and frozen before test.",
        }
        return tables, summary

    def select_policy_router(
        self,
        candidates: List[Dict[str, Any]],
        memory: ReliabilityMemory,
        pair_id: str,
        topic: str,
        variant: str = "router_balanced",
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if not candidates:
            return {}, {"selector_type": variant, "error": "no_candidates"}
        if not self.is_fit:
            return candidates[0], {"selector_type": variant, "error": "selector_not_fit"}
        table = self.policy_tables.get(variant, {}) or self.policy_tables.get("router_balanced", {})
        keys = self._policy_bucket_keys(candidates, topic)
        selected_key = "global"
        policy = table.get("global", "stateless")
        for key in keys:
            if key in table:
                selected_key = key
                policy = table[key]
                break
        selected = self._choose_by_policy(policy, candidates, memory, pair_id, topic)
        info = {
            "selector_type": variant,
            "router_bucket": selected_key,
            "routed_policy": policy,
            "selected_candidate_id": selected.get("candidate_id"),
            "selected_answer_norm": selected.get("normalized_answer"),
            "candidate_ids": [c.get("candidate_id") for c in candidates],
        }
        return selected, info

    def select(
        self,
        candidates: List[Dict[str, Any]],
        memory: ReliabilityMemory,
        pair_id: str,
        topic: str,
        variant: str = "adaptive_learned",
        params: Optional[SelectorParams] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if not candidates:
            return {}, {"selector_type": variant, "error": "no_candidates"}
        if not self.is_fit:
            return candidates[0], {"selector_type": variant, "error": "selector_not_fit"}
        params = params or self.best_params.get(variant) or self.best_params.get("adaptive_learned")
        if params is None:
            params = SelectorParams(1.0, 0.0, 0.0, 0.0)

        pred_rows = self._predict_candidate_scores(candidates, memory, pair_id, topic)
        answer_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in pred_rows:
            norm = _norm(row["candidate"])
            if norm:
                answer_groups[norm].append(row)

        if not answer_groups:
            return candidates[0], {"selector_type": variant, "error": "no_normalized_answers"}

        scored_answers = []
        for norm, rows in answer_groups.items():
            stages = {str(r["candidate"].get("stage", "")) for r in rows}
            has_cross = 1.0 if "independent" in stages and "post_debate" in stages else 0.0
            has_independent = 1.0 if "independent" in stages else 0.0
            support_count = len(rows)
            max_pc = max(r["p_correct"] for r in rows)
            min_risk = min(r["p_c2w"] for r in rows)
            # Answer-level score. Parameters are tuned on calibration only.
            answer_score = (
                max_pc
                - params.lambda_risk * min_risk
                + params.support_weight * math.log1p(support_count)
                + params.cross_stage_weight * has_cross
                + params.independent_weight * has_independent
            )
            scored_answers.append({
                "normalized_answer": norm,
                "answer_score": float(answer_score),
                "max_p_correct": float(max_pc),
                "min_p_c2w": float(min_risk),
                "support_count": int(support_count),
                "cross_stage_support": bool(has_cross),
                "has_independent_support": bool(has_independent),
                "rows": rows,
            })

        scored_answers.sort(key=lambda x: (x["answer_score"], x["max_p_correct"], -x["min_p_c2w"], x["support_count"]), reverse=True)
        best_answer = scored_answers[0]

        # Representative candidate for the selected answer.
        rep_rows = best_answer["rows"]
        def rep_key(row: Dict[str, Any]) -> Tuple[float, float, float, float]:
            c = row["candidate"]
            indep = 1.0 if c.get("stage") == "independent" else 0.0
            return (row["p_correct"] - params.lambda_risk * row["p_c2w"], row["p_correct"], _confidence(c), indep)
        rep_rows = sorted(rep_rows, key=rep_key, reverse=True)
        selected = rep_rows[0]["candidate"]

        info = {
            "selector_type": variant,
            "selected_candidate_id": selected.get("candidate_id"),
            "selected_answer_norm": best_answer["normalized_answer"],
            "params": params.to_dict(),
            "calibration_summary": self.calibration_summary,
            "answer_scores": [
                {k: v for k, v in ans.items() if k != "rows"}
                for ans in scored_answers
            ],
            "candidate_predictions": [
                {
                    "candidate_id": row["candidate"].get("candidate_id"),
                    "normalized_answer": row["candidate"].get("normalized_answer"),
                    "stage": row["candidate"].get("stage"),
                    "agent_id": row["candidate"].get("agent_id"),
                    "p_correct": row["p_correct"],
                    "p_c2w": row["p_c2w"],
                }
                for row in sorted(pred_rows, key=lambda r: (r["p_correct"] - params.lambda_risk * r["p_c2w"]), reverse=True)
            ],
        }
        return selected, info

    def to_dict(self) -> Dict[str, Any]:
        return {
            "topic_vocab": self.topic_vocab,
            "feature_names": self.feature_names,
            "best_params": {k: v.to_dict() for k, v in self.best_params.items()},
            "policy_tables": self.policy_tables,
            "calibration_summary": self.calibration_summary,
            "note": "Models and routers are recomputed from calibration records at run time; coefficients are not required for evaluation reruns.",
        }
