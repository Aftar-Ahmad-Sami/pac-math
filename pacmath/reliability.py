from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Tuple


Stage = str


@dataclass
class ReliabilityMemory:
    alpha: float = 10.0
    # Counts are keyed by tuples.
    counts: Dict[Tuple[str, ...], Dict[str, int]] = field(default_factory=lambda: defaultdict(lambda: {"correct": 0, "total": 0}))
    # Candidate-policy counts are used by safety selectors. They track how often
    # choosing a candidate id would be correct and how often it would create a
    # correct-to-wrong flip in calibration.
    candidate_counts: Dict[Tuple[str, ...], Dict[str, int]] = field(
        default_factory=lambda: defaultdict(lambda: {"correct": 0, "total": 0, "c2w_flip": 0, "c2w_risk": 0})
    )

    def update(self, pair_id: str, agent_id: str, model: str, topic: str, stage: str, correct: bool) -> None:
        c = 1 if correct else 0
        keys = [
            ("pair_agent_topic_stage", pair_id, agent_id, topic, stage),
            ("pair_agent_stage", pair_id, agent_id, stage),
            ("agent_topic_stage", model, topic, stage),
            ("agent_stage", model, stage),
            ("global_stage", stage),
        ]
        for key in keys:
            self.counts[key]["correct"] += c
            self.counts[key]["total"] += 1


    def update_candidate_policy(
        self,
        pair_id: str,
        agent_id: str,
        model: str,
        topic: str,
        stage: str,
        candidate_id: str,
        correct: bool,
        initial_any_correct: bool,
    ) -> None:
        """Update frozen calibration statistics for candidate-selection policies.

        If this candidate were selected and at least one initial answer was
        correct, then selecting an incorrect candidate would be a C2W flip.
        These statistics let us evaluate safety-first policies without using
        test labels.
        """
        c = 1 if correct else 0
        risk = 1 if initial_any_correct else 0
        flip = 1 if (initial_any_correct and not correct) else 0
        keys = [
            ("cand_pair_agent_topic_stage", pair_id, agent_id, topic, stage, candidate_id),
            ("cand_pair_agent_stage", pair_id, agent_id, stage, candidate_id),
            ("cand_agent_topic_stage", model, topic, stage, candidate_id),
            ("cand_agent_stage", model, stage, candidate_id),
            ("cand_stage", stage, candidate_id),
            ("cand_global", candidate_id),
        ]
        for key in keys:
            self.candidate_counts[key]["correct"] += c
            self.candidate_counts[key]["total"] += 1
            self.candidate_counts[key]["c2w_flip"] += flip
            self.candidate_counts[key]["c2w_risk"] += risk

    def _cand_rate(self, key: Tuple[str, ...], field_num: str, field_den: str, default: float) -> float:
        item = self.candidate_counts.get(key)
        if not item or int(item.get(field_den, 0)) <= 0:
            return default
        return (float(item.get(field_num, 0)) + 1.0) / (float(item.get(field_den, 0)) + 2.0)

    def score_candidate_safety(
        self,
        pair_id: str,
        agent_id: str,
        model: str,
        topic: str,
        stage: str,
        candidate_id: str,
    ) -> Dict[str, Any]:
        """Return calibration accuracy and C2W-risk estimates for a candidate.

        This selector directly targets the paper's primary failure mode. It is
        still fully frozen-memory: all values are computed only from calibration
        records and then applied to test records.
        """
        topic_key = ("cand_pair_agent_topic_stage", pair_id, agent_id, topic, stage, candidate_id)
        pair_key = ("cand_pair_agent_stage", pair_id, agent_id, stage, candidate_id)
        agent_key = ("cand_agent_stage", model, stage, candidate_id)
        global_key = ("cand_stage", stage, candidate_id)
        candidate_global_key = ("cand_global", candidate_id)

        # Accuracy prior hierarchy.
        prior_acc = self._cand_rate(pair_key, "correct", "total", default=None)  # type: ignore[arg-type]
        prior_source = "cand_pair_agent_stage"
        if prior_acc is None:
            prior_acc = self._cand_rate(agent_key, "correct", "total", default=None)  # type: ignore[arg-type]
            prior_source = "cand_agent_stage"
        if prior_acc is None:
            prior_acc = self._cand_rate(global_key, "correct", "total", default=None)  # type: ignore[arg-type]
            prior_source = "cand_stage"
        if prior_acc is None:
            prior_acc = self._cand_rate(candidate_global_key, "correct", "total", default=0.5)
            prior_source = "cand_global"

        # C2W prior hierarchy. Smaller is better.
        prior_c2w = self._cand_rate(pair_key, "c2w_flip", "c2w_risk", default=None)  # type: ignore[arg-type]
        c2w_prior_source = "cand_pair_agent_stage"
        if prior_c2w is None:
            prior_c2w = self._cand_rate(agent_key, "c2w_flip", "c2w_risk", default=None)  # type: ignore[arg-type]
            c2w_prior_source = "cand_agent_stage"
        if prior_c2w is None:
            prior_c2w = self._cand_rate(global_key, "c2w_flip", "c2w_risk", default=None)  # type: ignore[arg-type]
            c2w_prior_source = "cand_stage"
        if prior_c2w is None:
            prior_c2w = self._cand_rate(candidate_global_key, "c2w_flip", "c2w_risk", default=0.5)
            c2w_prior_source = "cand_global"

        item = self.candidate_counts.get(topic_key, {"correct": 0, "total": 0, "c2w_flip": 0, "c2w_risk": 0})
        correct = int(item.get("correct", 0))
        total = int(item.get("total", 0))
        flips = int(item.get("c2w_flip", 0))
        risks = int(item.get("c2w_risk", 0))
        acc = (correct + self.alpha * float(prior_acc)) / (total + self.alpha)
        c2w = (flips + self.alpha * float(prior_c2w)) / (risks + self.alpha)
        utility = acc - c2w
        return {
            "accuracy_score": float(acc),
            "c2w_score": float(c2w),
            "utility_score": float(utility),
            "topic_correct": correct,
            "topic_total": total,
            "topic_c2w_flip": flips,
            "topic_c2w_risk": risks,
            "accuracy_prior": float(prior_acc),
            "accuracy_prior_source": prior_source,
            "c2w_prior": float(prior_c2w),
            "c2w_prior_source": c2w_prior_source,
            "alpha": self.alpha,
        }

    def _rate(self, key: Tuple[str, ...], default: float = 0.5) -> float:
        item = self.counts.get(key)
        if not item or item["total"] <= 0:
            return default
        return (item["correct"] + 1.0) / (item["total"] + 2.0)

    def _total(self, key: Tuple[str, ...]) -> int:
        item = self.counts.get(key)
        if not item:
            return 0
        return int(item["total"])

    def score_pair_topic_stage(self, pair_id: str, agent_id: str, model: str, topic: str, stage: str) -> Dict[str, Any]:
        topic_key = ("pair_agent_topic_stage", pair_id, agent_id, topic, stage)
        pair_stage_key = ("pair_agent_stage", pair_id, agent_id, stage)
        agent_stage_key = ("agent_stage", model, stage)
        global_stage_key = ("global_stage", stage)

        # Prior level: pair-agent-stage; if missing, agent-stage; if missing, global-stage; else 0.5.
        prior = self._rate(pair_stage_key, default=None)  # type: ignore[arg-type]
        prior_source = "pair_agent_stage"
        if prior is None:
            prior = self._rate(agent_stage_key, default=None)  # type: ignore[arg-type]
            prior_source = "agent_stage"
        if prior is None:
            prior = self._rate(global_stage_key, default=0.5)
            prior_source = "global_stage"

        topic_counts = self.counts.get(topic_key, {"correct": 0, "total": 0})
        correct = int(topic_counts["correct"])
        total = int(topic_counts["total"])
        smoothed = (correct + self.alpha * float(prior)) / (total + self.alpha)
        return {
            "score": float(smoothed),
            "topic_correct": correct,
            "topic_total": total,
            "prior": float(prior),
            "prior_source": prior_source,
            "alpha": self.alpha,
        }

    def score_overall(self, pair_id: str, agent_id: str, model: str, topic: str, stage: str) -> Dict[str, Any]:
        key = ("pair_agent_stage", pair_id, agent_id, stage)
        score = self._rate(key, default=None)  # type: ignore[arg-type]
        source = "pair_agent_stage"
        if score is None:
            key = ("agent_stage", model, stage)
            score = self._rate(key, default=None)  # type: ignore[arg-type]
            source = "agent_stage"
        if score is None:
            key = ("global_stage", stage)
            score = self._rate(key, default=0.5)
            source = "global_stage"
        return {"score": float(score), "source": source}

    def score_agent_topic(self, pair_id: str, agent_id: str, model: str, topic: str, stage: str) -> Dict[str, Any]:
        # Ignore pair and use agent-topic-stage. This is an ablation.
        topic_key = ("agent_topic_stage", model, topic, stage)
        prior = self._rate(("agent_stage", model, stage), default=self._rate(("global_stage", stage), default=0.5))
        topic_counts = self.counts.get(topic_key, {"correct": 0, "total": 0})
        correct = int(topic_counts["correct"])
        total = int(topic_counts["total"])
        smoothed = (correct + self.alpha * float(prior)) / (total + self.alpha)
        return {"score": float(smoothed), "topic_correct": correct, "topic_total": total, "prior": float(prior)}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alpha": self.alpha,
            "counts": {"|".join(k): v for k, v in self.counts.items()},
            "candidate_counts": {"|".join(k): v for k, v in self.candidate_counts.items()},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReliabilityMemory":
        mem = cls(alpha=float(data.get("alpha", 10.0)))
        counts = data.get("counts", {})
        for key_str, value in counts.items():
            key = tuple(key_str.split("|"))
            mem.counts[key] = {"correct": int(value.get("correct", 0)), "total": int(value.get("total", 0))}
        cand_counts = data.get("candidate_counts", {})
        for key_str, value in cand_counts.items():
            key = tuple(key_str.split("|"))
            mem.candidate_counts[key] = {
                "correct": int(value.get("correct", 0)),
                "total": int(value.get("total", 0)),
                "c2w_flip": int(value.get("c2w_flip", 0)),
                "c2w_risk": int(value.get("c2w_risk", 0)),
            }
        return mem
