from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

from .adaptive_selector import AdaptiveCandidateSelector
from .answer_norm import canonicalize
from .reliability import ReliabilityMemory


def _cand_by_id(candidates: List[Dict[str, Any]], cid: str) -> Dict[str, Any]:
    for c in candidates:
        if c.get("candidate_id") == cid:
            return c
    raise KeyError(cid)


def _is_valid_candidate(c: Dict[str, Any]) -> bool:
    """Candidate is eligible for selectors that compare answer content.

    Parse failures must not become a majority answer such as PARSE_ERROR.
    Single-agent baselines still expose parse failures, but multi-candidate
    selectors should ignore invalid candidates whenever any valid candidate is
    available.
    """
    if not bool(c.get("parse_ok", True)):
        return False
    ans = str(c.get("answer", "")).strip().lower()
    if ans in {"", "parse_error", "error", "none", "null"}:
        return False
    norm = str(c.get("normalized_answer", "")).strip().lower()
    if norm in {"", "parse_error", "parseerror", "error", "none", "null"}:
        return False
    return True


def _eligible(candidates: List[Dict[str, Any]], allow_invalid_if_all_bad: bool = True) -> List[Dict[str, Any]]:
    valid = [c for c in candidates if _is_valid_candidate(c)]
    if valid or not allow_invalid_if_all_bad:
        return valid
    return list(candidates)


def _norm(c: Dict[str, Any]) -> str:
    if not _is_valid_candidate(c):
        return ""
    return str(c.get("normalized_answer", canonicalize(c.get("answer", "")).normalized))


def _confidence(c: Dict[str, Any]) -> float:
    try:
        return float(c.get("confidence", 0) or 0)
    except Exception:
        return 0.0


def choose_single_a(candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    return _cand_by_id(candidates, "A0")


def choose_single_b(candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    return _cand_by_id(candidates, "B0")


def choose_stateless_debate(candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Standard post-debate baseline: only A1/B1 are eligible.

    If one post-debate response failed parsing, use the other valid post-debate
    response. If both failed, fall back to the original A1/B1 confidence rule so
    the parse failure is visible as a model failure.
    """
    a1 = _cand_by_id(candidates, "A1")
    b1 = _cand_by_id(candidates, "B1")
    valid_post = _eligible([a1, b1], allow_invalid_if_all_bad=False)
    if len(valid_post) == 1:
        return valid_post[0]
    if len(valid_post) == 2:
        if _norm(valid_post[0]) and _norm(valid_post[0]) == _norm(valid_post[1]):
            return valid_post[0]
        return valid_post[0] if _confidence(valid_post[0]) >= _confidence(valid_post[1]) else valid_post[1]
    return a1 if _confidence(a1) >= _confidence(b1) else b1


def choose_4cand_majority(candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    eligible = _eligible(candidates)
    counts = Counter(_norm(c) for c in eligible if _norm(c))
    if not counts:
        return eligible[0]
    best_norm, _ = counts.most_common(1)[0]
    tied = [c for c in eligible if _norm(c) == best_norm]
    # Within majority answer, choose highest confidence; stable order breaks remaining ties.
    tied = sorted(tied, key=lambda c: (_confidence(c), 1 if c.get("stage") == "independent" else 0), reverse=True)
    return tied[0]


def choose_4cand_confidence(candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    eligible = _eligible(candidates)
    return sorted(eligible, key=lambda c: (_confidence(c), 1 if c.get("stage") == "independent" else 0), reverse=True)[0]


def choose_oracle(candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    for c in candidates:
        if _is_valid_candidate(c) and bool(c.get("is_correct", False)):
            return c
    eligible = _eligible(candidates)
    return eligible[0]


def _score_candidate(
    c: Dict[str, Any],
    memory: ReliabilityMemory,
    pair_id: str,
    topic: str,
    mode: str,
) -> Dict[str, Any]:
    agent_id = str(c.get("agent_id", ""))
    model = str(c.get("model", ""))
    stage = str(c.get("stage", ""))
    if mode == "overall":
        info = memory.score_overall(pair_id, agent_id, model, topic, stage)
    elif mode == "agent_topic":
        info = memory.score_agent_topic(pair_id, agent_id, model, topic, stage)
    elif mode == "pair_topic_stage":
        info = memory.score_pair_topic_stage(pair_id, agent_id, model, topic, stage)
    else:
        raise ValueError(f"Unknown reliability mode: {mode}")
    return info


def choose_reliability(
    candidates: List[Dict[str, Any]],
    memory: ReliabilityMemory,
    pair_id: str,
    topic: str,
    mode: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Candidate-level reliability selector.

    This is the original PAC selector: each A0/B0/A1/B1 candidate receives
    a historical reliability score. The highest-scoring candidate is selected.
    """
    candidates = _eligible(candidates)
    scored: List[Tuple[float, float, int, int, Dict[str, Any], Dict[str, Any]]] = []
    for idx, c in enumerate(candidates):
        info = _score_candidate(c, memory, pair_id, topic, mode)
        score = float(info["score"])
        conf = _confidence(c)
        indep = 1 if str(c.get("stage")) == "independent" else 0
        # Deterministic tie-breaking: score, confidence, independent, earlier candidate order.
        scored.append((score, conf, indep, -idx, c, info))
    scored_sorted = sorted(scored, key=lambda x: (x[0], x[1], x[2], x[3]), reverse=True)
    best = scored_sorted[0]
    selected = best[4]
    info = {
        "mode": mode,
        "selector_type": "candidate_max",
        "selected_candidate_id": selected.get("candidate_id"),
        "selected_score": best[0],
        "candidate_scores": [
            {
                "candidate_id": item[4].get("candidate_id"),
                "answer": item[4].get("answer"),
                "normalized_answer": item[4].get("normalized_answer"),
                "score": item[0],
                "confidence": item[1],
                "stage": item[4].get("stage"),
                "agent_id": item[4].get("agent_id"),
                "score_info": item[5],
            }
            for item in scored_sorted
        ],
    }
    return selected, info


def choose_reliability_support_sum(
    candidates: List[Dict[str, Any]],
    memory: ReliabilityMemory,
    pair_id: str,
    topic: str,
    mode: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Answer-level reliability selector using support aggregation.

    The original selector can over-trust one high-scoring candidate even when
    several other candidates support another answer. This ablation first sums
    reliability scores for candidates with the same normalized answer, then
    returns the highest-confidence representative of the winning answer.

    This is not used to hide negative results. It is a principled ablation that
    distinguishes candidate preservation, candidate-level reliability, and
    answer-level support aggregation.
    """
    candidates = _eligible(candidates)
    answer_scores: Dict[str, float] = defaultdict(float)
    answer_candidates: Dict[str, List[Tuple[Dict[str, Any], Dict[str, Any], float]]] = defaultdict(list)

    for c in candidates:
        norm = _norm(c)
        if not norm:
            continue
        info = _score_candidate(c, memory, pair_id, topic, mode)
        score = float(info["score"])
        answer_scores[norm] += score
        answer_candidates[norm].append((c, info, score))

    if not answer_scores:
        return candidates[0], {"mode": mode, "selector_type": "support_sum", "selected_candidate_id": candidates[0].get("candidate_id")}

    # Pick answer with largest summed reliability. Tie-break by answer support count and max confidence.
    def answer_key(norm: str) -> Tuple[float, int, float]:
        cands = answer_candidates[norm]
        max_conf = max(_confidence(c) for c, _, _ in cands)
        return (answer_scores[norm], len(cands), max_conf)

    best_norm = sorted(answer_scores.keys(), key=answer_key, reverse=True)[0]
    reps = [item[0] for item in answer_candidates[best_norm]]
    selected = sorted(
        reps,
        key=lambda c: (_confidence(c), 1 if c.get("stage") == "independent" else 0, -candidates.index(c)),
        reverse=True,
    )[0]

    info = {
        "mode": mode,
        "selector_type": "support_sum",
        "selected_candidate_id": selected.get("candidate_id"),
        "selected_answer_norm": best_norm,
        "selected_answer_score": float(answer_scores[best_norm]),
        "answer_scores": [
            {
                "normalized_answer": norm,
                "sum_score": float(answer_scores[norm]),
                "support_count": len(answer_candidates[norm]),
                "candidate_ids": [x[0].get("candidate_id") for x in answer_candidates[norm]],
            }
            for norm in sorted(answer_scores.keys(), key=answer_key, reverse=True)
        ],
    }
    return selected, info


def choose_persuasion_guard(
    candidates: List[Dict[str, Any]],
    memory: ReliabilityMemory,
    pair_id: str,
    topic: str,
    mode: str = "pair_topic_stage",
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Persuasion-aware conservative ablation.

    If a post-debate candidate changed away from its own independent answer,
    select it only if its reliability score is strictly higher than that same
    agent's independent candidate. This directly tests the hypothesis that
    post-debate changes are a persuasion risk.
    """
    base_selected, base_info = choose_reliability(candidates, memory, pair_id, topic, mode)
    cid = str(base_selected.get("candidate_id"))
    if cid not in {"A1", "B1"}:
        base_info["selector_type"] = "persuasion_guard"
        base_info["guard_triggered"] = False
        return base_selected, base_info

    own0_id = "A0" if cid == "A1" else "B0"
    own0 = _cand_by_id(candidates, own0_id)
    # If the answer did not actually change, there is no persuasion risk.
    if _norm(base_selected) == _norm(own0):
        base_info["selector_type"] = "persuasion_guard"
        base_info["guard_triggered"] = False
        return base_selected, base_info

    post_info = _score_candidate(base_selected, memory, pair_id, topic, mode)
    indep_info = _score_candidate(own0, memory, pair_id, topic, mode)
    if float(post_info["score"]) > float(indep_info["score"]):
        base_info["selector_type"] = "persuasion_guard"
        base_info["guard_triggered"] = False
        return base_selected, base_info

    guard_info = {
        "mode": mode,
        "selector_type": "persuasion_guard",
        "selected_candidate_id": own0.get("candidate_id"),
        "guard_triggered": True,
        "guarded_from": cid,
        "post_score": float(post_info["score"]),
        "independent_score": float(indep_info["score"]),
    }
    return own0, guard_info



def choose_independent_reliability(
    candidates: List[Dict[str, Any]],
    memory: ReliabilityMemory,
    pair_id: str,
    topic: str,
    mode: str = "pair_topic_stage",
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Select only from A0/B0 using frozen reliability.

    This is an important diagnostic baseline. If this performs better than
    post-debate selectors, debate is adding persuasion risk that selection is
    not yet controlling.
    """
    eligible = _eligible([_cand_by_id(candidates, "A0"), _cand_by_id(candidates, "B0")])
    return choose_reliability(eligible, memory, pair_id, topic, mode)


def choose_candidate_safety(
    candidates: List[Dict[str, Any]],
    memory: ReliabilityMemory,
    pair_id: str,
    topic: str,
    selector_type: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Safety-aware frozen-memory selectors.

    safety_first: minimize calibration C2W rate first, then maximize accuracy.
    utility: maximize calibration accuracy minus calibration C2W rate.

    These methods are not allowed to use test labels. They use only the frozen
    candidate-policy statistics built from calibration records.
    """
    candidates = _eligible(candidates)
    scored = []
    for idx, c in enumerate(candidates):
        info = memory.score_candidate_safety(
            pair_id=pair_id,
            agent_id=str(c.get("agent_id", "")),
            model=str(c.get("model", "")),
            topic=topic,
            stage=str(c.get("stage", "")),
            candidate_id=str(c.get("candidate_id", "")),
        )
        acc = float(info["accuracy_score"])
        c2w = float(info["c2w_score"])
        util = float(info["utility_score"])
        conf = _confidence(c)
        indep = 1 if str(c.get("stage")) == "independent" else 0
        if selector_type == "safety_first":
            # Lower C2W is better, so use -c2w in descending sort.
            key = (-c2w, acc, conf, indep, -idx)
        elif selector_type == "utility":
            key = (util, -c2w, acc, conf, indep, -idx)
        elif selector_type == "accuracy_first":
            key = (acc, -c2w, conf, indep, -idx)
        else:
            raise ValueError(f"Unknown candidate safety selector: {selector_type}")
        scored.append((key, c, info))

    scored_sorted = sorted(scored, key=lambda x: x[0], reverse=True)
    selected = scored_sorted[0][1]
    info = {
        "mode": "candidate_policy",
        "selector_type": selector_type,
        "selected_candidate_id": selected.get("candidate_id"),
        "candidate_scores": [
            {
                "candidate_id": item[1].get("candidate_id"),
                "normalized_answer": item[1].get("normalized_answer"),
                "stage": item[1].get("stage"),
                "agent_id": item[1].get("agent_id"),
                "accuracy_score": float(item[2]["accuracy_score"]),
                "c2w_score": float(item[2]["c2w_score"]),
                "utility_score": float(item[2]["utility_score"]),
                "score_info": item[2],
            }
            for item in scored_sorted
        ],
    }
    return selected, info



def choose_anchor_gate(
    candidates: List[Dict[str, Any]],
    memory: ReliabilityMemory,
    pair_id: str,
    topic: str,
    mode: str = "pair_topic_stage",
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Anchored debate gate.

    First run the normal stateless debate selector. Accept its post-debate answer
    only if that answer is anchored by at least one independent answer A0/B0.
    If the post-debate answer is new and unsupported by either independent
    candidate, fall back to reliability selection over A0/B0.

    Rationale: the pilot showed that harmful persuasion often appears as a
    post-debate answer dominating the selector. This gate tests whether debate
    is safer when it confirms or preserves an initial answer rather than creating
    an unsupported new one.
    """
    a0 = _cand_by_id(candidates, "A0")
    b0 = _cand_by_id(candidates, "B0")
    stateless = choose_stateless_debate(candidates)
    initial_norms = {_norm(a0), _norm(b0)}
    if _norm(stateless) and _norm(stateless) in initial_norms:
        return stateless, {
            "mode": mode,
            "selector_type": "anchor_gate",
            "selected_candidate_id": stateless.get("candidate_id"),
            "accepted_post_debate": True,
            "reason": "post_debate_answer_matches_independent_answer",
        }
    fallback, info = choose_independent_reliability(candidates, memory, pair_id, topic, mode)
    info = dict(info)
    info.update({
        "selector_type": "anchor_gate",
        "selected_candidate_id": fallback.get("candidate_id"),
        "accepted_post_debate": False,
        "rejected_candidate_id": stateless.get("candidate_id"),
        "reason": "post_debate_answer_not_anchored_by_independent_answer",
    })
    return fallback, info




def choose_cross_agent_anchor_gate(
    candidates: List[Dict[str, Any]],
    memory: ReliabilityMemory,
    pair_id: str,
    topic: str,
    mode: str = "pair_topic_stage",
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Stricter evidence-gated anchor selector.

    The earlier anchor_gate accepted a post-debate answer if it matched *any*
    independent answer. Under the evidence-gated protocol, A1 often equals A0
    and B1 often equals B0. That made anchor_gate almost collapse back into
    stateless_debate, because the selected post-debate answer was usually
    "anchored" only by the same agent's own unchanged answer.

    This stricter variant accepts the stateless post-debate answer only when the
    selected answer has cross-agent support: at least one A-side candidate and
    at least one B-side candidate share the same normalized answer. Otherwise it
    falls back to answer-level four-candidate majority, with independent-stage
    support preferred for ties.

    It is still test-label-free. It uses only observable agreement structure.
    """
    a0 = _cand_by_id(candidates, "A0")
    b0 = _cand_by_id(candidates, "B0")
    a1 = _cand_by_id(candidates, "A1")
    b1 = _cand_by_id(candidates, "B1")
    stateless = choose_stateless_debate(candidates)
    st_norm = _norm(stateless)

    def has_cross_agent_support(norm: str) -> bool:
        has_a = any(str(c.get("candidate_id", "")).startswith("A") and _norm(c) == norm for c in candidates)
        has_b = any(str(c.get("candidate_id", "")).startswith("B") and _norm(c) == norm for c in candidates)
        return bool(norm and has_a and has_b)

    def has_cross_stage_support(norm: str) -> bool:
        has_ind = any(c.get("stage") == "independent" and _norm(c) == norm for c in candidates)
        has_post = any(c.get("stage") == "post_debate" and _norm(c) == norm for c in candidates)
        return bool(norm and has_ind and has_post)

    if st_norm and has_cross_agent_support(st_norm) and has_cross_stage_support(st_norm):
        return stateless, {
            "mode": mode,
            "selector_type": "cross_agent_anchor_gate",
            "selected_candidate_id": stateless.get("candidate_id"),
            "accepted_stateless": True,
            "reason": "post_debate_answer_has_cross_agent_and_cross_stage_support",
        }

    # Fallback: observable answer support rather than one candidate's confidence.
    # This prevents a single high-confidence post-debate candidate from dominating.
    selected = choose_4cand_majority(candidates)
    return selected, {
        "mode": mode,
        "selector_type": "cross_agent_anchor_gate",
        "selected_candidate_id": selected.get("candidate_id"),
        "accepted_stateless": False,
        "rejected_candidate_id": stateless.get("candidate_id"),
        "reason": "stateless_answer_lacked_cross_agent_or_cross_stage_support; fallback_4cand_majority",
        "stateless_norm": st_norm,
        "a0_norm": _norm(a0),
        "b0_norm": _norm(b0),
        "a1_norm": _norm(a1),
        "b1_norm": _norm(b1),
    }

def choose_cross_stage_support(
    candidates: List[Dict[str, Any]],
    memory: ReliabilityMemory,
    pair_id: str,
    topic: str,
    mode: str = "pair_topic_stage",
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Select an answer only when it has cross-stage support.

    An answer is eligible if it appears in at least one independent candidate
    and at least one post-debate candidate. Among eligible answers, select the
    answer with the largest summed frozen reliability. If no answer has
    cross-stage support, fall back to A0/B0 reliability selection.
    """
    groups: Dict[str, List[Tuple[Dict[str, Any], Dict[str, Any], float]]] = defaultdict(list)
    for c in candidates:
        norm = _norm(c)
        if not norm:
            continue
        info = _score_candidate(c, memory, pair_id, topic, mode)
        groups[norm].append((c, info, float(info["score"])))

    eligible = {}
    for norm, items in groups.items():
        stages = {str(item[0].get("stage")) for item in items}
        if "independent" in stages and "post_debate" in stages:
            eligible[norm] = items

    if not eligible:
        fallback, info = choose_independent_reliability(candidates, memory, pair_id, topic, mode)
        info = dict(info)
        info.update({
            "selector_type": "cross_stage_support",
            "selected_candidate_id": fallback.get("candidate_id"),
            "used_cross_stage_answer": False,
            "reason": "no_answer_has_both_independent_and_post_debate_support",
        })
        return fallback, info

    def answer_key(norm: str) -> Tuple[float, int, float]:
        items = eligible[norm]
        sum_score = sum(x[2] for x in items)
        max_conf = max(_confidence(x[0]) for x in items)
        return (sum_score, len(items), max_conf)

    best_norm = sorted(eligible.keys(), key=answer_key, reverse=True)[0]
    reps = [x[0] for x in eligible[best_norm]]
    selected = sorted(
        reps,
        key=lambda c: (_confidence(c), 1 if c.get("stage") == "independent" else 0, -candidates.index(c)),
        reverse=True,
    )[0]
    return selected, {
        "mode": mode,
        "selector_type": "cross_stage_support",
        "selected_candidate_id": selected.get("candidate_id"),
        "selected_answer_norm": best_norm,
        "used_cross_stage_answer": True,
        "eligible_answers": [
            {
                "normalized_answer": norm,
                "sum_score": float(sum(x[2] for x in items)),
                "candidate_ids": [x[0].get("candidate_id") for x in items],
            }
            for norm, items in sorted(eligible.items(), key=lambda kv: answer_key(kv[0]), reverse=True)
        ],
    }


def choose_stage_gate(
    candidates: List[Dict[str, Any]],
    memory: ReliabilityMemory,
    pair_id: str,
    topic: str,
    mode: str = "pair_topic_stage",
    margin: float = 0.03,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Agent-wise stage gate.

    For each agent, consider its post-debate candidate only if frozen memory
    says the agent's post-debate reliability exceeds its independent reliability
    by at least `margin`, or if the answer did not change. Then select between
    the remaining one candidate per agent with reliability.

    This is stricter than the earlier persuasion_guard, which only intervened
    after a post-debate candidate had already won the global candidate race.
    """
    eligible: List[Dict[str, Any]] = []
    decisions = []
    for prefix in ["A", "B"]:
        c0 = _cand_by_id(candidates, f"{prefix}0")
        c1 = _cand_by_id(candidates, f"{prefix}1")
        info0 = _score_candidate(c0, memory, pair_id, topic, mode)
        info1 = _score_candidate(c1, memory, pair_id, topic, mode)
        s0 = float(info0["score"])
        s1 = float(info1["score"])
        unchanged = _norm(c0) == _norm(c1)
        if unchanged or s1 >= s0 + margin:
            chosen = c1
            reason = "unchanged" if unchanged else "post_debate_exceeds_independent_margin"
        else:
            chosen = c0
            reason = "independent_safer_or_post_debate_margin_not_met"
        eligible.append(chosen)
        decisions.append({
            "agent_prefix": prefix,
            "chosen_candidate_id": chosen.get("candidate_id"),
            "independent_score": s0,
            "post_debate_score": s1,
            "margin": margin,
            "unchanged": unchanged,
            "reason": reason,
        })

    selected, info = choose_reliability(eligible, memory, pair_id, topic, mode)
    info = dict(info)
    info.update({
        "selector_type": "stage_gate",
        "selected_candidate_id": selected.get("candidate_id"),
        "margin": margin,
        "stage_gate_decisions": decisions,
    })
    return selected, info


def apply_all_methods(record: Dict[str, Any], memory: Optional[ReliabilityMemory] = None, adaptive_selector: Optional[AdaptiveCandidateSelector] = None) -> Dict[str, Any]:
    candidates = record["candidates"]
    pair_id = record["pair_id"]
    topic = record["topic"]
    methods: Dict[str, Dict[str, Any]] = {}

    simple = {
        "single_A": choose_single_a,
        "single_B": choose_single_b,
        "stateless_debate": choose_stateless_debate,
        "4cand_majority": choose_4cand_majority,
        "4cand_confidence": choose_4cand_confidence,
        "oracle_candidate": choose_oracle,
    }
    for name, chooser in simple.items():
        c = chooser(candidates)
        methods[name] = _method_result(name, c)

    if memory is not None:
        for name, mode in [
            ("overall_reliability", "overall"),
            ("agent_topic_reliability", "agent_topic"),
            ("pac_math_pair_topic_stage", "pair_topic_stage"),
        ]:
            c, info = choose_reliability(candidates, memory, pair_id, topic, mode)
            res = _method_result(name, c)
            res["selection_info"] = info
            methods[name] = res

        c, info = choose_reliability_support_sum(candidates, memory, pair_id, topic, "pair_topic_stage")
        res = _method_result("pac_math_pair_topic_stage_support_sum", c)
        res["selection_info"] = info
        methods["pac_math_pair_topic_stage_support_sum"] = res

        c, info = choose_persuasion_guard(candidates, memory, pair_id, topic, "pair_topic_stage")
        res = _method_result("pac_math_pair_topic_stage_guard", c)
        res["selection_info"] = info
        methods["pac_math_pair_topic_stage_guard"] = res

        c, info = choose_independent_reliability(candidates, memory, pair_id, topic, "pair_topic_stage")
        res = _method_result("pac_math_independent_only", c)
        res["selection_info"] = info
        methods["pac_math_independent_only"] = res

        for method_name, selector_type in [
            ("pac_math_safety_first", "safety_first"),
            ("pac_math_utility", "utility"),
            ("pac_math_accuracy_first", "accuracy_first"),
        ]:
            c, info = choose_candidate_safety(candidates, memory, pair_id, topic, selector_type)
            res = _method_result(method_name, c)
            res["selection_info"] = info
            methods[method_name] = res

        for method_name, chooser in [
            ("pac_math_anchor_gate", choose_anchor_gate),
            ("pac_math_cross_agent_anchor_gate", choose_cross_agent_anchor_gate),
            ("pac_math_cross_stage_support", choose_cross_stage_support),
            ("pac_math_stage_gate", choose_stage_gate),
        ]:
            c, info = chooser(candidates, memory, pair_id, topic, "pair_topic_stage")
            res = _method_result(method_name, c)
            res["selection_info"] = info
            methods[method_name] = res

        if adaptive_selector is not None:
            for method_name, variant in [
                ("pac_math_adaptive_learned", "adaptive_learned"),
                ("pac_math_adaptive_accuracy", "adaptive_accuracy"),
                ("pac_math_adaptive_safety", "adaptive_safety"),
            ]:
                c, info = adaptive_selector.select(candidates, memory, pair_id, topic, variant=variant)
                res = _method_result(method_name, c)
                res["selection_info"] = info
                methods[method_name] = res

            for method_name, variant in [
                ("pac_math_router_balanced", "router_balanced"),
                ("pac_math_router_accuracy", "router_accuracy"),
                ("pac_math_router_safety", "router_safety"),
            ]:
                c, info = adaptive_selector.select_policy_router(candidates, memory, pair_id, topic, variant=variant)
                res = _method_result(method_name, c)
                res["selection_info"] = info
                methods[method_name] = res

    return methods


def _method_result(name: str, cand: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "method": name,
        "selected_candidate_id": cand.get("candidate_id"),
        "selected_answer": cand.get("answer"),
        "selected_normalized_answer": cand.get("normalized_answer"),
        "selected_confidence": cand.get("confidence"),
        "is_correct": bool(cand.get("is_correct", False)),
        "selected_agent_id": cand.get("agent_id"),
        "selected_model": cand.get("model"),
        "selected_stage": cand.get("stage"),
    }


def flip_annotations(candidates: List[Dict[str, Any]], final_correct: bool) -> Dict[str, Any]:
    a0 = _cand_by_id(candidates, "A0")
    b0 = _cand_by_id(candidates, "B0")
    a1 = _cand_by_id(candidates, "A1")
    b1 = _cand_by_id(candidates, "B1")
    initial_any_correct = bool(a0.get("is_correct")) or bool(b0.get("is_correct"))
    initial_both_wrong = (not bool(a0.get("is_correct"))) and (not bool(b0.get("is_correct")))
    post_any_correct = bool(a1.get("is_correct")) or bool(b1.get("is_correct"))
    return {
        "initial_any_correct": initial_any_correct,
        "initial_both_wrong": initial_both_wrong,
        "post_any_correct": post_any_correct,
        "correct_to_wrong": bool(initial_any_correct and not final_correct),
        "wrong_to_correct": bool(initial_both_wrong and final_correct),
        "preserved_correct": bool(initial_any_correct and final_correct),
        "candidate_oracle_possible": any(bool(c.get("is_correct")) for c in candidates),
    }


def _verifier_review_map(record: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    ver = record.get("verifier") or {}
    out: Dict[str, Dict[str, float]] = {}
    for r in ver.get("reviews", []) or []:
        if not isinstance(r, dict):
            continue
        cid = str(r.get("candidate_id", ""))
        try:
            validity = float(r.get("validity_score", 50)) / 100.0
        except Exception:
            validity = 0.5
        try:
            risk = float(r.get("error_risk", 50)) / 100.0
        except Exception:
            risk = 0.5
        out[cid] = {
            "validity": max(0.0, min(1.0, validity)),
            "risk": max(0.0, min(1.0, risk)),
        }
    return out


def choose_verifier_router(
    record: Dict[str, Any],
    memory: ReliabilityMemory,
    selector_type: str = "balanced",
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Verifier-enhanced answer-level PAC selector.

    This uses an extra no-gold LLM verification pass. The verifier sees only the
    problem and A0/B0/A1/B1, never the gold answer. Selection is answer-level:
    candidates with the same normalized answer share support.
    """
    candidates = record.get("candidates", [])
    pair_id = str(record.get("pair_id", ""))
    topic = str(record.get("topic", "unknown"))
    reviews = _verifier_review_map(record)
    if not candidates or not reviews:
        # If verifier is unavailable, fall back to the best non-oracle dynamic selector.
        return choose_reliability_support_sum(candidates, memory, pair_id, topic, "pair_topic_stage")

    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for c in candidates:
        n = _norm(c)
        if n:
            groups[n].append(c)
    if not groups:
        return candidates[0], {"selector_type": f"verifier_{selector_type}", "fallback": "empty_norms"}

    answer_rows = []
    for norm, cands in groups.items():
        support = len(cands) / 4.0
        indep_support = sum(1 for c in cands if c.get("stage") == "independent") / 2.0
        post_support = sum(1 for c in cands if c.get("stage") == "post_debate") / 2.0
        cross_stage = 1.0 if indep_support > 0 and post_support > 0 else 0.0
        max_valid = 0.0
        mean_valid = 0.0
        min_risk = 1.0
        mean_risk = 0.5
        rel_scores = []
        confs = []
        changed_unanchored = 0
        for c in cands:
            cid = str(c.get("candidate_id", ""))
            rev = reviews.get(cid, {"validity": 0.5, "risk": 0.5})
            max_valid = max(max_valid, rev["validity"])
            min_risk = min(min_risk, rev["risk"])
            confs.append(_confidence(c) / 100.0)
            info = _score_candidate(c, memory, pair_id, topic, "pair_topic_stage")
            rel_scores.append(float(info.get("score", 0.5)))
            if str(c.get("stage")) == "post_debate":
                own0 = _cand_by_id(candidates, "A0") if cid.startswith("A") else _cand_by_id(candidates, "B0")
                a0 = _cand_by_id(candidates, "A0")
                b0 = _cand_by_id(candidates, "B0")
                anchored = norm in {_norm(a0), _norm(b0)}
                if _norm(own0) != norm and not anchored:
                    changed_unanchored += 1
        vals = [reviews.get(str(c.get("candidate_id", "")), {"validity": 0.5})["validity"] for c in cands]
        risks = [reviews.get(str(c.get("candidate_id", "")), {"risk": 0.5})["risk"] for c in cands]
        mean_valid = sum(vals) / max(1, len(vals))
        mean_risk = sum(risks) / max(1, len(risks))
        rel = max(rel_scores) if rel_scores else 0.5
        conf = max(confs) if confs else 0.0

        if selector_type == "best":
            score = 0.70 * max_valid + 0.15 * rel + 0.10 * cross_stage + 0.05 * support - 0.15 * min_risk
        elif selector_type == "balanced":
            score = 0.50 * max_valid + 0.20 * rel + 0.12 * cross_stage + 0.08 * support + 0.05 * indep_support + 0.05 * conf - 0.25 * mean_risk - 0.10 * changed_unanchored
        elif selector_type == "safety":
            # Safety prioritizes low verifier risk and independent/cross-stage support.
            score = 0.35 * max_valid + 0.20 * (1.0 - mean_risk) + 0.15 * rel + 0.15 * cross_stage + 0.10 * indep_support + 0.05 * support - 0.20 * changed_unanchored
        else:
            raise ValueError(f"Unknown verifier selector: {selector_type}")
        answer_rows.append({
            "norm": norm,
            "score": float(score),
            "max_validity": float(max_valid),
            "mean_validity": float(mean_valid),
            "min_risk": float(min_risk),
            "mean_risk": float(mean_risk),
            "reliability": float(rel),
            "support": float(support),
            "cross_stage": float(cross_stage),
            "independent_support": float(indep_support),
            "changed_unanchored": int(changed_unanchored),
            "candidate_ids": [c.get("candidate_id") for c in cands],
            "candidates": cands,
        })

    # In strict safety mode, first prefer candidates under a verifier risk cap.
    if selector_type == "safety":
        safe_rows = [r for r in answer_rows if r["mean_risk"] <= 0.45 or r["max_validity"] >= 0.72]
        if safe_rows:
            answer_rows = safe_rows

    answer_rows = sorted(answer_rows, key=lambda r: (r["score"], r["max_validity"], r["cross_stage"], r["support"]), reverse=True)
    best = answer_rows[0]
    reps = best["candidates"]
    selected = sorted(reps, key=lambda c: (_confidence(c), 1 if c.get("stage") == "independent" else 0), reverse=True)[0]
    info = {
        "selector_type": f"verifier_{selector_type}",
        "selected_candidate_id": selected.get("candidate_id"),
        "selected_answer_norm": best["norm"],
        "answer_scores": [
            {k: v for k, v in row.items() if k != "candidates"}
            for row in answer_rows
        ],
        "verifier_parse_ok": bool((record.get("verifier") or {}).get("parse_ok", False)),
        "verifier_model": (record.get("verifier") or {}).get("verifier_model", ""),
    }
    return selected, info


# Patch apply_all_methods by wrapping the original function body is difficult after
# definition, so we replace it below with a version that includes verifier methods.
_old_apply_all_methods = apply_all_methods


def apply_all_methods(record: Dict[str, Any], memory: Optional[ReliabilityMemory] = None, adaptive_selector: Optional[AdaptiveCandidateSelector] = None) -> Dict[str, Any]:
    methods = _old_apply_all_methods(record, memory=memory, adaptive_selector=adaptive_selector)
    if memory is not None and record.get("verifier"):
        for method_name, variant in [
            ("pac_math_verifier_best", "best"),
            ("pac_math_verifier_balanced", "balanced"),
            ("pac_math_verifier_safety", "safety"),
        ]:
            c, info = choose_verifier_router(record, memory, selector_type=variant)
            res = _method_result(method_name, c)
            res["selection_info"] = info
            methods[method_name] = res
    return methods
