from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .eval_math import answer_record, is_correct
from .answer_norm import canonicalize, equivalent
from .io_utils import get_problem, get_topic
from .ollama_client import OllamaClient, safe_json_loads
from .prompts import SYSTEM_DEBATE, SYSTEM_SOLVER, SYSTEM_VERIFIER, critique_prompt, independent_prompt, revision_prompt, commitment_check_prompt, verification_prompt


@dataclass
class AgentSpec:
    agent_id: str
    model: str


def _protocol_metadata() -> Dict[str, Any]:
    """Return cache-version metadata for generated candidate records.

    This must change whenever the prompts, debate protocol, commitment rules,
    generation settings, or answer-finalization logic change. It prevents old
    cached A0/B0/A1/B1 records from being reused after a methodology patch.
    """
    try:
        import config as _cfg
        version = str(getattr(_cfg, "PROTOCOL_VERSION", "unknown_protocol"))
        return {
            "protocol_version": version,
            "evidence_gated_debate": bool(getattr(_cfg, "EVIDENCE_GATED_DEBATE", True)),
            "min_change_justification_score": float(getattr(_cfg, "MIN_CHANGE_JUSTIFICATION_SCORE", 70.0)),
            "min_revised_validity_score": float(getattr(_cfg, "MIN_REVISED_VALIDITY_SCORE", 60.0)),
            "max_initial_validity_for_change": float(getattr(_cfg, "MAX_INITIAL_VALIDITY_FOR_CHANGE", 65.0)),
            "generation_options": dict(getattr(_cfg, "GENERATION_OPTIONS", {})),
        }
    except Exception:
        return {"protocol_version": "unknown_protocol"}


def _as_conf(value: Any, default: float = 50.0) -> float:
    try:
        conf = float(value)
    except Exception:
        conf = default
    return max(0.0, min(100.0, conf))


def _bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1", "changed"}
    return bool(value)



def _same_answer_text(a: Any, b: Any) -> bool:
    """Compare answer strings using the same canonicalizer used for scoring."""
    try:
        na = canonicalize(a).normalized
        nb = canonicalize(b).normalized
        return bool(na and nb and na == nb)
    except Exception:
        return str(a).strip() == str(b).strip()


def _answer_changed(initial_answer: Any, final_answer: Any) -> bool:
    return not _same_answer_text(initial_answer, final_answer)

def _commitment_allowed(parsed: Dict[str, Any], cfg: Any = None) -> bool:
    """Programmatic backstop for the evidence-gated debate protocol.

    The LLM makes a dynamic judgment, but this hard guard prevents a changed
    answer from being accepted when the model admits the change is weak or
    based only on partner assertion. Thresholds are defined in config.py.
    """
    if not _bool_value(parsed.get("changed_answer", False)):
        return True
    evidence_type = str(parsed.get("error_evidence_type", "none")).strip().lower()
    weak_types = {"none", "partner_assertion_only", "assertion_only", "unknown", ""}
    try:
        import config as _cfg
        min_change = float(getattr(_cfg, "MIN_CHANGE_JUSTIFICATION_SCORE", 70.0))
        min_revised = float(getattr(_cfg, "MIN_REVISED_VALIDITY_SCORE", 60.0))
        max_initial = float(getattr(_cfg, "MAX_INITIAL_VALIDITY_FOR_CHANGE", 65.0))
    except Exception:
        min_change, min_revised, max_initial = 70.0, 60.0, 65.0
    change_score = _as_conf(parsed.get("change_justification_score", 0))
    revised_score = _as_conf(parsed.get("revised_answer_validity_score", 0))
    initial_score = _as_conf(parsed.get("initial_answer_validity_score", 100))
    if evidence_type in weak_types:
        return False
    if change_score < min_change:
        return False
    if revised_score < min_revised:
        return False
    if initial_score > max_initial and revised_score <= initial_score:
        return False
    return True


def _request_json(
    client: OllamaClient,
    model: str,
    prompt: str,
    system: str,
    options: Dict[str, Any],
    max_retries: int,
) -> Dict[str, Any]:
    last_text = ""
    token_info = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "duration_ns": 0}
    for attempt in range(max_retries + 1):
        use_prompt = prompt
        if attempt > 0:
            use_prompt = prompt + "\n\nYour previous response was not valid JSON. Return valid JSON only."
        resp = client.generate(model=model, prompt=use_prompt, system=system, options=options, json_mode=True)
        last_text = resp.text
        token_info = {
            "prompt_tokens": resp.prompt_eval_count,
            "completion_tokens": resp.eval_count,
            "total_tokens": resp.total_tokens,
            "duration_ns": resp.total_duration,
        }
        parsed = safe_json_loads(resp.text)
        if parsed is not None:
            parsed["_parse_ok"] = True
            parsed["_raw_response"] = last_text
            parsed["_tokens"] = token_info
            parsed["_attempts"] = attempt + 1
            return parsed
    return {
        "_parse_ok": False,
        "_raw_response": last_text,
        "_tokens": token_info,
        "_attempts": max_retries + 1,
    }


def solve_independent(
    client: OllamaClient,
    agent: AgentSpec,
    problem: str,
    options: Dict[str, Any],
    max_retries: int,
) -> Dict[str, Any]:
    parsed = _request_json(
        client=client,
        model=agent.model,
        prompt=independent_prompt(problem),
        system=SYSTEM_SOLVER,
        options=options,
        max_retries=max_retries,
    )
    answer = str(parsed.get("answer", "")) if parsed.get("_parse_ok") else "PARSE_ERROR"
    return {
        "agent_id": agent.agent_id,
        "model": agent.model,
        "stage": "independent",
        "answer": answer,
        "confidence": _as_conf(parsed.get("confidence", 0 if not parsed.get("_parse_ok") else 50)),
        "reasoning_summary": str(parsed.get("reasoning_summary", "")),
        "weak_point": str(parsed.get("weak_point", "")),
        "parse_ok": bool(parsed.get("_parse_ok", False)),
        "raw_response": parsed.get("_raw_response", ""),
        "tokens": parsed.get("_tokens", {}),
        "attempts": parsed.get("_attempts", 0),
    }


def run_debate(
    client: OllamaClient,
    agent_a: AgentSpec,
    agent_b: AgentSpec,
    problem: str,
    a0: Dict[str, Any],
    b0: Dict[str, Any],
    options: Dict[str, Any],
    max_retries: int,
) -> Dict[str, Any]:
    # Round 1: simultaneous critique.
    acrit = _request_json(
        client=client,
        model=agent_a.model,
        prompt=critique_prompt(problem, a0["answer"], a0.get("reasoning_summary", ""), b0["answer"], b0.get("reasoning_summary", "")),
        system=SYSTEM_DEBATE,
        options=options,
        max_retries=max_retries,
    )
    bcrit = _request_json(
        client=client,
        model=agent_b.model,
        prompt=critique_prompt(problem, b0["answer"], b0.get("reasoning_summary", ""), a0["answer"], a0.get("reasoning_summary", "")),
        system=SYSTEM_DEBATE,
        options=options,
        max_retries=max_retries,
    )

    acrit_text = json.dumps({k: v for k, v in acrit.items() if not k.startswith("_")}, ensure_ascii=False)
    bcrit_text = json.dumps({k: v for k, v in bcrit.items() if not k.startswith("_")}, ensure_ascii=False)

    # Round 2: each agent revises after seeing the partner's critique.
    arev = _request_json(
        client=client,
        model=agent_a.model,
        prompt=revision_prompt(problem, a0["answer"], a0.get("reasoning_summary", ""), b0["answer"], bcrit_text),
        system=SYSTEM_DEBATE,
        options=options,
        max_retries=max_retries,
    )
    brev = _request_json(
        client=client,
        model=agent_b.model,
        prompt=revision_prompt(problem, b0["answer"], b0.get("reasoning_summary", ""), a0["answer"], acrit_text),
        system=SYSTEM_DEBATE,
        options=options,
        max_retries=max_retries,
    )

    # Optional commitment check. This is the core evidence-gated debate protocol.
    # It makes post-debate revision dynamic but resistant to unsupported persuasion.
    try:
        import config as _cfg
        evidence_gate = bool(getattr(_cfg, "EVIDENCE_GATED_DEBATE", True))
    except Exception:
        evidence_gate = True

    a_revised_raw = str(arev.get("revised_answer", a0["answer"])) if arev.get("_parse_ok") else a0["answer"]
    b_revised_raw = str(brev.get("revised_answer", b0["answer"])) if brev.get("_parse_ok") else b0["answer"]

    a_commit = {}
    b_commit = {}
    if evidence_gate:
        a_commit = _request_json(
            client=client,
            model=agent_a.model,
            prompt=commitment_check_prompt(problem, a0["answer"], a_revised_raw, json.dumps({k: v for k, v in arev.items() if not k.startswith("_")}, ensure_ascii=False)),
            system=SYSTEM_DEBATE,
            options=options,
            max_retries=max_retries,
        )
        b_commit = _request_json(
            client=client,
            model=agent_b.model,
            prompt=commitment_check_prompt(problem, b0["answer"], b_revised_raw, json.dumps({k: v for k, v in brev.items() if not k.startswith("_")}, ensure_ascii=False)),
            system=SYSTEM_DEBATE,
            options=options,
            max_retries=max_retries,
        )

    def _finalize_post_debate(agent, initial, revision, revised_raw, commit):
        revision_parse_ok = bool(revision.get("_parse_ok", False))
        initial_answer = str(initial.get("answer", ""))
        proposed_answer = str(revised_raw if revised_raw is not None else initial_answer)
        final_conf = _as_conf(revision.get("revised_confidence", initial.get("confidence", 50)))

        # Standard-debate mode: accept the parsed revised answer directly.
        # This restores the non-evidence-gated protocol used in the stronger Phi-4 pilot.
        # v15 had a hidden bug: even when EVIDENCE_GATED_DEBATE=False, it still called
        # _commitment_allowed(), which rejected normal revisions because they did not
        # contain evidence-gate-only fields such as error_evidence_type. That made
        # standard debate silently behave like a conservative evidence gate.
        if not evidence_gate:
            final_answer = proposed_answer if revision_parse_ok else initial_answer
            accept_revision = bool(revision_parse_ok and _answer_changed(initial_answer, final_answer))
            return {
                "agent_id": agent.agent_id,
                "model": agent.model,
                "stage": "post_debate",
                "answer": final_answer,
                "confidence": final_conf,
                "changed_answer": _answer_changed(initial_answer, final_answer),
                "revision_reason": str(revision.get("revision_reason", "")),
                "error_evidence_type": str(revision.get("error_evidence_type", "standard_debate_no_gate")),
                "change_justification_score": _as_conf(revision.get("change_justification_score", 0)),
                "initial_answer_validity_score": _as_conf(revision.get("initial_answer_validity_score", 0)),
                "revised_answer_validity_score": _as_conf(revision.get("revised_answer_validity_score", 0)),
                "accept_revision": bool(accept_revision),
                "parse_ok": revision_parse_ok,
                "raw_response": revision.get("_raw_response", ""),
                "tokens": revision.get("_tokens", {}),
                "attempts": revision.get("_attempts", 0),
                "evidence_gate_note": "standard_debate_revision_accepted_without_commitment_gate",
            }

        allowed_by_revision = _commitment_allowed(revision)

        # The commitment check is allowed to choose only between the initial
        # answer and the already-proposed revised answer. Earlier versions used
        # commit["final_answer"] directly when accept_revision=True. That gave
        # the model a hidden third chance to invent a new answer during the safety
        # check, which is not part of the A0/B0/A1/B1 protocol and can corrupt
        # both candidate preservation and C2W accounting.
        final_answer = initial_answer
        accept_revision = False

        if evidence_gate and commit.get("_parse_ok"):
            commit_accepts = _bool_value(commit.get("accept_revision", False))
            commit_final = str(commit.get("final_answer", proposed_answer if commit_accepts else initial_answer))
            final_conf = _as_conf(commit.get("final_confidence", revision.get("revised_confidence", initial.get("confidence", 50))))

            if (
                revision_parse_ok
                and allowed_by_revision
                and commit_accepts
                and _answer_changed(initial_answer, proposed_answer)
                and _same_answer_text(commit_final, proposed_answer)
            ):
                final_answer = proposed_answer
                accept_revision = True
            else:
                final_answer = initial_answer
                accept_revision = False

            return {
                "agent_id": agent.agent_id,
                "model": agent.model,
                "stage": "post_debate",
                "answer": final_answer,
                "confidence": final_conf,
                "changed_answer": _answer_changed(initial_answer, final_answer),
                "revision_reason": str(revision.get("revision_reason", "")),
                "error_evidence_type": str(revision.get("error_evidence_type", "")),
                "change_justification_score": _as_conf(revision.get("change_justification_score", 0)),
                "initial_answer_validity_score": _as_conf(revision.get("initial_answer_validity_score", 0)),
                "revised_answer_validity_score": _as_conf(revision.get("revised_answer_validity_score", 0)),
                "commitment_check": {k: v for k, v in commit.items() if k != "_raw_response"},
                "accept_revision": bool(accept_revision),
                "parse_ok": revision_parse_ok and bool(commit.get("_parse_ok", False)),
                "raw_response": revision.get("_raw_response", ""),
                "tokens": revision.get("_tokens", {}),
                "commitment_tokens": commit.get("_tokens", {}),
                "attempts": int(revision.get("_attempts", 0) or 0) + int(commit.get("_attempts", 0) or 0),
                "evidence_gate_note": "commitment_final_constrained_to_initial_or_proposed_revision",
            }

        # No commitment check or failed commitment parse: do NOT accept a new
        # answer unless the revision itself parsed and passed the hard evidence
        # backstop. This is safer than accepting proposed revisions on commit
        # failure.
        if revision_parse_ok and allowed_by_revision and _answer_changed(initial_answer, proposed_answer):
            final_answer = proposed_answer
            accept_revision = True
        else:
            final_answer = initial_answer
            accept_revision = False

        return {
            "agent_id": agent.agent_id,
            "model": agent.model,
            "stage": "post_debate",
            "answer": final_answer,
            "confidence": _as_conf(revision.get("revised_confidence", initial.get("confidence", 50))),
            "changed_answer": _answer_changed(initial_answer, final_answer),
            "revision_reason": str(revision.get("revision_reason", "")),
            "error_evidence_type": str(revision.get("error_evidence_type", "")),
            "change_justification_score": _as_conf(revision.get("change_justification_score", 0)),
            "initial_answer_validity_score": _as_conf(revision.get("initial_answer_validity_score", 0)),
            "revised_answer_validity_score": _as_conf(revision.get("revised_answer_validity_score", 0)),
            "accept_revision": bool(accept_revision),
            "parse_ok": revision_parse_ok,
            "raw_response": revision.get("_raw_response", ""),
            "tokens": revision.get("_tokens", {}),
            "attempts": revision.get("_attempts", 0),
            "evidence_gate_note": "commitment_missing_or_disabled_revision_hard_guard_only",
        }

    a1 = _finalize_post_debate(agent_a, a0, arev, a_revised_raw, a_commit)
    b1 = _finalize_post_debate(agent_b, b0, brev, b_revised_raw, b_commit)

    return {
        "protocol": "evidence_gated_commitment" if evidence_gate else "standard_debate_v16",
        "a_critique": {k: v for k, v in acrit.items() if k != "_raw_response"},
        "b_critique": {k: v for k, v in bcrit.items() if k != "_raw_response"},
        "a_revision_raw": {k: v for k, v in arev.items() if k != "_raw_response"},
        "b_revision_raw": {k: v for k, v in brev.items() if k != "_raw_response"},
        "a_commitment_check": {k: v for k, v in a_commit.items() if k != "_raw_response"},
        "b_commitment_check": {k: v for k, v in b_commit.items() if k != "_raw_response"},
        "a1": a1,
        "b1": b1,
    }



def verify_candidate_answers(
    client: OllamaClient,
    verifier_model: str,
    problem: str,
    candidates: List[Dict[str, Any]],
    options: Dict[str, Any],
    max_retries: int,
) -> Dict[str, Any]:
    """Run a no-gold verifier over A0/B0/A1/B1.

    This is intentionally separate from answer correctness evaluation. It uses
    only the problem text and the candidate answers, never the gold answer.
    """
    lines = []
    for c in candidates:
        cid = str(c.get("candidate_id", ""))
        ans = str(c.get("answer", ""))
        norm = str(c.get("normalized_answer", ""))
        conf = c.get("confidence", "")
        stage = c.get("stage", "")
        lines.append(f"{cid} | stage={stage} | confidence={conf} | answer={ans} | normalized={norm}")
    prompt = verification_prompt(problem, "\n".join(lines))
    parsed = _request_json(
        client=client,
        model=verifier_model,
        prompt=prompt,
        system=SYSTEM_VERIFIER,
        options=options,
        max_retries=max_retries,
    )
    reviews_raw = parsed.get("reviews", []) if parsed.get("_parse_ok") else []
    reviews = []
    valid_ids = {str(c.get("candidate_id")) for c in candidates}
    if isinstance(reviews_raw, list):
        for r in reviews_raw:
            if not isinstance(r, dict):
                continue
            cid = str(r.get("candidate_id", ""))
            if cid not in valid_ids:
                continue
            reviews.append({
                "candidate_id": cid,
                "validity_score": _as_conf(r.get("validity_score", 50)),
                "error_risk": _as_conf(r.get("error_risk", 50)),
                "brief_reason": str(r.get("brief_reason", ""))[:500],
            })
    return {
        "verifier_model": verifier_model,
        "parse_ok": bool(parsed.get("_parse_ok", False)),
        "reviews": reviews,
        "best_candidate_id": str(parsed.get("best_candidate_id", "")) if parsed.get("_parse_ok") else "",
        "best_answer": str(parsed.get("best_answer", "")) if parsed.get("_parse_ok") else "",
        "tokens": parsed.get("_tokens", {}),
        "attempts": parsed.get("_attempts", 0),
        "raw_response": parsed.get("_raw_response", ""),
    }

def run_problem(
    client: OllamaClient,
    row: Dict[str, Any],
    pair_id: str,
    agent_a_model: str,
    agent_b_model: str,
    problem_cols: List[str],
    topic_cols: List[str],
    answer_cols: List[str],
    options: Dict[str, Any],
    max_retries: int,
    problem_index: int,
) -> Dict[str, Any]:
    problem = get_problem(row, problem_cols)
    topic = get_topic(row, topic_cols)
    gold_text = ""
    from .eval_math import get_gold_text
    gold_text = get_gold_text(row, answer_cols)

    agent_a = AgentSpec(agent_id=f"{pair_id}__A", model=agent_a_model)
    agent_b = AgentSpec(agent_id=f"{pair_id}__B", model=agent_b_model)

    start = time.time()
    a0 = solve_independent(client, agent_a, problem, options, max_retries)
    b0 = solve_independent(client, agent_b, problem, options, max_retries)
    debate = run_debate(client, agent_a, agent_b, problem, a0, b0, options, max_retries)
    a1 = debate["a1"]
    b1 = debate["b1"]

    candidates = [
        {"candidate_id": "A0", **a0},
        {"candidate_id": "B0", **b0},
        {"candidate_id": "A1", **a1},
        {"candidate_id": "B1", **b1},
    ]
    for cand in candidates:
        rec = answer_record(cand.get("answer", ""), gold_text)
        cand.update(rec)

    total_tokens = 0
    total_duration_ns = 0
    for obj in [a0, b0, a1, b1]:
        tok = obj.get("tokens", {}) or {}
        total_tokens += int(tok.get("total_tokens", 0) or 0)
        total_duration_ns += int(tok.get("duration_ns", 0) or 0)
        ctok = obj.get("commitment_tokens", {}) or {}
        total_tokens += int(ctok.get("total_tokens", 0) or 0)
        total_duration_ns += int(ctok.get("duration_ns", 0) or 0)
    for crit in [debate.get("a_critique", {}), debate.get("b_critique", {})]:
        tok = crit.get("_tokens", {}) or crit.get("tokens", {}) or {}
        total_tokens += int(tok.get("total_tokens", 0) or 0)
        total_duration_ns += int(tok.get("duration_ns", 0) or 0)

    protocol_meta = _protocol_metadata()

    return {
        "run_id": f"{pair_id}__{problem_index}",
        "protocol_version": protocol_meta.get("protocol_version", "unknown_protocol"),
        "protocol_metadata": protocol_meta,
        "problem_index": problem_index,
        "pair_id": pair_id,
        "agent_a_model": agent_a_model,
        "agent_b_model": agent_b_model,
        "topic": topic,
        "problem_hash": row.get("problem_hash", ""),
        "problem": problem,
        "gold_text": gold_text,
        "candidates": candidates,
        "debate": debate,
        "total_tokens": total_tokens,
        "total_duration_ns": total_duration_ns,
        "wall_time_seconds": time.time() - start,
    }
