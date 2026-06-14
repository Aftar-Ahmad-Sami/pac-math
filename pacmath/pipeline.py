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
            "nvidia_enable_thinking": bool(getattr(_cfg, "NVIDIA_ENABLE_THINKING", False)),
            "nvidia_reasoning_budget": int(getattr(_cfg, "NVIDIA_REASONING_BUDGET", 0)),
            "nvidia_rate_limit_rpm": float(getattr(_cfg, "NVIDIA_RATE_LIMIT_RPM", 0.0)),
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
    """Request one JSON object with robust fallbacks.

    v19 prioritizes /api/chat with top-level think=False for Qwen-style reasoning models.
    The debug log showed qwen3.6:27b returns empty message.content and long
    message.thinking unless think=False is set. With think=False, /api/chat returns
    valid JSON in message.content. /api/generate is retained only as fallback.

    The returned object always carries a compact request trace so blank-output
    failures are inspectable instead of becoming opaque PARSE_ERROR records.
    """
    try:
        import config as _cfg
        disable_thinking = bool(getattr(_cfg, "OLLAMA_DISABLE_THINKING", True))
        use_chat_fallback = bool(getattr(_cfg, "OLLAMA_CHAT_FALLBACK", True))
        prefer_chat = bool(getattr(_cfg, "OLLAMA_PREFER_CHAT", True))
        length_retry_enabled = bool(getattr(_cfg, "LENGTH_RETRY_ENABLED", True))
        length_retry_num_predict = int(getattr(_cfg, "LENGTH_RETRY_NUM_PREDICT", 2200))
    except Exception:
        disable_thinking = True
        use_chat_fallback = True
        prefer_chat = True
        length_retry_enabled = True
        length_retry_num_predict = 2200

    think_value = False if disable_thinking else None
    trace: List[Dict[str, Any]] = []
    last_text = ""
    last_length_limited = False
    token_info = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "duration_ns": 0}

    def _record_trace(kind: str, json_mode: bool, resp: Any = None, error: str = "") -> None:
        text = getattr(resp, "text", "") if resp is not None else ""
        raw = getattr(resp, "raw", None) if resp is not None else None
        item: Dict[str, Any] = {
            "kind": kind,
            "json_mode": bool(json_mode),
            "text_len": len(text or ""),
            "text_head": (text or "")[:500],
            "error": error,
        }
        if isinstance(raw, dict):
            item["raw_keys"] = sorted(list(raw.keys()))
            for key in ["done", "done_reason", "model", "created_at", "prompt_eval_count", "eval_count"]:
                if key in raw:
                    item[key] = raw.get(key)
            # Some future Ollama/build variants may expose content/thinking in different fields.
            if isinstance(raw.get("message"), dict):
                item["message_keys"] = sorted(list(raw["message"].keys()))
                msg_content = raw["message"].get("content", "") or ""
                item["message_content_len"] = len(msg_content)
                item["message_content_head"] = msg_content[:500]
                if "reasoning_content_len" in raw["message"]:
                    item["reasoning_content_len"] = raw["message"].get("reasoning_content_len")
                    item["reasoning_content_head"] = raw["message"].get("reasoning_content_head", "")
            if "response" in raw:
                item["response_len"] = len(raw.get("response") or "")
            if "thinking" in raw:
                think_text = raw.get("thinking") or ""
                item["thinking_len"] = len(think_text)
                item["thinking_head"] = str(think_text)[:500]
        trace.append(item)

    def _with_length_retry_options(base_options: Dict[str, Any]) -> Dict[str, Any]:
        retry_options = dict(base_options or {})
        current_np = int(retry_options.get("num_predict", retry_options.get("max_tokens", 0)) or 0)
        retry_cap = max(length_retry_num_predict, current_np)
        retry_options["num_predict"] = retry_cap
        retry_options["max_tokens"] = retry_cap
        return retry_options

    def _is_length_limited_response(resp: Any, call_options: Dict[str, Any]) -> bool:
        raw = getattr(resp, "raw", None) if resp is not None else None
        if not isinstance(raw, dict):
            return False
        reason = str(raw.get("done_reason", "") or raw.get("finish_reason", "")).strip().lower()
        if reason in {"length", "max_tokens", "max_completion_tokens", "context_length"}:
            return True
        try:
            requested = int(call_options.get("max_tokens", call_options.get("num_predict", 0)) or 0)
            eval_count = int(raw.get("eval_count", 0) or 0)
            return requested > 0 and eval_count >= requested
        except Exception:
            return False

    def _try_parse(
        kind: str,
        use_prompt: str,
        json_mode: bool,
        call_options: Optional[Dict[str, Any]] = None,
        trace_note: str = "",
    ) -> Optional[Dict[str, Any]]:
        nonlocal last_text, token_info, last_length_limited
        call_options = dict(call_options or options or {})
        try:
            if kind == "generate":
                resp = client.generate(
                    model=model,
                    prompt=use_prompt,
                    system=system,
                    options=call_options,
                    json_mode=json_mode,
                    think=think_value,
                )
            elif kind == "chat":
                resp = client.chat(
                    model=model,
                    prompt=use_prompt,
                    system=system,
                    options=call_options,
                    json_mode=json_mode,
                    think=think_value,
                )
            else:
                raise ValueError(f"Unknown request kind: {kind}")
        except Exception as exc:
            _record_trace(kind, json_mode, None, error=repr(exc))
            if trace and trace_note:
                trace[-1]["trace_note"] = trace_note
            return None

        last_text = resp.text or ""
        last_length_limited = _is_length_limited_response(resp, call_options)
        token_info = {
            "prompt_tokens": resp.prompt_eval_count,
            "completion_tokens": resp.eval_count,
            "total_tokens": resp.total_tokens,
            "duration_ns": resp.total_duration,
            "length_limited": bool(last_length_limited),
            "requested_num_predict": int(call_options.get("num_predict", call_options.get("max_tokens", 0)) or 0),
        }
        _record_trace(kind, json_mode, resp)
        if trace:
            trace[-1]["length_limited"] = bool(last_length_limited)
            trace[-1]["requested_num_predict"] = token_info["requested_num_predict"]
            if trace_note:
                trace[-1]["trace_note"] = trace_note

        parsed = safe_json_loads(last_text)
        if parsed is not None:
            parsed["_parse_ok"] = True
            parsed["_raw_response"] = last_text
            parsed["_tokens"] = token_info
            parsed["_attempts"] = len(trace)
            parsed["_request_trace"] = trace
            return parsed
        return None

    for attempt in range(max_retries + 1):
        use_prompt = prompt
        if attempt > 0:
            use_prompt = (
                prompt
                + "\n\nYour previous response was not valid JSON. Return exactly one minified JSON object only. "
                + "Do not include reasoning, markdown, code fences, <think> blocks, or any text outside the JSON object."
            )

        request_order = []
        if prefer_chat and use_chat_fallback:
            request_order.extend([
                ("chat", use_prompt, True, {"_chat_primary": True}),
                ("chat", use_prompt + "\n\nReturn only the JSON object. The first character must be { and the last must be }.", False, {"_chat_primary": True, "_json_repair_fallback": True}),
                ("generate", use_prompt, True, {"_generate_fallback": True}),
                ("generate", use_prompt + "\n\nReturn only the JSON object. The first character must be { and the last must be }.", False, {"_generate_fallback": True, "_json_repair_fallback": True}),
            ])
        else:
            request_order.extend([
                ("generate", use_prompt, True, {}),
                ("generate", use_prompt + "\n\nReturn only the JSON object. The first character must be { and the last must be }.", False, {"_json_repair_fallback": True}),
            ])
            if use_chat_fallback:
                request_order.extend([
                    ("chat", use_prompt, True, {"_chat_fallback": True}),
                    ("chat", use_prompt + "\n\nReturn only the JSON object. The first character must be { and the last must be }.", False, {"_chat_fallback": True, "_json_repair_fallback": True}),
                ])

        for kind, req_prompt, json_mode, flags in request_order:
            parsed = _try_parse(kind, req_prompt, json_mode=json_mode)
            if parsed is not None:
                parsed.update(flags)
                return parsed

            # If the endpoint reports a generation-length cutoff, retry the same
            # request once with a larger cap before moving to unrelated fallbacks.
            # This directly handles failures such as done_reason=length with
            # eval_count exactly equal to num_predict.
            if length_retry_enabled and last_length_limited:
                retry_options = _with_length_retry_options(options)
                parsed = _try_parse(
                    kind,
                    req_prompt + "\n\nReturn the complete JSON object. Do not stop before the closing }.",
                    json_mode=json_mode,
                    call_options=retry_options,
                    trace_note="length_retry",
                )
                if parsed is not None:
                    parsed.update(flags)
                    parsed["_length_retry_used"] = True
                    return parsed

    return {
        "_parse_ok": False,
        "_raw_response": last_text,
        "_tokens": token_info,
        "_attempts": len(trace),
        "_request_trace": trace,
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
        "request_trace": parsed.get("_request_trace", []),
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
                "request_trace": revision.get("_request_trace", []),
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
            "request_trace": revision.get("_request_trace", []),
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
