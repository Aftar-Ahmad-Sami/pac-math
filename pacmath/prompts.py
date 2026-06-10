from __future__ import annotations

SYSTEM_SOLVER = """You are a careful mathematical reasoning agent. Give a concise solution and final answer. Return valid JSON only."""

SYSTEM_DEBATE = """You are a careful mathematical reasoning agent participating in a structured two-agent math debate. Return valid JSON only."""


def _evidence_gate_enabled() -> bool:
    try:
        import config
        return bool(getattr(config, "EVIDENCE_GATED_DEBATE", False))
    except Exception:
        return False


def independent_prompt(problem: str) -> str:
    return f"""
Solve the math problem carefully.

Problem:
{problem}

Return only valid JSON with exactly these keys:
{{
  "answer": "your final answer only",
  "confidence": 0,
  "reasoning_summary": "brief reasoning summary, not too long",
  "weak_point": "the most uncertain part of your solution, or 'none'"
}}

Rules:
- confidence must be a number from 0 to 100.
- answer must be short and contain only the final answer, not the full solution.
- Do not include Markdown.
""".strip()


def critique_prompt(problem: str, own_answer: str, own_reasoning: str, partner_answer: str, partner_reasoning: str) -> str:
    return f"""
You solved this problem independently. Now inspect your partner's answer.

Problem:
{problem}

Your independent answer:
{own_answer}

Your reasoning summary:
{own_reasoning}

Partner answer:
{partner_answer}

Partner reasoning summary:
{partner_reasoning}

Return only valid JSON with exactly these keys:
{{
  "agreement": "what you agree with, or 'none'",
  "disagreement": "what you disagree with, or 'none'",
  "error_identified": "specific possible error in partner's solution, or 'none'",
  "critique_summary": "brief critique summary"
}}

Do not change your final answer yet. This round is critique only.
Do not include Markdown.
""".strip()


def revision_prompt(
    problem: str,
    own_initial_answer: str,
    own_initial_reasoning: str,
    partner_initial_answer: str,
    partner_critique: str,
) -> str:
    if not _evidence_gate_enabled():
        return f"""
You now revise or preserve your answer after reading your partner's critique.

Problem:
{problem}

Your independent answer:
{own_initial_answer}

Your independent reasoning summary:
{own_initial_reasoning}

Partner's independent answer:
{partner_initial_answer}

Partner's critique of your solution:
{partner_critique}

Return only valid JSON with exactly these keys:
{{
  "revised_answer": "your final answer after considering the critique",
  "revised_confidence": 0,
  "changed_answer": false,
  "revision_reason": "brief reason for preserving or changing your answer"
}}

Rules:
- revised_confidence must be a number from 0 to 100.
- changed_answer must be true or false.
- revised_answer must be short and contain only the final answer, not the full solution.
- Do not include Markdown.
""".strip()

    return f"""
You now decide whether to preserve or revise your independent answer.

Important rule: do not change your independent answer merely because your partner disagrees or sounds confident. Change only if you can identify a concrete mathematical error in your own solution, or if you can independently verify the partner's answer from the problem.

Problem:
{problem}

Your independent answer:
{own_initial_answer}

Your independent reasoning summary:
{own_initial_reasoning}

Partner's independent answer:
{partner_initial_answer}

Partner's critique of your solution:
{partner_critique}

Return only valid JSON with exactly these keys:
{{
  "revised_answer": "your final answer after considering the critique",
  "revised_confidence": 0,
  "changed_answer": false,
  "error_evidence_type": "none | arithmetic_error | algebraic_error | missed_case | misread_problem | independent_verification | partner_assertion_only",
  "change_justification_score": 0,
  "initial_answer_validity_score": 0,
  "revised_answer_validity_score": 0,
  "revision_reason": "brief reason for preserving or changing your answer"
}}

Rules:
- revised_confidence, change_justification_score, initial_answer_validity_score, and revised_answer_validity_score must be numbers from 0 to 100.
- changed_answer must be true or false.
- If error_evidence_type is partner_assertion_only or none, changed_answer should normally be false.
- revised_answer must be short and contain only the final answer, not the full solution.
- Do not include Markdown.
""".strip()


def commitment_check_prompt(
    problem: str,
    own_initial_answer: str,
    proposed_revised_answer: str,
    revision_json: str,
) -> str:
    return f"""
You are doing a final commitment check before submitting your post-debate answer.

Your task is to prevent harmful persuasion. Keep your independent answer unless the proposed revision is clearly better and supported by concrete mathematical evidence.

Problem:
{problem}

Your independent answer:
{own_initial_answer}

Your proposed revised answer:
{proposed_revised_answer}

Your previous revision record:
{revision_json}

Return only valid JSON with exactly these keys:
{{
  "final_answer": "the answer you commit to after the safety check",
  "final_confidence": 0,
  "accept_revision": false,
  "specific_error_in_initial": "specific error in your independent answer, or 'none'",
  "specific_support_for_revision": "specific mathematical support for revised answer, or 'none'",
  "persuasion_risk_score": 0,
  "commitment_reason": "brief explanation"
}}

Rules:
- final_confidence and persuasion_risk_score must be numbers from 0 to 100.
- accept_revision must be true only if a concrete error or independent verification supports the revised answer.
- If specific_error_in_initial is 'none' and specific_support_for_revision is 'none', keep the independent answer.
- final_answer must be short and contain only the final answer, not the full solution.
- Do not include Markdown.
""".strip()


SYSTEM_VERIFIER = """You are a strict mathematical verifier. You do not know the official answer. Your job is to inspect candidate final answers against the problem and rate how likely each candidate is correct. Return valid JSON only."""


def verification_prompt(problem: str, candidate_lines: str) -> str:
    return f"""
You must verify candidate final answers for this math problem.

Problem:
{problem}

Candidate answers:
{candidate_lines}

For each candidate, judge only whether the final answer appears mathematically valid for the problem. Do not assume the majority is correct. Penalize answers that are unsupported, inconsistent, or likely caused by persuasion. You do not have the official answer.

Return only valid JSON with exactly these keys:
{{
  "reviews": [
    {{
      "candidate_id": "A0",
      "validity_score": 0,
      "error_risk": 100,
      "brief_reason": "short reason"
    }}
  ],
  "best_candidate_id": "A0",
  "best_answer": "short final answer"
}}

Rules:
- validity_score is 0 to 100, where 100 means very likely correct.
- error_risk is 0 to 100, where 100 means very likely wrong or unsafe to trust.
- Include one review for each candidate id shown.
- Do not include Markdown.
""".strip()
