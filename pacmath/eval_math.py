from __future__ import annotations

from typing import Any, Dict, List

from .answer_norm import canonicalize, equivalent
from .io_utils import get_gold_answer_or_solution


def get_gold_text(row: Dict[str, Any], answer_cols: List[str]) -> str:
    return get_gold_answer_or_solution(row, answer_cols)


def is_correct(candidate_answer: Any, gold_text: Any) -> bool:
    # Gold field may be a direct answer or a full solution containing boxed answer.
    return equivalent(candidate_answer, gold_text)


def answer_record(raw_answer: Any, gold_text: Any) -> Dict[str, Any]:
    norm = canonicalize(raw_answer)
    gold_norm = canonicalize(gold_text)
    return {
        "raw_answer": norm.raw,
        "extracted_answer": norm.extracted,
        "normalized_answer": norm.normalized,
        "normalization_status": norm.status,
        "gold_extracted": gold_norm.extracted,
        "gold_normalized": gold_norm.normalized,
        "is_correct": equivalent(raw_answer, gold_text),
    }
