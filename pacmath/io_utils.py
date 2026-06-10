from __future__ import annotations

import json
import hashlib
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd


def normalize_text(text: Any) -> str:
    text = "" if text is None else str(text)
    return re.sub(r"\s+", " ", text).strip()


def problem_hash(problem: str) -> str:
    return hashlib.sha256(normalize_text(problem).encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_completed_ids(path: Path, key: str = "run_id") -> set[str]:
    if not path.exists():
        return set()
    completed: set[str] = set()
    for row in read_jsonl(path):
        if key in row:
            completed.add(str(row[key]))
    return completed


def find_first_column(row: Dict[str, Any], candidates: List[str]) -> Optional[str]:
    for col in candidates:
        if col in row and row[col] not in [None, ""]:
            return col
    return None


def get_problem(row: Dict[str, Any], candidates: List[str]) -> str:
    col = find_first_column(row, candidates)
    if col is None:
        raise ValueError(f"Could not find problem column. Available keys: {list(row.keys())}")
    return str(row[col])


def get_topic(row: Dict[str, Any], candidates: List[str]) -> str:
    col = find_first_column(row, candidates)
    if col is None:
        return "unknown"
    topic = normalize_text(row[col]).lower()
    topic = topic.replace(" ", "_").replace("/", "_")
    return topic or "unknown"


def get_gold_answer_or_solution(row: Dict[str, Any], candidates: List[str]) -> str:
    col = find_first_column(row, candidates)
    if col is None:
        return ""
    return str(row[col])


def subset_rows(rows: List[Dict[str, Any]], n: Optional[int]) -> List[Dict[str, Any]]:
    if n is None:
        return rows
    return rows[: max(0, min(n, len(rows)))]


def rows_to_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
