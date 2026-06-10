from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests


@dataclass
class OllamaResponse:
    text: str
    prompt_eval_count: int = 0
    eval_count: int = 0
    total_duration: int = 0
    raw: Optional[Dict[str, Any]] = None

    @property
    def total_tokens(self) -> int:
        return int(self.prompt_eval_count or 0) + int(self.eval_count or 0)


class OllamaClient:
    def __init__(self, base_url: str, timeout_seconds: int = 240):
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def generate(
        self,
        model: str,
        prompt: str,
        system: str = "",
        options: Optional[Dict[str, Any]] = None,
        json_mode: bool = False,
    ) -> OllamaResponse:
        payload: Dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
        }
        if system:
            payload["system"] = system
        if options:
            payload["options"] = options
        if json_mode:
            payload["format"] = "json"

        start = time.time()
        resp = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=self.timeout_seconds,
        )
        elapsed = time.time() - start
        if resp.status_code != 200:
            raise RuntimeError(f"Ollama HTTP {resp.status_code}: {resp.text[:1000]}")
        data = resp.json()
        return OllamaResponse(
            text=data.get("response", ""),
            prompt_eval_count=int(data.get("prompt_eval_count", 0) or 0),
            eval_count=int(data.get("eval_count", 0) or 0),
            total_duration=int(data.get("total_duration", 0) or int(elapsed * 1e9)),
            raw=data,
        )


def _strip_thinking_and_fences(text: str) -> str:
    """Remove common wrappers produced by reasoning-heavy local models.

    Some Qwen-family models may emit <think>...</think>, markdown fences, or
    text before/after the JSON object even when Ollama format=json is requested.
    This helper keeps the JSON-recovery path model-agnostic and prevents a
    parse failure from turning a valid answer into PARSE_ERROR.
    """
    text = (text or "").strip()

    # Remove hidden/reasoning blocks if the model exposes them in text.
    while "<think>" in text.lower() and "</think>" in text.lower():
        lower = text.lower()
        start = lower.find("<think>")
        end = lower.find("</think>", start)
        if start < 0 or end < 0:
            break
        text = text[:start] + text[end + len("</think>"):]
        text = text.strip()

    # Unwrap a fenced JSON block if present.
    if "```" in text:
        parts = text.split("```")
        # Prefer the largest fenced block containing braces.
        fenced = [part for part in parts if "{" in part and "}" in part]
        if fenced:
            text = max(fenced, key=len).strip()
            if text.lower().startswith("json"):
                text = text[4:].strip()

    return text.strip()


def _balanced_json_objects(text: str) -> list[str]:
    """Return balanced top-level JSON-object candidates from arbitrary text."""
    objects: list[str] = []
    start = None
    depth = 0
    in_string = False
    escape = False

    for i, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            continue

        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    objects.append(text[start : i + 1])
                    start = None

    return objects


def _coerce_json_like(text: str) -> Optional[Dict[str, Any]]:
    """Last-resort parser for JSON-like outputs.

    This intentionally handles only the required fields used by PAC-Math. It is
    not meant to be a broad natural-language answer extractor. It prevents a
    model that writes key-value lines instead of strict JSON from being marked
    as total parse failure when the required answer is present.
    """
    import re

    def grab_string(keys: list[str]) -> str:
        for key in keys:
            patterns = [
                rf'"{key}"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"',
                rf"'{key}'\s*:\s*'([^'\\]*(?:\\.[^'\\]*)*)'",
                rf"(?im)^\s*{key}\s*[:=]\s*(.+?)\s*$",
            ]
            for pat in patterns:
                m = re.search(pat, text)
                if m:
                    return m.group(1).strip().strip(',')
        return ""

    def grab_number(keys: list[str], default: float = 50.0) -> float:
        for key in keys:
            patterns = [
                rf'"{key}"\s*:\s*([0-9]+(?:\.[0-9]+)?)',
                rf"'{key}'\s*:\s*([0-9]+(?:\.[0-9]+)?)",
                rf"(?im)^\s*{key}\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)",
            ]
            for pat in patterns:
                m = re.search(pat, text)
                if m:
                    try:
                        return float(m.group(1))
                    except Exception:
                        pass
        return default

    ans = grab_string(["answer", "final_answer", "revised_answer"])
    rev = grab_string(["revised_answer", "final_answer", "answer"])

    # Independent-solve record.
    if ans:
        return {
            "answer": ans,
            "confidence": grab_number(["confidence", "revised_confidence", "final_confidence"], 50.0),
            "reasoning_summary": grab_string(["reasoning_summary", "summary", "reason"]),
            "weak_point": grab_string(["weak_point", "uncertainty"]),
        }

    # Revision record.
    if rev:
        return {
            "revised_answer": rev,
            "revised_confidence": grab_number(["revised_confidence", "confidence", "final_confidence"], 50.0),
            "changed_answer": "true" in text.lower(),
            "revision_reason": grab_string(["revision_reason", "reason"]),
        }

    return None


def safe_json_loads(text: str) -> Optional[Dict[str, Any]]:
    text = _strip_thinking_and_fences(text)
    if not text:
        return None

    # Fast path: exact JSON.
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Robust path: try each balanced JSON object. Prefer later/larger objects,
    # because reasoning text can contain small math-set braces before the final JSON.
    candidates = _balanced_json_objects(text)
    candidates = sorted(candidates, key=lambda x: (len(x), text.rfind(x)), reverse=True)
    for cand in candidates:
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue

    # Backward-compatible coarse substring fallback.
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            obj = json.loads(text[start : end + 1])
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    # Last-resort key-value recovery.
    return _coerce_json_like(text)
