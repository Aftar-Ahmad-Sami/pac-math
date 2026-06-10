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


def safe_json_loads(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
        return None
    except json.JSONDecodeError:
        pass

    # Fallback: recover the first JSON object-like substring.
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            obj = json.loads(text[start : end + 1])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            return None
    return None
