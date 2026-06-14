from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - openai is optional until NVIDIA is used.
    OpenAI = None  # type: ignore

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None  # type: ignore


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


class RateLimiter:
    """Small process-local RPM limiter for hosted APIs.

    NVIDIA's free endpoint can be limited to about 40 requests/minute.  We use a
    conservative default below the hard limit and apply it before every hosted
    call.  The project runs sequentially, but the lock also keeps this safe if a
    future script uses light parallelism.
    """

    def __init__(self, rpm: float):
        self.rpm = float(rpm or 0)
        self.min_interval = 0.0 if self.rpm <= 0 else 60.0 / self.rpm
        self._lock = threading.Lock()
        self._next_allowed = 0.0

    def wait(self) -> None:
        if self.min_interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            sleep_s = max(0.0, self._next_allowed - now)
            if sleep_s > 0:
                time.sleep(sleep_s)
                now = time.monotonic()
            self._next_allowed = max(now, self._next_allowed) + self.min_interval


class OllamaClient:
    def __init__(self, base_url: str, timeout_seconds: int = 240):
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def _post_ollama(self, endpoint: str, payload: Dict[str, Any]) -> tuple[Dict[str, Any], float]:
        start = time.time()
        resp = requests.post(
            f"{self.base_url}{endpoint}",
            json=payload,
            timeout=self.timeout_seconds,
        )
        elapsed = time.time() - start
        if resp.status_code != 200:
            raise RuntimeError(f"Ollama HTTP {resp.status_code}: {resp.text[:1000]}")
        return resp.json(), elapsed

    def generate(
        self,
        model: str,
        prompt: str,
        system: str = "",
        options: Optional[Dict[str, Any]] = None,
        json_mode: bool = False,
        think: Optional[bool] = None,
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
        if think is not None:
            payload["think"] = bool(think)

        data, elapsed = self._post_ollama("/api/generate", payload)
        text = data.get("response", "")
        if not text and isinstance(data.get("message"), dict):
            text = data.get("message", {}).get("content", "") or ""
        if not text:
            text = data.get("content", "") or ""

        return OllamaResponse(
            text=text,
            prompt_eval_count=int(data.get("prompt_eval_count", 0) or 0),
            eval_count=int(data.get("eval_count", 0) or 0),
            total_duration=int(data.get("total_duration", 0) or int(elapsed * 1e9)),
            raw=data,
        )

    def chat(
        self,
        model: str,
        prompt: str,
        system: str = "",
        options: Optional[Dict[str, Any]] = None,
        json_mode: bool = False,
        think: Optional[bool] = None,
    ) -> OllamaResponse:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
        }
        if options:
            payload["options"] = options
        if json_mode:
            payload["format"] = "json"
        if think is not None:
            payload["think"] = bool(think)

        data, elapsed = self._post_ollama("/api/chat", payload)
        msg = data.get("message", {}) if isinstance(data.get("message"), dict) else {}
        text = msg.get("content", "") or data.get("response", "") or data.get("content", "") or ""

        return OllamaResponse(
            text=text,
            prompt_eval_count=int(data.get("prompt_eval_count", 0) or 0),
            eval_count=int(data.get("eval_count", 0) or 0),
            total_duration=int(data.get("total_duration", 0) or int(elapsed * 1e9)),
            raw=data,
        )


class NvidiaClient:
    """OpenAI-compatible NVIDIA API client for build.nvidia.com models.

    The code collects streaming `reasoning_content` separately and returns only
    final visible content in `.text`.  PAC-Math parses `.text`, not hidden
    reasoning.  That keeps the main experiment under the same observable-output
    protocol as Ollama v19.
    """

    def __init__(
        self,
        base_url: str = "https://integrate.api.nvidia.com/v1",
        api_key: Optional[str] = None,
        api_key_env: str = "NVIDIA_API_KEY",
        timeout_seconds: int = 900,
        rpm: float = 38.0,
        stream: bool = True,
        enable_thinking: bool = False,
        reasoning_budget: int = 0,
        extra_body: Optional[Dict[str, Any]] = None,
    ):
        if load_dotenv is not None:
            load_dotenv()
        if OpenAI is None:
            raise RuntimeError("The 'openai' package is required for NVIDIA API models. Run: pip install openai python-dotenv")
        self.base_url = base_url.rstrip("/")
        self.api_key_env = api_key_env
        self.api_key = api_key or os.getenv(api_key_env)
        if not self.api_key:
            raise RuntimeError(f"Missing NVIDIA API key. Set {api_key_env}=... in your shell or .env file.")
        self.timeout_seconds = timeout_seconds
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=timeout_seconds)
        self.rate_limiter = RateLimiter(rpm)
        self.stream = bool(stream)
        self.enable_thinking = bool(enable_thinking)
        self.reasoning_budget = int(reasoning_budget or 0)
        self.extra_body = dict(extra_body or {})

    @staticmethod
    def _opt(options: Optional[Dict[str, Any]], key: str, default: Any) -> Any:
        if not options:
            return default
        return options.get(key, default)

    def _extra_body_for_call(self, think: Optional[bool]) -> Dict[str, Any]:
        extra = dict(self.extra_body)
        enable_thinking = self.enable_thinking if think is None else bool(think)
        chat_kwargs = dict(extra.get("chat_template_kwargs", {}) or {})
        chat_kwargs["enable_thinking"] = bool(enable_thinking)
        extra["chat_template_kwargs"] = chat_kwargs
        if enable_thinking and self.reasoning_budget > 0:
            extra["reasoning_budget"] = int(self.reasoning_budget)
        else:
            # Avoid accidentally inheriting a large hidden-reasoning budget when
            # the main PAC-Math protocol disables native thinking.
            extra.pop("reasoning_budget", None)
        return extra

    def _create_completion(
        self,
        *,
        model: str,
        messages: list[Dict[str, str]],
        options: Optional[Dict[str, Any]],
        json_mode: bool,
        think: Optional[bool],
        allow_response_format: bool = True,
    ) -> OllamaResponse:
        start = time.time()
        self.rate_limiter.wait()

        temperature = float(self._opt(options, "temperature", 0.2))
        top_p = float(self._opt(options, "top_p", 0.9))
        max_tokens = int(self._opt(options, "max_tokens", self._opt(options, "num_predict", 700)))
        extra_body = self._extra_body_for_call(think)

        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "extra_body": extra_body,
            "stream": self.stream,
        }
        # NVIDIA's endpoint is OpenAI-compatible, but not every hosted model
        # accepts response_format.  Try it when requested; if the endpoint rejects
        # it, the caller retries without response_format.
        if json_mode and allow_response_format:
            kwargs["response_format"] = {"type": "json_object"}

        try:
            completion = self.client.chat.completions.create(**kwargs)
        except Exception as exc:
            if json_mode and allow_response_format:
                return self._create_completion(
                    model=model,
                    messages=messages,
                    options=options,
                    json_mode=json_mode,
                    think=think,
                    allow_response_format=False,
                )
            raise RuntimeError(f"NVIDIA API error for {model}: {repr(exc)}") from exc

        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        finish_reason = ""
        prompt_tokens = 0
        completion_tokens = 0

        if self.stream:
            for chunk in completion:
                if not getattr(chunk, "choices", None):
                    usage = getattr(chunk, "usage", None)
                    if usage:
                        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
                        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
                    continue
                choice = chunk.choices[0]
                delta = getattr(choice, "delta", None)
                if delta is None:
                    continue
                reasoning = getattr(delta, "reasoning_content", None)
                if reasoning:
                    reasoning_parts.append(str(reasoning))
                content = getattr(delta, "content", None)
                if content is not None:
                    content_parts.append(str(content))
                fr = getattr(choice, "finish_reason", None)
                if fr:
                    finish_reason = str(fr)
        else:
            choice = completion.choices[0]
            msg = choice.message
            content = getattr(msg, "content", "") or ""
            reasoning = getattr(msg, "reasoning_content", "") or ""
            content_parts.append(str(content))
            if reasoning:
                reasoning_parts.append(str(reasoning))
            finish_reason = str(getattr(choice, "finish_reason", "") or "")
            usage = getattr(completion, "usage", None)
            if usage:
                prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
                completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)

        elapsed = time.time() - start
        text = "".join(content_parts).strip()
        reasoning_text = "".join(reasoning_parts)
        raw = {
            "provider": "nvidia",
            "model": model,
            "message": {
                "content": text,
                "reasoning_content_head": reasoning_text[:1000],
                "reasoning_content_len": len(reasoning_text),
            },
            "done_reason": finish_reason,
            "prompt_eval_count": prompt_tokens,
            "eval_count": completion_tokens,
            "total_duration": int(elapsed * 1e9),
            "stream": self.stream,
            "extra_body": extra_body,
            "response_format_requested": bool(json_mode and allow_response_format),
        }
        return OllamaResponse(
            text=text,
            prompt_eval_count=prompt_tokens,
            eval_count=completion_tokens,
            total_duration=int(elapsed * 1e9),
            raw=raw,
        )

    def chat(
        self,
        model: str,
        prompt: str,
        system: str = "",
        options: Optional[Dict[str, Any]] = None,
        json_mode: bool = False,
        think: Optional[bool] = None,
    ) -> OllamaResponse:
        messages: list[Dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self._create_completion(
            model=model,
            messages=messages,
            options=options,
            json_mode=json_mode,
            think=think,
        )

    def generate(
        self,
        model: str,
        prompt: str,
        system: str = "",
        options: Optional[Dict[str, Any]] = None,
        json_mode: bool = False,
        think: Optional[bool] = None,
    ) -> OllamaResponse:
        # NVIDIA's endpoint is chat-completions based.  Use chat for both paths
        # so the rest of PAC-Math can keep its v19 fallback structure unchanged.
        return self.chat(model=model, prompt=prompt, system=system, options=options, json_mode=json_mode, think=think)


class MultiProviderClient:
    """Route each model call to Ollama or NVIDIA by model name.

    The routing is position-independent: agent A and agent B can each be either
    an Ollama model or an NVIDIA-hosted OpenAI-compatible model.

    Rules:
    - explicit prefixes are supported: ``ollama:phi4:14b`` and
      ``nvidia:google/gemma-4-31b-it``; the prefix is stripped before the call.
    - explicit ``MODEL_PROVIDER_OVERRIDES`` in config.py wins.
    - known NVIDIA vendor prefixes such as ``nvidia/`` or ``google/`` route to NVIDIA.
    - if enabled, any slash-style model name (``vendor/model``) routes to NVIDIA.
      Ollama model names usually use colon tags (``gemma4:31b``), so this avoids
      accidental Ollama 404 errors for NVIDIA models like ``google/gemma-4-31b-it``.
    """

    def __init__(
        self,
        ollama: OllamaClient,
        nvidia_factory,
        nvidia_model_prefixes: Optional[list[str]] = None,
        provider_overrides: Optional[Dict[str, str]] = None,
        route_slash_models_to_nvidia: bool = True,
    ):
        self.ollama = ollama
        self._nvidia_factory = nvidia_factory
        self._nvidia_client: Optional[NvidiaClient] = None
        self.nvidia_model_prefixes = tuple(nvidia_model_prefixes or ["nvidia/"])
        self.provider_overrides = dict(provider_overrides or {})
        self.route_slash_models_to_nvidia = bool(route_slash_models_to_nvidia)

    def _nvidia(self) -> NvidiaClient:
        if self._nvidia_client is None:
            self._nvidia_client = self._nvidia_factory()
        return self._nvidia_client

    @staticmethod
    def _strip_provider_prefix(model: str) -> tuple[str, Optional[str]]:
        model_s = str(model).strip()
        lower = model_s.lower()
        if lower.startswith("nvidia:"):
            return model_s.split(":", 1)[1].strip(), "nvidia"
        if lower.startswith("ollama:"):
            return model_s.split(":", 1)[1].strip(), "ollama"
        return model_s, None

    def provider_for_model(self, model: str) -> str:
        clean_model, explicit_provider = self._strip_provider_prefix(model)
        if explicit_provider:
            return explicit_provider

        override = self.provider_overrides.get(clean_model) or self.provider_overrides.get(str(model))
        if override:
            return str(override).lower()

        if clean_model.startswith(self.nvidia_model_prefixes):
            return "nvidia"
        if self.route_slash_models_to_nvidia and "/" in clean_model:
            return "nvidia"
        return "ollama"

    def routed_model_name(self, model: str) -> str:
        clean_model, _ = self._strip_provider_prefix(model)
        return clean_model

    def chat(self, model: str, *args: Any, **kwargs: Any) -> OllamaResponse:
        provider = self.provider_for_model(model)
        routed_model = self.routed_model_name(model)
        if provider == "nvidia":
            return self._nvidia().chat(routed_model, *args, **kwargs)
        if provider == "ollama":
            return self.ollama.chat(routed_model, *args, **kwargs)
        raise ValueError(f"Unknown provider '{provider}' for model '{model}'")

    def generate(self, model: str, *args: Any, **kwargs: Any) -> OllamaResponse:
        provider = self.provider_for_model(model)
        routed_model = self.routed_model_name(model)
        if provider == "nvidia":
            return self._nvidia().generate(routed_model, *args, **kwargs)
        if provider == "ollama":
            return self.ollama.generate(routed_model, *args, **kwargs)
        raise ValueError(f"Unknown provider '{provider}' for model '{model}'")


def build_model_client(config_module: Any) -> MultiProviderClient:
    """Build the mixed Ollama + NVIDIA client from config.py."""
    ollama = OllamaClient(
        getattr(config_module, "OLLAMA_BASE_URL", "http://localhost:11434"),
        timeout_seconds=int(getattr(config_module, "OLLAMA_TIMEOUT_SECONDS", 900)),
    )

    def make_nvidia() -> NvidiaClient:
        return NvidiaClient(
            base_url=str(getattr(config_module, "NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")),
            api_key=os.getenv(str(getattr(config_module, "NVIDIA_API_KEY_ENV", "NVIDIA_API_KEY"))),
            api_key_env=str(getattr(config_module, "NVIDIA_API_KEY_ENV", "NVIDIA_API_KEY")),
            timeout_seconds=int(getattr(config_module, "NVIDIA_TIMEOUT_SECONDS", 900)),
            rpm=float(getattr(config_module, "NVIDIA_RATE_LIMIT_RPM", 38.0)),
            stream=bool(getattr(config_module, "NVIDIA_STREAM", True)),
            enable_thinking=bool(getattr(config_module, "NVIDIA_ENABLE_THINKING", False)),
            reasoning_budget=int(getattr(config_module, "NVIDIA_REASONING_BUDGET", 0)),
            extra_body=dict(getattr(config_module, "NVIDIA_EXTRA_BODY", {}) or {}),
        )

    return MultiProviderClient(
        ollama=ollama,
        nvidia_factory=make_nvidia,
        nvidia_model_prefixes=list(getattr(config_module, "NVIDIA_MODEL_PREFIXES", ["nvidia/"])),
        provider_overrides=dict(getattr(config_module, "MODEL_PROVIDER_OVERRIDES", {}) or {}),
        route_slash_models_to_nvidia=bool(getattr(config_module, "NVIDIA_ROUTE_SLASH_MODELS", True)),
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
