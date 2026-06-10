from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import sympy as sp


@dataclass
class NormalizedAnswer:
    raw: str
    extracted: str
    normalized: str
    status: str


BOX_PATTERNS = [
    re.compile(r"\\boxed\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}"),
    re.compile(r"boxed\s*\{([^{}]+)\}", re.IGNORECASE),
]

FINAL_PATTERNS = [
    re.compile(r"(?:final\s+answer|answer)\s*[:=]\s*([^\n]+)", re.IGNORECASE),
    re.compile(r"therefore\s*,?\s*(?:the\s+answer\s+is)?\s*([^\n.]+)", re.IGNORECASE),
]


def _strip_latex_wrappers(text: str) -> str:
    text = str(text).strip()
    text = text.replace("$", "")
    text = text.replace("\\left", "").replace("\\right", "")
    text = text.replace("\\,", " ").replace("\\;", " ")
    text = text.replace("\\!", "")
    text = text.replace("\\text", "text")
    text = re.sub(r"\\mathrm\{([^{}]+)\}", r"\1", text)
    text = re.sub(r"\\text\{([^{}]+)\}", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_answer(text: Any) -> Tuple[str, str]:
    raw = "" if text is None else str(text).strip()
    if not raw:
        return raw, "empty"

    for pat in BOX_PATTERNS:
        matches = pat.findall(raw)
        if matches:
            return str(matches[-1]).strip(), "boxed"

    for pat in FINAL_PATTERNS:
        matches = pat.findall(raw)
        if matches:
            return str(matches[-1]).strip().rstrip("."), "final_pattern"

    # If JSON answer field already short, keep it.
    if len(raw) <= 120 and "\n" not in raw:
        return raw, "raw_short"

    # Fallback to last numeric/expression-looking token.
    candidates = re.findall(r"[-+]?\d+(?:\.\d+)?(?:\s*/\s*[-+]?\d+(?:\.\d+)?)?|[-+]?\\?sqrt\{?\d+\}?", raw)
    if candidates:
        return candidates[-1].strip(), "last_numeric"

    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if lines:
        return lines[-1].strip().rstrip("."), "last_line"
    return raw, "raw_fallback"


def _latex_frac_to_plain(text: str) -> str:
    text = re.sub(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}", r"(\1)/(\2)", text)
    text = re.sub(r"\\dfrac\s*\{([^{}]+)\}\s*\{([^{}]+)\}", r"(\1)/(\2)", text)
    text = re.sub(r"\\tfrac\s*\{([^{}]+)\}\s*\{([^{}]+)\}", r"(\1)/(\2)", text)
    return text


def _latex_sqrt_to_plain(text: str) -> str:
    text = re.sub(r"\\sqrt\s*\{([^{}]+)\}", r"sqrt(\1)", text)
    return text


def canonicalize(text: Any) -> NormalizedAnswer:
    raw = "" if text is None else str(text)
    extracted, status = extract_answer(raw)
    s = _strip_latex_wrappers(extracted)
    s = _latex_frac_to_plain(s)
    s = _latex_sqrt_to_plain(s)
    s = s.replace("−", "-").replace("–", "-").replace("—", "-")
    s = s.replace("\\pi", "pi")
    s = s.replace("^", "**")
    s = re.sub(r"\b(x|y|n|k|m)\s*=\s*", "", s, flags=re.IGNORECASE)
    s = s.strip().strip(". ,;:")
    s = re.sub(r"\s+", "", s)

    # Normalize simple percentages only as text, not numeric conversion, to avoid accidental semantic changes.
    if not s:
        return NormalizedAnswer(raw=raw, extracted=extracted, normalized="", status="empty_after_norm")

    # Try SymPy for numeric/symbolic canonicalization.
    sympy_status = ""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            expr = sp.sympify(s, evaluate=True)
        if expr.is_number:
            expr = sp.nsimplify(expr)
        normalized = str(expr)
        sympy_status = "sympy"
        return NormalizedAnswer(raw=raw, extracted=extracted, normalized=normalized, status=f"{status}+{sympy_status}")
    except Exception:
        pass

    # Text fallback.
    normalized = s.lower()
    return NormalizedAnswer(raw=raw, extracted=extracted, normalized=normalized, status=f"{status}+string")


def equivalent(a: Any, b: Any) -> bool:
    na = canonicalize(a)
    nb = canonicalize(b)
    if not na.normalized or not nb.normalized:
        return False
    if na.normalized == nb.normalized:
        return True

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            ea = sp.sympify(na.normalized, evaluate=True)
            eb = sp.sympify(nb.normalized, evaluate=True)
        if ea == eb:
            return True
        diff = sp.simplify(ea - eb)
        if diff == 0:
            return True
        if ea.is_number and eb.is_number:
            return abs(float(ea) - float(eb)) <= 1e-6
    except Exception:
        return False
    return False
