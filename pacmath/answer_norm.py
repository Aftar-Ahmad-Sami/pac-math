from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
)


@dataclass
class NormalizedAnswer:
    raw: str
    extracted: str
    normalized: str
    status: str


FINAL_PATTERNS = [
    re.compile(r"(?:final\s+answer|answer)\s*[:=]\s*([^\n]+)", re.IGNORECASE),
    re.compile(r"therefore\s*,?\s*(?:the\s+answer\s+is)?\s*([^\n.]+)", re.IGNORECASE),
]

UNIT_WORDS = [
    "dollar", "dollars", "cent", "cents", "unit", "units", "squareunit", "squareunits",
    "squnits", "squnit", "squnits", "squnits", "degree", "degrees", "radian", "radians",
    "percent", "percentage", "textsqunits", "textsqunit", "textdollars", "textdollar",
]

_TRANSFORMS = standard_transformations + (implicit_multiplication_application, convert_xor)


def _repair_json_escape_artifacts(text: str) -> str:
    """Repair common LaTeX strings damaged by invalid JSON escaping.

    If a model writes JSON like {"answer":"\frac{1}{2}"}, JSON interprets
    \f as form-feed.  Similar issues occur for \boxed and \begin.  These repairs
    recover the intended LaTeX before answer normalization.
    """
    s = str(text)
    replacements = {
        "\x0crac": r"\frac",
        "\x0cr": r"\fr",
        "\x08egin": r"\begin",
        "\x08oxed": r"\boxed",
        "\x08ig": r"\big",
        "\x08ar": r"\bar",
        "\x08eta": r"\beta",
        "\x07lpha": r"\alpha",
        "\x0beta": r"\theta",
        "\x0bspace": r"\vspace",
    }
    for bad, good in replacements.items():
        s = s.replace(bad, good)
    return s


def _matching_brace(text: str, open_idx: int) -> int:
    depth = 0
    for i in range(open_idx, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
    return -1


def _last_boxed_content(text: str) -> Optional[str]:
    s = str(text)
    matches: list[str] = []
    for m in re.finditer(r"\\?boxed\s*\{", s, flags=re.IGNORECASE):
        open_idx = s.find("{", m.start())
        close_idx = _matching_brace(s, open_idx)
        if open_idx >= 0 and close_idx > open_idx:
            matches.append(s[open_idx + 1 : close_idx])
    return matches[-1].strip() if matches else None


def _strip_latex_wrappers(text: str) -> str:
    s = _repair_json_escape_artifacts(str(text).strip())
    s = s.replace("$", "")
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\,", " ").replace("\\;", " ").replace("\\!", "")
    s = s.replace("−", "-").replace("–", "-").replace("—", "-")
    s = re.sub(r"\\(?:mathrm|operatorname|text)\s*\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\btext\s*\{([^{}]*)\}", r"\1", s, flags=re.IGNORECASE)
    s = re.sub(r"\\(?:textbf|mathbf|mathit|mathrm)\s*\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\(?:circ|degree)\b", "", s)
    s = s.replace("^\\circ", "").replace("^{\\circ}", "").replace("^circ", "")
    s = s.replace("\\%", "%")
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def extract_answer(text: Any) -> Tuple[str, str]:
    raw = "" if text is None else str(text).strip()
    if not raw:
        return raw, "empty"

    boxed = _last_boxed_content(raw)
    if boxed:
        return boxed, "boxed"

    for pat in FINAL_PATTERNS:
        matches = pat.findall(raw)
        if matches:
            return str(matches[-1]).strip().rstrip("."), "final_pattern"

    # If JSON answer field already short, keep it.
    if len(raw) <= 160 and "\n" not in raw:
        return raw, "raw_short"

    # Fallback to last numeric/expression-looking token.
    candidates = re.findall(
        r"[-+]?\d+(?:\.\d+)?(?:\s*/\s*[-+]?\d+(?:\.\d+)?)?|[-+]?\\?sqrt\{?\d+\}?|[A-E]",
        raw,
    )
    if candidates:
        return candidates[-1].strip(), "last_numeric"

    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if lines:
        return lines[-1].strip().rstrip("."), "last_line"
    return raw, "raw_fallback"


def _replace_latex_two_arg_command(text: str, command: str, replacement) -> str:
    s = text
    i = 0
    cmd = "\\" + command
    while True:
        idx = s.find(cmd, i)
        if idx < 0:
            break
        j = idx + len(cmd)
        while j < len(s) and s[j].isspace():
            j += 1
        if j >= len(s) or s[j] != "{":
            i = j
            continue
        c1 = _matching_brace(s, j)
        if c1 < 0:
            break
        k = c1 + 1
        while k < len(s) and s[k].isspace():
            k += 1
        if k >= len(s) or s[k] != "{":
            i = k
            continue
        c2 = _matching_brace(s, k)
        if c2 < 0:
            break
        a = s[j + 1 : c1]
        b = s[k + 1 : c2]
        rep = replacement(a, b)
        s = s[:idx] + rep + s[c2 + 1 :]
        i = idx + len(rep)
    return s


def _replace_latex_one_arg_command(text: str, command: str, replacement) -> str:
    s = text
    i = 0
    cmd = "\\" + command
    while True:
        idx = s.find(cmd, i)
        if idx < 0:
            break
        j = idx + len(cmd)
        while j < len(s) and s[j].isspace():
            j += 1
        if j >= len(s) or s[j] != "{":
            i = j
            continue
        c1 = _matching_brace(s, j)
        if c1 < 0:
            break
        a = s[j + 1 : c1]
        rep = replacement(a)
        s = s[:idx] + rep + s[c1 + 1 :]
        i = idx + len(rep)
    return s


def _latex_to_plain(text: str) -> str:
    s = text.replace("√", r"\sqrt")
    # Recurse a few times because fractions can contain sqrt or nested fractions.
    for _ in range(5):
        old = s
        for cmd in ["dfrac", "tfrac", "frac"]:
            s = _replace_latex_two_arg_command(s, cmd, lambda a, b: f"(({a})/({b}))")
        s = _replace_latex_one_arg_command(s, "sqrt", lambda a: f"sqrt({a})")
        if s == old:
            break
    # Handle compact \dfrac23-style model outputs.
    s = re.sub(r"\\(?:dfrac|tfrac|frac)\s*([+-]?\d+)\s*([+-]?\d+)", r"((\1)/(\2))", s)
    s = re.sub(r"\\sqrt\s*([0-9A-Za-z]+)", r"sqrt(\1)", s)
    return s


def _remove_units_and_noise(s: str) -> str:
    s = s.lower().strip()
    s = s.replace("\\pi", "pi")
    s = s.replace("π", "pi")
    s = s.replace("^", "**")
    s = s.replace("%", "")
    s = s.replace("°", "")
    s = s.replace("$", "")
    s = re.sub(r"(?<![0-9])\.(?=[0-9])", "0.", s)
    s = re.sub(r"\b(x|y|n|k|m|r|a|b|c)\s*=\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\b(?:sq\.?\s*units?|square\s+units?|dollars?|degrees?)\b", "", s)
    s = re.sub(r"\b(?:" + "|".join(map(re.escape, UNIT_WORDS)) + r")\b", "", s)
    s = re.sub(r"\bsq\b", "", s)
    s = s.strip().strip(". ,;:")
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"\*+$", "", s)
    # implicit multiplication: 4sqrt(65), 2pi, 3(1+x)
    s = re.sub(r"(?<=\d)(?=sqrt\()", "*", s)
    s = re.sub(r"(?<=\d)(?=pi\b)", "*", s)
    s = re.sub(r"(?<=\d)(?=\()", "*", s)
    return s


def _split_top_level_commas(s: str) -> Optional[list[str]]:
    depth = 0
    parts: list[str] = []
    start = 0
    for i, ch in enumerate(s):
        if ch in "([{" :
            depth += 1
        elif ch in ")]}":
            depth = max(0, depth - 1)
        elif ch == "," and depth == 0:
            parts.append(s[start:i])
            start = i + 1
    if parts:
        parts.append(s[start:])
        return [p for p in (x.strip() for x in parts) if p]
    return None


def _parse_scalar(s: str):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        return parse_expr(s, transformations=_TRANSFORMS, evaluate=True)


def _canonical_scalar(s: str) -> tuple[str, bool]:
    if not s:
        return "", False
    # Multiple choice letter.
    if re.fullmatch(r"[a-e]", s, flags=re.IGNORECASE):
        return s.lower(), False
    try:
        expr = _parse_scalar(s)
        if getattr(expr, "is_number", False):
            expr = sp.nsimplify(expr)
        return str(expr), True
    except Exception:
        return s.lower(), False


def canonicalize(text: Any) -> NormalizedAnswer:
    raw = "" if text is None else str(text)
    extracted, status = extract_answer(raw)
    s = _strip_latex_wrappers(extracted)
    s = _latex_to_plain(s)
    s = _strip_latex_wrappers(s)

    # Normalize connector words for unordered answer lists.
    s = re.sub(r"\band\b", ",", s, flags=re.IGNORECASE)
    s = _remove_units_and_noise(s)

    if not s:
        return NormalizedAnswer(raw=raw, extracted=extracted, normalized="", status="empty_after_norm")

    # Ordered tuple/list/vector style.
    bracketed = (s.startswith("(") and s.endswith(")")) or (s.startswith("[") and s.endswith("]"))
    inner = s[1:-1] if bracketed else s
    parts = _split_top_level_commas(inner)
    if parts and len(parts) > 1:
        norm_parts = [_canonical_scalar(p)[0] for p in parts]
        prefix = "tuple" if bracketed else "list"
        return NormalizedAnswer(raw=raw, extracted=extracted, normalized=f"{prefix}(" + ",".join(norm_parts) + ")", status=f"{status}+sequence")

    normalized, numeric = _canonical_scalar(s)
    return NormalizedAnswer(raw=raw, extracted=extracted, normalized=normalized, status=f"{status}+{'sympy' if numeric else 'string'}")


def _try_expr(s: str):
    return _parse_scalar(s)


def _sequence_parts(norm: str) -> Optional[tuple[str, list[str]]]:
    m = re.fullmatch(r"(tuple|list)\((.*)\)", norm)
    if not m:
        return None
    kind, body = m.group(1), m.group(2)
    parts = body.split(",") if body else []
    return kind, parts


def equivalent(a: Any, b: Any) -> bool:
    na = canonicalize(a)
    nb = canonicalize(b)
    if not na.normalized or not nb.normalized:
        return False
    if na.normalized == nb.normalized:
        return True

    sa = _sequence_parts(na.normalized)
    sb = _sequence_parts(nb.normalized)
    if sa and sb and len(sa[1]) == len(sb[1]):
        # Ordered comparison for explicit tuples/coordinates.
        if sa[0] == "tuple" or sb[0] == "tuple":
            return all(equivalent(x, y) for x, y in zip(sa[1], sb[1]))
        # Non-bracketed comma/and answers are usually unordered sets/lists.
        return sorted(sa[1]) == sorted(sb[1])

    # Allow scalar-vs-singleton list.
    if sa and len(sa[1]) == 1:
        return equivalent(sa[1][0], nb.normalized)
    if sb and len(sb[1]) == 1:
        return equivalent(na.normalized, sb[1][0])

    try:
        ea = _try_expr(na.normalized)
        eb = _try_expr(nb.normalized)
        if ea == eb:
            return True
        diff = sp.simplify(ea - eb)
        if diff == 0:
            return True
        if getattr(ea, "is_number", False) and getattr(eb, "is_number", False):
            return abs(float(ea) - float(eb)) <= 1e-6
    except Exception:
        return False
    return False
