"""Core verification engine — checks LLM-extracted fields against source text.

For each extracted field, performs fuzzy text search in the original source
context and assigns a grounding status (GROUNDED / PARTIAL / UNGROUNDED).
No external dependencies — pure string matching + regex.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Literal


@dataclass
class FieldVerification:
    """Verification result for a single extracted field."""

    field_name: str
    extracted_value: str
    status: Literal["GROUNDED", "PARTIAL", "UNGROUNDED"]
    matched_snippet: str | None = None


@dataclass
class GroundingResult:
    """Overall verification result for an extraction."""

    grounding_score: float  # 0.0-1.0
    fields: list[FieldVerification] = field(default_factory=list)
    ungrounded_fields: list[str] = field(default_factory=list)


# ── Matching strategies ──────────────────────────────────────────────────────


def _match_name(
    value: str, source: str
) -> tuple[Literal["GROUNDED", "PARTIAL", "UNGROUNDED"], str | None]:
    """Case-insensitive substring match for names."""
    if not value or not value.strip():
        return "UNGROUNDED", None

    value_lower = value.strip().lower()
    source_lower = source.lower()

    # Exact substring match
    idx = source_lower.find(value_lower)
    if idx != -1:
        snippet = source[idx : idx + len(value)]
        return "GROUNDED", snippet

    # Try matching last name (for "John Smith" check if "Smith" appears)
    parts = value_lower.split()
    if len(parts) >= 2:
        last_name = parts[-1]
        if len(last_name) >= 3:  # avoid matching short fragments
            idx = source_lower.find(last_name)
            if idx != -1:
                # Extract surrounding context as snippet
                start = max(0, idx - 20)
                end = min(len(source), idx + len(last_name) + 20)
                return "PARTIAL", source[start:end].strip()

    return "UNGROUNDED", None


def _match_date(
    value: str, source: str
) -> tuple[Literal["GROUNDED", "PARTIAL", "UNGROUNDED"], str | None]:
    """Parse date and search for any common format within ±1 day."""
    if not value or not value.strip():
        return "UNGROUNDED", None

    # Try to parse the extracted date
    parsed_date = None
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y", "%B %d, %Y", "%b %d, %Y"):
        try:
            parsed_date = datetime.strptime(value.strip()[:10], fmt)
            break
        except ValueError:
            continue

    if parsed_date is None:
        return "UNGROUNDED", None

    # Generate date variants to search for (±1 day window)
    dates_to_check = [parsed_date + timedelta(days=d) for d in (-1, 0, 1)]
    source_lower = source.lower()

    for dt in dates_to_check:
        variants = [
            dt.strftime("%Y-%m-%d"),  # 2024-01-15
            dt.strftime("%m/%d/%Y"),  # 01/15/2024
            dt.strftime("%m/%d/%y"),  # 01/15/24
            dt.strftime("%B %d, %Y"),  # January 15, 2024
            dt.strftime("%b %d, %Y"),  # Jan 15, 2024
            dt.strftime("%B %d"),  # January 15
            dt.strftime("%b %d"),  # Jan 15
        ]
        for v in variants:
            idx = source_lower.find(v.lower())
            if idx != -1:
                status = "GROUNDED" if dt == parsed_date else "PARTIAL"
                snippet = source[idx : idx + len(v)]
                return status, snippet

    return "UNGROUNDED", None


def _match_number(
    value: str, source: str, tolerance: float = 0.01
) -> tuple[Literal["GROUNDED", "PARTIAL", "UNGROUNDED"], str | None]:
    """Search for a number in source text with ±tolerance (default 1%).

    Checks raw number, comma-formatted, and $-prefixed variants.
    """
    if not value or not str(value).strip():
        return "UNGROUNDED", None

    try:
        num = float(str(value).replace(",", "").replace("$", ""))
    except (ValueError, TypeError):
        return "UNGROUNDED", None

    if num == 0:
        return "UNGROUNDED", None

    # Find all numbers in source text
    # Match patterns like: 123, 1,234, 1234.56, $123.45, $1,234
    number_pattern = re.compile(r"\$?[\d,]+\.?\d*")
    for match in number_pattern.finditer(source):
        try:
            matched_str = match.group().replace(",", "").replace("$", "")
            matched_num = float(matched_str)
            if matched_num == 0:
                continue
            # Check within tolerance
            if abs(matched_num - num) / abs(num) <= tolerance:
                return "GROUNDED", match.group()
        except (ValueError, TypeError):
            continue

    return "UNGROUNDED", None


def _match_title(
    value: str, source: str, name: str | None = None
) -> tuple[Literal["GROUNDED", "PARTIAL", "UNGROUNDED"], str | None]:
    """Check if role keyword appears in source, optionally near a matched name."""
    if not value or not value.strip():
        return "UNGROUNDED", None

    role_keywords = {
        "CEO": ["ceo", "chief executive"],
        "CFO": ["cfo", "chief financial"],
        "COO": ["coo", "chief operating"],
        "CTO": ["cto", "chief technology"],
        "DIR": ["director", "dir"],
        "VP": ["vice president", "vp"],
        "PRES": ["president", "pres"],
        "SVP": ["senior vice president", "svp"],
        "EVP": ["executive vice president", "evp"],
        "10% OWNER": ["10%", "10 percent", "ten percent"],
    }

    value_upper = value.strip().upper()
    source_lower = source.lower()

    # Find keywords to search for
    keywords: list[str] = []
    for role_key, role_terms in role_keywords.items():
        if role_key in value_upper:
            keywords.extend(role_terms)

    # Fallback: use the value itself as a keyword
    if not keywords:
        keywords = [value.strip().lower()]

    for kw in keywords:
        idx = source_lower.find(kw)
        if idx == -1:
            continue

        snippet_start = max(0, idx - 30)
        snippet_end = min(len(source), idx + len(kw) + 30)
        snippet = source[snippet_start:snippet_end].strip()

        # If we have a name, check proximity (within 200 chars)
        if name:
            name_lower = name.lower()
            # Check last name proximity
            name_parts = name_lower.split()
            last_name = name_parts[-1] if name_parts else name_lower
            if len(last_name) >= 3:
                name_idx = source_lower.find(last_name)
                if name_idx != -1 and abs(name_idx - idx) < 200:
                    return "GROUNDED", snippet

        # No name to check proximity against — just confirm keyword exists
        return "PARTIAL", snippet

    return "UNGROUNDED", None


def _match_keyword(
    value: str, source: str
) -> tuple[Literal["GROUNDED", "PARTIAL", "UNGROUNDED"], str | None]:
    """Generic case-insensitive substring match for keywords like trade_type."""
    if not value or not value.strip():
        return "UNGROUNDED", None

    value_lower = value.strip().lower()
    source_lower = source.lower()

    idx = source_lower.find(value_lower)
    if idx != -1:
        snippet_start = max(0, idx - 10)
        snippet_end = min(len(source), idx + len(value) + 10)
        return "GROUNDED", source[snippet_start:snippet_end].strip()

    return "UNGROUNDED", None


# ── Field type detection ─────────────────────────────────────────────────────

_NAME_FIELDS = {"insider_name", "name", "politician", "author"}
_DATE_FIELDS = {"filing_date", "date", "transaction_date", "reported_date"}
_NUMBER_FIELDS = {"price", "qty", "value", "shares", "amount", "volume"}
_TITLE_FIELDS = {"title", "role", "position"}


def _detect_field_type(field_name: str) -> str:
    """Detect the matching strategy to use based on field name."""
    if field_name in _NAME_FIELDS:
        return "name"
    if field_name in _DATE_FIELDS:
        return "date"
    if field_name in _NUMBER_FIELDS:
        return "number"
    if field_name in _TITLE_FIELDS:
        return "title"
    return "keyword"


# ── Main entry point ─────────────────────────────────────────────────────────


def verify_extraction(
    extracted: dict,
    source_text: str,
    fields_to_check: list[str],
) -> GroundingResult:
    """Verify that LLM-extracted fields are grounded in the source text.

    Args:
        extracted: Dict of field_name -> extracted_value from the LLM.
        source_text: The original source text that was fed to the LLM.
        fields_to_check: List of field names to verify.

    Returns:
        GroundingResult with per-field verification and overall score.
    """
    fields: list[FieldVerification] = []
    ungrounded: list[str] = []

    # Get name value for title proximity checks
    name_value = None
    for nf in _NAME_FIELDS:
        if nf in extracted:
            name_value = str(extracted[nf])
            break

    for field_name in fields_to_check:
        raw_value = extracted.get(field_name)
        if raw_value is None:
            fields.append(
                FieldVerification(
                    field_name=field_name,
                    extracted_value="",
                    status="UNGROUNDED",
                )
            )
            ungrounded.append(field_name)
            continue

        str_value = str(raw_value)
        field_type = _detect_field_type(field_name)

        if field_type == "name":
            status, snippet = _match_name(str_value, source_text)
        elif field_type == "date":
            status, snippet = _match_date(str_value, source_text)
        elif field_type == "number":
            status, snippet = _match_number(str_value, source_text)
        elif field_type == "title":
            status, snippet = _match_title(str_value, source_text, name=name_value)
        else:
            status, snippet = _match_keyword(str_value, source_text)

        fields.append(
            FieldVerification(
                field_name=field_name,
                extracted_value=str_value,
                status=status,
                matched_snippet=snippet,
            )
        )
        if status == "UNGROUNDED":
            ungrounded.append(field_name)

    # Compute overall score
    if not fields:
        grounding_score = 0.0
    else:
        score_map = {"GROUNDED": 1.0, "PARTIAL": 0.5, "UNGROUNDED": 0.0}
        total = sum(score_map[f.status] for f in fields)
        grounding_score = total / len(fields)

    return GroundingResult(
        grounding_score=round(grounding_score, 3),
        fields=fields,
        ungrounded_fields=ungrounded,
    )


def verify_headline(headline: str, source_texts: list[str], threshold: float = 0.5) -> bool:
    """Check if a headline is grounded in any of the source texts.

    Uses word-overlap ratio: if ≥threshold of headline words appear in any source, it's grounded.
    """
    if not headline or not source_texts:
        return False

    # Normalize headline words
    words = set(re.findall(r"\b\w{3,}\b", headline.lower()))
    if not words:
        return False

    for source in source_texts:
        source_lower = source.lower()
        matched = sum(1 for w in words if w in source_lower)
        if matched / len(words) >= threshold:
            return True

    return False
