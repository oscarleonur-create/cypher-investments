"""Telling a written thesis from an untouched template.

Phase 3's story slot reported "you have a thesis" for CRDO. CRDO's thesis is
1,306 characters — byte for byte the length of the blank template, because it
*is* the blank template. 1,306 characters of instructions to the author and
nothing from the author.

Length proves nothing; the template is long by design. Matching phrases proves
little either — a first attempt at this leaked the continuation lines of
wrapped template paragraphs and passed CRDO as written. So the comparison is
against the real template: any line the author did not change is not theirs.
"""

from __future__ import annotations

import re

from advisor.thesis.template import TEMPLATE

# A heading, a checkbox, a blockquote or a blank line is structure, not content.
_STRUCTURAL = re.compile(r"^\s*(#{1,6}\s|[-*]\s*\[[ x]\]|>|\s*$)")

# Below this, whatever survives is a stray word rather than a view.
MIN_ORIGINAL_CHARS = 180


def _normalise(line: str) -> str:
    return " ".join(line.split()).strip().lower()


_TEMPLATE_LINES = frozenset(_normalise(line) for line in TEMPLATE.splitlines() if _normalise(line))


def original_prose(content: str) -> str:
    """Only what the author wrote: template lines and structure removed."""
    kept: list[str] = []
    for line in (content or "").splitlines():
        if _STRUCTURAL.match(line):
            continue
        normalised = _normalise(line)
        if not normalised or normalised in _TEMPLATE_LINES:
            continue
        kept.append(line.strip())
    return " ".join(kept)


def is_substantive(content: str) -> bool:
    """True when a human actually wrote something here.

    Deliberately conservative. A thesis wrongly called empty costs a nudge to
    go and fill it in; one wrongly called complete makes the advisor claim you
    hold a view you never formed, and then test events against nothing.
    """
    return len(original_prose(content)) >= MIN_ORIGINAL_CHARS


def template_lines_remaining(content: str) -> int:
    """How many template lines are still sitting there untouched."""
    lines = {_normalise(line) for line in (content or "").splitlines()}
    return len(_TEMPLATE_LINES & lines)


def completeness(content: str) -> float:
    """Rough share of the template the author has replaced, 0.0 to 1.0."""
    if not _TEMPLATE_LINES:
        return 0.0
    return 1.0 - (template_lines_remaining(content) / len(_TEMPLATE_LINES))
