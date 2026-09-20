"""Interim figures out of a 6-K exhibit, when there is no XBRL to read.

A foreign private issuer files a 20-F once a year and reports its quarters on
Form 6-K. Nebius is the live case: `latest_fundamentals` reads periodic
filings, found the December 2025 20-F, and valued the position on figures 263
days old — 108x revenue against a real 25x, and an implied growth rate of 33%
against a real 15%. The Q2 2026 numbers existed the whole time, inside
`nbis-20260812xex99d1.htm`, which nothing read past its headline.

The exhibit carries **no XBRL**. It is 50,000 characters of MD&A prose with
tables flattened into the text, label and figures interleaved:

    Total other 24.6 81.5 32.9 101.4 income, net

Parsing that layout is how you get a confident wrong number. So this module
reads the *sentences* instead, and takes only what the document confirms twice:

* **Self-checked** — "Total revenues increased by $477.2 million, or 454%,
  from $105.1 million in the second quarter of 2025 to $582.3 million in the
  second quarter of 2026." Five figures that must agree with each other:
  582.3 - 105.1 = 477.2, and 477.2 / 105.1 = 454%. A sentence we misread does
  not reconcile, and is dropped.

* **Table-confirmed** — a segment row gives levels but no arithmetic of its
  own, so it is accepted only when a separate sentence states the same change:
  the table says Nebius AI cloud went 9.5 -> 285.7, and the prose says
  "improved by $276.2 million". Two independent statements agreeing.

Nothing else is returned. A figure that cannot be established this way is
absent, never estimated — same rule the XBRL path follows for debt.

Scope, stated honestly: these sentence shapes are the register foreign
issuers' counsel write in, not a standard. NBIS is the only such filer in the
book, so generalisation is unproven. The reconciliation gate is what makes
that acceptable — a shape this module does not understand yields nothing,
which is the failure mode we want.
"""

from __future__ import annotations

import logging
import re
from enum import StrEnum

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Figures round to one decimal, so a difference of two roundings can be 0.2.
# The relative arm carries large numbers where a tenth is noise.
ABS_TOLERANCE = 0.3
REL_TOLERANCE = 0.005

# A stated percentage is rounded to a whole point more often than not, so the
# check has to be loose enough for 454% against a computed 454.04%, and tight
# enough to reject a sentence whose clauses were mismatched.
PCT_TOLERANCE = 1.5

_SCALE = {"million": 1e6, "billion": 1e9, "thousand": 1e3}

_UP = ("increased", "grew", "rose", "improved", "increase")
_DOWN = ("decreased", "declined", "fell", "decrease")


class Period(StrEnum):
    QUARTER = "QUARTER"
    HALF = "HALF"


class Basis(StrEnum):
    """How a figure earned its place. Both mean "confirmed twice"."""

    SELF_CHECKED = "SELF_CHECKED"  # one sentence whose own arithmetic agrees
    TABLE_CONFIRMED = "TABLE_CONFIRMED"  # a table row a sentence's delta agrees with


class InterimFigure(BaseModel):
    """One metric, one period, with the comparison the issuer stated."""

    metric: str
    period: Period
    prior: float
    current: float
    delta: float
    growth: float | None = None  # fraction, not points: 4.54 is +454%
    basis: Basis

    @property
    def turned_negative(self) -> bool:
        return self.current < 0 <= self.prior


class InterimResults(BaseModel):
    """Everything the exhibit confirmed, plus what it refused."""

    figures: list[InterimFigure] = Field(default_factory=list)
    rejected: list[str] = Field(default_factory=list)
    # Which filing the figures actually came from — not always the one asked
    # for, when a quarter is filed as a press release plus a financial report.
    source_accession: str | None = None

    def find(self, *, metric: str, period: Period = Period.QUARTER) -> InterimFigure | None:
        """First figure whose metric contains ``metric``, case-insensitively."""
        needle = metric.lower()
        for fig in self.figures:
            if fig.period is period and needle in fig.metric.lower():
                return fig
        return None


def _normalise(text: str) -> str:
    """Collapse the zero-width spaces and column padding the flattener leaves."""
    return re.sub(r"\s+", " ", text.replace("​", " ")).strip()


def _number(raw: str) -> float:
    return float(raw.replace(",", "").replace("−", "-").replace("–", "-"))


def _growth(prior: float, current: float) -> float | None:
    """Percentage change, or None when the base makes one meaningless.

    Total adjusted EBITDA went from -21.0 to 236.2. Dividing by the base gives
    +1,225%, which reads as spectacular growth and describes a sign flip. A
    change off a zero or negative base has no growth rate; the two levels say
    what happened and the percentage would only mislead.
    """
    if prior <= 0:
        return None
    return (current - prior) / prior


def _agrees(observed: float, expected: float) -> bool:
    return abs(observed - expected) <= max(ABS_TOLERANCE, abs(expected) * REL_TOLERANCE)


def _direction(verb: str) -> int:
    lowered = verb.lower()
    if any(lowered.startswith(w) for w in _DOWN):
        return -1
    if any(lowered.startswith(w) for w in _UP):
        return 1
    return 0


def _period_of(phrase: str) -> Period | None:
    lowered = phrase.lower()
    if "six month" in lowered or "half" in lowered or "nine month" in lowered:
        return Period.HALF
    if "quarter" in lowered or "three month" in lowered:
        return Period.QUARTER
    return None


def _clean_metric(raw: str) -> str:
    """Trim the connective words a sentence hangs its subject on."""
    out = raw.strip().strip(":;,").strip()
    out = re.sub(r"^(?:and|but|while|whereas|in addition,?)\s+", "", out, flags=re.I)
    # The flattener repeats a header that wraps two lines, so a subject can
    # arrive doubled: "Interest income Interest income".
    half = len(out) // 2
    if out[:half].strip() and out[:half].strip() == out[half:].strip():
        out = out[:half].strip()
    # "Revenues for the Nebius AI cloud business" and "Revenues from TripleTen"
    # are the segment's name; keeping the preposition keeps them distinct.
    return out


# "Total revenues increased by $477.2 million, or 454%, from $105.1 million in
#  the second quarter of 2025 to $582.3 million in the second quarter of 2026."
_SHAPE_A = re.compile(
    r"(?:^|[.:;]\s)(?P<metric>[A-Z][^.:;$]{2,90}?)\s+"
    r"(?P<verb>increased|decreased|grew|declined|rose|fell|improved)\s+by\s+"
    r"\$(?P<delta>[\d,]+(?:\.\d+)?)\s*(?P<dscale>million|billion|thousand)"
    r"\s*,?\s*or\s+(?P<pct>[\d,]+(?:\.\d+)?)\s*%\s*,?\s*"
    r"from\s+\$(?P<prior>[\d,]+(?:\.\d+)?)\s*(?P<pscale>million|billion|thousand)?\s+"
    r"in\s+(?P<p1>[^$]{3,70}?)\s+"
    r"to\s+\$(?P<current>[\d,]+(?:\.\d+)?)\s*(?P<cscale>million|billion|thousand)?\s+"
    r"in\s+(?P<p2>[^.]{3,70}?)\.",
    re.I,
)

# "Other income, net was $24.6 million and $81.5 million for the three months
#  ended June 30, 2025 and 2026, respectively."
_SHAPE_B = re.compile(
    r"(?:^|[.:;]\s)(?P<metric>[A-Z][^.:;$]{2,90}?)\s+was\s+"
    r"\$(?P<prior>[\d,]+(?:\.\d+)?)\s*(?P<pscale>million|billion|thousand)?\s+and\s+"
    r"\$(?P<current>[\d,]+(?:\.\d+)?)\s*(?P<cscale>million|billion|thousand)\s+"
    r"for\s+the\s+(?P<period>[^.]{3,80}?),?\s+respectively\.",
    re.I,
)

# "Adjusted EBITDA for the Nebius AI cloud business improved by $276.2 million
#  and $477.6 million in the three months ... and six months ..., respectively"
_SHAPE_C_PAIR = re.compile(
    r"(?:^|[.:;]\s)(?P<metric>[A-Z][^.:;$]{2,90}?)\s+"
    r"(?P<verb>increased|decreased|grew|declined|improved|worsened)\s+by\s+"
    r"\$(?P<d1>[\d,]+(?:\.\d+)?)\s*(?P<s1>million|billion|thousand)\s+and\s+"
    r"\$(?P<d2>[\d,]+(?:\.\d+)?)\s*(?P<s2>million|billion|thousand)\s+"
    r"in\s+the\s+(?P<periods>[^.]{3,120}?)\.",
    re.I,
)

# "Total Adjusted EBITDA loss for the Group improved by $257.2 million in the
#  second quarter of 2026 compared to the same period of 2025."
_SHAPE_C_ONE = re.compile(
    r"(?:^|[.:;]\s)(?P<metric>[A-Z][^.:;$]{2,90}?)\s+"
    r"(?P<verb>increased|decreased|grew|declined|improved|worsened)\s+by\s+"
    r"\$(?P<delta>[\d,]+(?:\.\d+)?)\s*(?P<scale>million|billion|thousand)\s+"
    r"in\s+(?P<period>[^.]{3,90}?)\.",
    re.I,
)

# A flattened table row is four columns — prior quarter, current quarter,
# prior half, current half — abutting whatever text precedes them.
_FOUR_COLUMNS = re.compile(
    r"(?P<q1>-?\d[\d,]*\.\d)\s+(?P<q2>-?\d[\d,]*\.\d)\s+"
    r"(?P<h1>-?\d[\d,]*\.\d)\s+(?P<h2>-?\d[\d,]*\.\d)(?!\s*\.?\d)"
)

# Words that end a table's unit header. A label must not reach past one.
_HEADER_TAIL = frozenset({"dollars", "thousands", "millions", "billions"})

# A row label is at most this many words. Longer means the scan ran into prose.
_MAX_LABEL_WORDS = 6


def _label_before(text: str, end: int) -> str:
    """The row label sitting immediately left of a row's figures.

    Matching the label as part of the row pattern absorbed whatever came
    before it — the Nebius AI cloud row came back as "ollars U.S. dollars
    Nebius AI cloud", which then matched no sentence and dropped the one
    segment figure the thesis needed. A label abuts its own figures, so read
    backwards from them and stop at the previous number or the unit header.
    """
    window = text[max(0, end - 90) : end]
    tail = re.split(r"\d", window)[-1]
    words = re.findall(r"[A-Za-z][A-Za-z()/&.\-']*", tail)
    out: list[str] = []
    for word in reversed(words):
        if word.lower().strip(".") in _HEADER_TAIL:
            break
        out.append(word)
        if len(out) >= _MAX_LABEL_WORDS:
            break
    return " ".join(reversed(out))


def _shape_a(text: str) -> tuple[list[InterimFigure], list[str]]:
    """The self-checking comparison: five figures that must agree."""
    figures: list[InterimFigure] = []
    rejected: list[str] = []
    for m in _SHAPE_A.finditer(text):
        sign = _direction(m.group("verb"))
        if sign == 0:
            continue
        # A scale stated once governs the clause; "from $105.1 million to
        # $582.3 million" often omits it on the first figure.
        dscale = _SCALE[m.group("dscale").lower()]
        pscale = _SCALE.get((m.group("pscale") or "").lower(), dscale)
        cscale = _SCALE.get((m.group("cscale") or "").lower(), dscale)

        delta = _number(m.group("delta")) * dscale * sign
        prior = _number(m.group("prior")) * pscale
        current = _number(m.group("current")) * cscale
        pct = _number(m.group("pct")) * sign

        period = _period_of(m.group("p1")) or _period_of(m.group("p2"))
        metric = _clean_metric(m.group("metric"))
        label = f"{metric} ({m.group('p1').strip()})"

        if period is None:
            rejected.append(f"{label}: no period named")
            continue
        if not _agrees(current - prior, delta):
            rejected.append(
                f"{label}: {current:,.0f} - {prior:,.0f} is not the stated change of {delta:,.0f}"
            )
            continue
        computed_pct = (delta / prior * 100.0) if prior else None
        if computed_pct is not None and abs(computed_pct - pct) > PCT_TOLERANCE:
            rejected.append(f"{label}: stated {pct:.0f}% against a computed {computed_pct:.1f}%")
            continue

        figures.append(
            InterimFigure(
                metric=metric,
                period=period,
                prior=prior,
                current=current,
                delta=delta,
                growth=_growth(prior, current),
                basis=Basis.SELF_CHECKED,
            )
        )
    return figures, rejected


def _shape_b(text: str) -> list[InterimFigure]:
    """ "X was $A and $B for the <period>, respectively" — two levels, one period.

    No internal arithmetic to check, so the delta is derived rather than
    confirmed. Kept because the two levels are stated outright, which is the
    same standard as a table row a sentence agrees with.
    """
    figures: list[InterimFigure] = []
    for m in _SHAPE_B.finditer(text):
        period = _period_of(m.group("period"))
        if period is None:
            continue
        cscale = _SCALE[m.group("cscale").lower()]
        pscale = _SCALE.get((m.group("pscale") or "").lower(), cscale)
        prior = _number(m.group("prior")) * pscale
        current = _number(m.group("current")) * cscale
        figures.append(
            InterimFigure(
                metric=_clean_metric(m.group("metric")),
                period=period,
                prior=prior,
                current=current,
                delta=current - prior,
                growth=_growth(prior, current),
                basis=Basis.TABLE_CONFIRMED,
            )
        )
    return figures


def _stated_deltas(text: str) -> list[tuple[str, Period, float]]:
    """Every change the prose states without giving the levels behind it."""
    out: list[tuple[str, Period, float]] = []
    for m in _SHAPE_C_PAIR.finditer(text):
        sign = _direction(m.group("verb"))
        if sign == 0:
            continue
        metric = _clean_metric(m.group("metric"))
        periods = m.group("periods")
        # "in the three months of 2026 and six months ended June 30, 2026" —
        # the quarter is named first, the half second.
        head, _, tail = periods.partition(" and ")
        for raw, scale_key, phrase in (
            (m.group("d1"), m.group("s1"), head),
            (m.group("d2"), m.group("s2"), tail),
        ):
            period = _period_of(phrase)
            if period is not None:
                out.append((metric, period, _number(raw) * _SCALE[scale_key.lower()] * sign))
    for m in _SHAPE_C_ONE.finditer(text):
        sign = _direction(m.group("verb"))
        period = _period_of(m.group("period"))
        if sign == 0 or period is None:
            continue
        out.append(
            (
                _clean_metric(m.group("metric")),
                period,
                _number(m.group("delta")) * _SCALE[m.group("scale").lower()] * sign,
            )
        )
    return out


def _table_rows(text: str, *, scale: float) -> list[tuple[str, dict[Period, tuple[float, float]]]]:
    """Flattened four-column rows: prior/current quarter, prior/current half."""
    rows = []
    for m in _FOUR_COLUMNS.finditer(text):
        label = _clean_metric(_label_before(text, m.start()))
        if not label or label.lower() in {"and", "the", "of"}:
            continue
        cols = [_number(m.group(g)) * scale for g in ("q1", "q2", "h1", "h2")]
        rows.append(
            (
                label,
                {
                    Period.QUARTER: (cols[0], cols[1]),
                    Period.HALF: (cols[2], cols[3]),
                },
            )
        )
    return rows


def _confirmed_rows(text: str, *, scale: float) -> tuple[list[InterimFigure], list[str]]:
    """Table rows a stated delta agrees with. Everything else is dropped."""
    deltas = _stated_deltas(text)
    figures: list[InterimFigure] = []
    rejected: list[str] = []
    for label, by_period in _table_rows(text, scale=scale):
        for period, (prior, current) in by_period.items():
            named = next(
                (
                    metric
                    for metric, p, d in deltas
                    if p is period and _mentions(metric, label) and _agrees(current - prior, d)
                ),
                None,
            )
            if named is None:
                continue
            # The name comes from the sentence, not the row. A row label is
            # only the segment — "Nebius AI cloud" heads both a revenue table
            # and an EBITDA table, and the flattener had already truncated the
            # total's label to "Total adjusted". The sentence that confirms the
            # change is the thing that says what was measured.
            figures.append(
                InterimFigure(
                    metric=named,
                    period=period,
                    prior=prior,
                    current=current,
                    delta=current - prior,
                    growth=_growth(prior, current),
                    basis=Basis.TABLE_CONFIRMED,
                )
            )
    confirmed_labels = {
        label
        for label, by_period in _table_rows(text, scale=scale)
        for metric, _, _ in deltas
        if _mentions(metric, label) and any(metric == f.metric for f in figures)
    }
    unconfirmed = {label for label, _ in _table_rows(text, scale=scale)} - confirmed_labels
    rejected.extend(f"{label}: no sentence states its change" for label in sorted(unconfirmed))
    return figures, rejected


def _mentions(sentence_metric: str, row_label: str) -> bool:
    """Whether a prose subject and a table row name the same thing.

    "Adjusted EBITDA for the Nebius AI cloud business" against the row
    "Nebius AI cloud". Matching on the row's words inside the sentence is the
    direction that works: the sentence is always the longer of the two.
    """
    words = [w for w in re.findall(r"[a-z]+", row_label.lower()) if len(w) > 2]
    if not words:
        return False
    lowered = sentence_metric.lower()
    return all(w in lowered for w in words)


def _units(text: str) -> float:
    """The scale a table's own header declares. Defaults to millions."""
    if re.search(r"in\s+(?:thousands|millions|billions)\s+of\s+U\.?S\.?\s*dollars", text, re.I):
        word = re.search(
            r"in\s+(thousands|millions|billions)\s+of\s+U\.?S\.?\s*dollars", text, re.I
        )
        assert word is not None
        return {"thousands": 1e3, "millions": 1e6, "billions": 1e9}[word.group(1).lower()]
    return 1e6


def parse_interim(text: str) -> InterimResults:
    """Every figure the exhibit confirms twice, and why the rest were refused."""
    body = _normalise(text)
    figures, rejected = _shape_a(body)
    figures.extend(_shape_b(body))
    confirmed, refused = _confirmed_rows(body, scale=_units(body))
    figures.extend(confirmed)
    rejected.extend(refused)

    # A metric stated both ways keeps the self-checked reading.
    best: dict[tuple[str, Period], InterimFigure] = {}
    for fig in figures:
        key = (fig.metric.lower(), fig.period)
        held = best.get(key)
        if held is None or (
            held.basis is Basis.TABLE_CONFIRMED and fig.basis is Basis.SELF_CHECKED
        ):
            best[key] = fig
    return InterimResults(figures=list(best.values()), rejected=rejected)


# Exhibit types that carry interim results. EX-99.1 is the near-universal
# choice; a filer occasionally splits the review from the statements.
_RESULTS_EXHIBITS = ("EX-99.1", "EX-99.2", "EX-99")


def _parse_attachments(filing) -> InterimResults | None:
    """The best-parsing results exhibit on one filing, or None."""
    best: InterimResults | None = None
    for attachment in getattr(filing, "attachments", []) or []:
        if str(getattr(attachment, "document_type", "")).upper() not in _RESULTS_EXHIBITS:
            continue
        try:
            text = attachment.text()
        except Exception as exc:  # noqa: BLE001
            logger.info("interim: an exhibit would not render: %s", exc)
            continue
        if not text:
            continue
        parsed = parse_interim(text)
        # Several exhibits can parse; the one with the most confirmed figures
        # is the operating review rather than a press release or a covenant.
        if best is None or len(parsed.figures) > len(best.figures):
            best = parsed
    return best


def _same_day_siblings(filing) -> list:
    """The filer's other reports accepted the same day.

    Nebius files its quarter as two 6-Ks minutes apart: a press release with
    the shareholder letter and slides, and the financial report with the
    operating review. The classifier marks the press release as the results
    filing, and the press release states "revenue of $582.3 million, was up
    454% year-over-year" — a level and a rate with nothing to check them
    against. The figures that reconcile are in the sibling.
    """
    try:
        company = filing.get_entity() if hasattr(filing, "get_entity") else None
        if company is None:
            from edgar import Company

            company = Company(filing.cik)
        others = company.get_filings(form="6-K", filing_date=str(filing.filing_date))
    except Exception as exc:  # noqa: BLE001
        logger.info("interim: could not list same-day filings: %s", exc)
        return []
    return [f for f in others if str(f.accession_no) != str(filing.accession_no)]


def interim_for_accession(accession: str) -> InterimResults | None:
    """Parse the results exhibit of a 6-K, or None when there is nothing to read.

    The 6-K itself is a cover page — `filing.text()` returns a paragraph
    saying an exhibit is attached. The figures are in the exhibit, which is
    why a results filing reached the event stream as a headline and no number.
    """
    try:
        from edgar import get_by_accession_number

        from advisor.research.config import get_settings
        from advisor.research.edgar import _ensure_identity

        # EDGAR refuses an unidentified client, and the failure surfaces deep
        # inside a retry stack rather than as a clean error.
        _ensure_identity(get_settings().edgar_user_agent)
        filing = get_by_accession_number(accession)
    except Exception as exc:  # noqa: BLE001
        logger.info("interim: could not load %s: %s", accession, exc)
        return None
    if filing is None:
        return None

    best = _parse_attachments(filing)
    if best is not None and best.figures:
        return best.model_copy(update={"source_accession": accession})

    for sibling in _same_day_siblings(filing):
        parsed = _parse_attachments(sibling)
        if parsed is not None and parsed.figures:
            logger.info(
                "interim: %s carried no confirmable figures; read %s instead",
                accession,
                sibling.accession_no,
            )
            return parsed.model_copy(update={"source_accession": str(sibling.accession_no)})
    return best


def headline_figures(results: InterimResults) -> dict[str, float | str]:
    """The figures any issuer reports, named the same way for every filer.

    Deliberately small. "Total revenues" and a group-level adjusted EBITDA are
    lines every operating review carries; a segment is named by whatever the
    issuer calls it, so those stay in ``figures`` and are not promoted to a
    payload key that a thesis could come to rely on.
    """
    out: dict[str, float | str] = {}
    revenue = results.find(metric="total revenue") or results.find(metric="total revenues")
    if revenue is not None:
        out["revenue_usd"] = round(revenue.current, 2)
        out["revenue_prior_usd"] = round(revenue.prior, 2)
        out["revenue_period"] = revenue.period.value
        if revenue.growth is not None:
            out["revenue_growth_yoy"] = round(revenue.growth, 4)
    ebitda = results.find(metric="total adjusted ebitda")
    if ebitda is not None:
        out["adjusted_ebitda_usd"] = round(ebitda.current, 2)
        out["adjusted_ebitda_prior_usd"] = round(ebitda.prior, 2)
    return out
