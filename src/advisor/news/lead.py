"""One or two sentences saying what a filing or article is actually about.

An 8-K classified from its item codes says *that* something happened and
never *what*. AAOI filed three 8-Ks in fifteen days, each archived as
"entered a material definitive agreement": two Houston factory leases, the
purchase of its Blue Ridge property, and a ten-year factory lease in Ningbo.
Three different facts about capacity build-out, rendered as three identical
rows. The substance was in the item text the whole time; nothing read it.

The lead is **extractive, never generated**. It is the filer's or publisher's
own opening words, cleaned of boilerplate and cut to length, so it can be shown
beside a Tier A event without a model having to be trusted. Anything that
cannot be read yields None rather than a guess — a missing lead renders as the
classification alone, exactly as before.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from advisor.news.models import SourceItem

logger = logging.getLogger(__name__)

# One 8-K sentence routinely runs 350 characters and puts the number last: AAOI's
# ATM agreement reads "...having an aggregate offering price of up to $600
# million". At 320 the lead stopped at "up to…".
LEAD_CHARS = 400

# Standard 8-K item titles, stripped from the front of an item's text so the
# lead starts at the first word that says something. Matched word by word,
# because filers truncate them: AAOI's Item 2.03 reads "Creation of a Direct
# Financial Obligation or an Obligation The information ...".
ITEM_TITLES: dict[str, str] = {
    "1.01": "Entry into a Material Definitive Agreement",
    "1.02": "Termination of a Material Definitive Agreement",
    "1.03": "Bankruptcy or Receivership",
    "1.04": "Mine Safety - Reporting of Shutdowns and Patterns of Violations",
    "1.05": "Material Cybersecurity Incidents",
    "2.01": "Completion of Acquisition or Disposition of Assets",
    "2.02": "Results of Operations and Financial Condition",
    "2.03": "Creation of a Direct Financial Obligation or an Obligation under an "
    "Off-Balance Sheet Arrangement of a Registrant",
    "2.04": "Triggering Events That Accelerate or Increase a Direct Financial "
    "Obligation or an Obligation under an Off-Balance Sheet Arrangement",
    "2.05": "Costs Associated with Exit or Disposal Activities",
    "2.06": "Material Impairments",
    "3.01": "Notice of Delisting or Failure to Satisfy a Continued Listing Rule or "
    "Standard; Transfer of Listing",
    "3.02": "Unregistered Sales of Equity Securities",
    "3.03": "Material Modification to Rights of Security Holders",
    "4.01": "Changes in Registrant's Certifying Accountant",
    "4.02": "Non-Reliance on Previously Issued Financial Statements or a Related "
    "Audit Report or Completed Interim Review",
    "5.01": "Changes in Control of Registrant",
    "5.02": "Departure of Directors or Certain Officers; Election of Directors; "
    "Appointment of Certain Officers; Compensatory Arrangements of Certain Officers",
    "5.03": "Amendments to Articles of Incorporation or Bylaws; Change in Fiscal Year",
    "5.07": "Submission of Matters to a Vote of Security Holders",
    "7.01": "Regulation FD Disclosure",
    "8.01": "Other Events",
    "9.01": "Financial Statements and Exhibits",
}

# Items that never carry the substance: exhibits lists, and FD boilerplate is
# handled by preferring the attached press release.
_SKIP_ITEMS = {"9.01"}

# "The information set forth in Item 1.01 ... is incorporated by reference" —
# a pointer, not a disclosure.
_BY_REFERENCE = re.compile(r"incorporated (herein )?by reference", re.IGNORECASE)

# The item hands its substance to a press release. Any mention will do: an
# item that names Exhibit 99 is either quoting it or pointing at it, and the
# release's own opening is the better statement of the news either way.
_POINTS_TO_EXHIBIT = re.compile(r"exhibit 99", re.IGNORECASE)

# Sentences that are legal scaffolding, not disclosure. Cerebras's Item 2.02
# consisted of nothing else.
_BOILERPLATE = re.compile(
    r"shall not be deemed|is being furnished|incorporated (herein )?by reference|"
    r"forward[- ]looking statements|section 18 of the securities exchange act",
    re.IGNORECASE,
)

# A sub-heading run into the first sentence: "Notes Offering On June 22, ...",
# "(c) Appointment of Principal Officer On June 5, ...".
_SUBHEADING = re.compile(
    r"^(?:\([a-z]\)\s*)?[A-Z][\w&'’.,()-]*\s+(?:[\w&'’.,()-]+\s+){0,9}?(?=(?:On|As of|Effective) "
    r"[A-Z][a-z]+ \d{1,2}, \d{4})"
)

# The masthead of a press release: "NEWS RELEASE Contact: ... (512) 705-1720
# brandi.martina@amd.com". Everything up to the last contact detail goes.
_RELEASE_BANNER = re.compile(r"^(news release|press release|for immediate release)\s*", re.I)
_CONTACT = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+|\(\d{3}\)\s*\d{3}-\d{4}|\+?\d[\d .-]{8,}\d")

# Defined-term declarations: (the "Company"), ("Global Technology"),
# (each, a "Lease" and collectively, the "Leases"). They make a first sentence
# half again as long and add nothing a reader needs.
_DEFINED_TERM = re.compile(
    r"\s*\((?:[^()]{0,40}?)[\"“][^\"”()]{1,60}[\"”][^()]{0,80}\)",
)

# Legal detail that lengthens a sentence without informing it.
_PAR_VALUE = re.compile(r",?\s*par value \$[\d.,]+ per share,?", re.IGNORECASE)

# Scraped-page furniture that arrives inside article bodies.
_JUNK_LINE = re.compile(
    r"^(#|!\[|\[|logo\b|author'?s avatar|article'?s main image|advertisement|"
    r"sign (in|up)|subscribe|share this|image source)",
    re.IGNORECASE,
)

# Words after which a period does not end a sentence. Only matters before a
# capital: "Alphabet Inc. (NASDAQ", "Co. LLC", "No. 227". Suffixes written
# without periods (LLC, PLC, LP) are left out — mid-sentence they take a comma,
# so a period after one ends the sentence.
_ABBREVIATIONS = {
    "inc", "corp", "co", "ltd", "l.l.c", "n.v", "s.a", "l.p",
    "no", "nos", "mr", "ms", "mrs", "dr", "st", "rd", "ave", "u.s", "u.k", "e.g",
    "i.e", "vs", "approx", "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep",
    "sept", "oct", "nov", "dec", "calif", "mass", "wash", "tex", "fla", "ill",
}  # fmt: skip


def _squash(text: str) -> str:
    return " ".join((text or "").split())


def _is_abbreviation(token: str) -> bool:
    token = token.lower().rstrip(".")
    # Initials: "J.P.", "U.S.", a lone "A." in a name.
    return token in _ABBREVIATIONS or bool(re.fullmatch(r"([a-z]\.)*[a-z]", token))


def _split(text: str) -> list[str]:
    """Split prose on sentence ends, respecting corporate abbreviations and initials."""
    out: list[str] = []
    start = 0
    for match in re.finditer(r"[.!?][\"”’)]?\s+(?=[\"“(]?[A-Z0-9$])", text):
        word = re.search(r"([A-Za-z.]+)\.$", text[start : match.start() + 1])
        if word and _is_abbreviation(word.group(1)):
            continue
        out.append(text[start : match.end()].strip())
        start = match.end()
    tail = text[start:].strip()
    if tail:
        out.append(tail)
    return out


def _sentences(text: str) -> list[tuple[str, bool]]:
    """Sentences, each flagged True when it opens a bulleted point.

    A results release is a headline followed by "• Q2 Net Revenue: ...
    • Q2 Gross Margin: ...". Treating each bullet as a sentence lets the lead
    stop between points instead of cutting one mid-word.
    """
    out: list[tuple[str, bool]] = []
    for index, chunk in enumerate(re.split(r"\s*[•●▪]\s*", text)):
        for position, sentence in enumerate(_split(chunk.strip())):
            out.append((sentence, index > 0 and position == 0))
    return out


def trim(text: str | None, *, max_chars: int = LEAD_CHARS) -> str | None:
    """The first sentence or two of ``text`` that fit in ``max_chars``.

    Legal boilerplate sentences are dropped first. A first sentence longer
    than the budget is cut at a word boundary and marked with an ellipsis, so
    a truncated lead never reads as complete.
    """
    text = _PAR_VALUE.sub("", _DEFINED_TERM.sub("", text or ""))
    # "Ningbo, China. ("Leased Property")." loses its parenthetical and would
    # otherwise end "China..".
    text = re.sub(r"(?<!\.)\.\s*\.(?!\.)", ".", text)
    text = _SUBHEADING.sub("", _squash(text))
    sentences = [(s, b) for s, b in _sentences(text) if not _BOILERPLATE.search(s)]
    if not sentences:
        return None
    lead = ""
    for sentence, bullet in sentences:
        joiner = " · " if bullet else " "
        candidate = f"{lead}{joiner}{sentence}" if lead else sentence
        if len(candidate) > max_chars:
            break
        lead = candidate
    if not lead:
        cut = sentences[0][0][: max_chars - 1]
        cut = cut[: cut.rfind(" ")] if " " in cut else cut
        lead = cut.rstrip(" ,;:") + "…"
    return lead


def _word(token: str) -> str:
    return token.strip(".;:,-–—").lower()


def _strip_item_heading(code: str, text: str) -> str:
    """Remove "Item 1.01 Entry into a Material Definitive Agreement." from the front."""
    text = re.sub(r"^\s*item\s+\d+\.\d+\.?[\s.:–—-]*", "", _squash(text), flags=re.IGNORECASE)
    text = text.lstrip("-–—: ")
    title = ITEM_TITLES.get(code)
    if not title:
        return text
    words = text.split(" ")
    consumed = 0
    for expected in title.split(" "):
        if consumed < len(words) and _word(words[consumed]) == _word(expected):
            consumed += 1
        else:
            break
    if consumed >= 2:
        text = " ".join(words[consumed:])
    return text.lstrip(".;: ")


def _strip_exhibit_heading(text: str) -> str:
    text = re.sub(r"^\s*exhibit\s+99(\.\d+)?\s*", "", _squash(text), flags=re.IGNORECASE)
    text = _RELEASE_BANNER.sub("", text)
    # A contact block leads the release: drop through its last phone or email.
    contacts = [m for m in _CONTACT.finditer(text[:500])]
    if contacts:
        text = text[contacts[-1].end() :].lstrip(" ,;")
    return text


def exhibit_lead(text: str | None) -> str | None:
    """Lead of a press-release exhibit (a 6-K's EX-99, or an 8-K's)."""
    return trim(_strip_exhibit_heading(text or ""))


def article_lead(text: str | None) -> str | None:
    """Lead of a news article body, with scraped-page furniture removed.

    Tavily returns page text, not an abstract: a GuruFocus body opened with
    "Logo Logo # <headline> Author's Avatar Article's Main Image" before its
    first real sentence. Only lines long enough to be prose survive.
    """
    if not text:
        return None
    lines = []
    for raw in re.split(r"\n+", text):
        line = raw.strip()
        if len(line) < 40 or _JUNK_LINE.match(line):
            continue
        lines.append(line)
    body = re.sub(r"\[\.\.\.\]|\[…\]", " ", " ".join(lines))
    return trim(body)


def eight_k_lead(filing: Any, codes: list[str]) -> str | None:
    """What an 8-K discloses, from its first substantive item.

    An item that merely points at an exhibit ("a copy of the press release is
    furnished as Exhibit 99.1") yields to the exhibit's opening, which is where
    a results or FD filing states its news. An item that incorporates another
    by reference is skipped outright.
    """
    try:
        report = filing.obj()
    except Exception as exc:  # noqa: BLE001
        logger.debug("lead: could not parse %s: %s", getattr(filing, "accession_no", "?"), exc)
        return None

    for code in codes:
        if code in _SKIP_ITEMS:
            continue
        try:
            raw = report[f"Item {code}"]
        except Exception:  # noqa: BLE001
            raw = None
        body = _strip_item_heading(code, str(raw or ""))
        if not body or (_BY_REFERENCE.search(body) and len(body) < 400):
            continue
        if _POINTS_TO_EXHIBIT.search(body):
            from advisor.news.foreign import exhibit_text

            exhibit = exhibit_lead(exhibit_text(filing, max_chars=3000))
            if exhibit:
                return exhibit
        lead = trim(body)
        if lead:
            return lead

    # Only exhibits, or only boilerplate: a 9.01-only 8-K may still carry a
    # press release.
    from advisor.news.foreign import exhibit_text

    return exhibit_lead(exhibit_text(filing, max_chars=3000))


def lead_for(item: SourceItem) -> str | None:
    """The lead for an archived item, from whatever its adapter stored in ``summary``.

    What ``summary`` holds depends on the source, so the reading does too: a
    6-K keeps its raw EX-99 opening (the classifier reads it), an 8-K already
    holds its lead, and a news body is page text or a publisher's abstract. A
    25-NSE's summary is the delisted security class, which is evidence for the
    classifier and not a lead.
    """
    doc_type = (item.doc_type or "").upper()
    if not item.summary or doc_type == "25-NSE":
        return None
    if doc_type.startswith("6-K"):
        return exhibit_lead(item.summary)
    if doc_type.startswith("8-K"):
        return trim(item.summary)
    if doc_type == "NEWS":
        return article_lead(item.summary)
    return None
