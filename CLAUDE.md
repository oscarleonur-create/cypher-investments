# CLAUDE.md

Project instructions for Claude Code. These override default behaviour.

---

## Merge policy — non-negotiable

**No feature is merged without live testing and corner-case testing.**

Unit tests passing is *necessary and not sufficient*. This system moves real
money in two live brokerage accounts. A green test suite has never once proved
that code works against the real world — in this repo, unit tests passed while
`get_regime()` could not run at all (an undeclared `hmmlearn` import), and while
the daemon scheduler mis-read every timestamp by an hour.

Before a PR is opened, both of the following must be done and their **evidence
pasted into the PR body**:

### 1. Live testing

Run the code against the real system — real database, real API, real CLI, real
market calendar. Not mocks. Not "the tests cover it."

- Execute the actual command or endpoint a user would hit, and paste the output.
- Confirm it works on the **installed dependency set**, not a stale venv:
  `poetry sync` after any `pyproject.toml` change, then re-run.
- Confirm existing data is intact — this repo shares one `data/research.db`
  across modules. Show row counts for tables the change touches.
- If the change cannot be exercised live (needs market hours, a filing that
  hasn't happened, a position you don't hold), say so explicitly in the PR and
  describe the closest real-world exercise you *did* run.

### 2. Corner-case testing

Write tests for the failure modes, not just the happy path. For this project
the recurring ones are:

| Class | Examples |
|---|---|
| **Time** | market closed, weekend, holiday, early close, DST transition, timezone mismatch between storage and logic, laptop asleep across a scheduled slot |
| **Empty / absent** | no positions, empty watchlist, no cached report, missing watermark, symbol with no options chain, zero-volume bar |
| **Broker & network** | API timeout, auth expiry, rate limit, partial response, account with zero net liq |
| **Numeric** | division by zero, negative prices, `None` where a float is expected, zero DTE, zero-width bid/ask |
| **Idempotency** | the same event ingested twice, two pollers racing, a job re-run after a crash |
| **Boundaries** | exactly at a threshold (21 DTE, the 09:30 bell, a strike precisely at the money) |

A corner case that is *deliberately* out of scope is fine — say so in the PR
and state what happens if it occurs.

### What not to do

- Do not report a feature as working when only unit tests were run.
- Do not merge your own PR without the user's explicit go-ahead.
- Do not claim verification you did not perform. If a step was skipped, say
  which and why.

---

## Project

An always-on portfolio advisor. A daemon watches held positions and the book's
macro exposure, evaluates events against stated theses, and pushes actionable
advice. The unit of monitoring is the **thesis**, not the ticker.

### Design constraints already decided

| Dimension | Choice |
|---|---|
| Cadence | daily digests + rare event-driven interrupts |
| Data sources | free tier only — TastyTrade, SEC EDGAR, yfinance, hand-maintained macro calendar |
| Host | the user's Mac, market hours, with watermark catch-up on wake |
| Delivery | Telegram bot (two-way) |
| Universe | open positions (accounts 5WI30382, 5WI47366) + the `watchlist` table |
| Interrupts | only when there is a concrete action with a deadline |
| Hedging advice | flag the exposure *and* name the fix; do not stage orders |
| Exposure limits | agent proposes, user approves |

### Build phases

1. Daemon spine — scheduler, event stream, watermarks, heartbeats ✅
2. Ingest → position mechanics **and** the factor/exposure model
3. Relevance gate over both pillars
4. Structured theses (drivers, invalidations, KPIs, macro drivers)
5. Action Cards (LLM)
6. Telegram delivery + suppression
7. Outcome scoring per event type
8. Autonomy ladder

Both pillars — position mechanics and macro exposure — are first-class. This is
not an options tool with macro bolted on.

### Conventions

- `advisor` CLI (Typer), entry point `src/advisor/cli/app.py`. Every command
  supports `--output json`.
- One SQLite file, `data/research.db`. Each module owns its own tables; no
  cross-module foreign keys.
- Daemon timestamps are timezone-aware in `America/New_York`. Never use naive
  `datetime.now()` inside `advisor/daemon/`.
- `poetry run pytest` and `poetry run ruff check src tests` must both be clean.
- Branch per feature: `feature/<slug>`, off `main`.
- `market_calendar.py` holidays and early closes are hardcoded through 2027 and
  need an annual refresh.
