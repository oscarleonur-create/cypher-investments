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
| Data sources | free tier only — TastyTrade, SEC EDGAR, yfinance, Tavily (news mode), hand-maintained macro calendar |
| Source trust | tier caps what an item may do: PRIMARY/BROKER may interrupt, AGGREGATOR reaches the digest, UNTAGGED is context and can never trigger |
| Entity matching | a match must be earned — CIK, provider tag, registered company name, cashtag, then a bare ticker; short and common-word tickers (the book holds **TE** = T1 Energy) match on name only |
| News retrieval | pulled to explain something a free detector already found, never polled. Keywords, never sentences — measured, a full-sentence query scored 0.393 against 0.924 for keywords |
| Sentiment | not scored. Classification comes from the SEC's own taxonomy (8-K items, form types); direction and thesis relevance wait for the thesis layer |
| Host | the user's Mac, market hours, with watermark catch-up on wake |
| Delivery | Telegram bot (two-way) |
| Universe | open positions (accounts 5WI30382, 5WI47366) + the `watchlist` table |
| Interrupts | only when there is a concrete action with a deadline |
| Hedging advice | flag the exposure *and* name the fix; do not stage orders |
| Exposure limits | agent proposes, user approves |

### Build phases

1. Daemon spine — scheduler, event stream, watermarks, heartbeats ✅
2. Ingest, in three parts, all shipped:
   - 2a position mechanics — edge-triggered crossings, concentration, drawdown ✅
   - 2b macro — nine-factor panel, ridge sensitivities, book exposure ✅
   - 2c external sources — tiered ingest, entity resolution, SEC classification,
     sized dilution, cross-source reconciliation ✅ (frontend ✅)
3. **Story assembler (deterministic)** — one anchor event plus position, price
   reaction, macro attribution and corroborating sources, assembled into the
   narrative for a ticker. No LLM.
4. Structured theses (drivers, invalidations, KPIs, macro drivers)
5. Relevance gate over both pillars, weighted by book impact **and** thesis
   relevance
6. Narration (LLM over the assembled story slots)
7. Telegram delivery + suppression
8. Outcome scoring per event type
9. Autonomy ladder

Both pillars — position mechanics and macro exposure — are first-class. This is
not an options tool with macro bolted on.

#### Why this order

The relevance gate moved after theses, and a deterministic story assembler
moved in front of both. The reasons are specific, and each came from live data
rather than from planning:

- **The story assembler needs nothing new.** Every slot but two is already in
  the store. Building it first makes the value of phases 4–6 concrete instead
  of theoretical, and it is the artefact Telegram will eventually send, so
  there is no separate message format to design later.
- **Building the gate before theses means building it twice.** Relevance is
  relative to something. Without a thesis the gate can only weight by book
  impact; with one it can ask whether an event touches a stated driver. TE
  (T1 Energy, 3% of net liq) generated more Tier A events in two months than
  AAOI and CRDO combined, because interrupt rate tracks corporate distress,
  not position size — book impact alone would still let a small distressed
  holding dominate attention.
- **Narration comes last because the skeleton must stand alone.** The
  assembled story is complete without a model; the LLM writes connective
  prose over filled slots and may use no number that is not already in one.
  `verification/grounding.py` is the gate. If the model is unavailable you
  still get the story, in tables.

#### Known gaps carried into phase 3

- **Position at event time.** The story currently reports the *current*
  position against a past event. `book_snapshots` only start 2026-09-04, so
  for older events there is nothing to read. The assembler must use the
  nearest snapshot at or before the event and say which date it used — never
  substitute today's silently.
- **Do not overstate the model.** A first prototype labelled AAOI's -13.77%
  session "idiosyncratic" from a residual z of -1.12, when the firing
  threshold is 2.0 and AAOI's residual vol (7.53%/day) makes that a routine
  day for it. Narrative confidence must not exceed what the statistic
  supports.

### Open decisions, unanswered

- **Threshold recalibration.** `-8%` stop and `-20%` drawdown do not
  discriminate on this book: 9 of 11 positions already exceed both. Proposed
  fix is per-position volatility-scaled thresholds. Awaiting the user.
- **A real market-session run.** Every run so far has been outside session
  hours, so the marked-price path has never executed live.
- **Options mechanics.** Unbuilt and unverifiable while the book holds none.

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
