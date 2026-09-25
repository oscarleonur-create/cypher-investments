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
| Entity matching | a match must be earned — CIK, provider tag, registered company name, cashtag, then a bare ticker; short and common-word tickers (the book holds **TE** = T1 Energy) match on name only. One exception the holder vouches for: an **angle** (a product or segment the holder confirmed, e.g. Grok for SPCX) matches as `ALIAS`, the weakest method, capped at Tier C |
| News retrieval | pulled to explain something a free detector already found. Keywords, never sentences — measured, a full-sentence query scored 0.393 against 0.924 for keywords. **One scheduled pull** (user decision, 2026-09-24): the review job searches each held position's confirmed angles once a day, largest position first, within a hard budget of 20 Tavily queries — Grok 4.7 shipped and was never seen because nothing asked about it |
| Sentiment | not scored. Classification comes from the SEC's own taxonomy (8-K items, form types); direction and thesis relevance wait for the thesis layer |
| Host | the user's Mac, market hours, with watermark catch-up on wake |
| Delivery | Telegram bot (two-way) |
| Universe | open positions (accounts 5WI30382, 5WI47366) + the `watchlist` table + the TastyTrade private watchlist **"Swing"** (user decision, 2026-09-25: names traded short-term that the user would also hold long). Research jobs (valuation, factor loadings, filings) cover all of it; a filing on a name not held is capped at tier B |
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
3. Story assembler (deterministic) — one anchor event plus position, price
   reaction, macro attribution and corroborating sources ✅
4. Structured theses — claims with triggers the event stream can test ✅
5. **Implied expectations** — what a price requires the business to deliver,
   recomputed from filings and price, monitored as a claim ✅
6. Relevance gate over both pillars, weighted by book impact **and** thesis
   relevance
7. Narration (LLM over the assembled story slots)
8. Telegram delivery + suppression
9. Outcome scoring per event type
10. Autonomy ladder

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

#### Why implied expectations came before the gate

Writing a real thesis for the book's largest position showed what was
missing. SPCX is 21% of net liq; its claims could say "the AI segment must
keep growing" but nothing could check the number, and "is the price fair" had
no home at all.

A discounted cash flow answers "what is this worth" with whatever assumptions
the author chose, and two analysts produce two numbers neither of which is
falsifiable. Running it backwards — *what growth would justify the price we
are being charged* — produces a number that is arithmetic rather than
judgement, and that becomes true or false quarter by quarter. That is exactly
the shape a thesis claim needs.

It also produced the fix for a bug phase 4 could not see: a claim triggered on
`RESIDUAL_DIVERGENCE` for SPCX was reported as machine-checked and can never
fire, because residual divergence needs a factor estimate and SPCX has 59
sessions against a 120-session floor. "Monitored" now means checkable *for
this symbol*.

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

#### Valuation rules

- **The entry zone is relative, by user decision (2026-09-25).** "In zone"
  means price-to-sales at or below the name's own two-year median, with
  revenue and shares as known on each day (SEC `companyconcept`). An
  absolute zone drawn from the generic 25x/25%-FCF scenario was tried and
  failed live: AMZN "required" -6.0% growth against a real FCF margin of
  -0.3%. The absolute requirement is shown only as context, at the
  company's own trailing margin, and "undefined" when that is negative.
- **Required growth is a range, never one number (user decision,
  2026-09-25).** It moves more with the assumed FCF margin than with anything
  else: META required -0.4%, +2.3% or +5.7% a year at its 3-year median
  (32.7%), the generic 25% and its trailing (18.0%) margins. The scorecard
  shows every margin the company offers, each labeled; a margin at or below
  zero is named and left out. The snapshot's base case stays generic, so
  thesis rules keep testing the same quantity. Nothing margin-sensitive may
  size a position: the model's CONSTRUCTIVE stance, which leans on these
  rows, adds no risk.
- **Never publish a fair value.** This project reports what a price requires,
  never what a business is worth. The first is arithmetic and falsifiable; the
  second is an opinion wearing a number.
- **XBRL is read undimensioned.** The same concept is tagged once per segment,
  instrument and class — SPCX's quarterly revenue appears sixteen times — so
  every extraction filters on the undimensioned fact. The one deliberate
  exception is share count, where the classes *are* the dimension and must be
  summed.
- **Balance figures are dated, never guessed.** A first version took the
  largest `LongTermDebt` value and read "Proceeds from debt and other
  financing obligations" ($51.8bn, a cash-flow line) as total debt ($39.4bn).
  Facts carry `period_key` of the form `instant_<date>`; use it.
- **Debt absent is not debt unparseable.** A filing that never mentions debt
  describes a debt-free company and zero is correct; one that mentions it but
  cannot be parsed must report the gap, because assuming zero overstates
  enterprise value in the flattering direction.

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
