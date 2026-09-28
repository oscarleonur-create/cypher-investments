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
| Data sources | free tier only — TastyTrade, SEC EDGAR, yfinance, Tavily (news mode), Google News RSS, the Nasdaq Trader trade-halts feed, hand-maintained macro calendar |
| Source trust | tier caps what an item may do: PRIMARY/BROKER may interrupt, AGGREGATOR reaches the digest, UNTAGGED is context and can never trigger |
| Entity matching | a match must be earned — CIK, provider tag, the company's names as headlines write them (`news.names`, a list the holder reviews: "SpaceX", not "space exploration"; "Coherent" only beside the ticker or a market word), cashtag, then a bare ticker; the registered name only for a symbol not on the list; short and common-word tickers (the book holds **TE** = T1 Energy) match on name only. One exception the holder vouches for: an **angle** (a product or segment the holder confirmed, e.g. Grok for SPCX) matches as `ALIAS`, the weakest method, capped at Tier C |
| News retrieval | pulled to explain something a free detector already found. Keywords, never sentences — measured, a full-sentence query scored 0.393 against 0.924 for keywords. **One scheduled pull** (user decision, 2026-09-24): the review job searches each held position's confirmed angles once a day, largest position first, within a hard budget of 20 Tavily queries — Grok 4.7 shipped and was never seen because nothing asked about it. **A second scheduled pull, for exits** (user decision, 2026-09-26): a distress sweep per held name at 08:15 and 12:30, weekends included (`bankruptcy default delisting`, `fraud investigation restructuring`); the model reads it and an exit-grade report from ≥2 independent outlets is an EXIT labeled unconfirmed, one outlet a REVIEW (`entry.distress`). **Primary sources act alone** (user decision, 2026-09-27): an exit-grade item on the company's own website is an EXIT; the sweep also runs one free Google News query per name, and press-release wires count as one outlet between them, because law firms' "investor alerts" run on the same wires. **Exchange halts** are polled from the Nasdaq Trader feed every 5 minutes, 04:00–20:05 on trading days: T12/H4/H9/H10/H11 on a held name is an EXIT while halted and a REVIEW once it resumes; T1/T6 (news pending) is a REVIEW until it resumes (`news.halts`). **News dates are checked before anything rests on them** (user decision, 2026-09-27: an entry is as grave as an exit): at ingestion each item's claimed date is compared with the publisher's own page; when the page cannot say, the same story is sought at other outlets and two agreeing within 24h corroborate it; otherwise it is UNVERIFIED and no reading, count or proposal may use it (`news.verify`). News events are dated at their checked publication. Polygon's free news was tried and dropped (2026-09-27): correct dates, but TE had nothing since July and its tags mean "mentioned", not "about" |
| Sentiment | **judged per item, context only** (user decision, 2026-09-27, replacing "not scored"). The news agent (`news/judge.py`) gives each archived item a direction *for the business* (not the headline's tone), materiality, novelty, basis and the thesis claims it supports or contradicts, quoting the item; the gate voids a judgment whose quote or numbers the item lacks. It reads only ingested items (no new searches) and changes no action: every judgment is scored against the price's excess over the name's own drift (`advisor news report`), and only a record that proves itself may earn a say. Filings are still classified by the SEC's own taxonomy |
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
  zero is named and left out. The snapshot's base case stays generic (25%
  FCF), so thesis rules keep testing the same kind of quantity — though since
  2026-09-27 it is discounted: at $148.75 SPCX's reads 44.0%, where the
  undiscounted arithmetic read 25.5% at the same price. Its "must not exceed
  25%" rule was written against the old number.
  Nothing margin-sensitive may size a position: the model's CONSTRUCTIVE
  stance, which leans on these rows, adds no risk.
- **Trailing figures are rebuilt from the filings.** A 10-Q's cash flow is
  year to date, so TTM is `FY + YTD − prior YTD`, or stated outright, or —
  across a fresh-start split like WOLF's 2025 bankruptcy exit — chained from
  contiguous periods. A period of odd length is annualised by its own length:
  WOLF's FY2026 10-K covers 272 days. A company too new for a fiscal year
  (SPCX, CBRS) is read over its longest reported span, labelled.
- **One engine prices everything (2026-09-27).** `valuation/dcf.py` is the only
  discounted-cash-flow code: the daemon's weekly valuation, the research
  workstation's DCF and reverse DCF, and the Bayesian Monte Carlo (a
  vectorised mirror pinned by a parity test) all run through it. Inputs come
  from `valuation/figures.py` — SEC filings for revenue, cash flow, margins,
  balance sheet and shares; the broker's close for price; Yahoo only where
  the SEC has nothing (IFRS filers), and labelled. Before this, the daemon did
  not discount (AMZN "required" -6%/yr) and the workstation took everything
  from `yfinance.info`, which when rate-limited defaulted shares to 1.0 and
  published JBL at $0.00.
- **Required growth is discounted.** At a stated 10% (the 10-year Treasury
  closed at 5.18% on 2026-09-24, plus a 4.8% premium) with 3% terminal
  growth, from trailing-twelve-month revenue. The rate is fixed so a thesis
  tests the same quantity week to week; it is the investor's hurdle, not a
  scenario, so it does not vary between bear and bull. Rows stored before the
  change are marked `method="undiscounted"` and never compared with new ones
  as a move.
- **A fair value is published only as a range, as an opinion (user decision,
  2026-09-27, replacing "never publish a fair value").** Bear / base / bull
  per share from the company's *own* filed margins — the lowest, median and
  highest of FCF TTM, three-year median FCF and operating margin after tax —
  with today's growth fading to terminal, ±25% (at least 3 points). Every
  assumption travels with the number. When the filings show fewer than two
  positive margins, or no year-over-year comparison, the range is refused
  with the reason — never built on a default. One reading cannot bracket the
  assumption that moves the answer most: COHR's lone 3.3% valued it at $0.00
  in all three cases, INTC's lone 5.0% at $5–$7 against $123. What the
  price *requires* stays the headline: it is arithmetic and falsifiable; the
  range is an opinion. The cases differ by how long today's growth lasts
  (held 0 / 3 / 5 years, then faded; inverted for a shrinking business) —
  not by multiplying it. The bull never assumes a margin the company has not
  filed: a price that needs one (AMZN 16.3% vs a best of 9.5%, 2026-09-28) is
  shown as the **market** case — what the price assumes — never used as a
  scenario, which would value the business at its price by construction.
- **One pricing view in the frontend (user decision, 2026-09-28).** The
  fair-price blend, DCF chart, Bayesian what-if, peer multiples, valuation-
  risk chart and upside columns were removed; the ticker page shows one
  `PriceRangeCard` (bear / base / bull / market + a deterministic rationale,
  `valuation/rationale.py`, `GET /api/daemon/symbol/{sym}/price-range`).
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
- **Every rule constant is versioned.** A constant added to a scanner or entry
  rule module must be declared in that package's `ruleset.py` with its kind —
  `threshold` (the learning loop may propose a change), `decided` (a limit the
  user set; never searched) or `model` (how something is measured) — or listed
  in `NOT_RULES` with the reason. `tests/test_learning/test_declared.py` fails
  otherwise. Every candidate and proposal carries the resulting `rules` stamp;
  `advisor learn rules` lists the versions and what each produced.
