# Trading Research Agent (web-first)

## Goal
Build a web-first research agent for trading idea generation and “buy-the-dip” analysis.
Inputs: ticker OR sector OR thesis text.
Outputs: evidence-backed Opportunity Cards (JSON + Markdown), plus persistence for “seen before”.

## Non-goals
- No brokerage/trade execution.
- No price prediction guarantees.
- Focus on evidence, not hype.

## Output contract (Opportunity Card)
Return both:
1) JSON (machine-readable)
2) Markdown (human-readable)

Fields:
- id (stable hash of input + date)
- input: {mode: ticker|sector|thesis, value: string}
- verdict: BUY_THE_DIP | WATCH | AVOID
- catalyst: {summary: string, date: YYYY-MM-DD}
- dip_type: TEMPORARY | STRUCTURAL | UNCLEAR
- bull_case: [string] (max 3)
- bear_case: [string] (max 3)
- key_metrics: {revenue_growth, margins, fcf, cash, debt, guidance_notes} (null if unknown)
- valuation_snapshot: {multiples, vs_history, vs_peers} (optional in MVP)
- risks: [string]
- invalidation: [string]
- sources: [{url, title, publisher, tier: 1|2|3, accessed_at, snippet}]

## Evidence rules
- Every non-trivial claim must map to at least one source in `sources`.
- Prefer Tier 1 (primary) sources when possible:
  Tier 1: company IR / filings / official releases
  Tier 2: reputable financial press / industry publications
  Tier 3: everything else (only as leads; must be corroborated)

## Engineering rules
- Provide a CLI entrypoint.
- Add caching + rate limiting for web fetch.
- Store runs + sources in SQLite.
- Tests must not hit the network: mock fetchers.
- Include `make test` and `make lint` (or documented equivalents).

## Definition of done for MVP
- `research_agent run --ticker <TICKER>` works end-to-end and writes:
  - `out/<run_id>.json`
  - `out/<run_id>.md`
  - SQLite records for run + sources
- `pytest` passes.
