// The guided research skeleton pre-filled into a new thesis. Markdown so the
// author can freely restructure — the headings are a starting frame, not a form.

export function thesisTemplate(symbol?: string): string {
  const heading = symbol ? `# ${symbol.toUpperCase()} — ` : "# ";
  return `${heading}<one-line thesis>

## Summary / Verdict
What's the call (buy / watch / avoid), the core reason in 2–3 sentences, and your
conviction. Write this last; lead with the punchline.

## Business & Moat
What the company does, how it makes money, and why that's durable (or not):
competitive advantage, switching costs, network effects, pricing power.

## Bull / Base / Bear
- **Bull —** what has to go right, and what it's worth then.
- **Base —** the central case you actually expect.
- **Bear —** what breaks the story, and the downside if it does.

## Valuation
Current price vs. your fair value. Method (DCF, multiples, sum-of-parts), key
assumptions (growth, margins, multiple), and the implied upside/downside.

## Catalysts
Dated, concrete events that could re-rate the stock (earnings, product launches,
approvals, capital returns). What and when.

## Key Risks
The handful of things most likely to be wrong. Be specific and honest.

## Invalidation Triggers
What you'll watch that would *kill* the thesis. Make these falsifiable:
- [ ] e.g. gross margin falls below 55% for two quarters
- [ ] e.g. net retention drops under 110%
- [ ] e.g. key customer concentration rises above 30%

## Sources
- [ ] 10-K / 10-Q reviewed
- [ ] Latest earnings call / transcript
- Links, notes, and references go here.
`;
}
