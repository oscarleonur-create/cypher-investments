import { Link, Route, Routes } from "react-router-dom";
import { LineChart } from "lucide-react";
import Portfolio from "./pages/Portfolio";
import Ticker from "./pages/Ticker";
import { useQuotes } from "./lib/useQuotes";

export default function App() {
  // One shared quote stream for the whole app.
  const quotes = useQuotes();

  return (
    <div className="min-h-full">
      <header className="sticky top-0 z-20 border-b border-border bg-bg/90 backdrop-blur">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-4 py-3">
          <Link to="/" className="flex items-center gap-2 font-semibold">
            <LineChart className="h-5 w-5 text-accent" />
            <span>Advisor</span>
            <span className="text-muted font-normal">Portfolio</span>
          </Link>
          <div className="flex items-center gap-2 text-xs">
            <span
              className={`h-2 w-2 rounded-full ${
                quotes.connected ? "bg-pos" : "bg-neg"
              }`}
            />
            <span className="text-muted">
              {quotes.connected ? (quotes.live ? "Live" : "Connected") : "Offline"}
            </span>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-6xl px-4 py-5">
        <Routes>
          <Route path="/" element={<Portfolio quotes={quotes} />} />
          <Route path="/ticker/:symbol" element={<Ticker quotes={quotes} />} />
        </Routes>
      </main>
    </div>
  );
}
