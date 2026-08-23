import { Link, NavLink, Route, Routes } from "react-router-dom";
import { LineChart } from "lucide-react";
import Portfolio from "./pages/Portfolio";
import Scalping from "./pages/Scalping";
import SwingTrade from "./pages/SwingTrade";
import Ticker from "./pages/Ticker";
import Theses from "./pages/Theses";
import ThesisEditor from "./pages/ThesisEditor";
import Watchlist from "./pages/Watchlist";
import { useQuotes } from "./lib/useQuotes";
import { cn } from "./lib/utils";

const tabClass = ({ isActive }: { isActive: boolean }) =>
  cn(
    "rounded-md px-3 py-1.5 text-sm font-medium transition-colors",
    isActive ? "bg-panel-2 text-text" : "text-muted hover:text-text"
  );

export default function App() {
  // One shared quote stream for the whole app.
  const quotes = useQuotes();

  return (
    <div className="min-h-full">
      <header className="sticky top-0 z-20 border-b border-border bg-bg/90 backdrop-blur">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-4 py-3">
          <div className="flex items-center gap-4">
            <Link to="/" className="flex items-center gap-2 font-semibold">
              <LineChart className="h-5 w-5 text-accent" />
              <span>Advisor</span>
            </Link>
            <nav className="flex items-center gap-1">
              <NavLink to="/" end className={tabClass}>
                Portfolio
              </NavLink>
              <NavLink to="/watchlist" className={tabClass}>
                Watchlist
              </NavLink>
              <NavLink to="/theses" className={tabClass}>
                Theses
              </NavLink>
              <NavLink to="/scalping" className={tabClass}>
                Scalping
              </NavLink>
              <NavLink to="/swing" className={tabClass}>
                Swing
              </NavLink>
            </nav>
          </div>
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
          <Route path="/watchlist" element={<Watchlist quotes={quotes} />} />
          <Route path="/theses" element={<Theses />} />
          <Route path="/scalping" element={<Scalping />} />
          <Route path="/swing" element={<SwingTrade />} />
          <Route path="/theses/new" element={<ThesisEditor />} />
          <Route path="/theses/:id" element={<ThesisEditor />} />
          <Route path="/ticker/:symbol" element={<Ticker quotes={quotes} />} />
        </Routes>
      </main>
    </div>
  );
}
