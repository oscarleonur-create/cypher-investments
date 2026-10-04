import { Link, NavLink, Route, Routes } from "react-router-dom";
import { LineChart } from "lucide-react";
import Actions from "./pages/Actions";
import Picks from "./pages/Picks";
import Portfolio from "./pages/Portfolio";
import Signals from "./pages/Signals";
import Ticker from "./pages/Ticker";
import Theses from "./pages/Theses";
import ThesisEditor from "./pages/ThesisEditor";
import Tracking from "./pages/Tracking";
import Watchlist from "./pages/Watchlist";
import { useQuotes } from "./lib/useQuotes";
import { cn } from "./lib/utils";

const tabClass = ({ isActive }: { isActive: boolean }) =>
  cn(
    "shrink-0 whitespace-nowrap rounded-md px-3 py-1.5 text-sm font-medium transition-colors",
    isActive ? "bg-panel-2 text-text" : "text-muted hover:text-text"
  );

export default function App() {
  // One shared quote stream for the whole app.
  const quotes = useQuotes();

  return (
    <div className="min-h-full">
      <header className="sticky top-0 z-20 border-b border-border bg-bg/90 backdrop-blur">
        {/* Phone: logo + status on one row, tabs below at full width and
            scrollable. From sm up, one row as before. */}
        <div className="mx-auto flex max-w-6xl flex-wrap items-center justify-between gap-x-4 gap-y-2 px-4 py-3">
          <Link to="/" className="order-1 flex items-center gap-2 font-semibold">
            <LineChart className="h-5 w-5 text-accent" />
            <span>Advisor</span>
          </Link>
          <nav className="order-3 -mx-1 flex w-full items-center gap-1 overflow-x-auto px-1 sm:order-2 sm:mx-0 sm:w-auto sm:flex-1 sm:px-0">
            <NavLink to="/" end className={tabClass}>
              Portfolio
            </NavLink>
            <NavLink to="/actions" className={tabClass}>
              Actions
            </NavLink>
            <NavLink to="/signals" className={tabClass}>
              Signals
            </NavLink>
            <NavLink to="/watchlist" className={tabClass}>
              Watchlist
            </NavLink>
            <NavLink to="/picks" className={tabClass}>
              Picks
            </NavLink>
            <NavLink to="/tracking" className={tabClass}>
              Tracking
            </NavLink>
            <NavLink to="/theses" className={tabClass}>
              Theses
            </NavLink>
          </nav>
          <div className="order-2 flex items-center gap-2 text-xs sm:order-3">
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
          <Route path="/actions" element={<Actions />} />
          <Route path="/signals" element={<Signals />} />
          <Route path="/watchlist" element={<Watchlist quotes={quotes} />} />
          <Route path="/picks" element={<Picks />} />
          <Route path="/tracking" element={<Tracking quotes={quotes} />} />
          <Route path="/theses" element={<Theses />} />
          <Route path="/theses/new" element={<ThesisEditor />} />
          <Route path="/theses/:id" element={<ThesisEditor />} />
          <Route path="/ticker/:symbol" element={<Ticker quotes={quotes} />} />
        </Routes>
      </main>
    </div>
  );
}
