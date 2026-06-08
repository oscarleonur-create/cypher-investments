"""Financials tab — 3-statement viewer + trend charts + quality charts.

Three views: Annual (from the cached research pipeline), Quarterly (fetched
on-demand from yfinance) and TTM (rolling 4-quarter sum for flow items).
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from advisor.dashboard import data as dashboard_data
from advisor.dashboard.theme import PALETTE, PLOTLY_TEMPLATE
from advisor.research.models import ResearchReport, StatementBundle

# yfinance row labels → display name. Multiple aliases per row; first one
# present wins. Quarterly data uses the same labels as annual data.
_INCOME_ROWS: list[tuple[list[str], str]] = [
    (["Total Revenue", "Revenue", "Operating Revenue"], "Revenue"),
    (["Cost Of Revenue", "Cost of Revenue", "Reconciled Cost Of Revenue"], "Cost of revenue"),
    (["Gross Profit"], "Gross profit"),
    (["Operating Expense", "Operating Expenses"], "Operating expenses"),
    (["Operating Income", "Total Operating Income As Reported"], "Operating income"),
    (["Interest Expense", "Interest Expense Non Operating"], "Interest expense"),
    (["Pretax Income", "Pre-Tax Income"], "Pretax income"),
    (["Tax Provision", "Income Tax Expense"], "Income tax"),
    (["Net Income", "Net Income Common Stockholders"], "Net income"),
    (["EBITDA", "Normalized EBITDA"], "EBITDA"),
    (["Diluted EPS"], "EPS (diluted)"),
    (["Diluted Average Shares"], "Shares (diluted)"),
]

_BALANCE_ROWS: list[tuple[list[str], str]] = [
    (["Cash And Cash Equivalents", "Cash"], "Cash & equivalents"),
    (["Other Short Term Investments", "Short Term Investments"], "ST investments"),
    (["Accounts Receivable", "Receivables"], "Accounts receivable"),
    (["Inventory"], "Inventory"),
    (["Current Assets", "Total Current Assets"], "Current assets"),
    (["Goodwill"], "Goodwill"),
    (["Other Intangible Assets", "Intangible Assets"], "Intangibles"),
    (["Total Assets"], "Total assets"),
    (["Accounts Payable", "Payables"], "Accounts payable"),
    (["Current Debt", "Short Long Term Debt"], "ST debt"),
    (["Current Liabilities", "Total Current Liabilities"], "Current liabilities"),
    (["Long Term Debt"], "LT debt"),
    (["Total Liabilities Net Minority Interest", "Total Liab"], "Total liabilities"),
    (["Stockholders Equity", "Total Stockholder Equity", "Common Stock Equity"], "Total equity"),
    (["Share Issued", "Ordinary Shares Number"], "Shares outstanding"),
]

_CASHFLOW_ROWS: list[tuple[list[str], str]] = [
    (["Operating Cash Flow", "Total Cash From Operating Activities"], "Operating CF"),
    (["Capital Expenditure", "Capital Expenditures"], "Capex"),
    (["Free Cash Flow"], "Free cash flow"),
    (["Investing Cash Flow", "Total Cashflows From Investing Activities"], "Investing CF"),
    (["Financing Cash Flow", "Total Cash From Financing Activities"], "Financing CF"),
    (["Cash Dividends Paid", "Dividends Paid"], "Dividends paid"),
    (["Repurchase Of Capital Stock", "Common Stock Repurchase"], "Share repurchases"),
    (["Changes In Cash", "Change In Cash"], "Net change in cash"),
]

# Balance-sheet rows are point-in-time (stocks); income & cashflow are flows
# (sum across a period). TTM rolls up flows; balance shows the most-recent
# snapshot.
_BALANCE_LABELS = {label for _, label in _BALANCE_ROWS}


def render_financials(report: ResearchReport) -> None:
    bundle = report.statements
    if bundle is None or not bundle.income:
        st.warning("No statement data available.")
        return

    view = st.radio(
        "View",
        options=["Annual", "Quarterly", "TTM"],
        index=0,
        horizontal=True,
        help=(
            "Annual = 5-yr 10-K data driving ratios + DCF. "
            "Quarterly + TTM are fetched on demand from yfinance for this tab only."
        ),
    )

    if view == "Annual":
        _render_annual(bundle)
        return

    frames = dashboard_data.load_quarterly_statements(report.symbol)
    if all(df.empty for df in frames.values()):
        st.warning(
            "No quarterly data available from yfinance for this ticker. "
            "Annual view is still populated."
        )
        return

    if view == "Quarterly":
        _render_periodic(frames, view="Quarterly")
    else:
        _render_periodic(frames, view="TTM")


# ── Annual view (unchanged from step 1) ─────────────────────────────────────


def _render_annual(bundle: StatementBundle) -> None:
    st.caption(f"Source: **{bundle.source}** · fetched {bundle.fetched_at:%Y-%m-%d %H:%M}")

    common_size = st.toggle(
        "Show income statement as % of revenue (common-size)",
        value=False,
        help="Divides each income-statement row by revenue for that period.",
    )

    _render_trend_chart(bundle)

    with st.expander("Income Statement", expanded=True):
        _render_income(bundle, common_size=common_size)
    with st.expander("Balance Sheet", expanded=False):
        _render_balance(bundle)
    with st.expander("Cash Flow Statement", expanded=False):
        _render_cashflow(bundle)

    st.divider()
    _render_quality_chart(bundle)


# ── Quarterly / TTM views ───────────────────────────────────────────────────


def _render_periodic(frames: dict[str, pd.DataFrame], view: str) -> None:
    st.caption(
        "Source: **yfinance** (live quarterly fetch · cached 1h). "
        "Quarters are labelled by calendar end-date — companies with non-calendar "
        "fiscal years (e.g. AAPL ends in September) will see the calendar quarter, "
        "not the fiscal one."
    )

    income_df = _periodic_df(frames["income"], _INCOME_ROWS, view)
    balance_df = _periodic_df(frames["balance"], _BALANCE_ROWS, view)
    cashflow_df = _periodic_df(frames["cashflow"], _CASHFLOW_ROWS, view)

    _render_quarterly_trend(frames["income"], frames["cashflow"], view)

    with st.expander(f"Income Statement — {view}", expanded=True):
        st.dataframe(_format_periodic(income_df), use_container_width=True)
    with st.expander(f"Balance Sheet — {view}", expanded=False):
        st.dataframe(_format_periodic(balance_df), use_container_width=True)
    with st.expander(f"Cash Flow Statement — {view}", expanded=False):
        st.dataframe(_format_periodic(cashflow_df), use_container_width=True)


def _periodic_df(
    raw: pd.DataFrame,
    rows: list[tuple[list[str], str]],
    view: str,
) -> pd.DataFrame:
    """Pivot the raw yfinance frame into our row order + quarter labels.

    For Quarterly: keep every period column.
    For TTM: roll up the most recent 4 columns. Income/cashflow rows are
      summed; balance-sheet rows are taken from the most recent column.
    """
    if raw is None or raw.empty:
        return pd.DataFrame()

    # Sort columns most-recent-first
    cols = sorted(raw.columns, reverse=True)
    raw = raw[cols]

    data: dict[str, list] = {}
    labels: list[str] = []

    if view == "Quarterly":
        labels = [_quarter_label(c) for c in cols]
        for aliases, display in rows:
            data[display] = [_first_present(raw, aliases, c) for c in cols]
    else:  # TTM
        last4 = cols[:4]
        labels = ["TTM", *[_quarter_label(c) for c in last4]]
        for aliases, display in rows:
            cell_values = [_first_present(raw, aliases, c) for c in last4]
            if display in _BALANCE_LABELS:
                # Stock variable — use most recent point
                ttm = cell_values[0] if cell_values else None
            else:
                # Flow variable — sum (skip Nones)
                clean = [v for v in cell_values if v is not None]
                ttm = sum(clean) if clean else None
            data[display] = [ttm, *cell_values]

    df = pd.DataFrame(data, index=labels).T
    return df


def _first_present(df: pd.DataFrame, aliases: list[str], col) -> float | None:
    """Return the value at (alias, col) for the first alias that exists. None otherwise."""
    for alias in aliases:
        if alias in df.index:
            v = df.at[alias, col]
            if pd.isna(v):
                return None
            try:
                return float(v)
            except (TypeError, ValueError):
                return None
    return None


def _quarter_label(ts) -> str:  # type: ignore[no-untyped-def]
    """Convert a period-end timestamp to a 'Q1 \\'26'-style label."""
    try:
        d = ts.date() if hasattr(ts, "date") else ts
        q = (d.month - 1) // 3 + 1
        return f"Q{q} '{d.year % 100:02d}"
    except Exception:  # noqa: BLE001
        return str(ts)[:10]


def _format_periodic(df: pd.DataFrame) -> pd.DataFrame:
    """Pretty-format numbers in the periodic DataFrame."""
    if df.empty:
        return df

    def _fmt(row: str, v) -> str:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return "—"
        if row in {"EPS (diluted)"}:
            return f"{v:.2f}"
        abs_v = abs(v)
        if abs_v >= 1e9:
            return f"{v / 1e9:,.2f}B"
        if abs_v >= 1e6:
            return f"{v / 1e6:,.1f}M"
        if abs_v >= 1e3:
            return f"{v / 1e3:,.1f}K"
        return f"{v:,.2f}" if abs_v < 10 else f"{v:,.0f}"

    out = df.copy()
    for row in out.index:
        out.loc[row] = [_fmt(row, v) for v in out.loc[row]]
    return out


def _render_quarterly_trend(
    income_df: pd.DataFrame,
    cashflow_df: pd.DataFrame,
    view: str,
) -> None:
    if income_df is None or income_df.empty:
        return
    cols = sorted(income_df.columns)
    labels = [_quarter_label(c) for c in cols]

    revenue = [_first_present(income_df, ["Total Revenue", "Revenue"], c) for c in cols]
    operating = [
        _first_present(income_df, ["Operating Income", "Total Operating Income As Reported"], c)
        for c in cols
    ]
    net_income = [_first_present(income_df, ["Net Income"], c) for c in cols]
    fcf = []
    if cashflow_df is not None and not cashflow_df.empty:
        cf_cols = sorted(cashflow_df.columns)
        fcf_labels = [_quarter_label(c) for c in cf_cols]
        fcf = [_first_present(cashflow_df, ["Free Cash Flow"], c) for c in cf_cols]
    else:
        fcf_labels = []

    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=revenue, name="Revenue", marker_color=PALETTE["accent"]))
    fig.add_trace(
        go.Scatter(
            x=labels,
            y=operating,
            mode="lines+markers",
            name="Operating income",
            line=dict(color=PALETTE["warn"]),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=labels,
            y=net_income,
            mode="lines+markers",
            name="Net income",
            line=dict(color=PALETTE["positive"]),
        )
    )
    if fcf:
        fig.add_trace(
            go.Scatter(
                x=fcf_labels,
                y=fcf,
                mode="lines+markers",
                name="Free cash flow",
                line=dict(color="#a78bfa", dash="dot"),
            )
        )

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=f"Quarterly trend ({view})",
        height=380,
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", y=-0.2),
        hovermode="x unified",
        barmode="group",
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Statement tables ─────────────────────────────────────────────────────────


def _render_income(bundle: StatementBundle, common_size: bool) -> None:
    fields = [
        ("revenue", "Revenue"),
        ("cost_of_revenue", "Cost of revenue"),
        ("gross_profit", "Gross profit"),
        ("operating_expenses", "Operating expenses"),
        ("operating_income", "Operating income"),
        ("interest_expense", "Interest expense"),
        ("pretax_income", "Pretax income"),
        ("income_tax", "Income tax"),
        ("net_income", "Net income"),
        ("ebitda", "EBITDA"),
        ("eps_diluted", "EPS (diluted)"),
        ("shares_diluted", "Shares (diluted)"),
    ]
    df = _to_df(bundle.income, fields)
    if common_size:
        df = _common_size(df, "Revenue")
    st.dataframe(_format_df(df, eps_rows={"EPS (diluted)"}), use_container_width=True)


def _render_balance(bundle: StatementBundle) -> None:
    fields = [
        ("cash_and_equivalents", "Cash & equivalents"),
        ("short_term_investments", "ST investments"),
        ("accounts_receivable", "Accounts receivable"),
        ("inventory", "Inventory"),
        ("current_assets", "Current assets"),
        ("goodwill", "Goodwill"),
        ("intangibles", "Intangibles"),
        ("total_assets", "Total assets"),
        ("accounts_payable", "Accounts payable"),
        ("short_term_debt", "ST debt"),
        ("current_liabilities", "Current liabilities"),
        ("long_term_debt", "LT debt"),
        ("total_liabilities", "Total liabilities"),
        ("total_equity", "Total equity"),
        ("shares_outstanding", "Shares outstanding"),
    ]
    df = _to_df(bundle.balance, fields)
    st.dataframe(_format_df(df), use_container_width=True)


def _render_cashflow(bundle: StatementBundle) -> None:
    fields = [
        ("operating_cash_flow", "Operating CF"),
        ("capex", "Capex"),
        ("free_cash_flow", "Free cash flow"),
        ("investing_cash_flow", "Investing CF"),
        ("financing_cash_flow", "Financing CF"),
        ("dividends_paid", "Dividends paid"),
        ("share_repurchases", "Share repurchases"),
        ("net_change_in_cash", "Net change in cash"),
    ]
    df = _to_df(bundle.cashflow, fields)
    st.dataframe(_format_df(df), use_container_width=True)


# ── Charts ──────────────────────────────────────────────────────────────────


def _render_trend_chart(bundle: StatementBundle) -> None:
    years = [p.fiscal_year for p in bundle.income]
    if not years:
        return
    # Income arrays are most-recent-first; reverse for chronological x-axis
    years_chrono = list(reversed(years))

    revenue = [p.revenue for p in reversed(bundle.income)]
    operating = [p.operating_income for p in reversed(bundle.income)]
    net_income = [p.net_income for p in reversed(bundle.income)]
    fcf = [p.free_cash_flow for p in reversed(bundle.cashflow)] if bundle.cashflow else []
    fcf_years = [p.fiscal_year for p in reversed(bundle.cashflow)] if bundle.cashflow else []

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=years_chrono,
            y=revenue,
            mode="lines+markers",
            name="Revenue",
            line=dict(color=PALETTE["accent"], width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=years_chrono,
            y=operating,
            mode="lines+markers",
            name="Operating income",
            line=dict(color=PALETTE["warn"]),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=years_chrono,
            y=net_income,
            mode="lines+markers",
            name="Net income",
            line=dict(color=PALETTE["positive"]),
        )
    )
    if fcf:
        fig.add_trace(
            go.Scatter(
                x=fcf_years,
                y=fcf,
                mode="lines+markers",
                name="Free cash flow",
                line=dict(color="#a78bfa", dash="dot"),
            )
        )

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title="Revenue · Operating income · Net income · Free cash flow",
        height=380,
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", y=-0.2),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_quality_chart(bundle: StatementBundle) -> None:
    if not bundle.cashflow or not bundle.income:
        return

    # Build aligned year → metric dict
    inc_by_year = {p.fiscal_year: p for p in bundle.income}
    cf_by_year = {p.fiscal_year: p for p in bundle.cashflow}
    bs_by_year = {p.fiscal_year: p for p in bundle.balance}

    years = sorted(set(inc_by_year) & set(cf_by_year))
    if not years:
        return

    fcf_ni = []
    shares = []
    for y in years:
        inc = inc_by_year[y]
        cf = cf_by_year[y]
        if inc.net_income and cf.free_cash_flow:
            fcf_ni.append(cf.free_cash_flow / inc.net_income)
        else:
            fcf_ni.append(None)
        bs = bs_by_year.get(y)
        shares.append(bs.shares_outstanding if bs and bs.shares_outstanding else None)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=years,
            y=fcf_ni,
            mode="lines+markers",
            name="FCF / Net income",
            line=dict(color=PALETTE["positive"], width=3),
            yaxis="y",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=years,
            y=shares,
            mode="lines+markers",
            name="Shares outstanding",
            line=dict(color=PALETTE["neutral"], dash="dot"),
            yaxis="y2",
        )
    )
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title="Cash quality · Dilution",
        height=320,
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", y=-0.2),
        yaxis=dict(title="FCF / NI"),
        yaxis2=dict(title="Shares outstanding", overlaying="y", side="right"),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)


# ── DataFrame helpers ───────────────────────────────────────────────────────


def _to_df(periods, fields: list[tuple[str, str]]) -> pd.DataFrame:
    if not periods:
        return pd.DataFrame()
    cols = [str(p.fiscal_year) for p in periods]
    data = {}
    for attr, label in fields:
        data[label] = [getattr(p, attr, None) for p in periods]
    return pd.DataFrame(data, index=cols).T


def _common_size(df: pd.DataFrame, base_row: str) -> pd.DataFrame:
    if base_row not in df.index:
        return df
    base = df.loc[base_row]
    out = df.div(base, axis=1)
    return out


def _format_df(df: pd.DataFrame, eps_rows: set[str] | None = None) -> pd.DataFrame:
    eps_rows = eps_rows or set()
    if df.empty:
        return df

    def _fmt(row_name: str, v):
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return "—"
        if row_name in eps_rows:
            return f"{v:.2f}"
        abs_v = abs(v)
        if abs_v >= 1e9:
            return f"{v / 1e9:,.2f}B"
        if abs_v >= 1e6:
            return f"{v / 1e6:,.1f}M"
        if abs_v >= 1e3:
            return f"{v / 1e3:,.1f}K"
        if abs_v < 10 and abs_v > 0:
            return f"{v:.2%}" if abs_v < 1 else f"{v:.2f}"
        return f"{v:,.0f}"

    out = df.copy()
    for row in out.index:
        out.loc[row] = [_fmt(row, v) for v in out.loc[row]]
    return out
