"""CLI commands for data management."""

from __future__ import annotations

from datetime import date
from typing import Annotated, Optional

import typer

from advisor.cli.formatters import console, output_error, output_json

app = typer.Typer(name="data", help="Fetch and inspect market data")


@app.command("fetch")
def data_fetch(
    symbol: Annotated[str, typer.Argument(help="Ticker symbol")],
    start: Annotated[str, typer.Option("--start", help="Start date (YYYY-MM-DD)")],
    end: Annotated[str, typer.Option("--end", help="End date (YYYY-MM-DD)")],
    output: Annotated[Optional[str], typer.Option("--output", help="Output format")] = None,
) -> None:
    """Fetch historical stock data."""
    from advisor.data.yahoo import YahooDataProvider

    try:
        start_date = date.fromisoformat(start)
        end_date = date.fromisoformat(end)
    except ValueError as e:
        output_error(f"Invalid date format: {e}")
        return

    try:
        provider = YahooDataProvider()
        df = provider.get_stock_history(symbol, start_date, end_date)
    except Exception as e:
        output_error(f"Failed to fetch data: {e}")
        return

    if output == "json":
        records = df.reset_index().to_dict(orient="records")
        output_json(
            {
                "symbol": symbol,
                "start": str(start_date),
                "end": str(end_date),
                "rows": len(df),
                "data": records,
            }
        )
    else:
        console.print(f"[green]Fetched {len(df)} rows for {symbol}[/green]")
        console.print(f"Period: {df.index[0].date()} to {df.index[-1].date()}")
        console.print(df.head(10).to_string())
        if len(df) > 10:
            console.print(f"... ({len(df) - 10} more rows)")


@app.command("options")
def data_options(
    symbol: Annotated[str, typer.Argument(help="Ticker symbol")],
    expiration: Annotated[
        Optional[str], typer.Option("--expiration", help="Expiration date")
    ] = None,
    output: Annotated[Optional[str], typer.Option("--output", help="Output format")] = None,
) -> None:
    """Fetch current options chain for a symbol."""
    from advisor.data.yahoo import YahooDataProvider

    try:
        provider = YahooDataProvider()
        exp_date = date.fromisoformat(expiration) if expiration else None
        chain = provider.get_options_chain(symbol, exp_date)
    except Exception as e:
        output_error(f"Failed to fetch options: {e}")
        return

    if output == "json":
        output_json(
            {
                "symbol": symbol,
                "calls": chain["calls"].to_dict(orient="records"),
                "puts": chain["puts"].to_dict(orient="records"),
            }
        )
    else:
        console.print(f"[cyan]Calls ({len(chain['calls'])}):[/cyan]")
        console.print(chain["calls"].head(10).to_string())
        console.print(f"\n[cyan]Puts ({len(chain['puts'])}):[/cyan]")
        console.print(chain["puts"].head(10).to_string())


@app.command("inspect")
def data_inspect(
    symbol: Annotated[str, typer.Argument(help="Ticker symbol")],
    output: Annotated[Optional[str], typer.Option("--output", help="Output format")] = None,
) -> None:
    """Inspect ticker metadata."""
    from advisor.data.yahoo import YahooDataProvider

    try:
        provider = YahooDataProvider()
        info = provider.get_ticker_info(symbol)
    except Exception as e:
        output_error(f"Failed to fetch info: {e}")
        return

    # Select key fields for display
    key_fields = [
        "shortName",
        "sector",
        "industry",
        "marketCap",
        "currentPrice",
        "fiftyTwoWeekHigh",
        "fiftyTwoWeekLow",
        "dividendYield",
        "beta",
        "trailingPE",
    ]
    summary = {k: info.get(k) for k in key_fields if info.get(k) is not None}

    if output == "json":
        output_json(info)
    else:
        from rich.table import Table

        table = Table(title=f"Ticker Info: {symbol}")
        table.add_column("Field", style="cyan")
        table.add_column("Value")
        for k, v in summary.items():
            table.add_row(k, str(v))
        console.print(table)


@app.command("cache")
def data_cache(
    clear: Annotated[bool, typer.Option("--clear", help="Clear the cache")] = False,
    output: Annotated[Optional[str], typer.Option("--output", help="Output format")] = None,
) -> None:
    """Manage the data cache."""
    from advisor.data.cache import DiskCache

    cache = DiskCache()

    if clear:
        count = cache.clear()
        msg = f"Cleared {count} cached files"
        if output == "json":
            output_json({"message": msg, "files_removed": count})
        else:
            console.print(f"[green]{msg}[/green]")
    else:
        # Show cache stats
        cache_dir = cache.cache_dir
        files = list(cache_dir.glob("*"))
        total_size = sum(f.stat().st_size for f in files if f.is_file())
        stats = {
            "cache_dir": str(cache_dir),
            "files": len(files),
            "total_size_mb": round(total_size / (1024 * 1024), 2),
        }
        if output == "json":
            output_json(stats)
        else:
            console.print(f"Cache dir: {cache_dir}")
            console.print(f"Files: {len(files)}")
            console.print(f"Total size: {stats['total_size_mb']} MB")
