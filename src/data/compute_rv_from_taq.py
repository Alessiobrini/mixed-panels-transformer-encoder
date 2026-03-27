"""
Compute daily realized variance measures from TAQ tick data on WRDS.

For each stock-day, downloads intraday trades from taqm_YYYY.ctm_YYYY,
resamples to 5-minute intervals (last price), then computes:
  - rv5:  realized variance (sum of squared 5-min log returns)
  - bv5:  bipower variation (Barndorff-Nielsen & Shephard 2004)
  - rsp5: realized semivariance, positive component
  - rsn5: realized semivariance, negative component
  - rk:   realized kernel (Parzen weights, Barndorff-Nielsen et al. 2008)

Market hours: 9:30–16:00 ET (78 five-minute intervals per day).

Output:
    data/raw/equity/taq_rv.parquet

Usage:
    python src/data/compute_rv_from_taq.py                    # all years
    python src/data/compute_rv_from_taq.py --years 2020 2021  # specific years
    python src/data/compute_rv_from_taq.py --resume            # resume from checkpoint
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

OUTPUT_DIR = project_root / "data" / "raw" / "equity"
CHECKPOINT_DIR = OUTPUT_DIR / "_taq_checkpoints"

MARKET_OPEN = "09:30:00"
MARKET_CLOSE = "16:00:00"
INTERVAL_MINUTES = 5
N_INTERVALS = 78  # 390 min / 5 min


def get_wrds_url() -> str:
    """Build WRDS connection URL from pgpass file or environment."""
    url = os.environ.get("WRDS_URL")
    if url:
        return url

    pgpass_path = Path.home() / "AppData" / "Roaming" / "postgresql" / "pgpass.conf"
    if not pgpass_path.exists():
        pgpass_path = Path.home() / ".pgpass"

    if pgpass_path.exists():
        for line in pgpass_path.read_text().strip().splitlines():
            parts = line.strip().split(":")
            if len(parts) >= 5 and "wrds" in parts[0]:
                host, port, db, user, pw = (
                    parts[0], parts[1], parts[2], parts[3], ":".join(parts[4:]),
                )
                return f"postgresql://{user}:{pw}@{host}:{port}/{db}?sslmode=require"

    raise RuntimeError(
        "No WRDS credentials found. Set WRDS_URL env var or create a pgpass file."
    )


# ---------------------------------------------------------------------------
# Realized variance computation
# ---------------------------------------------------------------------------
def compute_rv_measures(log_returns: np.ndarray) -> dict:
    """Compute realized variance measures from 5-min log returns.

    Args:
        log_returns: array of 5-min log returns for one stock-day.

    Returns:
        dict with rv5, bv5, rsp5, rsn5, rk.
    """
    r = log_returns
    n = len(r)

    if n < 2:
        return {"rv5": np.nan, "bv5": np.nan, "rsp5": np.nan, "rsn5": np.nan, "rk": np.nan}

    # RV5: realized variance
    rv5 = np.sum(r ** 2)

    # BV5: bipower variation (Barndorff-Nielsen & Shephard 2004)
    # BV = (pi/2) * sum(|r_j| * |r_{j-1}|) for j=2..n
    bv5 = (np.pi / 2) * np.sum(np.abs(r[1:]) * np.abs(r[:-1]))

    # RSP5 / RSN5: realized semivariance (positive/negative)
    rsp5 = np.sum(r[r > 0] ** 2)
    rsn5 = np.sum(r[r < 0] ** 2)

    # RK: realized kernel with Parzen weight function
    # Following Barndorff-Nielsen, Hansen, Lunde, Shephard (2008)
    rk = _realized_kernel_parzen(r)

    return {"rv5": rv5, "bv5": bv5, "rsp5": rsp5, "rsn5": rsn5, "rk": rk}


def _realized_kernel_parzen(r: np.ndarray) -> float:
    """Compute realized kernel with Parzen weight function.

    Bandwidth H is selected following the rule of thumb in
    Barndorff-Nielsen et al. (2009): H = c * n^(3/5) with c ≈ 0.4.
    """
    n = len(r)
    if n < 3:
        return np.sum(r ** 2)

    # Bandwidth
    H = max(1, int(0.4 * n ** 0.6))

    # Autocovariance at lag 0
    gamma_0 = np.sum(r ** 2)

    # Autocovariances at lags 1..H
    rk_val = gamma_0
    for h in range(1, H + 1):
        gamma_h = np.sum(r[h:] * r[:-h])
        weight = _parzen_weight(h / (H + 1))
        rk_val += 2 * weight * gamma_h

    return max(rk_val, 0.0)  # ensure non-negative


def _parzen_weight(x: float) -> float:
    """Parzen kernel weight function."""
    x = abs(x)
    if x <= 0.5:
        return 1 - 6 * x ** 2 + 6 * x ** 3
    elif x <= 1.0:
        return 2 * (1 - x) ** 3
    return 0.0


# ---------------------------------------------------------------------------
# TAQ data processing
# ---------------------------------------------------------------------------
def get_trading_dates_for_year(conn, year: int) -> list[str]:
    """Get all unique trading dates for a given year from the yearly ctm table."""
    table = f"taqm_{year}.ctm_{year}"
    query = text(f"""
        SELECT DISTINCT date
        FROM {table}
        ORDER BY date
    """)
    df = pd.read_sql(query, conn)
    return df["date"].astype(str).tolist()


def download_day_trades(
    conn, year: int, date: str, tickers: list[str]
) -> pd.DataFrame:
    """Download intraday trades for given tickers on a single day.

    Filters:
      - Regular trades only (tr_corr = '00')
      - Positive price
      - Market hours (9:30-16:00)
    """
    table = f"taqm_{year}.ctm_{year}"
    ticker_str = ", ".join(f"'{t}'" for t in tickers)

    query = text(f"""
        SELECT date, time_m, sym_root AS ticker, price
        FROM {table}
        WHERE date = :date
          AND sym_root IN ({ticker_str})
          AND tr_corr = '00'
          AND price > 0
          AND time_m >= :market_open
          AND time_m < :market_close
        ORDER BY sym_root, time_m
    """)
    return pd.read_sql(
        query, conn,
        params={
            "date": date,
            "market_open": MARKET_OPEN,
            "market_close": MARKET_CLOSE,
        },
    )


def resample_to_5min(trades_day: pd.DataFrame, ticker: str) -> np.ndarray:
    """Resample tick data to 5-minute last-price, compute log returns.

    Returns array of 5-min log returns.
    """
    stock = trades_day[trades_day["ticker"] == ticker].copy()
    if stock.empty:
        return np.array([])

    # Combine date + time_m into a single datetime
    # time_m comes as datetime.time objects from WRDS
    date_val = pd.Timestamp(stock["date"].iloc[0])
    stock["timestamp"] = stock["time_m"].apply(
        lambda t: date_val.replace(
            hour=t.hour, minute=t.minute, second=t.second,
            microsecond=t.microsecond,
        )
    )

    # Create 5-minute bins from 9:30 to 16:00
    date_str = date_val.strftime("%Y-%m-%d")
    bins = pd.date_range(
        start=f"{date_str} {MARKET_OPEN}",
        end=f"{date_str} {MARKET_CLOSE}",
        freq=f"{INTERVAL_MINUTES}min",
    )

    # Assign each trade to a bin
    stock["bin"] = pd.cut(
        stock["timestamp"], bins=bins, right=False, labels=bins[:-1]
    )
    stock = stock.dropna(subset=["bin"])

    if stock.empty:
        return np.array([])

    # Last price in each 5-min bin
    last_price = stock.groupby("bin", observed=False)["price"].last()

    # Forward-fill missing bins, then compute log returns
    last_price = last_price.reindex(bins[:-1]).ffill().dropna()

    if len(last_price) < 2:
        return np.array([])

    prices = last_price.values
    log_returns = np.diff(np.log(prices))

    return log_returns


def process_day(
    conn, year: int, date: str, tickers: list[str]
) -> list[dict]:
    """Process all tickers for a single trading day.

    Returns list of dicts with [ticker, date, rv5, bv5, rsp5, rsn5, rk].
    """
    trades = download_day_trades(conn, year, date, tickers)

    if trades.empty:
        return []

    results = []
    for ticker in tickers:
        log_returns = resample_to_5min(trades, ticker)

        if len(log_returns) < 5:
            # Not enough data for meaningful RV computation
            continue

        measures = compute_rv_measures(log_returns)
        measures["ticker"] = ticker
        measures["date"] = date
        results.append(measures)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Compute realized variance from TAQ tick data"
    )
    parser.add_argument(
        "--universe",
        default="data/raw/equity/universe_100.csv",
        help="Path to universe CSV",
    )
    parser.add_argument(
        "--years",
        nargs="*",
        type=int,
        default=None,
        help="Specific years to process (default: 2003-2026)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # Load universe
    universe_path = project_root / args.universe
    if not universe_path.exists():
        raise FileNotFoundError(
            f"Universe file not found: {universe_path}\n"
            "Run download_universe.py first."
        )
    universe = pd.read_csv(universe_path)
    tickers = sorted(universe["ticker"].unique().tolist())
    print(f"Universe: {len(tickers)} tickers")

    # Determine years to process
    if args.years:
        years = args.years
    else:
        years = list(range(2003, 2027))

    print(f"Years to process: {years}")

    engine = create_engine(get_wrds_url())

    for year in years:
        checkpoint_path = CHECKPOINT_DIR / f"rv_{year}.parquet"

        if args.resume and checkpoint_path.exists():
            print(f"\n  Year {year}: checkpoint exists, skipping")
            continue

        print(f"\n{'=' * 50}")
        print(f"  Processing year {year}")
        print(f"{'=' * 50}")

        year_results = []

        with engine.connect() as conn:
            # Check if the yearly table exists
            try:
                dates = get_trading_dates_for_year(conn, year)
            except Exception as e:
                print(f"  WARNING: could not access taqm_{year}.ctm_{year}: {e}")
                continue

            print(f"  {len(dates)} trading days")

            for i, date in enumerate(dates):
                day_results = process_day(conn, year, date, tickers)
                year_results.extend(day_results)

                if (i + 1) % 50 == 0 or i == len(dates) - 1:
                    n_obs = len(year_results)
                    print(
                        f"    Day {i + 1}/{len(dates)} ({date}): "
                        f"{n_obs:,} total observations"
                    )

        if year_results:
            year_df = pd.DataFrame(year_results)
            year_df["date"] = pd.to_datetime(year_df["date"])
            year_df.to_parquet(checkpoint_path, index=False)
            print(f"  Saved checkpoint: {checkpoint_path.name} ({len(year_df):,} rows)")
        else:
            print(f"  No data for year {year}")

    engine.dispose()

    # Combine all yearly checkpoints
    print("\nCombining yearly checkpoints...")
    all_dfs = []
    for cp in sorted(CHECKPOINT_DIR.glob("rv_*.parquet")):
        df = pd.read_parquet(cp)
        all_dfs.append(df)
        print(f"  {cp.name}: {len(df):,} rows")

    if not all_dfs:
        print("ERROR: No checkpoint files found!")
        return

    combined = pd.concat(all_dfs, ignore_index=True)
    combined = combined.sort_values(["ticker", "date"]).reset_index(drop=True)

    # Map ticker → permno from universe
    ticker_to_permno = dict(zip(universe["ticker"], universe["permno"]))
    combined["permno"] = combined["ticker"].map(ticker_to_permno)

    out_path = OUTPUT_DIR / "taq_rv.parquet"
    combined.to_parquet(out_path, index=False)

    print(f"\nSaved {len(combined):,} rows to {out_path}")
    print(f"  {combined['ticker'].nunique()} stocks")
    print(f"  {combined['date'].min()} to {combined['date'].max()}")

    # Summary statistics
    print("\nPer-stock summary:")
    summary = combined.groupby("ticker").agg(
        n_days=("date", "count"),
        first_date=("date", "min"),
        last_date=("date", "max"),
        rv5_mean=("rv5", "mean"),
    )
    for ticker, row in summary.iterrows():
        print(
            f"  {ticker:6s}  {row['n_days']:5d} days  "
            f"{row['first_date'].date()} to {row['last_date'].date()}  "
            f"avg rv5={row['rv5_mean']:.6f}"
        )


if __name__ == "__main__":
    main()
