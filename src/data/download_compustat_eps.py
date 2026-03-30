"""
Download quarterly EPS data from WRDS Compustat.

Reads permno/gvkey/ticker from a universe CSV (e.g. universe_500.csv).

Outputs:
    data/raw/equity/gvkey_ticker_map.csv   – gvkey ↔ ticker crosswalk
    data/raw/equity/compustat_fundq.csv    – quarterly fundamentals
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))


def get_wrds_url() -> str:
    """Build WRDS connection URL from pgpass file or environment."""
    # Try environment variable first
    url = os.environ.get("WRDS_URL")
    if url:
        return url

    # Read from pgpass file
    pgpass_path = Path.home() / "AppData" / "Roaming" / "postgresql" / "pgpass.conf"
    if not pgpass_path.exists():
        # Try Unix-style location as fallback
        pgpass_path = Path.home() / ".pgpass"

    if pgpass_path.exists():
        for line in pgpass_path.read_text().strip().splitlines():
            parts = line.strip().split(":")
            if len(parts) >= 5 and "wrds" in parts[0]:
                host, port, db, user, pw = parts[0], parts[1], parts[2], parts[3], ":".join(parts[4:])
                return f"postgresql://{user}:{pw}@{host}:{port}/{db}?sslmode=require"

    raise RuntimeError(
        "No WRDS credentials found. Set WRDS_URL env var or create a pgpass file at:\n"
        f"  {pgpass_path}\n"
        "Format: wrds-pgdata.wharton.upenn.edu:9737:wrds:USERNAME:PASSWORD"
    )

OUTPUT_DIR = project_root / "data" / "raw" / "equity"


def get_ticker_universe(
    universe_path: Path,
) -> tuple[list[str], list[str]]:
    """Return (tickers, gvkeys) from universe CSV."""
    if universe_path and universe_path.exists():
        df = pd.read_csv(universe_path)
        tickers = sorted(df["ticker"].unique().tolist())
        gvkeys = sorted(df["gvkey"].unique().tolist())
        print(f"Universe: {len(tickers)} tickers, {len(gvkeys)} gvkeys")
        return tickers, gvkeys

    raise FileNotFoundError(
        f"Universe file not found: {universe_path}. Provide --universe."
    )


def download_gvkey_map(engine, tickers: list[str]) -> pd.DataFrame:
    """Query comp.security to get gvkey for each ticker."""
    ticker_str = ", ".join(f"'{t}'" for t in tickers)
    query = f"""
        SELECT DISTINCT gvkey, tic
        FROM comp.security
        WHERE tic IN ({ticker_str})
    """
    df = pd.read_sql(query, engine)
    print(f"gvkey map: {len(df)} rows for {df['tic'].nunique()} unique tickers")

    # Flag any tickers not found
    found = set(df["tic"].unique())
    missing = set(tickers) - found
    if missing:
        print(f"WARNING: tickers not found in comp.security: {sorted(missing)}")

    return df


def download_fundq(
    engine, gvkeys: list[str], tickers: list[str],
    start_date: str = "2014-01-01",
    gvkey_ticker_map: dict = None,
) -> pd.DataFrame:
    """Query comp.fundq for quarterly EPS data.

    If gvkeys are provided, queries fundq directly by gvkey (avoids
    comp.security share-class duplicates). The ticker column is filled
    from gvkey_ticker_map if available.
    Otherwise falls back to ticker-based filtering via comp.security.
    """
    if gvkeys:
        gvkey_str = ", ".join(f"'{g}'" for g in gvkeys)
        query = f"""
            SELECT gvkey, datadate, rdq,
                   epspxq, epsfiq, fyearq, fqtr
            FROM comp.fundq
            WHERE gvkey IN ({gvkey_str})
              AND datadate >= '{start_date}'
              AND datafmt = 'STD'
              AND indfmt = 'INDL'
              AND consol = 'C'
              AND popsrc = 'D'
            ORDER BY gvkey, datadate
        """
        df = pd.read_sql(query, engine)
        df = df.drop_duplicates(subset=["gvkey", "datadate"], keep="first")
        # Map gvkey → ticker
        if gvkey_ticker_map:
            df["tic"] = df["gvkey"].map(gvkey_ticker_map)
        else:
            df["tic"] = df["gvkey"]
    else:
        ticker_str = ", ".join(f"'{t}'" for t in tickers)
        query = f"""
            SELECT a.gvkey, b.tic, a.datadate, a.rdq,
                   a.epspxq, a.epsfiq, a.fyearq, a.fqtr
            FROM comp.fundq a
            JOIN comp.security b ON a.gvkey = b.gvkey
            WHERE b.tic IN ({ticker_str})
              AND a.datadate >= '{start_date}'
              AND a.datafmt = 'STD'
              AND a.indfmt = 'INDL'
              AND a.consol = 'C'
              AND a.popsrc = 'D'
            ORDER BY b.tic, a.datadate
        """
        df = pd.read_sql(query, engine)
        df = df.drop_duplicates(subset=["tic", "datadate"], keep="first")

    print(f"fundq: {len(df)} rows for {df['tic'].nunique()} tickers")
    return df


def print_summary(fundq: pd.DataFrame) -> None:
    """Print a summary of the downloaded data."""
    print("\n--- Compustat Quarterly EPS Summary ---")
    print(f"Tickers:    {fundq['tic'].nunique()}")
    print(f"Date range: {fundq['datadate'].min()} to {fundq['datadate'].max()}")
    print(f"Total rows: {len(fundq)}")

    # Missing values
    n_rdq_null = fundq["rdq"].isna().sum()
    n_eps_null = fundq["epspxq"].isna().sum()
    print(f"Missing rdq:    {n_rdq_null} ({n_rdq_null / len(fundq):.1%})")
    print(f"Missing epspxq: {n_eps_null} ({n_eps_null / len(fundq):.1%})")

    # Per-ticker counts
    print("\nPer-ticker quarter counts:")
    counts = fundq.groupby("tic").size().sort_index()
    for tic, n in counts.items():
        flag = " ***" if n < 40 else ""
        print(f"  {tic:6s}  {n:3d} quarters{flag}")


def main():
    parser = argparse.ArgumentParser(description="Download Compustat quarterly EPS")
    parser.add_argument(
        "--universe", default="data/raw/equity/universe_500.csv",
        help="Path to universe CSV with permno/ticker/gvkey columns.",
    )
    parser.add_argument(
        "--start-date", default="2001-01-01",
        help="Earliest datadate (default: 2001-01-01)",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Get ticker universe
    universe_path = Path(args.universe)
    if not universe_path.is_absolute():
        universe_path = project_root / universe_path
    tickers, gvkeys = get_ticker_universe(universe_path)

    start_date = args.start_date

    # 2. Connect to WRDS via SQLAlchemy
    print("\nConnecting to WRDS...")
    wrds_url = get_wrds_url()
    engine = create_engine(wrds_url)

    with engine.connect() as conn:
        # 3. Download gvkey map (skip if universe already provides gvkeys)
        if not gvkeys:
            gvkey_map = download_gvkey_map(conn, tickers)
            gvkey_map_path = OUTPUT_DIR / "gvkey_ticker_map.csv"
            gvkey_map.to_csv(gvkey_map_path, index=False)
            print(f"Saved: {gvkey_map_path}")
            gvkeys = gvkey_map["gvkey"].unique().tolist()

        # 4. Download quarterly fundamentals
        # Build gvkey→ticker map from universe if available
        # Compustat gvkeys are zero-padded to 6 digits (e.g., '001690')
        gvkey_ticker_map = None
        if universe_path and universe_path.exists():
            udf = pd.read_csv(universe_path)
            gvkey_ticker_map = dict(
                zip(udf["gvkey"].astype(str).str.zfill(6), udf["ticker"])
            )
            gvkeys = [str(g).zfill(6) for g in gvkeys]
        fundq = download_fundq(
            conn, gvkeys, tickers, start_date=start_date,
            gvkey_ticker_map=gvkey_ticker_map,
        )
        fundq_path = OUTPUT_DIR / "compustat_fundq.csv"
        fundq.to_csv(fundq_path, index=False)
        print(f"Saved: {fundq_path}")

        # 5. Summary
        print_summary(fundq)

    engine.dispose()
    print("\nWRDS connection closed.")


if __name__ == "__main__":
    main()
