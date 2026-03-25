"""
Download quarterly EPS data from WRDS Compustat for the 40 VOLARE stocks.

Outputs:
    data/raw/equity/gvkey_ticker_map.csv   – gvkey ↔ ticker crosswalk
    data/raw/equity/compustat_fundq.csv    – quarterly fundamentals
"""

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

VOLARE_PATH = project_root / "data" / "raw" / "volare" / "realized_variance_stocks.csv"
OUTPUT_DIR = project_root / "data" / "raw" / "equity"


def get_ticker_universe(volare_path: Path) -> list[str]:
    """Read unique stock tickers from the VOLARE realized variance file."""
    df = pd.read_csv(volare_path, usecols=["symbol"])
    tickers = sorted(df["symbol"].unique().tolist())
    print(f"Ticker universe: {len(tickers)} stocks")
    return tickers


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


def download_fundq(engine, gvkeys: list[str], tickers: list[str]) -> pd.DataFrame:
    """Query comp.fundq for quarterly EPS data."""
    ticker_str = ", ".join(f"'{t}'" for t in tickers)
    query = f"""
        SELECT a.gvkey, b.tic, a.datadate, a.rdq,
               a.epspxq, a.epsfiq, a.fyearq, a.fqtr
        FROM comp.fundq a
        JOIN comp.security b ON a.gvkey = b.gvkey
        WHERE b.tic IN ({ticker_str})
          AND a.datadate >= '2014-01-01'
          AND a.datafmt = 'STD'
          AND a.indfmt = 'INDL'
          AND a.consol = 'C'
          AND a.popsrc = 'D'
        ORDER BY b.tic, a.datadate
    """
    df = pd.read_sql(query, engine)
    # Drop duplicate rows (multiple gvkeys for same ticker-date)
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
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Get ticker universe
    tickers = get_ticker_universe(VOLARE_PATH)

    # 2. Connect to WRDS via SQLAlchemy
    print("\nConnecting to WRDS...")
    wrds_url = get_wrds_url()
    engine = create_engine(wrds_url)

    with engine.connect() as conn:
        # 3. Download gvkey map
        gvkey_map = download_gvkey_map(conn, tickers)
        gvkey_map_path = OUTPUT_DIR / "gvkey_ticker_map.csv"
        gvkey_map.to_csv(gvkey_map_path, index=False)
        print(f"Saved: {gvkey_map_path}")

        # 4. Download quarterly fundamentals
        gvkeys = gvkey_map["gvkey"].unique().tolist()
        fundq = download_fundq(conn, gvkeys, tickers)
        fundq_path = OUTPUT_DIR / "compustat_fundq.csv"
        fundq.to_csv(fundq_path, index=False)
        print(f"Saved: {fundq_path}")

        # 5. Summary
        print_summary(fundq)

    engine.dispose()
    print("\nWRDS connection closed.")


if __name__ == "__main__":
    main()
