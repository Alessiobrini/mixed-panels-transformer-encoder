"""
Select a 100-stock equity universe from S&P 500 history on WRDS.

Strategy:
  1. Pull S&P 500 constituent history from CRSP
  2. Map permno → current ticker (crsp.dsenames)
  3. Map permno → gvkey (crsp.ccmxpf_lnkhist, linktype LC/LU)
  4. Filter by CRSP daily coverage (≥80%) and Compustat EPS coverage (≥60 quarters)
  5. Rank by market cap, take top N

Output:
    data/raw/equity/universe_100.csv   [permno, gvkey, ticker]
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

OUTPUT_DIR = project_root / "data" / "raw" / "equity"
START_DATE = "2003-01-01"


# ---------------------------------------------------------------------------
# WRDS connection (reuse pattern from download_compustat_eps.py)
# ---------------------------------------------------------------------------
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
# Data queries
# ---------------------------------------------------------------------------
def download_sp500_history(conn) -> pd.DataFrame:
    """Query S&P 500 constituent history from CRSP."""
    query = text("""
        SELECT permno,
               start  AS start_date,
               ending AS end_date
        FROM crsp.dsp500list
        WHERE ending >= :start_date
           OR ending IS NULL
    """)
    df = pd.read_sql(query, conn, params={"start_date": START_DATE})
    print(f"S&P 500 history: {len(df)} rows, {df['permno'].nunique()} unique permnos")
    return df


def download_permno_ticker_map(conn, permnos: list[int]) -> pd.DataFrame:
    """Get the most recent ticker for each permno from crsp.dsenames."""
    permno_str = ", ".join(str(p) for p in permnos)
    query = text(f"""
        SELECT permno, ticker, namedt, nameendt, shrcd, exchcd
        FROM crsp.dsenames
        WHERE permno IN ({permno_str})
        ORDER BY permno, nameendt DESC
    """)
    df = pd.read_sql(query, conn)
    # Keep only the most recent ticker per permno
    df = df.sort_values("nameendt", ascending=False).drop_duplicates(
        subset=["permno"], keep="first"
    )
    print(f"Ticker map: {len(df)} permnos with current tickers")
    return df


def download_ccm_link(conn, permnos: list[int]) -> pd.DataFrame:
    """Get permno → gvkey mapping from CRSP-Compustat Merged."""
    permno_str = ", ".join(str(p) for p in permnos)
    query = text(f"""
        SELECT lpermno AS permno, gvkey, linkdt, linkenddt
        FROM crsp.ccmxpf_lnkhist
        WHERE lpermno IN ({permno_str})
          AND linktype IN ('LC', 'LU')
          AND linkprim IN ('P', 'C')
        ORDER BY lpermno, linkenddt DESC
    """)
    df = pd.read_sql(query, conn)
    # Keep the most recent link per permno
    df = df.sort_values("linkenddt", ascending=False, na_position="first")
    df = df.drop_duplicates(subset=["permno"], keep="first")
    print(f"CCM link: {len(df)} permno-gvkey pairs")
    return df


def get_crsp_daily_coverage(conn, permnos: list[int]) -> pd.DataFrame:
    """Count available CRSP daily observations per permno since START_DATE."""
    permno_str = ", ".join(str(p) for p in permnos)
    query = text(f"""
        SELECT permno, COUNT(*) AS n_days,
               MIN(date) AS first_date, MAX(date) AS last_date
        FROM crsp.dsf
        WHERE permno IN ({permno_str})
          AND date >= :start_date
        GROUP BY permno
    """)
    df = pd.read_sql(query, conn, params={"start_date": START_DATE})
    print(f"CRSP coverage: {len(df)} permnos with daily data")
    return df


def get_compustat_coverage(conn, gvkeys: list[str]) -> pd.DataFrame:
    """Count available Compustat quarterly EPS observations per gvkey."""
    gvkey_str = ", ".join(f"'{g}'" for g in gvkeys)
    query = text(f"""
        SELECT gvkey, COUNT(*) AS n_quarters
        FROM comp.fundq
        WHERE gvkey IN ({gvkey_str})
          AND datadate >= '2001-01-01'
          AND datafmt = 'STD'
          AND indfmt = 'INDL'
          AND consol = 'C'
          AND popsrc = 'D'
          AND epspxq IS NOT NULL
        GROUP BY gvkey
    """)
    df = pd.read_sql(query, conn)
    print(f"Compustat coverage: {len(df)} gvkeys with EPS data")
    return df


def get_market_cap(conn, permnos: list[int], ref_date: str = "2024-12-31") -> pd.DataFrame:
    """Get market cap (abs(prc) * shrout) at a reference date for ranking."""
    permno_str = ", ".join(str(p) for p in permnos)
    query = text(f"""
        SELECT permno, date, ABS(prc) * shrout AS mktcap
        FROM crsp.dsf
        WHERE permno IN ({permno_str})
          AND date <= :ref_date
        ORDER BY permno, date DESC
    """)
    df = pd.read_sql(query, conn, params={"ref_date": ref_date})
    # Keep most recent observation per permno
    df = df.drop_duplicates(subset=["permno"], keep="first")
    return df[["permno", "mktcap"]]


# ---------------------------------------------------------------------------
# Universe selection
# ---------------------------------------------------------------------------
def select_universe(
    sp500_hist: pd.DataFrame,
    ticker_map: pd.DataFrame,
    ccm_link: pd.DataFrame,
    crsp_cov: pd.DataFrame,
    compustat_cov: pd.DataFrame,
    mktcap: pd.DataFrame,
    min_daily_coverage: float = 0.80,
    min_quarterly_obs: int = 60,
    top_n: int = 100,
) -> pd.DataFrame:
    """Apply filters and return the final universe."""

    # Total possible trading days (approx 252/year * 22 years ≈ 5544)
    max_days = crsp_cov["n_days"].max()
    coverage_threshold = int(max_days * min_daily_coverage)

    # Start with all S&P 500 permnos
    permnos = set(sp500_hist["permno"].unique())
    print(f"\nUniverse selection:")
    print(f"  S&P 500 permnos (ever): {len(permnos)}")

    # Filter: must have ticker mapping
    has_ticker = set(ticker_map["permno"].unique())
    permnos &= has_ticker
    print(f"  After ticker filter: {len(permnos)}")

    # Filter: must have CCM link (gvkey)
    has_gvkey = set(ccm_link["permno"].unique())
    permnos &= has_gvkey
    print(f"  After CCM link filter: {len(permnos)}")

    # Filter: sufficient CRSP daily coverage
    sufficient_crsp = set(
        crsp_cov[crsp_cov["n_days"] >= coverage_threshold]["permno"]
    )
    permnos &= sufficient_crsp
    print(f"  After CRSP coverage (>={coverage_threshold} days): {len(permnos)}")

    # Filter: sufficient Compustat quarterly observations
    # Need to map permno → gvkey for this check
    perm_gvkey = ccm_link[ccm_link["permno"].isin(permnos)][["permno", "gvkey"]]
    sufficient_eps = set(
        compustat_cov[compustat_cov["n_quarters"] >= min_quarterly_obs]["gvkey"]
    )
    perm_with_eps = set(
        perm_gvkey[perm_gvkey["gvkey"].isin(sufficient_eps)]["permno"]
    )
    permnos &= perm_with_eps
    print(f"  After Compustat coverage (>={min_quarterly_obs} quarters): {len(permnos)}")

    # Rank by market cap, take top N
    ranked = mktcap[mktcap["permno"].isin(permnos)].sort_values(
        "mktcap", ascending=False
    )
    top_permnos = set(ranked.head(top_n)["permno"])
    print(f"  After market cap ranking (top {top_n}): {len(top_permnos)}")

    # Assemble final universe
    result = (
        ticker_map[ticker_map["permno"].isin(top_permnos)][["permno", "ticker"]]
        .merge(ccm_link[["permno", "gvkey"]], on="permno", how="left")
        .merge(ranked[["permno", "mktcap"]], on="permno", how="left")
        .sort_values("mktcap", ascending=False)
        .drop(columns=["mktcap"])
        .reset_index(drop=True)
    )

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Select equity universe from S&P 500 history")
    parser.add_argument("--top-n", type=int, default=100, help="Number of stocks to select")
    parser.add_argument("--min-daily-coverage", type=float, default=0.80)
    parser.add_argument("--min-quarterly-obs", type=int, default=60)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Connecting to WRDS...")
    engine = create_engine(get_wrds_url())

    with engine.connect() as conn:
        # 1. S&P 500 history
        sp500_hist = download_sp500_history(conn)
        all_permnos = sp500_hist["permno"].unique().tolist()

        # 2. Ticker mapping
        ticker_map = download_permno_ticker_map(conn, all_permnos)

        # 3. CCM link (permno → gvkey)
        ccm_link = download_ccm_link(conn, all_permnos)

        # 4. CRSP daily coverage
        print("\nChecking CRSP daily coverage (this may take a moment)...")
        crsp_cov = get_crsp_daily_coverage(conn, all_permnos)

        # 5. Compustat coverage
        all_gvkeys = ccm_link["gvkey"].unique().tolist()
        compustat_cov = get_compustat_coverage(conn, all_gvkeys)

        # 6. Market cap for ranking
        print("\nFetching market cap for ranking...")
        mktcap = get_market_cap(conn, all_permnos)

    engine.dispose()

    # 7. Select universe
    universe = select_universe(
        sp500_hist, ticker_map, ccm_link, crsp_cov, compustat_cov, mktcap,
        min_daily_coverage=args.min_daily_coverage,
        min_quarterly_obs=args.min_quarterly_obs,
        top_n=args.top_n,
    )

    # 8. Save
    out_path = OUTPUT_DIR / f"universe_{args.top_n}.csv"
    universe.to_csv(out_path, index=False)
    print(f"\nSaved {len(universe)} stocks to {out_path}")
    print("\nTicker list:")
    for _, row in universe.iterrows():
        print(f"  {row['ticker']:6s}  permno={row['permno']}  gvkey={row['gvkey']}")


if __name__ == "__main__":
    main()
