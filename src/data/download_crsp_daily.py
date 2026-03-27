"""
Download CRSP daily stock data for the equity universe.

Reads the universe file and downloads daily returns, prices, and volume
from crsp.dsf in chunks to avoid WRDS query timeouts.

Output:
    data/raw/equity/crsp_dsf.parquet
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text
from tqdm import tqdm

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

OUTPUT_DIR = project_root / "data" / "raw" / "equity"
START_DATE = "2014-01-01"


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


def download_chunk(conn, permnos: list[int], start_date: str) -> pd.DataFrame:
    """Download crsp.dsf for a chunk of permnos."""
    permno_str = ", ".join(str(p) for p in permnos)
    query = text(f"""
        SELECT permno, date, ret, ABS(prc) AS prc, vol, shrout
        FROM crsp.dsf
        WHERE permno IN ({permno_str})
          AND date >= :start_date
        ORDER BY permno, date
    """)
    return pd.read_sql(query, conn, params={"start_date": start_date})


def main():
    parser = argparse.ArgumentParser(description="Download CRSP daily data")
    parser.add_argument(
        "--universe",
        default="data/raw/equity/universe_100.csv",
        help="Path to universe CSV",
    )
    parser.add_argument("--chunk-size", type=int, default=20)
    args = parser.parse_args()

    universe_path = project_root / args.universe
    if not universe_path.exists():
        raise FileNotFoundError(
            f"Universe file not found: {universe_path}\n"
            "Run download_universe.py first."
        )

    universe = pd.read_csv(universe_path)
    permnos = universe["permno"].unique().tolist()
    print(f"Downloading CRSP daily data for {len(permnos)} permnos...")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    engine = create_engine(get_wrds_url())
    chunks = []

    n_chunks = (len(permnos) + args.chunk_size - 1) // args.chunk_size
    with engine.connect() as conn:
        for i in tqdm(range(0, len(permnos), args.chunk_size),
                      total=n_chunks, desc="CRSP chunks", unit="chunk"):
            batch = permnos[i : i + args.chunk_size]
            df = download_chunk(conn, batch, START_DATE)
            chunks.append(df)

    engine.dispose()

    combined = pd.concat(chunks, ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"])

    # Drop rows with missing returns (e.g., first listing day)
    combined = combined.dropna(subset=["ret"])

    out_path = OUTPUT_DIR / "crsp_dsf.parquet"
    combined.to_parquet(out_path, index=False)

    print(f"\nSaved {len(combined):,} rows to {out_path}")
    print(f"  {combined['permno'].nunique()} stocks")
    print(f"  {combined['date'].min().date()} to {combined['date'].max().date()}")


if __name__ == "__main__":
    main()
