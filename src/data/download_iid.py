"""
Download pre-computed realized variance measures from WRDS Intraday Indicators.

Includes automatic schema discovery to find the correct table and column names,
since WRDS may restructure the wrdsapps_iid schema over time.

Output:
    data/raw/equity/iid_rv.parquet
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
START_DATE = "2003-09-01"  # Daily TAQ millisecond precision begins Sept 2003

# Our target variable names and candidate IID column name patterns.
# Each entry: our_name -> list of candidate IID column names (checked in order).
COLUMN_CANDIDATES = {
    "rv5": ["rv5", "rv_5", "rv5_ss", "rv"],
    "bv5": ["bv5", "bv_5", "bv5_ss", "bv"],
    "rsp5": ["rsp5", "rsp_5", "rsp5_ss", "rsp"],
    "rsn5": ["rsn5", "rsn_5", "rsn5_ss", "rsn"],
    "rk": ["rk_parzen", "rk_twoscale", "rk", "rk_th2"],
}

# Candidate schemas and table name patterns to search
CANDIDATE_SCHEMAS = ["wrdsapps_iid", "wrdsapps", "taq"]
CANDIDATE_TABLE_PATTERNS = ["iid_rv", "rv", "intraday_indicators"]


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
# Schema discovery
# ---------------------------------------------------------------------------
def discover_tables(conn) -> list[dict]:
    """Find all tables in candidate IID schemas."""
    tables = []
    for schema in CANDIDATE_SCHEMAS:
        query = text("""
            SELECT table_schema, table_name
            FROM information_schema.tables
            WHERE table_schema = :schema
            ORDER BY table_name
        """)
        df = pd.read_sql(query, conn, params={"schema": schema})
        for _, row in df.iterrows():
            tables.append({
                "schema": row["table_schema"],
                "table": row["table_name"],
            })
    return tables


def discover_columns(conn, schema: str, table: str) -> list[str]:
    """List all columns in a given table."""
    query = text("""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = :schema
          AND table_name = :table
        ORDER BY ordinal_position
    """)
    df = pd.read_sql(query, conn, params={"schema": schema, "table": table})
    return df["column_name"].tolist()


def find_rv_table(conn) -> tuple[str, str, dict[str, str]]:
    """Auto-discover the IID realized variance table and map columns.

    Returns:
        (schema, table_name, column_mapping)
        where column_mapping maps our names (rv5, bv5, ...) to IID column names.
    """
    print("\n--- IID Schema Discovery ---")
    all_tables = discover_tables(conn)

    if not all_tables:
        raise RuntimeError(
            "No tables found in candidate schemas: "
            f"{CANDIDATE_SCHEMAS}. Check your WRDS subscription."
        )

    # Print all discovered tables
    print(f"Found {len(all_tables)} tables:")
    for t in all_tables[:30]:  # limit output
        print(f"  {t['schema']}.{t['table']}")
    if len(all_tables) > 30:
        print(f"  ... and {len(all_tables) - 30} more")

    # Try to find a table matching our patterns
    best_match = None
    best_mapping = None
    best_score = 0

    for t in all_tables:
        table_name = t["table"].lower()
        # Check if table name matches any candidate pattern
        if not any(p in table_name for p in CANDIDATE_TABLE_PATTERNS):
            continue

        columns = discover_columns(conn, t["schema"], t["table"])
        columns_lower = [c.lower() for c in columns]

        # Try to map our variables to available columns
        mapping = {}
        for our_name, candidates in COLUMN_CANDIDATES.items():
            for cand in candidates:
                if cand.lower() in columns_lower:
                    # Get the actual column name (preserving case)
                    idx = columns_lower.index(cand.lower())
                    mapping[our_name] = columns[idx]
                    break

        score = len(mapping)
        if score > best_score:
            best_score = score
            best_match = t
            best_mapping = mapping

            print(f"\n  Candidate: {t['schema']}.{t['table']}")
            print(f"  Columns: {columns}")
            print(f"  Mapped {score}/{len(COLUMN_CANDIDATES)}: {mapping}")

    if best_match is None or best_score < 3:
        # Print all columns from all candidate tables for debugging
        print("\n--- Could not auto-map. Available tables and columns: ---")
        for t in all_tables:
            table_name = t["table"].lower()
            if any(p in table_name for p in CANDIDATE_TABLE_PATTERNS + ["rv", "vol"]):
                cols = discover_columns(conn, t["schema"], t["table"])
                print(f"  {t['schema']}.{t['table']}: {cols}")

        raise RuntimeError(
            f"Could not find a suitable IID table with at least 3 of our target "
            f"columns ({list(COLUMN_CANDIDATES.keys())}). See table listing above "
            f"and update COLUMN_CANDIDATES or CANDIDATE_TABLE_PATTERNS."
        )

    # Warn about unmapped columns
    unmapped = set(COLUMN_CANDIDATES.keys()) - set(best_mapping.keys())
    if unmapped:
        print(f"\n  WARNING: Could not map: {unmapped}")
        print("  These will be filled with NaN in the output.")

    schema = best_match["schema"]
    table = best_match["table"]
    print(f"\n  Using: {schema}.{table}")
    print(f"  Final mapping: {best_mapping}")

    return schema, table, best_mapping


# ---------------------------------------------------------------------------
# Data download
# ---------------------------------------------------------------------------
def _find_permno_column(conn, schema: str, table: str) -> str:
    """Find the permno-like column in the IID table."""
    columns = discover_columns(conn, schema, table)
    candidates = ["permno", "lpermno", "permno_", "cusip", "ticker"]
    for c in candidates:
        if c.lower() in [col.lower() for col in columns]:
            idx = [col.lower() for col in columns].index(c.lower())
            return columns[idx]
    raise RuntimeError(
        f"No permno-like column found in {schema}.{table}. "
        f"Available columns: {columns}"
    )


def _find_date_column(conn, schema: str, table: str) -> str:
    """Find the date column in the IID table."""
    columns = discover_columns(conn, schema, table)
    candidates = ["date", "trading_date", "dt", "tdate"]
    for c in candidates:
        if c.lower() in [col.lower() for col in columns]:
            idx = [col.lower() for col in columns].index(c.lower())
            return columns[idx]
    raise RuntimeError(
        f"No date column found in {schema}.{table}. "
        f"Available columns: {columns}"
    )


def download_chunk(
    conn,
    permnos: list[int],
    schema: str,
    table: str,
    column_map: dict[str, str],
    permno_col: str,
    date_col: str,
    start_date: str,
) -> pd.DataFrame:
    """Download IID data for a chunk of permnos."""
    permno_str = ", ".join(str(p) for p in permnos)

    # Build SELECT clause from column mapping
    select_cols = [f"{permno_col} AS permno", f"{date_col} AS date"]
    for our_name, iid_name in column_map.items():
        select_cols.append(f"{iid_name} AS {our_name}")

    select_clause = ", ".join(select_cols)

    query = text(f"""
        SELECT {select_clause}
        FROM {schema}.{table}
        WHERE {permno_col} IN ({permno_str})
          AND {date_col} >= :start_date
        ORDER BY {permno_col}, {date_col}
    """)
    return pd.read_sql(query, conn, params={"start_date": start_date})


def main():
    parser = argparse.ArgumentParser(description="Download WRDS Intraday Indicators")
    parser.add_argument(
        "--universe",
        default="data/raw/equity/universe_100.csv",
        help="Path to universe CSV",
    )
    parser.add_argument("--chunk-size", type=int, default=10)
    parser.add_argument(
        "--discover-only",
        action="store_true",
        help="Only run schema discovery, do not download data",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    engine = create_engine(get_wrds_url())

    with engine.connect() as conn:
        # Schema discovery
        schema, table, column_map = find_rv_table(conn)

        if args.discover_only:
            print("\nDiscovery complete. Exiting (--discover-only).")
            engine.dispose()
            return

        # Find identifier columns
        permno_col = _find_permno_column(conn, schema, table)
        date_col = _find_date_column(conn, schema, table)
        print(f"  Permno column: {permno_col}")
        print(f"  Date column: {date_col}")

        # Load universe
        universe_path = project_root / args.universe
        if not universe_path.exists():
            raise FileNotFoundError(
                f"Universe file not found: {universe_path}\n"
                "Run download_universe.py first."
            )
        universe = pd.read_csv(universe_path)
        permnos = universe["permno"].unique().tolist()
        print(f"\nDownloading IID data for {len(permnos)} permnos...")

        # Download in chunks
        chunks = []
        for i in range(0, len(permnos), args.chunk_size):
            batch = permnos[i : i + args.chunk_size]
            print(f"  Chunk {i // args.chunk_size + 1}: permnos {batch[0]}..{batch[-1]}")
            df = download_chunk(
                conn, batch, schema, table, column_map,
                permno_col, date_col, START_DATE,
            )
            chunks.append(df)
            print(f"    {len(df)} rows")

    engine.dispose()

    combined = pd.concat(chunks, ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"])

    out_path = OUTPUT_DIR / "iid_rv.parquet"
    combined.to_parquet(out_path, index=False)

    print(f"\nSaved {len(combined):,} rows to {out_path}")
    print(f"  {combined['permno'].nunique()} stocks")
    print(f"  {combined['date'].min().date()} to {combined['date'].max().date()}")
    print(f"  Columns: {list(combined.columns)}")


if __name__ == "__main__":
    main()
