"""
Build per-stock long-format CSVs for the equity earnings dataset.

Sources daily data from CRSP + WRDS IID (Intraday Indicators).

Combines:
  - Daily returns/volume from CRSP (crsp_dsf.parquet)
  - Daily realized variance from WRDS Intraday Indicators (iid_rv.parquet)
  - Monthly stock-level aggregates (computed from daily)
  - Cross-sectional monthly aggregates (computed across all stocks)
  - Monthly macro features from FRED-MD (transf_md.csv)
  - Quarterly macro features from FRED-QD (transf_qd.csv)
  - Quarterly EPS target from Compustat

Output: data/processed/equity/long_format_{TKR}_{N}D_{M}M_{Q}Q.csv

Usage:
    python src/data/build_equity_long_v2.py
    python src/data/build_equity_long_v2.py --universe data/raw/equity/universe_100.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import skew
from tqdm import tqdm

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

# ---------------------------------------------------------------------------
# Paths (defaults, can be overridden via args)
# ---------------------------------------------------------------------------
DEFAULT_UNIVERSE = project_root / "data" / "raw" / "equity" / "universe_100.csv"
DEFAULT_CRSP = project_root / "data" / "raw" / "equity" / "crsp_dsf.parquet"
DEFAULT_IID = project_root / "data" / "raw" / "equity" / "taq_rv.parquet"
DEFAULT_COMPUSTAT = project_root / "data" / "raw" / "equity" / "compustat_fundq.csv"
FRED_MD_PATH = project_root / "data" / "raw" / "fred_data" / "transf_md.csv"
FRED_QD_PATH = project_root / "data" / "raw" / "fred_data" / "transf_qd.csv"
OUTPUT_DIR = project_root / "data" / "processed" / "equity"

START_DATE = "2014-01-01"
MIN_MONTHLY_OBS = 15  # minimum daily observations to form a valid month

# Same macro variable lists as build_equity_long.py
MACRO_MONTHLY_VARS = [
    "RPI", "INDPRO", "CUMFNS", "HWI", "CLF16OV", "CE16OV", "UEMPMEAN",
    "CLAIMSx", "PAYEMS", "CES0600000007", "CES0600000008", "CES2000000008",
    "CES3000000008", "AWOTMAN", "AWHMAN", "HOUST", "DPCERA3M086SBEA",
    "BUSINVx", "RETAILx", "CMRMTSPLx", "M2REAL", "TOTRESNS", "BUSLOANS",
    "NONREVSL", "FEDFUNDS", "GS1", "GS10", "BAA", "PCEPI", "WPSFD49207",
    "OILPRICEx", "S&P 500", "S&P PE ratio", "TB3MS", "TB6MS",
]

MACRO_QUARTERLY_VARS = [
    "GDPC1", "GPDIC1", "PCECC96", "DPIC96", "OUTNFB", "UNRATE",
    "PCECTPI", "PCEPILFE", "CPIAUCSL", "CPILFESL", "FPIx", "EXPGSC1", "IMPGSC1",
]


# ---------------------------------------------------------------------------
# Daily features
# ---------------------------------------------------------------------------
def build_daily_features(
    crsp_stock: pl.DataFrame,
    iid_stock: pl.DataFrame,
    ticker: str,
) -> pd.DataFrame:
    """Build 7 daily features from CRSP + IID for one stock.

    Returns long-format pd.DataFrame [Timestamp, Variable, Value, Frequency].
    """
    # Inner-join on date — only keep days with both CRSP and IID data
    daily = crsp_stock.join(iid_stock, on="date", how="inner").sort("date")

    if daily.is_empty():
        print(f"  WARNING: no overlapping CRSP+IID data for {ticker}")
        return pd.DataFrame(columns=["Timestamp", "Variable", "Value", "Frequency"])

    # Compute log return from CRSP simple return: log(1 + ret)
    daily = daily.with_columns(
        pl.col("ret").map_elements(
            lambda r: float(np.log(1 + r)) if r is not None and r > -1 else None,
            return_dtype=pl.Float64,
        ).alias("log_ret")
    )

    # Compute log volume
    daily = daily.with_columns(
        pl.col("vol").clip(lower_bound=1).log().alias("logvol")
    )

    daily = daily.drop_nulls(subset=["log_ret"])

    # Build feature columns with ticker prefix
    feature_map = {
        f"{ticker}_ret": "log_ret",
        f"{ticker}_rv5": "rv5",
        f"{ticker}_bv5": "bv5",
        f"{ticker}_rsp5": "rsp5",
        f"{ticker}_rsn5": "rsn5",
        f"{ticker}_rk": "rk",
        f"{ticker}_logvol": "logvol",
    }

    # Convert to pandas for melt (small per-stock data)
    pdf = daily.select(["date"] + list(feature_map.values())).to_pandas()
    pdf = pdf.rename(columns={v: k for k, v in feature_map.items()})

    long = pdf.melt(
        id_vars="date",
        var_name="Variable",
        value_name="Value",
    )
    long.rename(columns={"date": "Timestamp"}, inplace=True)
    long["Frequency"] = "D"
    return long.dropna(subset=["Value"])


# ---------------------------------------------------------------------------
# Monthly stock-level aggregates
# ---------------------------------------------------------------------------
def build_monthly_stock_features(
    crsp_stock: pl.DataFrame,
    iid_stock: pl.DataFrame,
    ticker: str,
) -> pd.DataFrame:
    """Aggregate daily data to monthly per-stock features.

    Returns wide pd.DataFrame with Timestamp + 5 feature columns.
    """
    daily = crsp_stock.join(iid_stock, on="date", how="inner").sort("date")

    if daily.is_empty():
        return pd.DataFrame()

    daily = daily.with_columns(
        pl.col("ret").map_elements(
            lambda r: float(np.log(1 + r)) if r is not None and r > -1 else None,
            return_dtype=pl.Float64,
        ).alias("log_ret"),
        pl.col("date").dt.strftime("%Y-%m").alias("ym"),
    )

    daily = daily.with_columns(
        ((pl.col("rv5") - pl.col("bv5")) / pl.col("rv5").clip(lower_bound=1e-12))
        .alias("jump")
    )

    daily = daily.drop_nulls(subset=["log_ret"])

    # Group by month
    monthly = (
        daily.group_by("ym")
        .agg(
            pl.col("date").max().alias("Timestamp"),
            pl.col("log_ret").sum().alias("ret_m"),
            pl.col("rv5").mean().alias("rv5_mean"),
            pl.col("rv5").max().alias("rv5_max"),
            pl.col("log_ret").alias("_rets_list"),
            pl.col("jump").mean().alias("jump_ratio"),
            pl.len().alias("n_obs"),
        )
        .filter(pl.col("n_obs") >= MIN_MONTHLY_OBS)
        .sort("Timestamp")
    )

    if monthly.is_empty():
        return pd.DataFrame()

    # Compute skewness from the list of returns
    pdf = monthly.to_pandas()
    pdf["ret_skew"] = pdf["_rets_list"].apply(
        lambda x: skew(x, bias=False) if len(x) >= 3 else 0.0
    )

    # Rename with ticker prefix
    result = pd.DataFrame({
        "Timestamp": pdf["Timestamp"],
        f"{ticker}_ret_m": pdf["ret_m"],
        f"{ticker}_rv5_mean": pdf["rv5_mean"],
        f"{ticker}_rv5_max": pdf["rv5_max"],
        f"{ticker}_ret_skew": pdf["ret_skew"],
        f"{ticker}_jump_ratio": pdf["jump_ratio"],
    })

    return result


def compute_all_monthly_stock_features(
    crsp: pl.DataFrame,
    iid: pl.DataFrame,
    universe: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Compute monthly stock features for all tickers."""
    result = {}
    for _, row in universe.iterrows():
        permno = row["permno"]
        ticker = row["ticker"]

        crsp_stock = crsp.filter(pl.col("permno") == permno)
        iid_stock = iid.filter(pl.col("permno") == permno)

        if crsp_stock.is_empty():
            print(f"  WARNING: no CRSP data for {ticker} (permno={permno})")
            continue

        mf = build_monthly_stock_features(crsp_stock, iid_stock, ticker)
        if not mf.empty:
            result[ticker] = mf

    return result


# ---------------------------------------------------------------------------
# Cross-sectional monthly aggregates
# ---------------------------------------------------------------------------
def build_cross_sectional_monthly(
    monthly_by_ticker: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Compute equal-weighted cross-sectional monthly aggregates.

    Returns long-format DataFrame with MKT_ret_m, MKT_rv5_mean, MKT_rv5_disp.
    """
    all_ret = {}
    all_rv5 = {}

    for ticker, mdf in monthly_by_ticker.items():
        ts = mdf.set_index("Timestamp")
        all_ret[ticker] = ts[f"{ticker}_ret_m"]
        all_rv5[ticker] = ts[f"{ticker}_rv5_mean"]

    ret_panel = pd.DataFrame(all_ret)
    rv5_panel = pd.DataFrame(all_rv5)

    cross = pd.DataFrame({
        "Timestamp": ret_panel.index,
        "MKT_ret_m": ret_panel.mean(axis=1).values,
        "MKT_rv5_mean": rv5_panel.mean(axis=1).values,
        "MKT_rv5_disp": rv5_panel.std(axis=1).values,
    })

    return cross


# ---------------------------------------------------------------------------
# Macro features (same as build_equity_long.py, with configurable start)
# ---------------------------------------------------------------------------
def build_macro_monthly(fred_md_path: Path) -> pd.DataFrame:
    """Load FRED-MD monthly data, filter to >=2003 and the 35 macro vars."""
    md = pd.read_csv(fred_md_path, parse_dates=["date"]).sort_values("date")
    md = md[md["date"] >= START_DATE].copy()
    md = md.ffill()

    available = [v for v in MACRO_MONTHLY_VARS if v in md.columns]
    missing = set(MACRO_MONTHLY_VARS) - set(available)
    if missing:
        print(f"  WARNING: macro vars not found in transf_md.csv: {sorted(missing)}")

    md_subset = md[["date"] + available].copy()
    md_subset = md_subset.dropna(subset=available, how="all")

    long = md_subset.melt(
        id_vars="date",
        value_vars=available,
        var_name="Variable",
        value_name="Value",
    )
    long.rename(columns={"date": "Timestamp"}, inplace=True)
    long["Frequency"] = "M"
    return long.dropna(subset=["Value"])


def build_macro_quarterly(fred_qd_path: Path) -> pd.DataFrame:
    """Load FRED-QD quarterly data, filter to >=2003 and the 13 macro vars."""
    qd = pd.read_csv(fred_qd_path, parse_dates=["date"]).sort_values("date")
    qd = qd[qd["date"] >= START_DATE].copy()
    qd = qd.ffill()

    available = [v for v in MACRO_QUARTERLY_VARS if v in qd.columns]
    missing = set(MACRO_QUARTERLY_VARS) - set(available)
    if missing:
        print(f"  WARNING: macro vars not found in transf_qd.csv: {sorted(missing)}")

    qd_subset = qd[["date"] + available].copy()
    qd_subset = qd_subset.dropna(subset=available, how="all")

    long = qd_subset.melt(
        id_vars="date",
        value_vars=available,
        var_name="Variable",
        value_name="Value",
    )
    long.rename(columns={"date": "Timestamp"}, inplace=True)
    long["Frequency"] = "Q"
    return long.dropna(subset=["Value"])


# ---------------------------------------------------------------------------
# Quarterly EPS target (same logic as build_equity_long.py)
# ---------------------------------------------------------------------------
def build_quarterly_target(compustat: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Compute YoY EPS growth for one stock from Compustat quarterly data."""
    df = compustat[compustat["tic"] == ticker].copy()
    if df.empty:
        print(f"  WARNING: no Compustat data for {ticker}")
        return pd.DataFrame(columns=["Timestamp", "Variable", "Value", "Frequency"])

    df["datadate"] = pd.to_datetime(df["datadate"])
    df["rdq"] = pd.to_datetime(df["rdq"])
    df = df.sort_values("datadate").drop_duplicates(subset=["datadate"], keep="first")

    df = df.reset_index(drop=True)
    df["epspxq_lag4"] = df["epspxq"].shift(4)

    denom = df["epspxq_lag4"].abs().clip(lower=0.01)
    df["eps_yoy"] = (df["epspxq"] - df["epspxq_lag4"]) / denom
    df["eps_yoy"] = df["eps_yoy"].clip(lower=-2.0, upper=2.0)

    df = df.dropna(subset=["eps_yoy", "rdq"])
    df = df[df["rdq"] >= START_DATE]

    result = pd.DataFrame({
        "Timestamp": df["rdq"],
        "Variable": f"{ticker}_eps_yoy",
        "Value": df["eps_yoy"],
        "Frequency": "Q",
    })
    return result.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Assembly (same logic as build_equity_long.py)
# ---------------------------------------------------------------------------
def assemble_stock_long(
    ticker: str,
    daily_long: pd.DataFrame,
    monthly_stock: pd.DataFrame,
    cross_sectional: pd.DataFrame,
    macro_monthly_long: pd.DataFrame,
    macro_quarterly_long: pd.DataFrame,
    quarterly_target_long: pd.DataFrame,
) -> pd.DataFrame:
    """Assemble all features for one stock into a single long-format DataFrame."""
    # Melt monthly stock features to long format
    stock_monthly_cols = [c for c in monthly_stock.columns if c != "Timestamp"]
    monthly_stock_long = monthly_stock.melt(
        id_vars="Timestamp",
        value_vars=stock_monthly_cols,
        var_name="Variable",
        value_name="Value",
    )
    monthly_stock_long["Frequency"] = "M"

    # Melt cross-sectional features to long format
    cs_cols = [c for c in cross_sectional.columns if c != "Timestamp"]
    cs_long = cross_sectional.melt(
        id_vars="Timestamp",
        value_vars=cs_cols,
        var_name="Variable",
        value_name="Value",
    )
    cs_long["Frequency"] = "M"

    # Combine all blocks
    all_blocks = [
        daily_long, monthly_stock_long, cs_long, macro_monthly_long,
        macro_quarterly_long, quarterly_target_long,
    ]
    combined = pd.concat(all_blocks, ignore_index=True)

    combined["Timestamp"] = pd.to_datetime(combined["Timestamp"])

    freq_order = {"D": 0, "M": 1, "Q": 2}
    combined["_freq_sort"] = combined["Frequency"].map(freq_order)
    combined = (
        combined
        .sort_values(["Timestamp", "_freq_sort", "Variable"])
        .drop(columns=["_freq_sort"])
        .reset_index(drop=True)
    )

    combined = combined.dropna(subset=["Value"])
    return combined[["Timestamp", "Variable", "Value", "Frequency"]]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Build extended equity long-format CSVs")
    parser.add_argument("--universe", default=str(DEFAULT_UNIVERSE))
    parser.add_argument("--crsp", default=str(DEFAULT_CRSP))
    parser.add_argument("--iid", "--rv", default=str(DEFAULT_IID),
                        help="Path to RV parquet (from IID or TAQ)")
    parser.add_argument("--compustat", default=str(DEFAULT_COMPUSTAT))
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load universe
    universe = pd.read_csv(args.universe)
    print(f"Universe: {len(universe)} stocks")

    # Load raw data with polars (fast)
    print("\nLoading CRSP daily data...")
    crsp = pl.read_parquet(args.crsp)
    print(f"  {len(crsp):,} rows, {crsp['permno'].n_unique()} stocks")

    print("Loading IID realized variance data...")
    iid = pl.read_parquet(args.iid)
    print(f"  {len(iid):,} rows, {iid['permno'].n_unique()} stocks")

    print("Loading Compustat data...")
    compustat = pd.read_csv(args.compustat)
    print(f"  {len(compustat)} rows, {compustat['tic'].nunique()} tickers")

    print("Loading FRED-MD macro data (monthly)...")
    macro_monthly_long = build_macro_monthly(FRED_MD_PATH)
    print(f"  {macro_monthly_long['Variable'].nunique()} monthly macro vars")

    print("Loading FRED-QD macro data (quarterly)...")
    macro_quarterly_long = build_macro_quarterly(FRED_QD_PATH)
    print(f"  {macro_quarterly_long['Variable'].nunique()} quarterly macro vars")

    # Compute all monthly stock features
    print("\nComputing monthly stock features for all tickers...")
    monthly_by_ticker = compute_all_monthly_stock_features(crsp, iid, universe)
    print(f"  {len(monthly_by_ticker)} tickers with monthly data")

    print("Computing cross-sectional monthly aggregates...")
    cross_sectional = build_cross_sectional_monthly(monthly_by_ticker)
    print(f"  {len(cross_sectional)} months")

    # Build per-stock long-format files
    print(f"\nBuilding per-stock long-format CSVs in {OUTPUT_DIR}/")
    for _, row in tqdm(universe.iterrows(), total=len(universe),
                       desc="Building CSVs", unit="stock"):
        permno = row["permno"]
        ticker = row["ticker"]

        # Daily features
        crsp_stock = crsp.filter(pl.col("permno") == permno)
        iid_stock = iid.filter(pl.col("permno") == permno)
        daily_long = build_daily_features(crsp_stock, iid_stock, ticker)

        # Monthly stock features
        if ticker not in monthly_by_ticker:
            print(f"  SKIP {ticker}: no monthly data")
            continue
        monthly_stock = monthly_by_ticker[ticker]

        # Quarterly target
        quarterly_target_long = build_quarterly_target(compustat, ticker)

        # Assemble
        combined = assemble_stock_long(
            ticker, daily_long, monthly_stock, cross_sectional,
            macro_monthly_long, macro_quarterly_long, quarterly_target_long,
        )

        # Count features
        n_daily = combined[combined["Frequency"] == "D"]["Variable"].nunique()
        n_monthly = combined[combined["Frequency"] == "M"]["Variable"].nunique()
        n_quarterly = combined[combined["Frequency"] == "Q"]["Variable"].nunique()
        suffix = f"{n_daily}D_{n_monthly}M_{n_quarterly}Q"

        out_path = OUTPUT_DIR / f"long_format_{ticker}_{suffix}.csv"
        combined.to_csv(out_path, index=False)

        n_targets = len(quarterly_target_long)
        tqdm.write(
            f"  {ticker}: {len(combined):>7,} rows  ({n_daily}D, {n_monthly}M, {n_quarterly}Q)  "
            f"{n_targets} quarterly targets  -> {out_path.name}"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
