"""
Build per-stock long-format CSVs for the equity earnings dataset.

Combines:
  - Daily features from VOLARE realized variance data
  - Monthly stock-level aggregates + cross-sectional aggregates
  - Monthly macro features from FRED-MD (transf_md.csv)
  - Quarterly macro features from FRED-QD (transf_qd.csv)
  - Quarterly EPS target from Compustat

Output: data/processed/equity/long_format_{TKR}_7D_43M_14Q.csv  (one per stock)

Usage:
    python src/data/build_equity_long.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import skew

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
VOLARE_PATH = project_root / "data" / "raw" / "volare" / "realized_variance_stocks.csv"
COMPUSTAT_PATH = project_root / "data" / "raw" / "equity" / "compustat_fundq.csv"
FRED_MD_PATH = project_root / "data" / "raw" / "fred_data" / "transf_md.csv"
FRED_QD_PATH = project_root / "data" / "raw" / "fred_data" / "transf_qd.csv"
OUTPUT_DIR = project_root / "data" / "processed" / "equity"

# The 35 monthly macro variables from the existing FRED-MD setup
MACRO_MONTHLY_VARS = [
    "RPI", "INDPRO", "CUMFNS", "HWI", "CLF16OV", "CE16OV", "UEMPMEAN",
    "CLAIMSx", "PAYEMS", "CES0600000007", "CES0600000008", "CES2000000008",
    "CES3000000008", "AWOTMAN", "AWHMAN", "HOUST", "DPCERA3M086SBEA",
    "BUSINVx", "RETAILx", "CMRMTSPLx", "M2REAL", "TOTRESNS", "BUSLOANS",
    "NONREVSL", "FEDFUNDS", "GS1", "GS10", "BAA", "PCEPI", "WPSFD49207",
    "OILPRICEx", "S&P 500", "S&P PE ratio", "TB3MS", "TB6MS",
]

# The 13 quarterly macro variables from the existing FRED-QD setup
MACRO_QUARTERLY_VARS = [
    "GDPC1", "GPDIC1", "PCECC96", "DPIC96", "OUTNFB", "UNRATE",
    "PCECTPI", "PCEPILFE", "CPIAUCSL", "CPILFESL", "FPIx", "EXPGSC1", "IMPGSC1",
]

# Daily features to extract from VOLARE (besides computed ones)
VOLARE_RV_COLS = ["rv5", "bv5", "rsp5", "rsn5", "rk"]


# ---------------------------------------------------------------------------
# Daily features
# ---------------------------------------------------------------------------
def build_daily_features(volare_stock: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Build daily feature DataFrame for one stock from VOLARE data.

    Returns a long-format DataFrame with columns:
        Timestamp, Variable, Value, Frequency
    """
    df = volare_stock.sort_values("date").copy()

    # Log return from close prices
    df[f"{ticker}_ret"] = np.log(df["close_price"] / df["close_price"].shift(1))

    # RV measures (rename to ticker-prefixed)
    for col in VOLARE_RV_COLS:
        df[f"{ticker}_{col}"] = df[col]

    # Log volume
    df[f"{ticker}_logvol"] = np.log(df["volume"].clip(lower=1))

    # Drop the first row (NaN return)
    df = df.dropna(subset=[f"{ticker}_ret"])

    # Select only the feature columns
    feature_cols = [f"{ticker}_ret"] + [f"{ticker}_{c}" for c in VOLARE_RV_COLS] + [f"{ticker}_logvol"]

    long = df[["date"] + feature_cols].melt(
        id_vars="date",
        var_name="Variable",
        value_name="Value",
    )
    long.rename(columns={"date": "Timestamp"}, inplace=True)
    long["Frequency"] = "D"
    return long


# ---------------------------------------------------------------------------
# Monthly stock-level aggregates
# ---------------------------------------------------------------------------
def build_monthly_stock_features(volare_stock: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Aggregate daily data to monthly per-stock features.

    Returns a wide DataFrame indexed by year-month with columns:
        {TKR}_ret_m, {TKR}_rv5_mean, {TKR}_rv5_max, {TKR}_ret_skew, {TKR}_jump_ratio
    Plus a 'Timestamp' column (last trading day of each month).
    """
    df = volare_stock.sort_values("date").copy()
    df["log_ret"] = np.log(df["close_price"] / df["close_price"].shift(1))
    df = df.dropna(subset=["log_ret"])
    df["ym"] = df["date"].dt.to_period("M")

    monthly = df.groupby("ym").agg(
        Timestamp=("date", "max"),
        ret_m=("log_ret", "sum"),
        rv5_mean=("rv5", "mean"),
        rv5_max=("rv5", "max"),
        ret_skew=("log_ret", lambda x: skew(x, bias=False) if len(x) >= 3 else 0.0),
        jump_ratio_mean=("rv5", lambda x: np.nan),  # placeholder, computed below
    ).reset_index(drop=True)

    # Jump ratio: mean of (rv5 - bv5) / rv5, computed from daily data
    jump = df.copy()
    jump["jump"] = (jump["rv5"] - jump["bv5"]) / jump["rv5"].clip(lower=1e-12)
    jump_monthly = jump.groupby("ym")["jump"].mean().reset_index()
    jump_monthly.columns = ["ym_tmp", "jump_ratio"]

    # Re-derive ym for join
    monthly["ym_tmp"] = df.groupby("ym")["date"].first().values
    # Actually, simpler: just merge on index order since both are sorted by ym
    monthly["jump_ratio"] = jump_monthly["jump_ratio"].values
    monthly.drop(columns=["jump_ratio_mean", "ym_tmp"], inplace=True)

    # Rename to ticker-prefixed
    monthly.rename(columns={
        "ret_m": f"{ticker}_ret_m",
        "rv5_mean": f"{ticker}_rv5_mean",
        "rv5_max": f"{ticker}_rv5_max",
        "ret_skew": f"{ticker}_ret_skew",
        "jump_ratio": f"{ticker}_jump_ratio",
    }, inplace=True)

    return monthly


def compute_all_monthly_stock_features(volare: pd.DataFrame, tickers: list[str]) -> dict[str, pd.DataFrame]:
    """Compute monthly stock features for all tickers. Returns dict of ticker -> DataFrame."""
    result = {}
    for ticker in tickers:
        stock_data = volare[volare["symbol"] == ticker].copy()
        if stock_data.empty:
            print(f"  WARNING: no VOLARE data for {ticker}")
            continue
        result[ticker] = build_monthly_stock_features(stock_data, ticker)
    return result


# ---------------------------------------------------------------------------
# Cross-sectional monthly aggregates
# ---------------------------------------------------------------------------
def build_cross_sectional_monthly(monthly_by_ticker: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Compute equal-weighted cross-sectional monthly aggregates.

    Returns long-format DataFrame with variables:
        MKT_ret_m, MKT_rv5_mean, MKT_rv5_disp
    """
    # Build a panel: rows = months, columns = tickers, for each metric
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
# Macro monthly features from FRED-MD
# ---------------------------------------------------------------------------
def build_macro_monthly(fred_md_path: Path) -> pd.DataFrame:
    """
    Load FRED-MD transformed monthly data, restrict to 2015+ and the 35 macro vars.

    Returns long-format DataFrame with Frequency="M".
    """
    md = pd.read_csv(fred_md_path, parse_dates=["date"]).sort_values("date")
    md = md[md["date"] >= "2015-01-01"].copy()
    md = md.ffill()

    # Select only the columns that exist in the file
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
    long = long.dropna(subset=["Value"])
    return long


# ---------------------------------------------------------------------------
# Macro quarterly features from FRED-QD
# ---------------------------------------------------------------------------
def build_macro_quarterly(fred_qd_path: Path) -> pd.DataFrame:
    """
    Load FRED-QD transformed quarterly data, restrict to 2015+ and the 13 macro vars.

    Returns long-format DataFrame with Frequency="Q".
    """
    qd = pd.read_csv(fred_qd_path, parse_dates=["date"]).sort_values("date")
    qd = qd[qd["date"] >= "2015-01-01"].copy()
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
    long = long.dropna(subset=["Value"])
    return long


# ---------------------------------------------------------------------------
# Quarterly EPS target
# ---------------------------------------------------------------------------
def build_quarterly_target(compustat: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Compute YoY EPS growth for one stock from Compustat quarterly data.

    Returns long-format DataFrame with Frequency="Q", timestamped at rdq.
    """
    df = compustat[compustat["tic"] == ticker].copy()
    if df.empty:
        print(f"  WARNING: no Compustat data for {ticker}")
        return pd.DataFrame(columns=["Timestamp", "Variable", "Value", "Frequency"])

    df["datadate"] = pd.to_datetime(df["datadate"])
    df["rdq"] = pd.to_datetime(df["rdq"])
    df = df.sort_values("datadate").drop_duplicates(subset=["datadate"], keep="first")

    # YoY growth: compare to same fiscal quarter one year ago (4 quarters back)
    df = df.reset_index(drop=True)
    df["epspxq_lag4"] = df["epspxq"].shift(4)

    # Compute growth with denominator floor
    denom = df["epspxq_lag4"].abs().clip(lower=0.01)
    df["eps_yoy"] = (df["epspxq"] - df["epspxq_lag4"]) / denom

    # Winsorize at [-2, 2]
    df["eps_yoy"] = df["eps_yoy"].clip(lower=-2.0, upper=2.0)

    # Drop rows without valid YoY (first 4 quarters) or missing rdq
    df = df.dropna(subset=["eps_yoy", "rdq"])

    # Restrict to VOLARE time range (rdq >= 2015-01-01)
    df = df[df["rdq"] >= "2015-01-01"]

    result = pd.DataFrame({
        "Timestamp": df["rdq"],
        "Variable": f"{ticker}_eps_yoy",
        "Value": df["eps_yoy"],
        "Frequency": "Q",
    })
    return result.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Assembly
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
    """
    Assemble all features for one stock into a single long-format DataFrame.
    """
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

    # Ensure Timestamp is datetime
    combined["Timestamp"] = pd.to_datetime(combined["Timestamp"])

    # Sort: Timestamp, then frequency order (D < M < Q), then variable name
    freq_order = {"D": 0, "M": 1, "Q": 2}
    combined["_freq_sort"] = combined["Frequency"].map(freq_order)
    combined = (
        combined
        .sort_values(["Timestamp", "_freq_sort", "Variable"])
        .drop(columns=["_freq_sort"])
        .reset_index(drop=True)
    )

    # Drop any NaN values
    combined = combined.dropna(subset=["Value"])

    return combined[["Timestamp", "Variable", "Value", "Frequency"]]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load raw data
    print("Loading VOLARE data...")
    volare = pd.read_csv(VOLARE_PATH, parse_dates=["date"])
    tickers = sorted(volare["symbol"].unique().tolist())
    print(f"  {len(tickers)} tickers, {len(volare)} rows, "
          f"{volare['date'].min().date()} to {volare['date'].max().date()}")

    print("Loading Compustat data...")
    compustat = pd.read_csv(COMPUSTAT_PATH)
    print(f"  {compustat['tic'].nunique()} tickers, {len(compustat)} rows")

    print("Loading FRED-MD macro data (monthly)...")
    macro_monthly_long = build_macro_monthly(FRED_MD_PATH)
    print(f"  {macro_monthly_long['Variable'].nunique()} monthly macro vars, {len(macro_monthly_long)} rows")

    print("Loading FRED-QD macro data (quarterly)...")
    macro_quarterly_long = build_macro_quarterly(FRED_QD_PATH)
    print(f"  {macro_quarterly_long['Variable'].nunique()} quarterly macro vars, {len(macro_quarterly_long)} rows")

    # Compute all monthly stock features (needed for cross-sectional aggregates)
    print("\nComputing monthly stock features for all tickers...")
    monthly_by_ticker = compute_all_monthly_stock_features(volare, tickers)

    print("Computing cross-sectional monthly aggregates...")
    cross_sectional = build_cross_sectional_monthly(monthly_by_ticker)
    print(f"  {len(cross_sectional)} months")

    # Build per-stock long-format files
    print(f"\nBuilding per-stock long-format CSVs in {OUTPUT_DIR}/")
    for ticker in tickers:
        stock_volare = volare[volare["symbol"] == ticker].copy()

        # Daily
        daily_long = build_daily_features(stock_volare, ticker)

        # Monthly (stock-level)
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
        print(f"  {ticker}: {len(combined):>6,} rows  ({n_daily}D, {n_monthly}M, {n_quarterly}Q)  "
              f"{n_targets} quarterly targets  -> {out_path.name}")

    print("\nDone.")


if __name__ == "__main__":
    main()
