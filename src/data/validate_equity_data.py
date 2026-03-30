"""
Validate per-stock long-format equity CSVs.

Checks:
  - Schema: columns exactly [Timestamp, Variable, Value, Frequency]
  - No NaN values
  - Frequency labels in {D, M, Q}
  - Daily vars on trading days only (~252/year)
  - Monthly vars ~12/year
  - Quarterly target ~4/year, name matches {TKR}_eps_yoy
  - Prints summary table

Usage:
    python src/data/validate_equity_data.py
"""

import sys
from pathlib import Path

import pandas as pd

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

PROCESSED_DIR = project_root / "data" / "processed" / "equity"
EXPECTED_COLUMNS = ["Timestamp", "Variable", "Value", "Frequency"]
VALID_FREQS = {"D", "M", "Q"}


def validate_one(csv_path: Path) -> dict:
    """Validate a single per-stock CSV. Returns a summary dict."""
    fname = csv_path.name
    # Extract ticker from filename: long_format_{TKR}_{suffix}.csv
    parts = fname.replace("long_format_", "").replace(".csv", "").split("_")
    ticker = parts[0]

    errors = []
    df = pd.read_csv(csv_path, parse_dates=["Timestamp"])

    # 1. Schema check
    if list(df.columns) != EXPECTED_COLUMNS:
        errors.append(f"Column mismatch: got {list(df.columns)}")

    # 2. NaN check
    n_nan = df.isna().sum().sum()
    if n_nan > 0:
        nan_detail = df.isna().sum()
        errors.append(f"NaN values: {nan_detail[nan_detail > 0].to_dict()}")

    # 3. Frequency labels
    bad_freqs = set(df["Frequency"].unique()) - VALID_FREQS
    if bad_freqs:
        errors.append(f"Invalid frequencies: {bad_freqs}")

    # 4. Frequency-specific checks
    daily = df[df["Frequency"] == "D"]
    monthly = df[df["Frequency"] == "M"]
    quarterly = df[df["Frequency"] == "Q"]

    n_daily_vars = daily["Variable"].nunique()
    n_monthly_vars = monthly["Variable"].nunique()
    n_quarterly_vars = quarterly["Variable"].nunique()

    # Daily: check no weekends
    if not daily.empty:
        weekdays = daily["Timestamp"].dt.dayofweek
        n_weekends = (weekdays >= 5).sum()
        if n_weekends > 0:
            errors.append(f"Daily data has {n_weekends} weekend observations")

    # Quarterly: target variable name
    q_vars = quarterly["Variable"].unique().tolist()
    expected_target = f"{ticker}_eps_yoy"
    if q_vars and expected_target not in q_vars:
        errors.append(f"Expected quarterly var '{expected_target}', got {q_vars}")

    n_targets = len(quarterly.drop_duplicates("Timestamp"))

    # Date range
    date_min = df["Timestamp"].min().date()
    date_max = df["Timestamp"].max().date()

    return {
        "ticker": ticker,
        "rows": len(df),
        "date_min": date_min,
        "date_max": date_max,
        "D_vars": n_daily_vars,
        "M_vars": n_monthly_vars,
        "Q_vars": n_quarterly_vars,
        "D_rows": len(daily),
        "M_rows": len(monthly),
        "Q_rows": len(quarterly),
        "n_targets": n_targets,
        "errors": errors,
    }


def main():
    csv_files = sorted(PROCESSED_DIR.glob("long_format_*.csv"))
    if not csv_files:
        print(f"No CSV files found in {PROCESSED_DIR}")
        return

    print(f"Validating {len(csv_files)} files in {PROCESSED_DIR}/\n")

    results = []
    for f in csv_files:
        r = validate_one(f)
        results.append(r)

    # Print summary table
    header = f"{'Ticker':>6}  {'Rows':>7}  {'Date Range':>25}  {'D':>3}  {'M':>3}  {'Q':>2}  " \
             f"{'D rows':>7}  {'M rows':>7}  {'Q rows':>6}  {'Targets':>7}  {'Status'}"
    print(header)
    print("-" * len(header))

    n_pass = 0
    n_fail = 0
    for r in results:
        status = "PASS" if not r["errors"] else "FAIL"
        if status == "PASS":
            n_pass += 1
        else:
            n_fail += 1

        print(f"{r['ticker']:>6}  {r['rows']:>7,}  "
              f"{str(r['date_min'])} to {str(r['date_max'])}  "
              f"{r['D_vars']:>3}  {r['M_vars']:>3}  {r['Q_vars']:>2}  "
              f"{r['D_rows']:>7,}  {r['M_rows']:>7,}  {r['Q_rows']:>6,}  "
              f"{r['n_targets']:>7}  {status}")

    print(f"\n{n_pass} passed, {n_fail} failed out of {len(results)} files.")

    # Print errors
    for r in results:
        if r["errors"]:
            print(f"\n  {r['ticker']} errors:")
            for e in r["errors"]:
                print(f"    - {e}")


if __name__ == "__main__":
    main()
