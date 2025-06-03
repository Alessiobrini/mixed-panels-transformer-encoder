import pandas as pd
from pathlib import Path

def long_to_wide_with_frequency_rows(
    input_path: Path,
    output_path: Path,
    time_col: str = "Timestamp",
    variable_col: str = "Variable",
    value_col: str = "Value",
    freq_col: str = "Frequency",
    fillna: bool = True
) -> pd.DataFrame:
    """
    Convert long-format mixed-frequency data to a wide format:
    - Each original observation retained (multiple rows per timestamp allowed)
    - One column per variable
    - Frequency column preserved correctly
    """
    df = pd.read_csv(input_path, parse_dates=[time_col])
    print(f"Loaded {len(df)} rows from {input_path}")

    # Ensure all necessary columns exist
    for col in [time_col, variable_col, value_col, freq_col]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Assign a unique ID to each group of (Timestamp, Frequency)
    df["_row_id"] = df.groupby([time_col, freq_col], sort=False).ngroup()

    # Pivot the variables to columns
    value_wide = df.pivot(index="_row_id", columns=variable_col, values=value_col)

    # Extract corresponding metadata
    meta = df[[time_col, freq_col, "_row_id"]].drop_duplicates("_row_id").set_index("_row_id")

    # Join metadata with pivoted values
    wide_df = meta.join(value_wide).reset_index(drop=True)

    # Optional: sort by time and frequency priority
    freq_priority = {"D": 1, "ME": 2, "QE": 3}
    wide_df["freq_sort"] = wide_df[freq_col].map(freq_priority)
    wide_df = wide_df.sort_values(by=[time_col, "freq_sort"]).drop(columns="freq_sort")

    # Forward-fill if requested
    if fillna:
        wide_df = wide_df.sort_values(time_col)
        wide_df.update(wide_df.ffill())

    # Save to disk
    output_path.parent.mkdir(parents=True, exist_ok=True)
    wide_df.to_csv(output_path, index=False)
    print(f"Saved wide-format data to {output_path.resolve()}")

    return wide_df

# --------------------------------------
# Runner logic
# --------------------------------------

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    input_csv = project_root / "data" / "processed" / "toy_mixed_frequency_long.csv"

    output_csv = project_root / "data" / "processed" / "mixed_freq_wide.csv"

    long_to_wide_with_frequency_rows(input_path=input_csv, output_path=output_csv)
