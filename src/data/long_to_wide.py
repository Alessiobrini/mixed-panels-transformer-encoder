import pandas as pd
from pathlib import Path

def long_to_wide(
    input_path: Path,
    output_path: Path,
    time_col: str = "Timestamp",
    variable_col: str = "Variable",
    value_col: str = "Value",
    freq_col: str = "Frequency",
    fillna: bool = True
) -> pd.DataFrame:
    """
    Convert long-format mixed-frequency data to wide format:
    - One row per timestamp
    - One column per variable
    - Frequency label retained
    """
    df = pd.read_csv(input_path, parse_dates=[time_col])
    print(f"Loaded {len(df)} rows from {input_path}")

    wide_df = df.pivot(index=time_col, columns=variable_col, values=value_col)
    wide_df = wide_df.sort_index()

    freq_lookup = df.drop_duplicates(time_col).set_index(time_col)[freq_col]
    wide_df[freq_col] = freq_lookup.reindex(wide_df.index)

    if fillna:
        wide_df.ffill(inplace=True)

    wide_df = wide_df.reset_index()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    wide_df.to_csv(output_path, index=False)
    print(f"Saved wide-format data to {output_path.resolve()}")

    return wide_df

# --------------------------------------
# Runner logic
# --------------------------------------

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    input_csv = project_root / "data" / "raw" / "toy_mixed_frequency.csv"
    output_csv = project_root / "data" / "processed" / "mixed_freq_wide.csv"

    long_to_wide(input_path=input_csv, output_path=output_csv)
