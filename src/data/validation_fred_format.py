import pandas as pd
from pathlib import Path
from src.utils.config import Config

def validate_variable_coverage(df: pd.DataFrame) -> None:
    # Determine overall coverage range
    global_start = df['Timestamp'].min()
    global_end = df['Timestamp'].max()

    print(f"\nGlobal date coverage: {global_start.date()} to {global_end.date()}")

    # Frequency-specific expectations (from config)
    expected_freq = vars(config.validation.expected_freq)

    for freq in ['M', 'Q']:
        print(f"\nChecking variables with frequency '{freq}'")

        # Expected full date range for this frequency
        pd_freq = expected_freq[freq]
        expected_dates = pd.date_range(start=global_start, end=global_end, freq=pd_freq)

        for var in sorted(df[df['Frequency'] == freq]['Variable'].unique()):
            var_dates = (
                df[(df['Frequency'] == freq) & (df['Variable'] == var)]
                ['Timestamp']
                .drop_duplicates()
                .sort_values()
            )
            var_start = var_dates.min()
            var_end = var_dates.max()

            # Check for start/end misalignment
            start_msg = "OK"
            end_msg = "OK"

            if var_start > global_start:
                start_msg = f"starts late (from {var_start.date()})"
            if var_end < global_end:
                end_msg = f"ends early (until {var_end.date()})"

            # Check for missing dates in expected range
            expected_var_range = expected_dates[
                (expected_dates >= var_start) & (expected_dates <= var_end)
            ]
            missing_dates = expected_var_range.difference(var_dates)

            if missing_dates.empty and start_msg == "OK" and end_msg == "OK":
                print(f"  {var}: OK — full coverage")
            else:
                print(f"  {var}:")
                print(f"    - Start check: {start_msg}")
                print(f"    - End check: {end_msg}")
                print(f"    - Missing {len(missing_dates)} date(s) in range")

if __name__ == "__main__":
    # Set project root based on script location
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent

    # Load config
    cfg_path = project_root / "src" / "config" / "cfg.yml"
    config = Config(cfg_path)

    # Load long-format CSV from config
    input_path = project_root / config.paths.data_processed_long
    long_df = pd.read_csv(input_path, parse_dates=['Timestamp'])

    # Validate
    validate_variable_coverage(long_df)
