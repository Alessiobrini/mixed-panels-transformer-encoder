import pandas as pd
from pathlib import Path
from src.utils.config import Config

def create_long_format_FRED(
    md_path: Path,
    qd_path: Path,
    monthly_vars: list,
    quarterly_vars: list
) -> pd.DataFrame:
    """
    Transforms selected monthly and quarterly variables into a unified long-format DataFrame,
    preserving natural frequencies and ordering quarterly entries after monthly ones on the same date.

    Parameters:
        md_path (Path): path to the transformed monthly CSV
        qd_path (Path): path to the transformed quarterly CSV
        monthly_vars (list): list of variable names to extract from monthly data
        quarterly_vars (list): list of variable names to extract from quarterly data

    Returns:
        pd.DataFrame: long-format DataFrame with columns: Timestamp, Variable, Value, Frequency (M or Q)
    """
    # Load datasets
    md = pd.read_csv(md_path, parse_dates=['date'])
    qd = pd.read_csv(qd_path, parse_dates=['date'])

    # Filter and drop missing independently
    md_filtered = md[['date'] + monthly_vars].dropna(subset=monthly_vars)
    qd_filtered = qd[['date'] + quarterly_vars].dropna(subset=quarterly_vars)

    # Melt into long format
    md_long = md_filtered.melt(
        id_vars='date',
        var_name='Variable',
        value_name='Value'
    )
    qd_long = qd_filtered.melt(
        id_vars='date',
        var_name='Variable',
        value_name='Value'
    )

    # Tag frequencies
    md_long['Frequency'] = 'M'
    qd_long['Frequency'] = 'Q'

    # Combine and rename
    long_df = pd.concat([md_long, qd_long], ignore_index=True)
    long_df.rename(columns={'date': 'Timestamp'}, inplace=True)

    # Ensure M comes before Q for the same timestamp
    freq_order = {'M': 0, 'Q': 1}
    long_df['FreqSort'] = long_df['Frequency'].map(freq_order)
    long_df = (
        long_df
        .sort_values(['Timestamp', 'FreqSort', 'Variable'])
        .drop(columns='FreqSort')
        .reset_index(drop=True)
    )

    return long_df

if __name__ == "__main__":
    # Set project root and load config
    project_root = Path(__file__).resolve().parents[2]
    cfg_path = project_root / "src" / "config" / "cfg.yml"
    config = Config(cfg_path)

    # Paths to raw FRED files from config
    md_path = project_root / config.paths.data_raw_fred_monthly
    qd_path = project_root / config.paths.data_raw_fred_quarterly

    # Variable lists from config
    monthly_vars   = config.features.monthly_vars
    quarterly_vars = config.features.quarterly_vars

    # Create long-format DataFrame
    long_df = create_long_format_FRED(
        md_path,
        qd_path,
        monthly_vars,
        quarterly_vars
    )

    # Preview output
    print(long_df.head())

    # Save to CSV (driven by config)
    output_path = project_root / config.paths.data_processed_long
    long_df.to_csv(output_path, index=False)
    print(f"\nSaved long-format data to: {output_path.resolve()}")
