import sys
import pandas as pd
from pathlib import Path

# ensure project root on path so `import src...` works
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))
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
    """
    md = pd.read_csv(md_path, parse_dates=['date'])
    qd = pd.read_csv(qd_path, parse_dates=['date'])

    md_filtered = md[['date'] + monthly_vars].dropna(subset=monthly_vars)
    qd_filtered = qd[['date'] + quarterly_vars].dropna(subset=quarterly_vars)

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

    md_long['Frequency'] = 'M'
    qd_long['Frequency'] = 'Q'

    long_df = pd.concat([md_long, qd_long], ignore_index=True)
    long_df.rename(columns={'date': 'Timestamp'}, inplace=True)

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
    # Setup
    cfg_path = project_root / "src" / "config" / "cfg.yml"
    config   = Config(cfg_path)

    # Paths to raw FRED files
    md_path = project_root / config.paths.data_raw_fred_monthly
    qd_path = project_root / config.paths.data_raw_fred_quarterly

    # Read header to discover columns
    md_cols = pd.read_csv(md_path, nrows=0).columns.tolist()

    # Build the variable lists
    if config.features.all_monthly:
        monthly_vars   = [c for c in md_cols if c != 'date']
        quarterly_vars = []
    else:
        monthly_vars   = config.features.monthly_vars
        quarterly_vars = config.features.quarterly_vars

    # Create an informative suffix: e.g. "3M_1Q"
    suffix = f"{len(monthly_vars)}M_{len(quarterly_vars)}Q"

    # Build the long-format DataFrame
    long_df = create_long_format_FRED(
        md_path,
        qd_path,
        monthly_vars,
        quarterly_vars
    )

    print(long_df.head())

    # Save with dynamic file name
    output_path = project_root / config.paths.data_processed_template.format(suffix=suffix)
    long_df.to_csv(output_path, index=False)
    print(f"\nSaved long-format data to: {output_path.resolve()}")
