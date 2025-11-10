import sys
import pandas as pd
from pathlib import Path

# ensure project root on path so `import src...` works
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))
from src.utils.config import Config
from src.utils.data_paths import resolve_variable_lists

def create_long_format_FRED(
    md_path: Path,
    qd_path: Path,
    monthly_vars: list,
    quarterly_vars: list,
) -> pd.DataFrame:
    """
    Transforms selected monthly and quarterly variables into a unified long-format DataFrame,
    preserving natural frequencies and ordering quarterly entries after monthly ones on the same date.
    """
    monthly_vars = list(monthly_vars or [])
    quarterly_vars = list(quarterly_vars or [])

    frames = []

    if monthly_vars:
        md = pd.read_csv(md_path, parse_dates=["date"]).sort_values("date").ffill()
        md.dropna(subset=monthly_vars, inplace=True)
        md_melt = md.melt(
            id_vars="date",
            value_vars=monthly_vars,
            var_name="Variable",
            value_name="Value",
        )
        md_long = md_melt.dropna(subset=["Value"]).assign(Frequency="M")
        frames.append(md_long)

    if quarterly_vars:
        qd = pd.read_csv(qd_path, parse_dates=["date"]).sort_values("date").ffill()
        qd.dropna(subset=quarterly_vars, inplace=True)
        qd_melt = qd.melt(
            id_vars="date",
            value_vars=quarterly_vars,
            var_name="Variable",
            value_name="Value",
        )
        qd_long = qd_melt.dropna(subset=["Value"]).assign(Frequency="Q")
        frames.append(qd_long)

    if not frames:
        raise ValueError("At least one of `monthly_vars` or `quarterly_vars` must be non-empty.")

    # Combine and rename
    long_df = pd.concat(frames, ignore_index=True)
    long_df.rename(columns={'date': 'Timestamp'}, inplace=True)

    # Sorting keys: frequency order then variable order
    freq_order = {'M': 0, 'Q': 1}
    long_df['FreqSort'] = long_df['Frequency'].map(freq_order)
    # Preserve the exact order you listed in the config
    var_order = {var: idx for idx, var in enumerate(monthly_vars + quarterly_vars)}
    long_df['VarOrder'] = long_df['Variable'].map(var_order)

    # Final sort and cleanup
    long_df = (
        long_df
        .sort_values(['Timestamp', 'FreqSort', 'VarOrder'])
        .drop(columns=['FreqSort', 'VarOrder'])
        .reset_index(drop=True)
    )

    return long_df

if __name__ == "__main__":
    # Setup
    cfg_path = project_root / "src" / "config" / "cfg.yaml"
    config   = Config(cfg_path)

    # Paths to raw FRED files
    md_path = project_root / config.paths.data_raw_fred_monthly
    qd_path = project_root / config.paths.data_raw_fred_quarterly

    # Build the variable lists from the config
    monthly_vars, quarterly_vars = resolve_variable_lists(config, project_root)

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
