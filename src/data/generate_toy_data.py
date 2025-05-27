import pandas as pd
import numpy as np
from pathlib import Path
from typing import List

def generate_variable_series(
    start: str,
    end: str,
    freq: str,
    var_names: List[str],
    mean: float,
    std: float
) -> pd.DataFrame:
    """
    Generic generator for any frequency and variable list.

    Parameters:
        - start: start date string (e.g., '2022-01-01')
        - end: end date string (e.g., '2022-12-31')
        - freq: pandas frequency string ('D', 'M', 'Q', etc.)
        - var_names: list of variable names to generate
        - mean: average value of the variable
        - std: standard deviation

    Returns:
        - DataFrame in long format with Timestamp, Variable, Value, Frequency
    """
    dates = pd.date_range(start=start, end=end, freq=freq)
    data = []

    for var in var_names:
        values = np.random.normal(loc=mean, scale=std, size=len(dates))
        rows = zip(dates, [var]*len(dates), values, [freq]*len(dates))
        data.extend(rows)

    return pd.DataFrame(data, columns=["Timestamp", "Variable", "Value", "Frequency"])

if __name__ == "__main__":
    
    project_root = Path(__file__).resolve().parent.parent.parent  # go up from src/data to root
    output_dir = project_root / "data" / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)

    start_date = "2015-01-01"
    end_date = "2022-12-31"

    df_daily = generate_variable_series(start_date, end_date, "D", ["D1"], mean=5.0, std=1.0)
    df_monthly = generate_variable_series(start_date, end_date, "ME", ["M1", "M2", "M3"], mean=50.0, std=10.0)
    # df_quarterly = generate_variable_series(start_date, end_date, "QE", ["Y"], mean=200.0, std=30.0)
    
    
    # Compute quarterly averages
    df_d1_q = df_daily[df_daily["Variable"] == "D1"].copy()
    df_d1_q["Quarter"] = df_d1_q["Timestamp"].dt.to_period("Q")
    avg_d1_q = df_d1_q.groupby("Quarter")["Value"].mean()
    
    df_m1_q = df_monthly[df_monthly["Variable"] == "M1"].copy()
    df_m1_q["Quarter"] = df_m1_q["Timestamp"].dt.to_period("Q")
    avg_m1_q = df_m1_q.groupby("Quarter")["Value"].mean()
    
    # Create synthetic Y
    quarterly_index = pd.date_range(start=start_date, end=end_date, freq="QE")
    y_values = []
    
    for q_date in quarterly_index:
        q_str = q_date.to_period("Q")
        d1_avg = avg_d1_q.get(q_str, np.nan)
        m1_avg = avg_m1_q.get(q_str, np.nan)
        if pd.isna(d1_avg) or pd.isna(m1_avg):
            continue
        noise = np.random.normal(loc=0.0, scale=5.0)
        y = 0.6 * d1_avg + 0.4 * m1_avg + noise
        y_values.append([q_date, "Y", y, "QE"])
    
    df_quarterly = pd.DataFrame(y_values, columns=["Timestamp", "Variable", "Value", "Frequency"])
    
    

    (df_daily.sort_values("Timestamp")
     .to_csv(output_dir / "toy_daily.csv", index=False))
    
    (df_monthly.sort_values("Timestamp")
     .to_csv(output_dir / "toy_monthly.csv", index=False))
    
    (df_quarterly.sort_values("Timestamp")
     .to_csv(output_dir / "toy_quarterly.csv", index=False))
    

    df_all = pd.concat([df_daily, df_monthly, df_quarterly], axis=0)
    df_all = df_all.sort_values("Timestamp").reset_index(drop=True)

    output_path = output_dir / "toy_mixed_frequency.csv"
    df_all.to_csv(output_path, index=False)
    print(f"Toy dataset saved to {output_path.resolve()}")
