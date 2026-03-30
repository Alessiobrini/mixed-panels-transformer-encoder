import pandas as pd
from pandas.api.types import CategoricalDtype
import numpy as np
from pathlib import Path
from typing import List, Dict
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

def generate_mean_reverting_series(
    start: str,
    end: str,
    freq: str,
    var_params: Dict[str, Dict[str, float]],
) -> pd.DataFrame:
    """
    Generates time series for variables following mean-reverting dynamics.
    
    var_params: dict of var_name -> {mean, std, phi}
    """
    dates = pd.date_range(start=start, end=end, freq=freq)
    data = []

    for var, params in var_params.items():
        mu = params["mean"]
        std = params["std"]
        phi = params["phi"]

        values = [mu]  # Start at mean
        for _ in range(1, len(dates)):
            prev = values[-1]
            noise = np.random.normal(0, std)
            val = mu + phi * (prev - mu) + noise
            values.append(val)

        rows = zip(dates, [var] * len(dates), values, [freq] * len(dates))
        data.extend(rows)

    return pd.DataFrame(data, columns=["Timestamp", "Variable", "Value", "Frequency"])

if __name__ == "__main__":

    project_root = Path(__file__).resolve().parent.parent.parent
    output_dir = project_root / "data" / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)

    start_date = "1980-01-01"
    end_date = "2024-12-31"

    daily_vars = {
        "D1": {"mean": 5.0, "std": 0.5, "phi": 0.7}
    }
    monthly_vars = {
        "M1": {"mean": 6.0, "std": 0.4, "phi": 0.8},
        "M2": {"mean": 6.5, "std": 0.6, "phi": 0.75},
        "M3": {"mean": 5.5, "std": 0.3, "phi": 0.9}
    }

    df_daily = generate_mean_reverting_series(start_date, end_date, "D", daily_vars)
    df_monthly = generate_mean_reverting_series(start_date, end_date, "ME", monthly_vars)

    # Compute quarterly averages for lags
    df_d1_q = df_daily[df_daily["Variable"] == "D1"].copy()
    df_d1_q["Quarter"] = df_d1_q["Timestamp"].dt.to_period("Q")
    avg_d1_q = df_d1_q.groupby("Quarter")["Value"].mean()

    df_m1_q = df_monthly[df_monthly["Variable"] == "M1"].copy()
    df_m1_q["Quarter"] = df_m1_q["Timestamp"].dt.to_period("Q")
    avg_m1_q = df_m1_q.groupby("Quarter")["Value"].mean()

    # Create synthetic Y with temporal memory
    quarterly_index = pd.date_range(start=start_date, end=end_date, freq="Q")
    y_values = []
    y_prev = 10.0  # Initial Y_0 baseline

    for q_date in quarterly_index:
        q_str = q_date.to_period("Q")
        d1_lag = avg_d1_q.get(q_str - 1, np.nan)
        m1_lag = avg_m1_q.get(q_str - 1, np.nan)
        if pd.isna(d1_lag) or pd.isna(m1_lag):
            continue
        noise = np.random.normal(loc=0.0, scale=0.02)
        y = 0.5 * y_prev + 0.3 * d1_lag + 0.2 * m1_lag + noise
        y_values.append([q_date, "Y", y, "QE"])
        y_prev = y

    df_quarterly = pd.DataFrame(y_values, columns=["Timestamp", "Variable", "Value", "Frequency"])

    (df_daily.sort_values("Timestamp")
     .to_csv(output_dir / "toy_daily.csv", index=False))
    (df_monthly.sort_values("Timestamp")
     .to_csv(output_dir / "toy_monthly.csv", index=False))
    (df_quarterly.sort_values("Timestamp")
     .to_csv(output_dir / "toy_quarterly.csv", index=False))

    df_all = pd.concat([df_daily, df_monthly, df_quarterly], axis=0)

    var_order = CategoricalDtype(
        categories=["D1", "M1", "M2", "M3", "Y"],
        ordered=True
    )
    df_all["Variable"] = df_all["Variable"].astype(var_order)
    df_all = df_all.sort_values(["Timestamp", "Variable"]).reset_index(drop=True)

    processed_dir = project_root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    output_path = processed_dir / "toy_mixed_frequency_long.csv"
    df_all.to_csv(output_path, index=False)
    print(f"Toy dataset with memory saved to {output_path.resolve()}")

    # Autocorrelation of Y
    y_series = df_quarterly.set_index("Timestamp")["Value"]
    plot_acf(y_series.dropna(), lags=20)
    plt.title("Autocorrelation of Y")
    plt.show()

    # Correlation with other variables
    pivot_df = df_all.pivot(index="Timestamp", columns="Variable", values="Value")
    correlation_matrix = pivot_df.corr()
    print("Correlation with Y:")
    print(correlation_matrix["Y"].drop("Y").round(3))
