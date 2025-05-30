import sys
from pathlib import Path
import pandas as pd
import torch
import matplotlib.pyplot as plt

# Setup project path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.data.mixed_frequency_dataset import MixedFrequencyDataset

# -------------------
# Config
# -------------------
CSV_PATH = project_root / "data" / "processed" / "toy_mixed_frequency_long.csv"
CONTEXT_DAYS = 90
TARGET_VARIABLE = "Y"

# -------------------
# Load Dataset
# -------------------
dataset = MixedFrequencyDataset(
    csv_path=CSV_PATH,
    context_days=CONTEXT_DAYS,
    target_variable=TARGET_VARIABLE
)

# -------------------
# Inspect N Samples
# -------------------
N = 5  # Number of samples to inspect
for i in range(N):
    sample = dataset[i]

    print(f"\n=== Sample {i} ===")
    print(f"Context window length: {sample['value'].shape[0]}")
    print(f"Target value: {sample['target'].item():.4f}")

    df = pd.DataFrame({
        "value": sample["value"].numpy(),
        "var_id": sample["var_id"].numpy(),
        "freq_id": sample["freq_id"].numpy(),
        "time_id": sample["time_id"].numpy()
    })
    print(df.head())

    # Optional: plot the values
    plt.figure(figsize=(6, 2.5))
    plt.plot(df["value"], label="Input values")
    plt.title(f"Sample {i} – Context Window")
    plt.xlabel("Time step")
    plt.ylabel("Scaled value")
    plt.tight_layout()
    plt.grid()
    plt.legend()
    plt.show()
