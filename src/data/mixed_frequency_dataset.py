import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, Dict
from sklearn.preprocessing import StandardScaler

class MixedFrequencyDataset(Dataset):
    """
    PyTorch Dataset for long-format mixed-frequency time series.

    Each row corresponds to a single observation (variable, time, value, frequency).
    """

    def __init__(
        self,
        csv_path: Union[str, Path],
        time_column: str = "Timestamp",
        variable_column: str = "Variable",
        freq_column: str = "Frequency",
        value_column: str = "Value",
        target_variable: str = "Y"
    ) -> None:
        self.df = pd.read_csv(csv_path, parse_dates=[time_column])

        # Frequency, time, and variable mappings
        self.freq_map = {freq: i for i, freq in enumerate(self.df[freq_column].unique())}
        self.var_map = {var: i for i, var in enumerate(self.df[variable_column].unique())}

        self.time_ids = (self.df[time_column] - self.df[time_column].min()).dt.days
        print("Max time ID:", self.time_ids.max())

        self.freq_ids = self.df[freq_column].map(self.freq_map).values
        self.var_ids = self.df[variable_column].map(self.var_map).values

        # Value: scale it to stabilize learning
        self.scaler = StandardScaler()
        self.values = self.scaler.fit_transform(self.df[[value_column]]).astype(np.float32).flatten()

        # Define is_target mask: only true for target variable "Y"
        self.is_target_row = (self.df[variable_column] == target_variable).values
        self.targets = self.df[value_column].fillna(0).astype(np.float32).values

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "value": torch.tensor(self.values[idx], dtype=torch.float32),
            "var_id": torch.tensor(self.var_ids[idx], dtype=torch.long),
            "freq_id": torch.tensor(self.freq_ids[idx], dtype=torch.long),
            "time_id": torch.tensor(self.time_ids[idx], dtype=torch.long),
            "target": torch.tensor(self.targets[idx], dtype=torch.float32),
            "is_target": torch.tensor(self.is_target_row[idx], dtype=torch.bool)
        }
