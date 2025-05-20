import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from typing import List, Optional, Dict, Union

class MixedFrequencyDataset(Dataset):
    """
    PyTorch Dataset for wide-format mixed-frequency time series.

    Each row is a timestamped observation containing multiple variables
    from different frequencies (e.g., daily, monthly, quarterly).
    """

    def __init__(
        self,
        csv_path: Union[str, Path],
        time_column: str = "Timestamp",
        freq_column: str = "Frequency",
        target_column: str = "Y",
        feature_columns: Optional[List[str]] = None,
    ) -> None:
        self.df = pd.read_csv(csv_path, parse_dates=[time_column])
        self.freq_map: Dict[str, int] = {freq: i for i, freq in enumerate(self.df[freq_column].unique())}

        # Time index: integer starting from 0
        self.time_ids = (self.df[time_column] - self.df[time_column].min()).dt.days

        # Frequency ID (0 = daily, 1 = monthly, 2 = quarterly, etc.)
        self.freq_ids = self.df[freq_column].map(self.freq_map).values

        # Extract target values
        self.targets = self.df[target_column].fillna(0).astype(np.float32).values

        # Determine features to use
        self.feature_columns = feature_columns or [
            col for col in self.df.columns if col not in [time_column, freq_column, target_column]
        ]

        # Standardize features
        self.scaler = StandardScaler()
        self.features = self.scaler.fit_transform(
            self.df[self.feature_columns].fillna(method="ffill")
        ).astype(np.float32)
        
        
        self.is_target_row = self.df[target_column].notna().astype(np.bool_).values
        self.targets = self.df[target_column].fillna(0).astype(np.float32).values


    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        raw_input = self.features[idx]
        freq_id = self.freq_ids[idx]
        time_id = self.time_ids[idx]
        target = self.targets[idx]
        

        return {
            "raw_input": torch.tensor(raw_input, dtype=torch.float32),
            "freq_id": torch.tensor(freq_id, dtype=torch.long),
            "time_id": torch.tensor(time_id, dtype=torch.long),
            "target": torch.tensor(target, dtype=torch.float32),
            "is_target": torch.tensor(self.is_target_row[idx], dtype=torch.bool)
        }
