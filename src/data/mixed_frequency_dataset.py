import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, Dict, List
from sklearn.preprocessing import StandardScaler

class MixedFrequencySequenceDataset(Dataset):
    """
    Dataset for mixed-frequency forecasting:
    - Each item is a 90-day context window of long-format tokens
    - The target is the next Y (quarterly) observed after the window
    """

    def __init__(
        self,
        csv_path: Union[str, Path],
        context_days: int = 90,
        time_column: str = "Timestamp",
        variable_column: str = "Variable",
        freq_column: str = "Frequency",
        value_column: str = "Value",
        target_variable: str = "Y"
    ):
        self.df = pd.read_csv(csv_path, parse_dates=[time_column])
        self.df = self.df.sort_values(time_column).reset_index(drop=True)

        self.time_column = time_column
        self.variable_column = variable_column
        self.freq_column = freq_column
        self.value_column = value_column
        self.target_variable = target_variable
        self.context_days = context_days

        # Time IDs: integer day from start
        self.df["time_id"] = (self.df[time_column] - self.df[time_column].min()).dt.days
        self.time_ids = self.df["time_id"].values

        # Embedding maps
        self.freq_map = {f: i for i, f in enumerate(self.df[freq_column].unique())}
        self.var_map = {v: i for i, v in enumerate(self.df[variable_column].unique())}
        self.df["freq_id"] = self.df[freq_column].map(self.freq_map)
        self.df["var_id"] = self.df[variable_column].map(self.var_map)

        # Standardize values
        self.scaler = StandardScaler()
        self.df["scaled_value"] = self.scaler.fit_transform(self.df[[value_column]])

        # Build list of sequence/target pairs
        self.sequence_windows = self._build_sequence_targets()

    def _build_sequence_targets(self) -> List[Dict]:
        """
        Build sequences and their corresponding future Y target.
        Each sequence includes all tokens in a rolling 90-day window.
        """
        result = []
        unique_days = sorted(self.df["time_id"].unique())
        max_day = self.df["time_id"].max()

        for start_day in unique_days:
            end_day = start_day + self.context_days
            if end_day >= max_day:
                break

            # Context window tokens
            context_df = self.df[(self.df["time_id"] >= start_day) & (self.df["time_id"] < end_day)]

            # Find the next available Y after the window
            target_row = self.df[
                (self.df["time_id"] > end_day) &
                (self.df[self.variable_column] == self.target_variable)
            ].head(1)

            if target_row.empty:
                continue

            result.append({
                "context": context_df,
                "target_value": float(target_row[self.value_column].values[0])
            })

        return result

    def __len__(self) -> int:
        return len(self.sequence_windows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.sequence_windows[idx]
        context = item["context"]

        return {
            "value": torch.tensor(context["scaled_value"].values, dtype=torch.float32),
            "var_id": torch.tensor(context["var_id"].values, dtype=torch.long),
            "freq_id": torch.tensor(context["freq_id"].values, dtype=torch.long),
            "time_id": torch.tensor(context["time_id"].values, dtype=torch.long),
            "target": torch.tensor(item["target_value"], dtype=torch.float32)
        }
