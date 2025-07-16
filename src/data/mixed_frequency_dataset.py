import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, Dict, List
from sklearn.preprocessing import StandardScaler
import pdb

class MixedFrequencyDataset(Dataset):
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

        # Standardize values: one scaler per variable
        self.scalers = {}
        scaled_values = []
        for var in self.df[self.variable_column].unique():
            mask = self.df[self.variable_column] == var
            scaler = StandardScaler()
            scaled_col = scaler.fit_transform(self.df.loc[mask, [value_column]])
        
            self.scalers[var] = scaler
            scaled_values.append(pd.Series(scaled_col.flatten(), index=self.df[mask].index))
        
        # Merge all scaled values and assign in original order
        self.df["scaled_value"] = pd.concat(scaled_values).sort_index()
        
        # Keep the scaler for the target variable to use later during inverse transform
        self.scaler = self.scalers[self.target_variable]
 
        # Build list of sequence/target pairs
        self.sequence_windows = self._build_sequence_targets()

    def inverse_transform(self, var_name: str, values: np.ndarray) -> np.ndarray:
        return self.scalers[var_name].inverse_transform(values.reshape(-1, 1)).flatten()


    def _build_sequence_targets(self) -> List[Dict]:
        """
        Build context-target pairs where:
        - Context: the 100-day window before each quarterly Y observation
        - Target: the Y value at that quarterly timestamp
        """
        
        result = []
    
        # Identify all rows where the target variable appears (i.e., Y observations)
        # target_rows = self.df[self.df[self.variable_column] == self.target_variable]
        target_rows = self.df[
                        (self.df[self.variable_column] == self.target_variable) &
                        (self.df[self.freq_column] == 'Q')
                    ]


        for _, row in target_rows.iterrows():
            target_time_id = row["time_id"]
            context_start = target_time_id - self.context_days
            # context_end = target_time_id  # exclusive
            context_end = self.df[
                                (self.df[self.variable_column] == self.target_variable) &
                                (self.df["time_id"] < target_time_id)
                            ]["time_id"].max()

 
            if context_start < 0:
                continue  # Not enough history to build context
    
            # Extract context window
            context_df = self.df[
                (self.df["time_id"] >= context_end - self.context_days) & 
                (self.df["time_id"] <= context_end)
            ]

    
            if context_df.empty:
                continue  # Skip if context has no data
 
            result.append({
                "context": context_df,
                "target_value": float(row["scaled_value"])

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
