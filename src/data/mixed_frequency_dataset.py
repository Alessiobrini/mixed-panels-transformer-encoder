import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, Dict, List
from sklearn.preprocessing import StandardScaler
from pandas.api.types import is_datetime64_any_dtype
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
        target_variable: str = "Y",
        allowed_variables: Union[None, List[str], set] = None,
        allowed_frequencies: Union[None, List[str], set] = None,
    ):
        self.df = pd.read_csv(csv_path, parse_dates=[time_column])

        if allowed_variables is not None:
            allowed_variables = set(allowed_variables)
            self.df = self.df[self.df[variable_column].isin(allowed_variables)].reset_index(drop=True)

        if allowed_frequencies is not None:
            allowed_frequencies = set(allowed_frequencies)
            self.df = self.df[self.df[freq_column].isin(allowed_frequencies)].reset_index(drop=True)


        self.time_column = time_column
        self.variable_column = variable_column
        self.freq_column = freq_column
        self.value_column = value_column
        self.target_variable = target_variable
        self.requested_context_days = context_days
        
        # Time IDs: integer offset from earliest timestamp / identifier
        time_series = self.df[time_column]

        if is_datetime64_any_dtype(time_series):
            time_delta = time_series - time_series.min()
            self.df["time_id"] = (time_delta / pd.Timedelta(days=1)).astype(int)
            self.uses_calendar_days = True
        else:
            numeric_time = pd.to_numeric(time_series, errors="coerce")
            if numeric_time.isna().any():
                raise ValueError(
                    "Time column must be convertible to datetime or numeric values to build time ids."
                )
            time_offsets = numeric_time - numeric_time.min()
            if not np.allclose(time_offsets, np.round(time_offsets)):
                raise ValueError(
                    "Numeric time column must contain integer-like values to derive time ids."
                )
            time_offsets = np.round(time_offsets).astype(int)
            self.df[time_column] = numeric_time
            self.df["time_id"] = time_offsets
            self.uses_calendar_days = False
        self.time_ids = self.df["time_id"].values
        
        self.context_days = self._resolve_context_days(self.requested_context_days)

        # Embedding maps
        self.freq_map = {f: i for i, f in enumerate(self.df[freq_column].unique())}
        self.var_map = {v: i for i, v in enumerate(self.df[variable_column].unique())}
        self.df["freq_id"] = self.df[freq_column].map(self.freq_map)
        self.df["var_id"] = self.df[variable_column].map(self.var_map)

        self.scalers = {}
        self.scaler = None
        self.df["scaled_value"] = np.nan  # will be filled after train-only fit
 
        # Build list of sequence/target pairs
        self.skipped_context = 0
        self.sequence_windows = self._build_sequence_targets()
        
        
        

    def inverse_transform(self, var_name: str, values: np.ndarray) -> np.ndarray:
        return self.scalers[var_name].inverse_transform(values.reshape(-1, 1)).flatten()


    def _resolve_context_days(self, requested_span: int) -> int:
        """Return an effective context span measured in the dataset's time units."""

        if getattr(self, "uses_calendar_days", False):
            return max(1, int(round(requested_span)))

        quarter_in_days = 90.0

        target_mask = self.df[self.variable_column] == self.target_variable
        if self.freq_column in self.df.columns:
            freq_series = self.df[self.freq_column].astype(str).str.upper()
            if "Q" in freq_series.unique():
                target_mask &= freq_series == "Q"
            else:
                target_mask &= self.df[self.freq_column].notna()

        target_time_ids = (
            self.df.loc[target_mask, "time_id"].dropna().astype(int).unique()
        )
        if target_time_ids.size < 2:
            return max(1, int(round(requested_span)))

        target_time_ids.sort()
        diffs = np.diff(target_time_ids)
        diffs = diffs[diffs > 0]
        if diffs.size == 0:
            return max(1, int(round(requested_span)))

        median_gap = float(np.median(diffs))
        if np.isclose(median_gap, quarter_in_days, rtol=0.1):
            return max(1, int(round(requested_span)))

        scaled_context = int(round((requested_span / quarter_in_days) * median_gap))
        return max(1, scaled_context)

    def _build_sequence_targets(self) -> List[Dict]:
        """
        Build context-target pairs where:
        - Context: the 100-day window before each quarterly Y observation
        - Target: the Y value at that quarterly timestamp
        """
        
        result = []
    
        # Identify all rows where the target variable appears (i.e., Y observations)
        target_rows = self.df[
                        (self.df[self.variable_column] == self.target_variable) &
                        (self.df[self.freq_column] == 'Q')
                    ]

        for _, row in target_rows.iterrows():
            target_time_id = row["time_id"]
            context_end = self.df[
                (self.df[self.variable_column] == self.target_variable)
                & (self.df["time_id"] < target_time_id)
            ]["time_id"].max()

            if pd.notna(context_end):
                context_end = int(context_end)
            else:
                self.skipped_context += 1
                continue
            context_start = context_end - self.context_days

            if context_start < 0:
                self.skipped_context += 1
                continue  # Not enough history to build context

            context_idx = self.df.index[
                (self.df["time_id"] >= context_start) & (self.df["time_id"] <= context_end)
            ]
            if len(context_idx) == 0:
                self.skipped_context += 1
                continue
 
            result.append({
                "context_idx": context_idx.to_numpy(),
                "target_idx": int(row.name)

            })
  
    
        return result
    
    def __len__(self) -> int:
        return len(self.sequence_windows)

    def __getitem__(self, idx: int):
        item = self.sequence_windows[idx]
        ctx = self.df.loc[item["context_idx"]]                  
        y  = float(self.df.loc[item["target_idx"], "scaled_value"])
    
        return {
            "value":  torch.tensor(ctx["scaled_value"].values, dtype=torch.float32),
            "var_id": torch.tensor(ctx["var_id"].values,       dtype=torch.long),
            "freq_id":torch.tensor(ctx["freq_id"].values,      dtype=torch.long),
            "time_id":torch.tensor(ctx["time_id"].values,      dtype=torch.long),
            "target": torch.tensor(y,                           dtype=torch.float32),
        }
    

    def fit_scalers_from_train_items(self, train_item_indices: List[int]) -> None:
        """
        Fit one StandardScaler per variable using ONLY the observations that
        appear in the context windows of the given train items. Then apply
        those scalers to transform the entire df into 'scaled_value'.
        """
        # collect all row indices used in train contexts
        ctx_idx = pd.Index([])
        for i in train_item_indices:
            ctx_idx = ctx_idx.union(pd.Index(self.sequence_windows[i]["context_idx"]))
        cutoff = int(self.df.loc[ctx_idx, "time_id"].max()) if len(ctx_idx) else int(self.df["time_id"].min())

        # (re)fit per-variable scalers on train-context rows only
        self.scalers = {}
        for var in self.df[self.variable_column].unique():
            mask_var_trainctx = (self.df[self.variable_column] == var) & (self.df.index.isin(ctx_idx))
            # fallback: if a variable never appears in train contexts, fit on earliest available rows of that var
            if not mask_var_trainctx.any():
                mask_var_trainctx = (self.df[self.variable_column] == var) & (self.df["time_id"] <= cutoff)
            scaler = StandardScaler().fit(self.df.loc[mask_var_trainctx, [self.value_column]])
            self.scalers[var] = scaler
        
        # write 'scaled_value' for ALL rows using train-only scalers
        scaled_values = []
        for var, scaler in self.scalers.items():
            m = (self.df[self.variable_column] == var)
            vals = scaler.transform(self.df.loc[m, [self.value_column]]).ravel()
            scaled_values.append(pd.Series(vals, index=self.df[m].index))
        self.df["scaled_value"] = pd.concat(scaled_values).sort_index()

    
        # keep a handy handle to the target scaler for inverse_transform in evaluation
        self.scaler = self.scalers[self.target_variable]
