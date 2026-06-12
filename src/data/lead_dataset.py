"""LeadMixedFrequencyDataset: a temporary subclass that augments each quarterly target's
information set with the target quarter's EARLY monthly observations (a high-frequency
"lead"), without touching the shipped MixedFrequencyDataset.

Standard behaviour ends the context at the previous quarter (context_end). With lead L,
the MONTHLY rows are extended to context_end + L time-ids (L months into the target
quarter), while the QUARTERLY rows still end at context_end (the target quarter's
quarterly siblings are not yet released). The target itself is never included. One
observation per target; the lead never reaches the target's own quarter-end month.

This is standard mixed-frequency *forecasting* with high-frequency leads, not nowcasting:
the target and horizon are unchanged. If it works we will fold a `lead` option into the
main class (toggle: truncate at previous quarter, or extend monthly into the next quarter).
"""
from __future__ import annotations

import pandas as pd

from src.data.mixed_frequency_dataset import MixedFrequencyDataset


class LeadMixedFrequencyDataset(MixedFrequencyDataset):
    def __init__(self, *args, lead: int = 0, **kwargs):
        self.lead = int(lead)
        super().__init__(*args, **kwargs)

    def _build_sequence_targets(self):
        if not getattr(self, "lead", 0):
            return super()._build_sequence_targets()

        fcol, vcol = self.freq_column, self.variable_column
        result = []
        target_rows = self.df[(self.df[vcol] == self.target_variable) & (self.df[fcol] == "Q")]

        for _, row in target_rows.iterrows():
            target_time_id = row["time_id"]
            context_end = self.df[
                (self.df[vcol] == self.target_variable) & (self.df["time_id"] < target_time_id)
            ]["time_id"].max()
            if pd.isna(context_end):
                self.skipped_context += 1
                continue
            context_end = int(context_end)
            context_start = context_end - self.context_days
            if context_start < 0:
                self.skipped_context += 1
                continue

            # Monthly rows may run `lead` steps into the target quarter, but never reach the
            # target's own quarter-end month (so the concurrent month is excluded).
            m_end = min(context_end + self.lead, int(target_time_id) - 1)

            is_q = self.df[fcol] == "Q"
            in_window = self.df["time_id"] >= context_start
            q_ok = is_q & (self.df["time_id"] <= context_end)
            m_ok = (~is_q) & (self.df["time_id"] <= m_end)
            context_idx = self.df.index[in_window & (q_ok | m_ok)]
            if len(context_idx) == 0:
                self.skipped_context += 1
                continue

            result.append({"context_idx": context_idx.to_numpy(), "target_idx": int(row.name)})

        return result
