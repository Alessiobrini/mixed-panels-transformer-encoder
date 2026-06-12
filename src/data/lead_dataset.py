"""LeadMixedFrequencyDataset: a subclass that augments each quarterly target's information
set with the target quarter's EARLY monthly observations (a high-frequency "lead"), without
touching the shipped MixedFrequencyDataset.

Standard behaviour ends the context at the previous quarter (``context_end``). With lead L,
the MONTHLY rows are extended to include the FIRST L monthly observations of the target
quarter, while the QUARTERLY rows still end at ``context_end`` (the target quarter's
quarterly siblings are not yet released). The target's own quarter-end month is always
excluded (the concurrent month is not observed at forecast time). The number of training
windows is unchanged.

`lead` is measured in MONTHS (number of within-quarter monthly observations to expose), and
is unit-agnostic: it selects the first L *distinct monthly time-ids* strictly after
``context_end`` and strictly before the target's quarter-end month. This matters because the
dataset's ``time_id`` is in whatever unit the panel uses -- months for the integer-indexed
simulation, calendar DAYS for the date-indexed FRED panel. A naive ``context_end + L``
(time-ids) would add L months in the simulation but only L *days* (i.e. nothing) in FRED.
Selecting the first L monthly observations gives the intended 2-months-into-the-quarter lead
in both, and is identical to the previous ``context_end + L`` rule on the monthly-indexed
simulation (verified), so it does not change the simulation results.

This is standard mixed-frequency *forecasting* with high-frequency leads, not nowcasting: the
target and horizon are unchanged.
"""
from __future__ import annotations

import numpy as np
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
        is_q_all = self.df[fcol] == "Q"
        month_tids_all = self.df.loc[~is_q_all, "time_id"]  # cache for the lead lookup

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

            # First `lead` monthly observations strictly inside the target quarter:
            # time-ids > context_end and < target quarter-end month (concurrent month excluded).
            lead_tids = np.sort(month_tids_all[
                (month_tids_all > context_end) & (month_tids_all < int(target_time_id))
            ].unique())
            if lead_tids.size == 0:
                m_end = context_end  # no within-quarter months available -> no lead
            else:
                m_end = int(lead_tids[min(self.lead, lead_tids.size) - 1])

            in_window = self.df["time_id"] >= context_start
            q_ok = is_q_all & (self.df["time_id"] <= context_end)
            m_ok = (~is_q_all) & (self.df["time_id"] <= m_end)
            context_idx = self.df.index[in_window & (q_ok | m_ok)]
            if len(context_idx) == 0:
                self.skipped_context += 1
                continue

            result.append({"context_idx": context_idx.to_numpy(), "target_idx": int(row.name)})

        return result
