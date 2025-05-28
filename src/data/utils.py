import torch
from torch.nn.utils.rnn import pad_sequence

def collate_batch(batch):
    """
    Pads variable-length token sequences in the batch.
    Assumes each item is a dict with 'value', 'var_id', 'freq_id', 'time_id', and 'target'.
    """

    # Pad sequences
    value = pad_sequence([item["value"] for item in batch], batch_first=True)  # (B, L, )
    var_id = pad_sequence([item["var_id"] for item in batch], batch_first=True)
    freq_id = pad_sequence([item["freq_id"] for item in batch], batch_first=True)
    time_id = pad_sequence([item["time_id"] for item in batch], batch_first=True)

    # No padding for targets (one scalar per sample)
    target = torch.stack([item["target"] for item in batch])

    return {
        "value": value,
        "var_id": var_id,
        "freq_id": freq_id,
        "time_id": time_id,
        "target": target
    }
