import sys
import torch
from pathlib import Path

# Add project root to Python path BEFORE importing anything from src
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.data.mixed_frequency_dataset import MixedFrequencyDataset
from src.models.mixed_frequency_transformer import MixedFrequencyTransformer
from src.data.utils import collate_batch

from torch.utils.data import DataLoader
from torch import nn

# Load dataset
csv_path = project_root / "data" / "processed" / "toy_mixed_frequency_long.csv"
dataset = MixedFrequencyDataset(csv_path)
dataloader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=collate_batch)

# Get a batch
batch = next(iter(dataloader))

# Vocab sizes
freq_vocab_size = len(dataset.freq_map)
var_vocab_size = len(dataset.var_map)
max_len = batch["value"].shape[1]  # sequence length (T)

# Initialize model with sinusoidal position encoding
model = MixedFrequencyTransformer(
    freq_vocab_size=freq_vocab_size,
    var_vocab_size=var_vocab_size,
    max_len=max_len,
    d_freq=4,
    d_var=4,
    d_model=64
)

# Loss function
criterion = nn.MSELoss()

# Sanity check
print("value shape:", batch["value"].shape)       # [B, T]
print("time_id shape:", batch["time_id"].shape)   # [B, T]

# Forward pass
pred = model(
    value=batch["value"],         # [B, T]
    var_id=batch["var_id"],       # [B, T]
    freq_id=batch["freq_id"],     # [B, T]
)

# Compute loss
target = batch["target"]          # [B]
loss = criterion(pred, target)

print("target:", target)
print("prediction:", pred)
print(f"Forward pass OK — loss = {loss.item():.4f}")
