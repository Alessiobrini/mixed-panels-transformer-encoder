import sys
import torch
from pathlib import Path

# Add project root to Python path BEFORE importing anything from src
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.data.mixed_frequency_dataset import MixedFrequencyDataset
from src.models.mixed_frequency_transformer import MixedFrequencyTransformer

from torch.utils.data import DataLoader
from torch import nn


# Load dataset
csv_path = project_root / "data" / "processed" / "mixed_freq_wide.csv"
dataset = MixedFrequencyDataset(csv_path)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

time_vocab_size = int(dataset.time_ids.max()) + 1


# Initialize model (dimensions must match your dataset)
model = MixedFrequencyTransformer(
    raw_input_dim=4,           # Adjust depending on your feature count (e.g., D1, M1–M3, etc.)
    freq_vocab_size=3,         # D, M, Q
    time_vocab_size=time_vocab_size,       # enough to cover your time index range
    d_freq=4,
    d_time=8,
    d_model=64
)

# Loss function
criterion = nn.MSELoss()

# Get a batch
batch = next(iter(dataloader))

print("Any NaNs in raw_input?", torch.isnan(batch["raw_input"]).any())


# Forward pass
pred = model(
    raw_input=batch["raw_input"],
    freq_id=batch["freq_id"],
    time_id=batch["time_id"]
)

# Apply mask to select valid targets
valid_idx = batch["is_target"]
valid_pred = pred[valid_idx]
valid_target = batch["target"][valid_idx]

print("valid_target:", valid_target)
print("valid_pred:", valid_pred)
print("Count of valid targets:", valid_target.numel())

# Compute loss
loss = criterion(valid_pred, valid_target)

print(f"Forward pass OK — masked loss = {loss.item():.4f}")
