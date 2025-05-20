import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[1]  # from src/ to repo root
sys.path.append(str(project_root))

from src.data.mixed_frequency_dataset import MixedFrequencyDataset
from src.models.mixed_frequency_transformer import MixedFrequencyTransformer

import torch
from torch.utils.data import DataLoader
from torch import nn
import logging

# ------------------------
# Logging setup
# ------------------------
log_path = project_root / "training.log"
logging.basicConfig(
    filename=log_path,
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger().addHandler(console)

# ------------------------
# Config
# ------------------------
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-3
TARGET = "Y"

# ------------------------
# Data
# ------------------------
csv_path = project_root / "data" / "processed" / "mixed_freq_wide.csv"
dataset = MixedFrequencyDataset(csv_path, target_column=TARGET)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ------------------------
# Model
# ------------------------
model = MixedFrequencyTransformer(
    raw_input_dim=4,
    freq_vocab_size=3,
    time_vocab_size=400,
    d_freq=4,
    d_time=8,
    d_model=64
)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ------------------------
# Training Loop
# ------------------------
logging.info("Starting training...")

for epoch in range(1, EPOCHS + 1):
    total_loss = 0.0
    valid_batches = 0

    for batch in dataloader:
        raw_input = batch["raw_input"]
        freq_id = batch["freq_id"]
        time_id = batch["time_id"]
        target = batch["target"]
        is_target = batch["is_target"]

        pred = model(raw_input, freq_id, time_id)

        valid_pred = pred[is_target]
        valid_target = target[is_target]

        if valid_target.numel() > 0:
            loss = criterion(valid_pred, valid_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            valid_batches += 1

    if valid_batches > 0:
        avg_loss = total_loss / valid_batches
        logging.info(f"Epoch {epoch:2d}/{EPOCHS} - Avg. Masked Loss = {avg_loss:.4f}")
    else:
        logging.warning(f"Epoch {epoch:2d} - No valid target batches")

logging.info("Training complete.")
