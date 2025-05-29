import sys
import torch
from pathlib import Path
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch import nn

# Setup path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.data.mixed_frequency_dataset import MixedFrequencyDataset
from src.models.mixed_frequency_transformer import MixedFrequencyTransformer
from src.data.utils import collate_batch

# ------------------------
# Config
# ------------------------
BATCH_SIZE = 8
EPOCHS = 200
LEARNING_RATE = 5e-4
CONTEXT_DAYS = 600
TARGET = "Y"

# ------------------------
# Load Dataset
# ------------------------
csv_path = project_root / "data" / "processed" / "toy_mixed_frequency_long.csv"
full_dataset = MixedFrequencyDataset(csv_path, context_days=CONTEXT_DAYS, target_variable=TARGET)

# Ensure sequential split (no random shuffle)
n = len(full_dataset)
train_size = int(0.8 * n)
test_size = n - train_size
train_indices = list(range(train_size))
test_indices = list(range(train_size, n))

train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
test_dataset = torch.utils.data.Subset(full_dataset, test_indices)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)

# ------------------------
# Model
# ------------------------
model = MixedFrequencyTransformer(
    freq_vocab_size=len(full_dataset.freq_map),
    time_vocab_size=int(full_dataset.time_ids.max()) + 1,
    var_vocab_size=len(full_dataset.var_map),
    d_value=8,
    d_freq=4,
    d_time=8,
    d_var=4,
    d_model=64
)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ------------------------
# Training Loop
# ------------------------
print("Starting training...")
model.train()
for epoch in range(1, EPOCHS + 1):
    total_loss = 0.0
    for batch in train_loader:
        pred = model(
            value=batch["value"],
            var_id=batch["var_id"],
            freq_id=batch["freq_id"],
            time_id=batch["time_id"]
        )
        target = batch["target"]
        loss = criterion(pred, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch:2d} - Train Loss = {avg_loss:.4f}")

print("Training complete.")

# ------------------------
# Evaluation & Plot
# ------------------------
model.eval()
preds, targets = [], []

with torch.no_grad():
    for batch in test_loader:
        pred = model(
            value=batch["value"],
            var_id=batch["var_id"],
            freq_id=batch["freq_id"],
            time_id=batch["time_id"]
        )
        preds.extend(pred.tolist())
        targets.extend(batch["target"].tolist())

# Plot predictions vs. targets
plt.figure(figsize=(10, 6))
plt.plot(targets, label="True", marker='o')
plt.plot(preds, label="Predicted", marker='x')
plt.legend()
plt.title("Model Forecasts vs True Targets")
plt.xlabel("Sample")
plt.ylabel("Scaled Target Value")
plt.grid()
plt.tight_layout()
plt.savefig(project_root / "forecast_vs_true.png")
plt.show()
