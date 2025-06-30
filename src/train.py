import sys
import torch
from pathlib import Path
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch import nn
import random
import numpy as np
import pandas as pd
from src.data.utils import collate_batch
import pdb


# Setup path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.data.mixed_frequency_dataset import MixedFrequencyDataset
from src.models.mixed_frequency_transformer import MixedFrequencyTransformer
from src.data.utils import collate_batch

from src.utils.config import Config
# ------------------------
# Config
# ------------------------
cfg_path = project_root / "src" / "config" / "cfg.yml"
config = Config(cfg_path)

SEED = config.training.seed
BATCH_SIZE = config.training.batch_size
EPOCHS = config.training.epochs
LEARNING_RATE = config.training.lr
CONTEXT_DAYS = config.data.context_days
TARGET = config.features.target
# Reproducibility settings
SEED = config.training.seed
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


# ------------------------
# Determine which long-format file to load based on config
# ------------------------
raw_md_path = project_root / config.paths.data_raw_fred_monthly
md_cols     = pd.read_csv(raw_md_path, nrows=0).columns.tolist()

if config.features.all_monthly:
    n_monthly   = len([c for c in md_cols if c != 'date'])
    n_quarterly = 0
else:
    n_monthly   = len(config.features.monthly_vars)
    n_quarterly = len(config.features.quarterly_vars)

suffix   = f"{n_monthly}M_{n_quarterly}Q"
csv_path = project_root / config.paths.data_processed_template.format(suffix=suffix)

# ------------------------
# Load Dataset
# ------------------------

full_dataset = MixedFrequencyDataset(csv_path, context_days=CONTEXT_DAYS, target_variable=TARGET)

# Ensure sequential split (no random shuffle)
n = len(full_dataset)
train_size = int(config.data.train_ratio * n)
test_size = n - train_size
train_indices = list(range(train_size))
test_indices = list(range(train_size, n))

train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
test_dataset = torch.utils.data.Subset(full_dataset, test_indices)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)

# ------------------------
# Determine max_len for positional encoding
# ------------------------
seq_lens = [batch["value"].shape[1] for batch in train_loader]
max_len = max(seq_lens)

# ------------------------
# Model
# ------------------------
tv = len(full_dataset.var_map)
tf = len(full_dataset.freq_map)

# log2 heuristic
def emb_dim(vocab_size):
    return min(50, int(np.ceil(np.log2(vocab_size))))

d_freq = emb_dim(tf)
d_var  = emb_dim(tv)

model = MixedFrequencyTransformer(
    freq_vocab_size=tf,
    var_vocab_size=tv,
    max_len=max_len,
    d_freq=d_freq,
    d_var=d_var,
    d_model=config.model.transformer.d_model,
    nhead=config.model.transformer.nhead,
    num_layers=config.model.transformer.num_layers,
    dropout=config.model.transformer.dropout,
)


criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ------------------------
# Training Loop
# ------------------------
print("Starting training...")
model.train()

for epoch in range(1, EPOCHS + 1):
    total_train_loss = 0.0

    # ---- Training Phase ----
    model.train()
    for batch in train_loader:
        # pdb.set_trace()
        pred = model(
            value=batch["value"],
            var_id=batch["var_id"],
            freq_id=batch["freq_id"],
        )
        target = batch["target"]
        loss = criterion(pred, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)

    # ---- Evaluation Phase ----
    model.eval()
    total_test_loss = 0.0
    with torch.no_grad():
        for batch in test_loader:
            pred = model(
                value=batch["value"],
                var_id=batch["var_id"],
                freq_id=batch["freq_id"],
            )
            target = batch["target"]
            loss = criterion(pred, target)
            total_test_loss += loss.item()

    avg_test_loss = total_test_loss / len(test_loader)

    print(f"Epoch {epoch:2d} - Train Loss = {avg_train_loss:.8f} | Test Loss = {avg_test_loss:.8f}")


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
        )
        preds.extend(pred.tolist())
        targets.extend(batch["target"].tolist())


# Inverse transform predictions and targets using the existing scaler
scaler = full_dataset.scaler
preds_unscaled = scaler.inverse_transform(torch.tensor(preds).reshape(-1, 1)).flatten()
targets_unscaled = scaler.inverse_transform(torch.tensor(targets).reshape(-1, 1)).flatten()

# Extract timestamps corresponding to test targets
target_mask = full_dataset.df["Variable"] == TARGET
target_timestamps = full_dataset.df[target_mask]["Timestamp"].reset_index(drop=True)
test_dates = target_timestamps.iloc[test_indices].reset_index(drop=True)

# Save results with date
results_df = pd.DataFrame({
    "date": test_dates,
    "target": targets_unscaled,
    "predicted": preds_unscaled
})
results_df.to_csv(project_root / config.paths.outputs.transformer_preds, index=False)

# Plot predictions vs. targets
plt.figure(figsize=(10, 6))
plt.plot(targets_unscaled, label="True", marker='o')
plt.plot(preds_unscaled, label="Predicted", marker='x')
plt.legend()
plt.title("Model Forecasts vs True Targets (Unscaled)")
plt.xlabel("Sample")
plt.ylabel("Original Target Value")

# Save figure using the same suffix as the data file
viz_dir = project_root / config.paths.visualization
viz_dir.mkdir(exist_ok=True, parents=True)
fig_path = viz_dir / f"forecast_vs_true_{suffix}.pdf"
plt.savefig(fig_path, dpi=300, bbox_inches="tight")
