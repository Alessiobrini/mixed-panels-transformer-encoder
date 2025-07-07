import sys
import torch
import optuna
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch import nn
import random
import numpy as np
import pandas as pd
from datetime import datetime
from functools import partial
import shutil


# Setup path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.data.utils import collate_batch
from src.data.mixed_frequency_dataset import MixedFrequencyDataset
from src.models.mixed_frequency_transformer import MixedFrequencyTransformer
from src.utils.config import Config

# ------------------------
# Helpers
# ------------------------

def emb_dim(vocab_size):
    return min(50, int(np.ceil(np.log2(vocab_size))))


def prepare_data(csv_path, config):
    full_dataset = MixedFrequencyDataset(
        csv_path,
        context_days=config.data.context_days,
        target_variable=config.features.target
    )
    n = len(full_dataset)
    train_size = int(config.data.train_ratio * n)
    train_indices = list(range(train_size))
    test_indices = list(range(train_size, n))

    train_loader = DataLoader(
        torch.utils.data.Subset(full_dataset, train_indices),
        batch_size=config.training.batch_size,
        shuffle=False,
        collate_fn=collate_batch
    )
    test_loader = DataLoader(
        torch.utils.data.Subset(full_dataset, test_indices),
        batch_size=config.training.batch_size,
        shuffle=False,
        collate_fn=collate_batch
    )
    return full_dataset, train_loader, test_loader, test_indices


def build_model(full_dataset, config, d_model, nhead, num_layers, dropout):
    # Determine positional encoding length
    loader = DataLoader(
        torch.utils.data.Subset(full_dataset, list(range(int(config.data.train_ratio * len(full_dataset))))),
        batch_size=config.training.batch_size,
        shuffle=False,
        collate_fn=collate_batch
    )
    seq_lens = [batch['value'].shape[1] for batch in loader]
    max_len = max(seq_lens)

    tv = len(full_dataset.var_map)
    tf = len(full_dataset.freq_map)

    return MixedFrequencyTransformer(
        freq_vocab_size=tf,
        var_vocab_size=tv,
        max_len=max_len,
        d_freq=emb_dim(tf),
        d_var=emb_dim(tv),
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dropout=dropout
    )


def evaluate_and_save(model, test_loader, full_dataset, test_indices, exp_path, suffix, title):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in test_loader:
            out = model(
                value=batch['value'],
                var_id=batch['var_id'],
                freq_id=batch['freq_id'],
            )
            preds.extend(out.tolist())
            targets.extend(batch['target'].tolist())

    scaler = full_dataset.scaler
    preds_unscaled = scaler.inverse_transform(
        torch.tensor(preds).reshape(-1, 1)
    ).flatten()
    targets_unscaled = scaler.inverse_transform(
        torch.tensor(targets).reshape(-1, 1)
    ).flatten()

    mask = full_dataset.df['Variable'] == full_dataset.target_variable
    timestamps = full_dataset.df[mask]['Timestamp'].reset_index(drop=True)
    test_dates = timestamps.iloc[test_indices].reset_index(drop=True)

    df_out = pd.DataFrame({
        'date': test_dates,
        'target': targets_unscaled,
        'predicted': preds_unscaled
    })
    df_out.to_csv(exp_path / f"transformer_preds_{suffix}.csv", index=False)

    plt.figure(figsize=(10, 6))
    plt.plot(targets_unscaled, label='True', marker='o')
    plt.plot(preds_unscaled, label='Predicted', marker='x')
    plt.legend()
    plt.title(title)
    plt.xlabel('Sample')
    plt.ylabel('Value')
    plt.savefig(exp_path / f"forecast_vs_true_{suffix}.pdf", dpi=300, bbox_inches='tight')

# ------------------------
# Optuna Objective
# ------------------------

def objective(trial, config, csv_path, exp_path, suffix):
    # Sample hyperparameters
    params = {
        'd_model': trial.suggest_categorical('d_model', config.hyperopt.d_model),
        'nhead': trial.suggest_categorical('nhead', config.hyperopt.nhead),
        'num_layers': trial.suggest_int(
            'num_layers', min(config.hyperopt.num_layers), max(config.hyperopt.num_layers)
        ),
        'dropout': trial.suggest_float(
            'dropout', min(config.hyperopt.dropout), max(config.hyperopt.dropout)
        ),
        'lr': trial.suggest_float(
            'lr', min(config.hyperopt.lr), max(config.hyperopt.lr), log=True
        )
    }
    # Prepare data
    full_dataset, train_loader, test_loader, test_indices = prepare_data(csv_path, config)
    # Build model
    model = build_model(
        full_dataset, config,
        params['d_model'], params['nhead'],
        params['num_layers'], params['dropout']
    )
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])

    best_loss = float('inf')
    patience = getattr(config.training, 'patience', 5)
    wait = 0

    # Training loop
    for epoch in range(1, config.training.epochs + 1):
        model.train()
        for batch in train_loader:
            out = model(
                value=batch['value'], var_id=batch['var_id'], freq_id=batch['freq_id']
            )
            loss = criterion(out, batch['target'])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Eval
        model.eval()
        total_test = 0
        with torch.no_grad():
            for batch in test_loader:
                out = model(
                    value=batch['value'], var_id=batch['var_id'], freq_id=batch['freq_id']
                )
                total_test += criterion(out, batch['target']).item()
        avg_test = total_test / len(test_loader)
        trial.report(avg_test, epoch)

        # Early stopping
        if avg_test < best_loss:
            best_loss = avg_test
            wait = 0
            torch.save(model.state_dict(), exp_path / 'best_model.pt')
        else:
            wait += 1
            if wait >= patience:
                break

    return best_loss

# ------------------------
# Training Modes
# ------------------------

def run_standard_training(config, csv_path, exp_path, suffix):
    full_dataset, train_loader, test_loader, test_indices = prepare_data(csv_path, config)
    model = build_model(
        full_dataset, config,
        config.model.transformer.d_model,
        config.model.transformer.nhead,
        config.model.transformer.num_layers,
        config.model.transformer.dropout
    )
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.lr)

    best_loss = float('inf')
    patience = getattr(config.training, 'patience', 5)
    wait = 0

    print("Starting standard training...")
    for epoch in range(1, config.training.epochs + 1):
        model.train()
        total_train = 0
        for batch in train_loader:
            out = model(
                value=batch['value'], var_id=batch['var_id'], freq_id=batch['freq_id']
            )
            loss = criterion(out, batch['target'])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train += loss.item()
        avg_train = total_train / len(train_loader)

        # Eval
        model.eval()
        total_test = 0
        with torch.no_grad():
            for batch in test_loader:
                out = model(
                    value=batch['value'], var_id=batch['var_id'], freq_id=batch['freq_id']
                )
                total_test += criterion(out, batch['target']).item()
        avg_test = total_test / len(test_loader)

        print(f"Epoch {epoch:2d} - Train {avg_train:.6f} | Test {avg_test:.6f}")

        if avg_test < best_loss:
            best_loss = avg_test
            wait = 0
            torch.save(model.state_dict(), exp_path / 'best_model.pt')
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    model.load_state_dict(torch.load(exp_path / 'best_model.pt'))
    evaluate_and_save(
        model, test_loader, full_dataset,
        test_indices, exp_path, suffix,
        'Forecast vs True (Standard)'
    )


def run_optuna(config, csv_path, exp_path, suffix):
    print("Starting Optuna optimization...")
    algo = partial(
        objective,
        config=config,
        csv_path=csv_path,
        exp_path=exp_path,
        suffix=suffix
    )
    study = optuna.create_study(direction='minimize')
    study.optimize(algo, n_trials=config.hyperopt.n_trials)

    best_params = study.best_trial.params
    with open(exp_path / 'best_params.yaml', 'w') as f:
        yaml.dump(best_params, f)

    print(f"Best RMSE: {study.best_value}\nBest params: {best_params}")

    # Final evaluation
    full_dataset, _, test_loader, test_indices = prepare_data(csv_path, config)
    model = build_model(
        full_dataset, config,
        best_params['d_model'], best_params['nhead'],
        best_params['num_layers'], best_params['dropout']
    )
    model.load_state_dict(torch.load(exp_path / 'best_model.pt'))
    evaluate_and_save(
        model, test_loader, full_dataset,
        test_indices, exp_path, suffix,
        'Forecast vs True (Optimized)'
    )

# ------------------------
# Main
# ------------------------
if __name__ == '__main__':
    cfg_path = project_root / 'src' / 'config' / 'cfg.yaml'
    config = Config(cfg_path)

    random.seed(config.training.seed)
    np.random.seed(config.training.seed)
    torch.manual_seed(config.training.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    raw_md_path = project_root / config.paths.data_raw_fred_monthly
    md_cols = pd.read_csv(raw_md_path, nrows=0).columns.tolist()
    if config.features.all_monthly:
        n_monthly = len([c for c in md_cols if c != 'date'])
        n_quarterly = len(config.features.quarterly_vars)
    else:
        n_monthly = len(config.features.monthly_vars)
        n_quarterly = len(config.features.quarterly_vars)
    suffix = f"{n_monthly}M_{n_quarterly}Q"
    csv_path = project_root / config.paths.data_processed_template.format(suffix=suffix)

    mode = 'optuna' if config.training.optimize else 'run'
    if config.training.experiment_name:
        exp_name = config.training.experiment_name
    else:
        exp_name = f"{mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    exp_path = project_root / 'outputs' / 'experiments' / exp_name
    exp_path.mkdir(parents=True, exist_ok=True)
    
    shutil.copy(cfg_path, exp_path / 'used_config.yaml')

    if config.training.optimize:
        run_optuna(config, csv_path, exp_path, suffix)
    else:
        run_standard_training(config, csv_path, exp_path, suffix)
