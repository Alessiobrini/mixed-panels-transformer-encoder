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
from src.utils.data_paths import resolve_data_paths

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# ------------------------
# Helpers
# ------------------------

def emb_dim(vocab_size, override=None):
    if override is not None:
        return override
    return min(50, int(np.ceil(np.log2(vocab_size))))


def make_dataloader(dataset, indices, batch_size, collate_fn):
    return DataLoader(
        torch.utils.data.Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

def compute_split_indices(n, train_ratio, val_ratio=0.0, optimize=False):
    n_total_train = int(train_ratio * n)

    if optimize:
        n_val = int(val_ratio * n_total_train)
        n_train = n_total_train - n_val

        return (
            list(range(n_train)),                                 # train
            list(range(n_train, n_train + n_val)),                # val (end of train set)
            list(range(n_total_train, n))                         # test (fixed end of data)
        )
    else:
        return (
            list(range(n_total_train)),     # train
            [],                             # val (not used)
            list(range(n_total_train, n))   # test (fixed end of data)
        )


def prepare_data(csv_path, config):
    full_dataset = MixedFrequencyDataset(
        csv_path,
        context_days=config.data.context_days,
        target_variable=config.features.target
    )
    n = len(full_dataset)

    train_ratio = config.data.train_ratio
    val_ratio = getattr(config.data, 'val_ratio', 0.0)
    optimize = config.training.optimize

    train_idx, val_idx, test_idx = compute_split_indices(n, train_ratio, val_ratio, optimize)

    full_dataset.fit_scalers_from_train_items(train_idx)

    train_loader = make_dataloader(full_dataset, train_idx, config.training.batch_size, collate_batch)
    test_loader  = make_dataloader(full_dataset, test_idx, config.training.batch_size, collate_batch)

    val_loader = (
        make_dataloader(full_dataset, val_idx, config.training.batch_size, collate_batch)
        if val_idx else None
    )
    # print(f"[Split Info] Total: {n} | Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")

    if optimize:
        return full_dataset, train_loader, val_loader, test_loader, test_idx
    else:
        return full_dataset, train_loader, test_loader, test_idx
    



def build_model(
    full_dataset,
    config,
    d_model,
    nhead,
    num_layers,
    dropout,
    train_loader=None,
    d_freq=None,
    d_var=None,
    dim_feedforward=2048,
    activation="relu",
    use_nonlinearity=True,
    use_attention=True,
):
    if train_loader is None:
        train_loader = DataLoader(
            torch.utils.data.Subset(full_dataset, list(range(int(config.data.train_ratio * len(full_dataset))))),
            batch_size=config.training.batch_size,
            shuffle=False,
            collate_fn=collate_batch
        )
    seq_lens = [batch['value'].shape[1] for batch in train_loader]
    max_len = max(seq_lens)

    tv = len(full_dataset.var_map)
    tf = len(full_dataset.freq_map)
    
    transformer_cfg = getattr(config.model, "transformer", None)

    default_d_freq = None if transformer_cfg is None else getattr(transformer_cfg, "d_freq", None)
    default_d_var = None if transformer_cfg is None else getattr(transformer_cfg, "d_var", None)

    d_freq = emb_dim(tf, override=d_freq if d_freq is not None else default_d_freq)
    d_var  = emb_dim(tv, override=d_var if d_var is not None else default_d_var)


    return MixedFrequencyTransformer(
        freq_vocab_size=tf,
        var_vocab_size=tv,
        max_len=max_len,
        d_freq=d_freq,
        d_var=d_var,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dropout=dropout,
        dim_feedforward=dim_feedforward,
        activation=activation,
        use_nonlinearity=use_nonlinearity,
        use_attention=use_attention,
    )


def evaluate_and_save(model, test_loader, full_dataset, test_indices, exp_path, suffix, title, device):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
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
    test_indices = [int(i) for i in np.array(test_indices)+full_dataset.skipped_context]
    test_dates = timestamps[-len(test_indices):].reset_index(drop=True)

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
    # Prepare data early to get vocab sizes
    full_dataset, train_loader, val_loader, test_loader, test_indices = prepare_data(csv_path, config)
    tv = len(full_dataset.var_map)
    tf = len(full_dataset.freq_map)

    transformer_cfg = getattr(config.model, "transformer", None)
    default_d_freq = None if transformer_cfg is None else getattr(transformer_cfg, "d_freq", None)
    default_d_var = None if transformer_cfg is None else getattr(transformer_cfg, "d_var", None)
    default_use_nonlinearity = True if transformer_cfg is None else getattr(transformer_cfg, "use_nonlinearity", True)
    default_use_attention = True if transformer_cfg is None else getattr(transformer_cfg, "use_attention", True)

    # d_model (required for valid nhead)
    if hasattr(config.hyperopt, 'd_model'):
        d_model = trial.suggest_categorical('d_model', config.hyperopt.d_model)
    else:
        d_model = config.model.transformer.d_model

    # nhead depends on d_model, so adjust its options
    if hasattr(config.hyperopt, 'nhead'):
        valid_heads = [h for h in config.hyperopt.nhead if d_model % h == 0]
        if not valid_heads:
            raise optuna.TrialPruned(f"No valid nhead for d_model={d_model}")
        nhead = trial.suggest_categorical('nhead', valid_heads)
    else:
        nhead = config.model.transformer.nhead

    # Sample or fallback
    params = {
        'd_model': d_model,
        'nhead': nhead,
        'num_layers': (
            trial.suggest_categorical('num_layers', [int(x) for x in config.hyperopt.num_layers])
            if hasattr(config.hyperopt, 'num_layers')
            else config.model.transformer.num_layers
        ),
        'dropout': (
            trial.suggest_categorical('dropout', [float(x) for x in config.hyperopt.dropout])
            if hasattr(config.hyperopt, 'dropout')
            else config.model.transformer.dropout
        ),
        'lr': (
            trial.suggest_categorical('lr', [float(x) for x in config.hyperopt.lr])
            if hasattr(config.hyperopt, 'lr')
            else config.training.lr
        ),
        'd_freq': (
            trial.suggest_categorical('d_freq', [int(x) for x in config.hyperopt.d_freq])
            if hasattr(config.hyperopt, 'd_freq')
            else (default_d_freq if default_d_freq is not None else emb_dim(tf))
        ),
        'd_var': (
            trial.suggest_categorical('d_var', [int(x) for x in config.hyperopt.d_var])
            if hasattr(config.hyperopt, 'd_var')
            else (default_d_var if default_d_var is not None else emb_dim(tv))
        ),
        'dim_feedforward': (
            trial.suggest_categorical('dim_feedforward', [int(x) for x in config.hyperopt.dim_feedforward])
            if hasattr(config.hyperopt, 'dim_feedforward')
            else 2048
        ),
        'activation': (
            trial.suggest_categorical('activation', config.hyperopt.activation)
            if hasattr(config.hyperopt, 'activation')
            else 'relu'
        )

    }


    print(f"[Trial {trial.number}] Sampled or default hyperparameters:")
    for k, v in params.items():
        print(f"  {k}: {v}")


    # Build model
    model = build_model(
        full_dataset, config,
        params['d_model'], params['nhead'],
        params['num_layers'], params['dropout'],
        train_loader=train_loader,
        d_freq=params['d_freq'],
        d_var=params['d_var'],
        dim_feedforward=params['dim_feedforward'],
        activation=params['activation'],
        use_nonlinearity=default_use_nonlinearity,
        use_attention=default_use_attention,
    )
    model.to(device)

    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])

    best_loss = float('inf')
    patience = getattr(config.training, 'patience', 5)
    wait = 0

    # Training loop
    for epoch in range(1, config.training.epochs + 1):
        model.train()
        for batch in train_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            out = model(
                value=batch['value'], var_id=batch['var_id'], freq_id=batch['freq_id']
            )
            loss = criterion(out, batch['target'])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Eval
        model.eval()
        total_val = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                out = model(
                    value=batch['value'], var_id=batch['var_id'], freq_id=batch['freq_id']
                )
                total_val += criterion(out, batch['target']).item()
        avg_val = total_val / len(val_loader)
        trial.report(avg_val, epoch)

        # Early stopping
        if avg_val < best_loss:
            best_loss = avg_val
            wait = 0
            trial.set_user_attr("best_model_path", str(exp_path / f'best_model_trial_{trial.number}.pt'))
            torch.save(model.state_dict(), exp_path / f'best_model_trial_{trial.number}.pt')
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

    transformer_cfg = getattr(config.model, "transformer", None)
    default_use_nonlinearity = True if transformer_cfg is None else getattr(transformer_cfg, "use_nonlinearity", True)
    default_use_attention = True if transformer_cfg is None else getattr(transformer_cfg, "use_attention", True)

    model = build_model(
        full_dataset, config,
        config.model.transformer.d_model,
        config.model.transformer.nhead,
        config.model.transformer.num_layers,
        config.model.transformer.dropout,
        dim_feedforward=config.model.transformer.dim_feedforward,
        activation=config.model.transformer.activation,
        use_nonlinearity=default_use_nonlinearity,
        use_attention=default_use_attention,
    )
    model.to(device)
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
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
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
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
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

    model.load_state_dict(torch.load(exp_path / 'best_model.pt', map_location=device))
    evaluate_and_save(
        model, test_loader, full_dataset,
        test_indices, exp_path, suffix,
        'Forecast vs True (Standard)',
        device
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
    study = optuna.create_study(
        study_name=config.hyperopt.study_name,
        direction='minimize'
    )
    study.optimize(algo, n_trials=config.hyperopt.n_trials)

    best_params = study.best_trial.params
    with open(exp_path / 'best_params.yaml', 'w') as f:
        yaml.dump(best_params, f)

    print(f"Best RMSE: {study.best_value}\nBest params: {best_params}")

    # Final evaluation
    full_dataset, train_loader, val_loader, test_loader, test_indices = prepare_data(csv_path, config)
    tv = len(full_dataset.var_map)
    tf = len(full_dataset.freq_map)

    transformer_cfg = getattr(config.model, "transformer", None)
    default_d_freq = None if transformer_cfg is None else getattr(transformer_cfg, "d_freq", None)
    default_d_var = None if transformer_cfg is None else getattr(transformer_cfg, "d_var", None)
    default_use_nonlinearity = True if transformer_cfg is None else getattr(transformer_cfg, "use_nonlinearity", True)
    default_use_attention = True if transformer_cfg is None else getattr(transformer_cfg, "use_attention", True)

    complete_params = {
        'd_model': best_params.get('d_model', config.model.transformer.d_model),
        'nhead': best_params.get('nhead', config.model.transformer.nhead),
        'num_layers': best_params.get('num_layers', config.model.transformer.num_layers),
        'dropout': best_params.get('dropout', config.model.transformer.dropout),
        'lr': best_params.get('lr', config.training.lr),
        'd_freq': best_params.get('d_freq', default_d_freq if default_d_freq is not None else emb_dim(tf)),
        'd_var': best_params.get('d_var', default_d_var if default_d_var is not None else emb_dim(tv)),
        'dim_feedforward': best_params.get('dim_feedforward', 2048),
        'activation': best_params.get('activation', 'relu'),

    }

    model = build_model(
        full_dataset, config,
        complete_params['d_model'], complete_params['nhead'],
        complete_params['num_layers'], complete_params['dropout'],
        train_loader=train_loader,
        d_freq=complete_params['d_freq'],
        d_var=complete_params['d_var'],
        dim_feedforward=complete_params['dim_feedforward'],
        activation=complete_params['activation'],
        use_nonlinearity=default_use_nonlinearity,
        use_attention=default_use_attention,
    )

    best_model_path = Path(study.best_trial.user_attrs["best_model_path"])
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.to(device)


    evaluate_and_save(
        model, test_loader, full_dataset,
        test_indices, exp_path, suffix,
        'Forecast vs True (Optimized)',
        device
    )

    # Save all trials for inspection
    trials_df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    trials_df.to_csv(exp_path / "optuna_trials.csv", index=False)

    # Optionally save the merged config + best params for reproducibility
    with open(exp_path / 'full_final_params.yaml', 'w') as f:
        yaml.dump(complete_params, f)


    # Clean up all but best model checkpoint
    print("Cleaning up non-best model checkpoints...")
    best_trial_number = study.best_trial.number
    for path in exp_path.glob("best_model_trial_*.pt"):
        trial_num = int(path.stem.split("_")[-1])
        if trial_num != best_trial_number:
            try:
                path.unlink()
            except Exception as e:
                print(f"Failed to delete {path.name}: {e}")


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

    csv_path, suffix, _, _ = resolve_data_paths(config, project_root)

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
