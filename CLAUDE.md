# CLAUDE.md

## Project Overview

This repository implements **MPTE (Mixed-Panels-Transformer Encoder)**, a Transformer-based framework for mixed-frequency time series forecasting. It accompanies the paper:

> **A Nonlinear Target-Factor Model with Attention Mechanism for Mixed-Frequency Data**
> Alessio Brini, Ekaterina Seregina — [arXiv:2601.16274](https://arxiv.org/abs/2601.16274)

MPTE estimates factor models in panel datasets where variables are observed at different frequencies (monthly, quarterly) and may contain nonlinear signals. It replaces fixed linear PCA with Transformer attention for adaptive, context-aware signal construction. The paper shows that with linear activations, MPTE nests Target PCA as a special case; the nonlinear extension captures complex hierarchical interactions.

Two empirical applications:
1. **Macro forecasting** — 13 macroeconomic targets using 48 monthly and quarterly FRED series
2. **Equity earnings** — quarterly EPS growth for ~500 S&P 500 equities using daily + monthly + quarterly features (CRSP daily + WRDS IID realized variance, Compustat EPS, FRED-MD/QD macro)

## Repository Structure

```
src/
  config/
    cfg.yaml                       # FRED macro experiment configuration
    cfg_equity.yaml                # Equity earnings experiment configuration
  data/
    simulate_to_long.py            # Synthetic data generation (VAR-based)
    convert_fred_to_long.py        # FRED CSV → long format
    download_compustat_eps.py      # WRDS Compustat EPS download (SQLAlchemy)
    build_equity_long.py           # Build per-stock long-format CSVs
    validate_equity_data.py        # Schema validation for equity CSVs
    mixed_frequency_dataset.py     # PyTorch Dataset for mixed-freq sequences
    long_to_wide.py                # Long → wide format conversion
    utils.py                       # Collate/batch utilities
  models/
    mixed_frequency_transformer.py # Main Transformer encoder architecture
    ar.py                          # AR baseline (statsmodels) — callable via run_ar_baseline()
    single_freq_models.py          # OLS / XGBoost / NN baselines — callable via run_single_freq_baselines()
    midas.R                        # MIDAS regression baseline (R) — accepts config path arg
  evaluation/
    evaluate_forecasts.py          # RMSE & Diebold-Mariano tests
    aggregate_attention.py         # Attention pattern analysis
    batch_inspect.py               # Batch model inspection
  utils/
    config.py                      # YAML config loader
    data_paths.py                  # Path/suffix resolution (FRED + equity modes)
  train.py                         # Training script (standard or Optuna mode, --config arg)
  run_pipeline.py                  # FRED pipeline: data → train → baselines → eval
  run_equity_pipeline.py           # Equity pipeline: per-ticker MPTE + AR + MIDAS + OLS/XGB/NN + eval
  univariate_targets.sh            # SLURM batch for FRED multi-target experiments
  equity_targets.sh                # SLURM batch for equity S&P 500 experiments
data/
  raw/
    fred_data/                     # Raw FRED monthly/quarterly CSVs
    equity/                        # CRSP daily, WRDS IID RV, Compustat EPS, universe CSV
  processed/
    long_format_fred_*.csv         # FRED long-format datasets
    equity/                        # Per-stock long-format: long_format_{TKR}_7D_43M_14Q.csv
outputs/
  experiments/                     # Per-experiment results, checkpoints, plots
tests/                             # pytest tests
notebooks/                         # Exploration notebooks
```

## Tech Stack

- **Python 3.11** with conda
- **PyTorch** — model, training loop, dataset
- **Optuna** — hyperparameter optimization (toggle `training.optimize` in cfg.yaml)
- **statsmodels** — AR baseline
- **scikit-learn** — scalers, linear baselines
- **R** — MIDAS baseline (`src/models/midas.R`, requires Rscript)
- **pandas / numpy** — data processing

## Configuration

Two config files, same structure:

- **`src/config/cfg.yaml`** — FRED macro experiments (simulation toggle, FRED variable lists, context_days=360)
- **`src/config/cfg_equity.yaml`** — Equity earnings experiments (equity ticker list, context_days=120, max_lag=12)

Key sections: `simulation`, `features`, `data`, `training`, `model.transformer`, `model.ar`, `model.midas`, `hyperopt`. Equity config adds `equity` section (active_ticker, tickers, midas_monthly_vars_template).

## Common Commands

```bash
# === FRED Macro Pipeline ===
python src/run_pipeline.py                      # Full FRED pipeline
python src/train.py                             # FRED training only
python src/train.py --config src/config/cfg_equity.yaml  # Train with any config
sbatch src/univariate_targets.sh                # HPC: 13 macro targets

# === Equity Earnings Pipeline ===
python src/run_equity_pipeline.py --ticker AAPL  # Single ticker (all 6 models + eval)
python src/run_equity_pipeline.py                # All ~500 S&P 500 tickers
sbatch src/equity_targets.sh                     # HPC: all S&P 500 tickers

# === Shared ===
python src/evaluation/evaluate_forecasts.py
python src/evaluation/aggregate_attention.py
pytest tests/
```

## Model Architecture

The Transformer encoder processes mixed-frequency sequences:

1. **Input**: each token is (value, variable_id, frequency_id) from the long-format panel
2. **Embeddings**: learned variable + frequency embeddings concatenated with the scalar value
3. **Projection**: linear layer → LayerNorm → sinusoidal positional encoding
4. **Encoder**: standard multi-head self-attention Transformer encoder (configurable depth/heads)
5. **Pooling**: mean over the temporal dimension
6. **Head**: linear projection → scalar forecast

Ablation flags (`use_attention`, `use_nonlinearity`, `use_positional_encoding`) allow disabling components for controlled experiments.

## Data Formats

All data passes through a **long format** with columns: `Timestamp`, `Variable`, `Frequency` (M/Q), `Value`. The `MixedFrequencyDataset` class creates context windows of configurable length, pairing them with the next quarterly target observation.

## Equity Earnings Dataset

### Data Sources
- **CRSP Daily** (`data/raw/equity/crsp_dsf.parquet`) — Daily returns and volume for ~500 S&P 500 stocks via WRDS (`crsp.dsf`). Downloaded with `src/data/download_crsp_daily.py`.
- **WRDS IID** (`data/raw/equity/taq_rv.parquet`) — Daily realized variance measures (rv5, bv5, rsp5, rsn5, rk) from WRDS Intraday Indicators. Downloaded with `src/data/download_iid.py`.
- **Compustat** (`data/raw/equity/compustat_fundq.csv`) — Quarterly EPS via WRDS (`comp.fundq`). Downloaded with `src/data/download_compustat_eps.py`.
- **FRED-MD/QD** — Same 35 monthly + 13 quarterly macro variables as the FRED experiment.

### Features per Stock (long_format_{TKR}_7D_43M_14Q.csv)
| Freq | Count | Variables |
|------|-------|-----------|
| D | 7 | {TKR}_ret, {TKR}_rv5, {TKR}_bv5, {TKR}_rsp5, {TKR}_rsn5, {TKR}_rk, {TKR}_logvol |
| M | 43 | 5 stock agg + 3 cross-sectional (MKT_*) + 35 FRED-MD macro |
| Q | 14 | 13 FRED-QD macro + {TKR}_eps_yoy (target) |

### Target
`{TKR}_eps_yoy` = YoY EPS growth: `(epspxq_t - epspxq_{t-4}) / max(|epspxq_{t-4}|, 0.01)`, winsorized [-2, 2], timestamped at report date (`rdq`).

### Models Benchmarked
| Model | Freq Input | Script |
|-------|------------|--------|
| MPTE | D + M + Q | `train.py` |
| AR | Q only | `ar.py` → `run_ar_baseline()` |
| MIDAS | M → Q | `midas.R` (8 monthly predictors, 3 lags) |
| OLS | Q only | `single_freq_models.py` → `run_single_freq_baselines()` |
| XGBoost | Q only | same |
| NN | Q only | same |

### Known Gotchas
- **WRDS**: Use SQLAlchemy + pgpass, not `wrds.Connection()` (interactive prompt fails in non-interactive shells)
- **MIDAS asynchronous dates**: EPS report dates don't fall on regular quarter boundaries. The pipeline writes a temp CSV with dates floored to quarter-start. R forecast loop has `tryCatch` for lags beyond available data.
- **Single-freq baselines**: Quarterly wide data needs forward-fill before lagging (EPS dates ≠ macro dates). See `_build_quarterly_wide()`.
- **Evaluation alignment**: Models produce different date formats. `_evaluate_ticker()` normalizes to quarter periods before joining.
- **AR max_lag=12** (not 40) — only ~44 quarterly obs per stock.
- **MIDAS uses 8 monthly vars** (not all 43) — 5 stock-specific + 3 market. 43 vars × 3 lags = 130 params >> 35 training obs.
### Next Steps
- Cross-sectional ConcatDataset model (designed, not yet wired)
- Potential future target: earnings surprise from I/B/E/S

## Code Style

- Formatting: **black**, **autopep8**
- Linting: **flake8**, **pylint**
