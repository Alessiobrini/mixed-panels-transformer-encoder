# Mixed-Panels-Transformer Encoder (MPTE)

Mixed-Frequency Attention is a research codebase for forecasting low-frequency targets (e.g., quarterly) using a mix of higher-frequency predictors (e.g., monthly) with a Transformer-style encoder. The repository supports both **synthetic simulations** and **FRED macroeconomic data**, and benchmarks the Transformer against classical baselines like AR and MIDAS regressions.

## What’s in this repo

- **Mixed-frequency Transformer** with frequency + variable embeddings, positional encoding, and masked loss.
- **Data pipelines** for:
  - synthetic simulations (`src/data/simulate_to_long.py`)
  - FRED CSV conversion (`src/data/convert_fred_to_long.py`)
- **Baselines**: autoregression (`src/models/ar.py`) and MIDAS (`src/models/midas.R`).
- **Evaluation & diagnostics**: forecast scoring and attention aggregation (`src/evaluation/*`).

## Project layout

```
data/               # raw + processed datasets
notebooks/          # exploration and experiment tracking
src/
  config/           # cfg.yaml (main experiment configuration)
  data/             # simulation + preprocessing
  models/           # transformer + baselines
  evaluation/       # scoring + analysis
  visualization/    # plotting helpers
tests/              # unit/integration tests (if present)
```

## Setup

Create a Python environment and install dependencies:

```bash
conda create -n tsa-dev python=3.11
conda activate tsa-dev
pip install -r requirements.txt
```

> **Note:** Running the MIDAS baseline requires an R installation and a valid `Rscript` path in `src/run_pipeline.py`.

## Configuration

Most experiments are driven by `src/config/cfg.yaml`. Key sections:

- `simulation`: toggles synthetic data generation and its parameters.
- `features`: selects monthly/quarterly variables for FRED data.
- `training`: Transformer hyperparameters and Optuna toggle.
- `model`: architecture choices and baseline settings.

Update this file before running the pipeline.

## Quick start

Run the full pipeline (data prep → transformer → baselines → evaluation):

```bash
python src/run_pipeline.py
```

The pipeline will:

1. **Generate data** (synthetic or FRED depending on config)
2. **Train the Transformer** (`src/train.py`)
3. **Run baselines** (`src/models/ar.py`, `src/models/midas.R`)
4. **Evaluate forecasts** (`src/evaluation/evaluate_forecasts.py`)

## Data generation options

### Synthetic simulation

Enable simulation in `cfg.yaml`:

```yaml
simulation:
  simulate: true
```

Then run:

```bash
python src/data/simulate_to_long.py
```

### FRED conversion

Ensure raw CSVs exist at the paths in `cfg.yaml`, then run:

```bash
python src/data/convert_fred_to_long.py
```

Both scripts output a **long-format** dataset under `data/processed/`.

## Training the Transformer

```bash
python src/train.py
```

The training script automatically:

- builds the dataset from processed long-format data
- handles scaling and masked loss
- logs outputs to the console and experiment directories

## Evaluation

Run forecast scoring and metrics:

```bash
python src/evaluation/evaluate_forecasts.py
```

For attention analyses and batch inspections, see `src/evaluation/aggregate_attention.py` and `src/evaluation/batch_inspect.py`.

## Outputs

Generated artifacts are written to:

- `data/processed/` – long-format datasets
- `outputs/` – predictions from transformer, AR, and MIDAS
- experiment-specific folders under `outputs/` (created by training/evaluation scripts)

## Notes

This codebase is research-focused and optimized for experimentation rather than packaging. Expect changes to configuration and structure over time.
