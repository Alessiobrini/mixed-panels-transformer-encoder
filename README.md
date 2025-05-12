# TimeSeriesAttention

This repository contains research code and experiments related to handling **mixed-frequency time series data** using **attention-based models** (e.g., Transformers).

## Project Structure

```
data/               # Raw and processed datasets
notebooks/          # Exploratory notebooks and experiment tracking
src/                # Source code (data processing, models, utils, etc.)
tests/              # Unit and integration tests
```

## Getting Started

### 1. Create and activate your conda environment

```bash
conda create -n tsa-dev python=3.11
conda activate tsa-dev
pip install -r requirements.txt  # TO ADD
```

### 2. Launch development tools

- Use **Spyder** or **JupyterLab** for development and exploration. Any other IDE can be used, as long as it is installed.
- Make sure to select the `tsa-dev` kernel if using notebooks.

## Goals

- Build a unified Transformer architecture for mixed-frequency inputs.
- Compare model performance across time series forecasting benchmarks.
- Explore attention-based frequency and temporal encoding strategies.

## Notes

This is an early-stage research project. Folder structure and code will evolve as the work progresses.


