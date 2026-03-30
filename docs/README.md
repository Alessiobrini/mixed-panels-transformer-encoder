# Mixed-Frequency Transformer – Codebase Overview

This repository implements a Transformer-based model for handling mixed-frequency time series data (e.g., daily, monthly, quarterly). The goal is to forecast a low-frequency target while leveraging higher-frequency predictors using a unified transformer encoder.

## Project Structure

mixed-frequency-attention/
│
├── data/
│   ├── raw/                         # Generated raw toy datasets (daily, monthly, quarterly)
│   └── processed/                   # Wide-format merged dataset (1 row per timestamp)
│
├── docs/
│   └── README.md                    # Documentation file
│
├── src/
│   ├── data/
│   │   ├── generate_toy_data.py         # Generates synthetic long-format data
│   │   ├── long_to_wide.py              # Converts long-format to wide-format with forward filling
│   │   └── mixed_frequency_dataset.py   # PyTorch Dataset with feature scaling and masking
│   │
│   ├── models/
│   │   ├── mixed_frequency_transformer.py  # Transformer model for mixed-frequency input
│   │   └── test_model_forward.py           # Validates model-dataset integration and loss masking
│   │
│   └── train.py                        # Full training script with masked loss and logging
│
└── training.log                        # Training log output

## Completed Components

1. **Data Generation**
   - `generate_toy_data.py` creates synthetic daily, monthly, and quarterly variables in long format.
   - Output: `toy_mixed_frequency.csv` plus separate files for each frequency.

2. **Preprocessing**
   - `long_to_wide.py` pivots the long-format data into wide format and fills missing values.
   - Output: `mixed_freq_wide.csv` in `data/processed/`.

3. **Dataset**
   - `MixedFrequencyDataset` loads wide-format data, scales features, and masks loss on missing targets.

4. **Model**
   - `MixedFrequencyTransformer` combines raw features, frequency embeddings, and time embeddings.
   - Output layer maps encoded representations to scalar predictions.

5. **Testing**
   - `test_model_forward.py` verifies the model runs on a batch and computes masked loss.

6. **Training**
   - `train.py` runs the full training loop with logging to console and file.
