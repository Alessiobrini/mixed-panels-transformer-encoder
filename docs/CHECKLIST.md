# Development and Validation Checklist

## 1. Validate Architecture Alignment

- [ ] Forecasting a low-frequency target (e.g., quarterly `Y_t`) using higher-frequency predictors
- [ ] Unified Transformer encoder consumes all time steps and frequencies
- [ ] Masked loss is applied only to rows with defined target values
- [ ] Frequency and time encodings are learned and concatenated, not summed
- [ ] Forecasting strategy is defined (e.g., predict all observed `Y_t` or future `Y_t`)

## 2. Evaluate Model Performance

- [ ] Implement train/validation split (preferably time-based)
- [ ] Compute validation RMSE on `Y_t` values
- [ ] Log predictions at target time steps for comparison
- [ ] Save predictions to CSV for inspection or visualization
- [ ] Log average training loss per epoch

## 3. Model Extensions (Optional)

- [ ] Visualize attention weights for interpretability
- [ ] Replace learned time embeddings with sinusoidal encodings
- [ ] Enable future-step prediction (e.g., autoregressive or decoder-style forecasting)

## 4. Prepare for Real-World Data

- [ ] Replace toy data with a real mixed-frequency dataset
- [ ] Ensure timestamps and frequencies are aligned and consistent
- [ ] Adapt data pipeline to handle raw inputs and transformations
- [ ] Validate model behavior on realistic input ranges
