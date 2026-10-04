# Predictive Model Evaluation — Reproducible Analysis

**Core Responsibility Demonstrated:** Building reproducible predictive analyses and evaluating models for regression, classification, and uncertainty quantification.

## What This Does

1. **Reproducible Pipeline**: Fixed random state ensures identical results across runs
2. **Regression Model**: Linear regression with RMSE evaluation
3. **Classification Model**: Logistic regression with accuracy and ROC AUC
4. **Uncertainty Quantification**: Residual std dev + false positive rate document model confidence

## Run It

```bash
pip install -r requirements.txt
python main.py
```

**Output:**
- Console: Model metrics and uncertainty summary
- `raw_data.csv`: Input features and targets
- `uncertainty.json`: Structured uncertainty estimates

## Reproducibility

Run the script multiple times—output will be identical. Random seed is fixed; train/test split is deterministic.

## Uncertainty in Detail

- **Regression**: Predictions have ±σ confidence interval (68% of predictions fall within residual_std)
- **Classification**: False positive rate shows decision risk even at high accuracy
