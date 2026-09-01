# Credit Card Fraud Detection

An end-to-end machine learning pipeline for detecting fraudulent credit card transactions in a highly imbalanced dataset.

## Overview

This project implements a complete fraud detection workflow, from exploratory data analysis to production-ready model evaluation. It benchmarks multiple classification algorithms, addresses severe class imbalance, and provides model explainability through SHAP.

## Project Structure

```
.
├── fraud_detection.ipynb   # Main analysis and modeling notebook
├── creditcard.csv          # Transaction dataset (not included, add locally)
└── README.md
```

## Features

- Exploratory data analysis of transaction patterns, amounts, and timing
- Feature scaling and engineering
- Stratified train/test splitting to preserve class ratios
- Class imbalance handling using SMOTE oversampling
- Benchmarking of four models: Logistic Regression, Random Forest, XGBoost, and LightGBM
- Evaluation using ROC-AUC, PR-AUC, Precision, Recall, and F1-score
- ROC and Precision-Recall curve comparisons
- Confusion matrix analysis
- Optimal decision threshold tuning based on the Precision-Recall curve
- Feature importance ranking
- Model explainability using SHAP summary plots

## Requirements

```
numpy
pandas
matplotlib
seaborn
scikit-learn
imbalanced-learn
xgboost
lightgbm
shap
```

Install with:

```bash
pip install -r requirements.txt
```

## Usage

1. Place your transaction dataset as `creditcard.csv` in the project root. The dataset should contain anonymized/PCA-transformed numerical features, a `Time` column, an `Amount` column, and a binary `Class` label (0 = legitimate, 1 = fraud).
2. Open `fraud_detection.ipynb` in Jupyter or JupyterLab.
3. Run all cells sequentially.

## Results Summary

The notebook automatically ranks all trained models by PR-AUC (the appropriate metric for imbalanced classification) and selects the best-performing model for detailed evaluation, threshold optimization, and explainability analysis.

## Methodology Notes

- **Why PR-AUC over ROC-AUC:** With fraud rates below 1%, ROC-AUC can be overly optimistic. Precision-Recall AUC gives a more realistic picture of model performance on the minority class.
- **Why SMOTE:** Synthetic oversampling of the minority class during training (applied only to the training set) prevents the model from being biased toward the majority class while avoiding information leakage into the test set.
- **Why threshold tuning:** The default 0.5 classification threshold is rarely optimal for imbalanced problems. The optimal threshold is derived from the Precision-Recall curve to balance false positives against false negatives according to business cost.

## License

MIT License

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss the proposed changes.
