# train_pipeline.py

Small script to train a tabular model (regression or classification) from a CSV or Excel file.

Basic usage (only `--file` is required; sheet, target and drop-columns have sensible defaults):

```bash
python train_pipeline.py --file insurance_data_with_features.xlsx
```

What it does:

- Loads CSV or XLSX (Excel)
- Drops any columns you pass with --drop
- Imputes missing values (median for numbers, most_frequent for categoricals)
- One-hot encodes categoricals and standard-scales numeric features
- Trains simple candidate models and picks the best one
- Saves the best pipeline to disk (default: best_model.joblib)

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Prediction

To make predictions using the trained model:

```bash
python3 predict_using_model.py --model best_model.joblib --input mock_input.csv --output preds.csv
```

This command loads the saved model from `best_model.joblib`, reads the input data from `mock_input.csv`, generates predictions, and saves them to `preds.csv`.

If the `--output` argument is omitted, predictions are printed to stdout instead.
