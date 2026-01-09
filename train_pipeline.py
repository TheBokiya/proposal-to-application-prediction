#!/usr/bin/env python3
"""
train_pipeline.py

Train a small model (regression or classification) from a CSV/XLSX file.

Usage examples:
  python train_pipeline.py --file insurance_data_with_features.xlsx --sheet Sheet1 --target Converted
  python train_pipeline.py --file data.csv --target price --model rf

The script will:
 - load data (CSV or Excel)
 - drop user-specified columns
 - impute missing values (median for numeric, most_frequent for categorical)
 - OneHotEncode categorical features and StandardScale numeric features
 - train candidate models and pick the best
 - save the best pipeline with joblib
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score

# Defaults (these are the values you said are always the same)
DEFAULT_SHEET = "Sheet1"
DEFAULT_TARGET = "Converted"
DEFAULT_DROP_COLUMNS = ["Proposal Number", "Application Number"]


def to_str(X):
    """Convert input array or DataFrame to string dtype (top-level for pickling)."""
    return X.astype(str)


def make_onehot(**kwargs):
    """Top-level OneHotEncoder factory that handles sklearn API differences.

    Keeps encoder creation at module scope so pipelines remain picklable.
    """
    try:
        return OneHotEncoder(**kwargs)
    except TypeError:
        opts = dict(kwargs)
        # If caller passed sparse_output (modern sklearn), map to sparse for older versions
        if "sparse_output" in opts:
            opts.pop("sparse_output")
            opts["sparse"] = False
        return OneHotEncoder(**opts)


def load_data(file_path: str, sheet_name: str | None = None) -> pd.DataFrame:
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")
    if p.suffix.lower() in [".xls", ".xlsx"]:
        return pd.read_excel(file_path, sheet_name=sheet_name, engine="openpyxl")
    return pd.read_csv(file_path)


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # OneHotEncoder parameter name changed in sklearn 1.2+: `sparse` -> `sparse_output`.
    # Use module-level helper `make_onehot` below to keep objects picklable.

    # Ensure categorical values are uniformly strings (prevents mixed-type encoder errors)
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("to_str", FunctionTransformer(to_str)),
        ("onehot", make_onehot(handle_unknown="ignore", sparse_output=False))
    ])

    transformers = []
    if numeric_cols:
        transformers.append(("num", numeric_pipeline, numeric_cols))
    if categorical_cols:
        transformers.append(("cat", categorical_pipeline, categorical_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def evaluate_regression(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return {"mse": float(mse), "rmse": float(rmse), "r2": float(r2)}


def evaluate_classification(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")
    return {"accuracy": float(acc), "f1_weighted": float(f1)}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train a simple model from a tabular dataset")
    parser.add_argument("--file", required=True, help="Path to CSV or XLSX file")
    parser.add_argument("--sheet", default=DEFAULT_SHEET, help=f"Sheet name for Excel files (default: {DEFAULT_SHEET})")
    parser.add_argument("--target", default=DEFAULT_TARGET, help=f"Name of the target column (default: {DEFAULT_TARGET})")
    parser.add_argument("--drop", nargs="*", default=DEFAULT_DROP_COLUMNS,
                        help=("Columns to drop (identifiers, dates, etc.). "
                              f"Default: {DEFAULT_DROP_COLUMNS}"))
    parser.add_argument("--model", choices=["auto", "linear", "rf"], default="auto",
                        help="Model hint: 'linear' or 'rf' (random forest). 'auto' chooses sensible default")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--save", default="best_model.joblib", help="Path to save trained pipeline")

    args = parser.parse_args(argv)

    df = load_data(args.file, sheet_name=args.sheet)
    if args.target not in df.columns:
        print(f"Target column '{args.target}' not found in data. Available columns:\n{df.columns.tolist()}")
        sys.exit(1)

    # Drop user-specified columns if present
    drop_cols = [c for c in args.drop if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    X = df.drop(columns=[args.target])
    y = df[args.target]

    # Decide task type
    if pd.api.types.is_numeric_dtype(y):
        task = "regression" if y.nunique() > 10 else "classification"
    else:
        task = "classification"

    print(f"Detected task: {task}")

    preprocessor = build_preprocessor(X)

    # Split
    if task == "classification":
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=args.random_state
        )

    # Candidate models
    candidates = []
    if task == "regression":
        candidates = [("linear", LinearRegression()), ("rf", RandomForestRegressor(n_estimators=100, random_state=args.random_state))]
    else:
        candidates = [("logreg", LogisticRegression(max_iter=1000)), ("rf", RandomForestClassifier(n_estimators=100, random_state=args.random_state))]

    best_pipeline = None
    best_score = None
    best_name = None
    results = []

    for name, estimator in candidates:
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", estimator)])
        print(f"Training candidate: {name}")
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        if task == "regression":
            metrics = evaluate_regression(y_test, y_pred)
            score = metrics["rmse"]
            print(f"  RMSE={metrics['rmse']:.4f}, R2={metrics['r2']:.4f}")
        else:
            metrics = evaluate_classification(y_test, y_pred)
            score = metrics["accuracy"]
            print(f"  Accuracy={metrics['accuracy']:.4f}, F1(weighted)={metrics['f1_weighted']:.4f}")

        results.append({"name": name, "metrics": metrics})

        if best_pipeline is None:
            best_pipeline = pipeline
            best_score = score
            best_name = name
        else:
            if task == "regression":
                if score < best_score:
                    best_pipeline = pipeline
                    best_score = score
                    best_name = name
            else:
                if score > best_score:
                    best_pipeline = pipeline
                    best_score = score
                    best_name = name

    print(f"Best model: {best_name}")
    joblib.dump(best_pipeline, args.save)
    print(f"Saved best pipeline to {args.save}")

    print("\nSummary of candidate results:")
    for r in results:
        print(r)


if __name__ == "__main__":
    main()
