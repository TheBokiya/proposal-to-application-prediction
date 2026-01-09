#!/usr/bin/env python3
"""
train_logistic.py

Train only a LogisticRegression classifier on a tabular dataset using the same
preprocessing as `train_pipeline.py`.

Usage:
  python3 train_logistic.py --file dataset.xlsx

Saves the pipeline (preprocessor + logistic regression) to disk (default: logistic_model.joblib)
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# Defaults
DEFAULT_SHEET = "Sheet1"
DEFAULT_TARGET = "Converted"
DEFAULT_DROP_COLUMNS = ["Proposal Number", "Application Number"]


def to_str(X):
    return X.astype(str)


def make_onehot(**kwargs):
    try:
        return OneHotEncoder(**kwargs)
    except TypeError:
        opts = dict(kwargs)
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


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train LogisticRegression with standard preprocessing")
    parser.add_argument("--file", required=True, help="Path to CSV or XLSX file")
    parser.add_argument("--sheet", default=DEFAULT_SHEET, help=f"Sheet name for Excel files (default: {DEFAULT_SHEET})")
    parser.add_argument("--target", default=DEFAULT_TARGET, help=f"Name of the target column (default: {DEFAULT_TARGET})")
    parser.add_argument("--drop", nargs="*", default=DEFAULT_DROP_COLUMNS,
                        help=("Columns to drop (identifiers, dates, etc.). " f"Default: {DEFAULT_DROP_COLUMNS}"))
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--save", default="logistic_model.joblib", help="Path to save trained pipeline")
    parser.add_argument("--output", default="test_predictions.csv", help="Path to write test-set predictions CSV")

    args = parser.parse_args(argv)

    df = load_data(args.file, sheet_name=args.sheet)
    if args.target not in df.columns:
        print(f"Target column '{args.target}' not found in data. Available columns:\n{df.columns.tolist()}")
        sys.exit(1)

    # Drop columns
    drop_cols = [c for c in args.drop if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    X = df.drop(columns=[args.target])
    y = df[args.target]

    print("Detected task: classification (forcing logistic regression)")

    preprocessor = build_preprocessor(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, 
        random_state=args.random_state, 
        stratify=y if pd.api.types.is_integer_dtype(y) or pd.api.types.is_categorical_dtype(y) or pd.api.types.is_object_dtype(y) else None
    )

    clf = LogisticRegression(max_iter=1000)
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", clf)])

    print("Training logistic regression...")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    print(f"Accuracy={acc:.4f}, F1(weighted)={f1:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Build a DataFrame with test features + actual + prediction (+ prob for positive class when available)
    out = X_test.copy().reset_index(drop=True)
    out["actual"] = y_test.reset_index(drop=True)
    out["pred"] = y_pred

    # Add prediction probabilities if supported
    if hasattr(pipeline, "predict_proba"):
        try:
            probs = pipeline.predict_proba(X_test)
            if probs is not None:
                if probs.shape[1] == 2:
                    out["prob_pos"] = probs[:, 1]
                else:
                    # multiclass: add probs per class
                    for i in range(probs.shape[1]):
                        out[f"prob_class_{i}"] = probs[:, i]
        except Exception:
            # ignore probability errors
            pass

    out.to_csv(args.output, index=False)
    print(f"Wrote test predictions to {args.output}")

    joblib.dump(pipeline, args.save)
    print(f"Saved logistic pipeline to {args.save}")


if __name__ == "__main__":
    main()
