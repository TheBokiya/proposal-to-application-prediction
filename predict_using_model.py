#!/usr/bin/env python3
"""
predict_using_model.py

Load a saved joblib pipeline and run predictions on a CSV. Writes predictions to stdout or a CSV file.

Usage:
  python3 predict_using_model.py --model best_model.joblib --input mock_input.csv --output preds.csv
"""
from __future__ import annotations

import argparse
import sys
import joblib
import pandas as pd

# Helpers that may be referenced by a saved pipeline when it was trained as a script
# If the pipeline was pickled while the trainer was running as __main__, the
# unpickler will look for these callables in the loading script's module. Define
# compatible helpers here so joblib.load succeeds.
from sklearn.preprocessing import OneHotEncoder


def to_str(X):
    """Ensure input is coerced to string dtype (kept at module scope for pickling)."""
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


def main(argv=None):
    parser = argparse.ArgumentParser(description="Load a joblib pipeline and predict on CSV input")
    parser.add_argument("--model", required=True, help="Path to joblib model (pipeline)")
    parser.add_argument("--input", required=True, help="Path to input CSV with feature columns")
    parser.add_argument("--output", default=None, help="Path to write predictions CSV (if omitted, prints to stdout)")

    args = parser.parse_args(argv)

    pipe = joblib.load(args.model)

    df = pd.read_csv(args.input)

    preds = pipe.predict(df)
    out = df.copy()
    out["prediction"] = preds

    if args.output:
        out.to_csv(args.output, index=False)
        print(f"Wrote predictions to {args.output}")
    else:
        print(out.to_csv(index=False))


if __name__ == "__main__":
    main()
