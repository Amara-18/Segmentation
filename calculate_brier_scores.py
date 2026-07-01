#!/usr/bin/env python3
"""Compute Brier scores from prediction_probabilities.csv."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss


def find_probability_columns(df: pd.DataFrame) -> list[str]:
    prob_cols = [col for col in df.columns if col.endswith("_prob")]
    if not prob_cols:
        raise ValueError("No prediction probability columns ending with _prob were found")
    return prob_cols


def compute_brier_scores(df: pd.DataFrame) -> pd.DataFrame:
    required = {"Dataset", "mRS_binary"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Input file is missing required columns: {', '.join(sorted(missing))}")

    prob_cols = find_probability_columns(df)
    rows = []

    for dataset_name, dataset_df in df.groupby("Dataset", sort=True):
        y_true = pd.to_numeric(dataset_df["mRS_binary"], errors="coerce")
        for prob_col in prob_cols:
            y_prob = pd.to_numeric(dataset_df[prob_col], errors="coerce")
            valid = y_true.notna() & y_prob.notna() & np.isfinite(y_prob)
            if not valid.any():
                continue

            model_name = prob_col.removesuffix("_prob")
            rows.append(
                {
                    "Dataset": dataset_name,
                    "Model": model_name,
                    "N": int(valid.sum()),
                    "Event_rate": float(y_true[valid].mean()),
                    "Brier_score": float(brier_score_loss(y_true[valid].astype(int), y_prob[valid])),
                }
            )

    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError("No valid data available for Brier score calculation")
    return result.reset_index(drop=True)


def main(input_path: Path, output_path: Path) -> pd.DataFrame:
    df = pd.read_csv(input_path, encoding="utf-8-sig")
    result = compute_brier_scores(df)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False, encoding="utf-8-sig")

    print("\nBrier score results:")
    print(result.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"\nResults saved to: {output_path}")
    return result


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent
    INPUT_PATH = BASE_DIR / "xgboost_mrs_results" / "prediction_probabilities.csv"
    OUTPUT_PATH = BASE_DIR / "xgboost_mrs_results" / "table_brier_scores.csv"

    main(input_path=INPUT_PATH, output_path=OUTPUT_PATH)
