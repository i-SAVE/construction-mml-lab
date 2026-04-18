from __future__ import annotations

import argparse
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.data.quality import build_data_report, clean_training_frame
from src.data.tabular_loader import load_csv
from src.models.baseline_xgb import build_regressor
from src.utils.metrics import regression_metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--target", type=str, default="SalePrice")
    parser.add_argument(
        "--model",
        type=str,
        default="xgboost",
        choices=["xgboost", "gradient_boosting"],
        help="Model to train: xgboost or sklearn gradient boosting.",
    )
    parser.add_argument(
        "--missing-threshold",
        type=float,
        default=0.4,
        help="Drop feature columns with missing share greater than this value.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    raw_df = load_csv(args.data)

    report = build_data_report(raw_df, args.target)
    print("Dataset report:")
    for key, value in report.items():
        print(f"{key}: {value}")

    df, quality_summary = clean_training_frame(
        raw_df,
        target=args.target,
        missing_threshold=args.missing_threshold,
    )
    print("\nCleaning summary:")
    print(f"rows_before: {quality_summary.rows_before}")
    print(f"rows_after: {quality_summary.rows_after}")
    print(f"duplicates_removed: {quality_summary.duplicates_removed}")
    print(f"dropped_columns: {quality_summary.dropped_columns}")

    if args.target not in df.columns:
        raise ValueError(f"Target column '{args.target}' not found")

    if df[args.target].isna().any():
        raise ValueError("Target contains missing values after cleaning. Fill or remove them first.")

    X = df.drop(columns=[args.target])
    y = np.log1p(df[args.target])

    if "Id" in X.columns:
        X = X.drop(columns=["Id"])

    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=["number"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), num_cols),
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]),
                cat_cols,
            ),
        ]
    )

    model = build_regressor(model_name=args.model)
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)

    pred_log = pipeline.predict(X_valid)
    y_true = np.expm1(y_valid)
    y_pred = np.expm1(pred_log)
    metrics = regression_metrics(y_true, y_pred)

    print("\nModel:", args.model)
    print("Validation metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()
