from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler


ROOT = Path(".")
PROCESSED_DIR = ROOT / "processed_data"

INPUT_TABLE = PROCESSED_DIR / "monthly_point_regression_table.csv"
METRICS_PATH = PROCESSED_DIR / "point_regression_metrics.csv"
COEFFICIENTS_PATH = PROCESSED_DIR / "point_regression_coefficients.csv"
PREDICTIONS_PATH = PROCESSED_DIR / "point_regression_predictions.csv"

TARGET = "sic"
BASE_FEATURES = ["tos", "tas", "rlds", "rsds", "uas", "vas"]
RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0]


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_pred - y_true))


def evaluate_split(
    model_name: str,
    point_id: int,
    point_name: str,
    features: list[str],
    split: str,
    data: pd.DataFrame,
    y_pred: np.ndarray,
    alpha: float | None = None,
) -> dict[str, object]:
    y_true = data[TARGET].to_numpy(dtype=float)
    return {
        "point_id": point_id,
        "point_name": point_name,
        "model": model_name,
        "split": split,
        "features": ",".join(features),
        "n_features": len(features),
        "n_samples": len(data),
        "alpha": alpha,
        "rmse": rmse(y_true, y_pred),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)) if len(data) >= 2 else np.nan,
        "bias": bias(y_true, y_pred),
    }


def choose_ridge_alpha(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
) -> float:
    best_alpha = RIDGE_ALPHAS[0]
    best_score = np.inf

    for alpha in RIDGE_ALPHAS:
        model = Ridge(alpha=alpha)
        model.fit(x_train, y_train)
        val_pred = model.predict(x_val)
        score = rmse(y_val, val_pred)
        if score < best_score:
            best_score = score
            best_alpha = alpha

    return best_alpha


def main() -> None:
    if not INPUT_TABLE.exists():
        raise FileNotFoundError(
            f"Input table not found: {INPUT_TABLE}\n"
            "Run scripts/01_extract_monthly_point_data.py first."
        )

    table = pd.read_csv(INPUT_TABLE)

    metrics_rows = []
    coefficient_rows = []
    prediction_rows = []

    for point_id, point_table in table.groupby("point_id", sort=True):
        point_name = str(point_table["point_name"].iloc[0])

        features = [
            feature
            for feature in BASE_FEATURES
            if not point_table[feature].isna().all()
        ]

        model_table = point_table.dropna(subset=features + [TARGET]).copy()
        train = model_table[model_table["split"] == "train"].copy()
        val = model_table[model_table["split"] == "val"].copy()
        test = model_table[model_table["split"] == "test"].copy()

        if train.empty or val.empty or test.empty:
            print(f"Skipping point {point_id} ({point_name}): empty split after NaN removal.")
            continue

        scaler = StandardScaler()
        x_train = scaler.fit_transform(train[features])
        x_val = scaler.transform(val[features])
        x_test = scaler.transform(test[features])

        y_train = train[TARGET].to_numpy(dtype=float)
        y_val = val[TARGET].to_numpy(dtype=float)
        y_test = test[TARGET].to_numpy(dtype=float)

        models = {
            "linear": (LinearRegression(), None),
        }

        ridge_alpha = choose_ridge_alpha(x_train, y_train, x_val, y_val)
        models["ridge"] = (Ridge(alpha=ridge_alpha), ridge_alpha)

        for model_name, (model, alpha) in models.items():
            model.fit(x_train, y_train)

            split_data = {
                "train": (train, x_train, y_train),
                "val": (val, x_val, y_val),
                "test": (test, x_test, y_test),
            }

            for split, (split_frame, split_x, _) in split_data.items():
                pred = np.clip(model.predict(split_x), 0.0, 1.0)
                metrics_rows.append(
                    evaluate_split(
                        model_name=model_name,
                        point_id=int(point_id),
                        point_name=point_name,
                        features=features,
                        split=split,
                        data=split_frame,
                        y_pred=pred,
                        alpha=alpha,
                    )
                )

                pred_frame = split_frame[
                    ["time_index", "date", "year", "month", "split", "point_id", "point_name", TARGET]
                ].copy()
                pred_frame["model"] = model_name
                pred_frame["prediction"] = pred
                pred_frame["residual"] = pred_frame["prediction"] - pred_frame[TARGET]
                prediction_rows.extend(pred_frame.to_dict(orient="records"))

            intercept = float(model.intercept_)
            for feature, coef in zip(features, model.coef_):
                coefficient_rows.append(
                    {
                        "point_id": int(point_id),
                        "point_name": point_name,
                        "model": model_name,
                        "alpha": alpha,
                        "feature": feature,
                        "coefficient_standardized": float(coef),
                        "intercept": intercept,
                    }
                )

    metrics = pd.DataFrame(metrics_rows)
    coefficients = pd.DataFrame(coefficient_rows)
    predictions = pd.DataFrame(prediction_rows)

    metrics.to_csv(METRICS_PATH, index=False, encoding="utf-8")
    coefficients.to_csv(COEFFICIENTS_PATH, index=False, encoding="utf-8")
    predictions.to_csv(PREDICTIONS_PATH, index=False, encoding="utf-8")

    print(f"Saved metrics: {METRICS_PATH}")
    print(f"Saved coefficients: {COEFFICIENTS_PATH}")
    print(f"Saved predictions: {PREDICTIONS_PATH}")

    test_metrics = metrics[metrics["split"] == "test"].copy()
    print("\nTest metrics:")
    print(
        test_metrics[
            ["point_id", "point_name", "model", "n_features", "n_samples", "alpha", "rmse", "mae", "r2", "bias"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
