from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(".")
PROCESSED_DIR = ROOT / "processed_data"
FIGURE_DIR = ROOT / "figures" / "regression"

METRICS_PATH = PROCESSED_DIR / "point_regression_metrics.csv"
COEFFICIENTS_PATH = PROCESSED_DIR / "point_regression_coefficients.csv"
PREDICTIONS_PATH = PROCESSED_DIR / "point_regression_predictions.csv"

COEFFICIENT_MODEL_TO_PLOT = "linear"
MODEL_COLORS = {
    "linear": "#d95f02",
    "ridge": "#1b9e77",
}


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}\n"
            "Run scripts/01_extract_monthly_point_data.py and "
            "scripts/02_run_point_regression.py first."
        )


def save_figure(fig: plt.Figure, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temp_output = output.with_name(f"{output.stem}.{uuid4().hex}.tmp.png")
    with open(str(temp_output), "wb") as file:
        fig.savefig(file, format="png", dpi=220)
    temp_output.replace(output)


def save_test_rmse(metrics: pd.DataFrame) -> Path:
    test_metrics = metrics[metrics["split"] == "test"].copy()
    test_metrics = test_metrics.sort_values(["point_id", "model"])

    pivot = test_metrics.pivot(index="point_id", columns="model", values="rmse")
    point_names = (
        test_metrics.drop_duplicates("point_id")
        .sort_values("point_id")
        .set_index("point_id")["point_name"]
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    pivot.plot(kind="bar", ax=ax, width=0.78)
    ax.set_title("Test RMSE by NSR bottleneck point")
    ax.set_xlabel("Point")
    ax.set_ylabel("RMSE")
    ax.set_xticklabels(
        [f"{point_id}\n{point_names.loc[point_id]}" for point_id in pivot.index],
        rotation=0,
        ha="center",
        fontsize=8,
    )
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="Model")

    output = FIGURE_DIR / "test_rmse_by_point.png"
    fig.tight_layout()
    save_figure(fig, output)
    plt.close(fig)
    return output


def save_test_metrics_overview(metrics: pd.DataFrame) -> Path:
    test_metrics = metrics[metrics["split"] == "test"].copy()
    test_metrics = test_metrics.sort_values(["point_id", "model"])

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True)
    metric_specs = [
        ("rmse", "RMSE"),
        ("mae", "MAE"),
        ("r2", "R2"),
    ]

    for ax, (metric, label) in zip(axes, metric_specs):
        pivot = test_metrics.pivot(index="point_id", columns="model", values=metric)
        pivot.plot(kind="bar", ax=ax, width=0.78)
        ax.set_title(label)
        ax.set_xlabel("Point")
        ax.set_ylabel(label)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(title="Model")
        ax.tick_params(axis="x", rotation=0)

    fig.suptitle("Test-set regression metrics by point")
    output = FIGURE_DIR / "test_metrics_overview.png"
    fig.tight_layout()
    save_figure(fig, output)
    plt.close(fig)
    return output


def save_prediction_timeseries(predictions: pd.DataFrame) -> list[Path]:
    outputs = []
    plot_data = predictions.copy()
    plot_data["date_dt"] = pd.to_datetime(plot_data["date"] + "-01")

    for point_id, point_table in plot_data.groupby("point_id", sort=True):
        point_name = str(point_table["point_name"].iloc[0])
        point_table = point_table.sort_values("date_dt")
        observed = point_table.drop_duplicates("date_dt").sort_values("date_dt")

        fig, ax = plt.subplots(figsize=(13, 5))
        ax.plot(
            observed["date_dt"],
            observed["sic"],
            color="#111111",
            linewidth=1.8,
            label="Observed SIC",
        )

        for model_name, model_table in point_table.groupby("model", sort=True):
            color = MODEL_COLORS.get(model_name)
            model_table = model_table.sort_values("date_dt")
            ax.plot(
                model_table["date_dt"],
                model_table["prediction"],
                linewidth=1.35,
                color=color,
                label=f"Predicted SIC ({model_name})",
            )

        for split_name, color in [
            ("train", "#e8eef5"),
            ("val", "#fff3bf"),
            ("test", "#fde2e2"),
        ]:
            split_table = observed[observed["split"] == split_name]
            if split_table.empty:
                continue
            ax.axvspan(
                split_table["date_dt"].min(),
                split_table["date_dt"].max(),
                color=color,
                alpha=0.45,
                linewidth=0,
                label=split_name,
            )

        ax.set_title(f"Observed vs predicted SIC - Point {point_id}: {point_name}")
        ax.set_xlabel("Date")
        ax.set_ylabel("SIC")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.25)

        handles, labels = ax.get_legend_handles_labels()
        dedup = dict(zip(labels, handles))
        ax.legend(dedup.values(), dedup.keys(), loc="upper right")

        output = FIGURE_DIR / f"prediction_timeseries_point_{int(point_id):02d}.png"
        fig.tight_layout()
        save_figure(fig, output)
        plt.close(fig)
        outputs.append(output)

    return outputs


def save_prediction_timeseries_by_model(predictions: pd.DataFrame) -> list[Path]:
    outputs = []
    plot_data = predictions.copy()
    plot_data["date_dt"] = pd.to_datetime(plot_data["date"] + "-01")

    for model_name, model_data in plot_data.groupby("model", sort=True):
        model_dir = FIGURE_DIR / f"timeseries_{model_name}"
        model_dir.mkdir(parents=True, exist_ok=True)
        color = MODEL_COLORS.get(model_name, "#d95f02")

        for point_id, point_table in model_data.groupby("point_id", sort=True):
            point_name = str(point_table["point_name"].iloc[0])
            point_table = point_table.sort_values("date_dt")

            fig, ax = plt.subplots(figsize=(13, 5))
            ax.plot(
                point_table["date_dt"],
                point_table["sic"],
                color="#111111",
                linewidth=1.8,
                label="Observed SIC",
            )
            ax.plot(
                point_table["date_dt"],
                point_table["prediction"],
                color=color,
                linewidth=1.45,
                label=f"Predicted SIC ({model_name})",
            )

            for split_name, split_color in [
                ("train", "#e8eef5"),
                ("val", "#fff3bf"),
                ("test", "#fde2e2"),
            ]:
                split_table = point_table[point_table["split"] == split_name]
                if split_table.empty:
                    continue
                ax.axvspan(
                    split_table["date_dt"].min(),
                    split_table["date_dt"].max(),
                    color=split_color,
                    alpha=0.45,
                    linewidth=0,
                    label=split_name,
                )

            ax.set_title(
                f"Observed vs predicted SIC ({model_name}) - Point {point_id}: {point_name}"
            )
            ax.set_xlabel("Date")
            ax.set_ylabel("SIC")
            ax.set_ylim(-0.05, 1.05)
            ax.grid(alpha=0.25)

            handles, labels = ax.get_legend_handles_labels()
            dedup = dict(zip(labels, handles))
            ax.legend(dedup.values(), dedup.keys(), loc="upper right")

            output = model_dir / f"prediction_timeseries_{model_name}_point_{int(point_id):02d}.png"
            fig.tight_layout()
            save_figure(fig, output)
            plt.close(fig)
            outputs.append(output)

    return outputs


def save_standardized_coefficients(coefficients: pd.DataFrame) -> Path:
    coef = coefficients[coefficients["model"] == COEFFICIENT_MODEL_TO_PLOT].copy()
    coef = coef.sort_values(["point_id", "feature"])

    points = coef[["point_id", "point_name"]].drop_duplicates().sort_values("point_id")
    features = list(coef["feature"].drop_duplicates())

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=False, sharey=True)
    axes = axes.ravel()

    for ax, (_, point) in zip(axes, points.iterrows()):
        point_id = point["point_id"]
        point_name = point["point_name"]
        point_coef = coef[coef["point_id"] == point_id].set_index("feature")
        values = [point_coef.loc[feature, "coefficient_standardized"] if feature in point_coef.index else 0 for feature in features]
        colors = ["#1f77b4" if value >= 0 else "#d62728" for value in values]

        ax.bar(features, values, color=colors)
        ax.axhline(0, color="#333333", linewidth=0.8)
        ax.set_title(f"Point {point_id}: {point_name}", fontsize=10)
        ax.tick_params(axis="x", rotation=35)
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle(f"Standardized coefficients ({COEFFICIENT_MODEL_TO_PLOT})")
    fig.supxlabel("Feature")
    fig.supylabel("Coefficient")

    output = FIGURE_DIR / "standardized_coefficients.png"
    fig.tight_layout()
    save_figure(fig, output)
    plt.close(fig)
    return output


def main() -> None:
    for path in [METRICS_PATH, COEFFICIENTS_PATH, PREDICTIONS_PATH]:
        require_file(path)

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    metrics = pd.read_csv(METRICS_PATH)
    coefficients = pd.read_csv(COEFFICIENTS_PATH)
    predictions = pd.read_csv(PREDICTIONS_PATH)

    outputs = [
        save_test_rmse(metrics),
        save_test_metrics_overview(metrics),
        save_standardized_coefficients(coefficients),
    ]
    outputs.extend(save_prediction_timeseries(predictions))
    outputs.extend(save_prediction_timeseries_by_model(predictions))

    print("Saved figures:")
    for output in outputs:
        print(f"  {output}")


if __name__ == "__main__":
    main()
