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

PREDICTIONS_PATH = PROCESSED_DIR / "point_regression_predictions.csv"

NAVIGATION_MONTHS = [7, 8, 9, 10]
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


def save_navigation_season_predictions_by_model(predictions: pd.DataFrame) -> list[Path]:
    outputs = []
    plot_data = predictions[predictions["month"].isin(NAVIGATION_MONTHS)].copy()
    plot_data["date_dt"] = pd.to_datetime(plot_data["date"] + "-01")

    for model_name, model_data in plot_data.groupby("model", sort=True):
        model_dir = FIGURE_DIR / f"navigation_season_{model_name}"
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
                marker="o",
                markersize=2.5,
                label="Observed SIC",
            )
            ax.plot(
                point_table["date_dt"],
                point_table["prediction"],
                color=color,
                linewidth=1.45,
                marker="o",
                markersize=2.2,
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

            month_text = ", ".join(f"{month:02d}" for month in NAVIGATION_MONTHS)
            ax.set_title(
                f"Navigation-season SIC ({model_name}) - Point {point_id}: {point_name}"
            )
            ax.set_xlabel(f"Date, months {month_text}")
            ax.set_ylabel("SIC")
            ax.set_ylim(-0.05, 1.05)
            ax.grid(alpha=0.25)

            handles, labels = ax.get_legend_handles_labels()
            dedup = dict(zip(labels, handles))
            ax.legend(dedup.values(), dedup.keys(), loc="upper right")

            output = model_dir / (
                f"navigation_season_prediction_{model_name}_point_{int(point_id):02d}.png"
            )
            fig.tight_layout()
            save_figure(fig, output)
            plt.close(fig)
            outputs.append(output)

    return outputs


def main() -> None:
    require_file(PREDICTIONS_PATH)
    predictions = pd.read_csv(PREDICTIONS_PATH)

    outputs = save_navigation_season_predictions_by_model(predictions)

    print("Saved navigation-season figures:")
    for output in outputs:
        print(f"  {output}")


if __name__ == "__main__":
    main()
