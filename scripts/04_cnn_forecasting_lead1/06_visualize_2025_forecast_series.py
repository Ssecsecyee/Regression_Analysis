from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from forecasting_common import (
    FIGURES_DIR,
    Lead1ForecastDataset,
    build_model,
    ensure_output_dirs,
    load_index_rows,
    load_normalization_stats,
    load_ocean_mask,
)


def parse_args() -> argparse.Namespace:
    """2025년 관측/예측 비교 그림 생성 옵션을 받는다."""
    parser = argparse.ArgumentParser(description="Visualize observed vs predicted SIC for all 2025 target months.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--device", default="cuda:3")
    return parser.parse_args()


def experiment_id_from_checkpoint(path: Path, checkpoint: dict) -> str:
    """checkpoint config 또는 경로에서 experiment_id를 복원한다."""
    config = checkpoint.get("config", {})
    if "experiment_id" in config:
        return str(config["experiment_id"])
    return path.parent.name


def draw_month_pair(actual: np.ndarray, predicted: np.ndarray, mask: np.ndarray, output_path: Path, title: str) -> None:
    """한 달의 관측 SIC와 예측 SIC를 나란히 저장한다."""
    actual_plot = np.where(mask, actual, np.nan)
    predicted_plot = np.where(mask, predicted, np.nan)
    error_plot = np.where(mask, predicted - actual, np.nan)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), constrained_layout=True)
    panels = [
        ("Observed SIC", actual_plot, 0.0, 1.0, "viridis"),
        ("Predicted SIC", predicted_plot, 0.0, 1.0, "viridis"),
        ("Prediction Error", error_plot, -0.5, 0.5, "coolwarm"),
    ]
    for axis, (panel_title, data, vmin, vmax, cmap) in zip(axes, panels):
        image = axis.imshow(data, origin="lower", vmin=vmin, vmax=vmax, cmap=cmap)
        axis.set_title(panel_title)
        axis.set_xticks([])
        axis.set_yticks([])
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    fig.suptitle(title)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def draw_single_map(
    data: np.ndarray,
    output_path: Path,
    title: str,
    vmin: float,
    vmax: float,
    cmap: str,
    colorbar_label: str,
) -> None:
    """관측, 예측, 오차 중 하나의 지도만 독립 PNG로 저장한다."""
    fig, axis = plt.subplots(figsize=(5, 6), constrained_layout=True)
    image = axis.imshow(data, origin="lower", vmin=vmin, vmax=vmax, cmap=cmap)
    axis.set_title(title)
    axis.set_xticks([])
    axis.set_yticks([])
    colorbar = fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label(colorbar_label)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def draw_year_grid(records: list[dict], output_path: Path, title: str) -> None:
    """2025년 12개월 관측/예측을 하나의 큰 grid 그림으로 저장한다."""
    fig, axes = plt.subplots(len(records), 2, figsize=(8, 34), constrained_layout=True)
    for row_index, record in enumerate(records):
        for col_index, key in enumerate(["actual", "predicted"]):
            axis = axes[row_index, col_index]
            image = axis.imshow(record[key], origin="lower", vmin=0.0, vmax=1.0, cmap="viridis")
            axis.set_title(f"{record['date']} {'Observed' if key == 'actual' else 'Predicted'}")
            axis.set_xticks([])
            axis.set_yticks([])
            fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.suptitle(title)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


@torch.no_grad()
def main() -> None:
    """2025년 1월부터 12월까지 관측 SIC와 lead-1 예측 SIC 비교 그림을 저장한다."""
    ensure_output_dirs()
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint["config"]
    experiment_id = experiment_id_from_checkpoint(checkpoint_path, checkpoint)

    rows = [
        row
        for row in load_index_rows(split=args.split)
        if int(row["target_year"]) == args.year
    ]
    rows = sorted(rows, key=lambda row: row["target_month"])
    if not rows:
        raise ValueError(f"No target rows found for year={args.year}, split={args.split}.")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dataset = Lead1ForecastDataset(
        rows=rows,
        stats=load_normalization_stats(),
        ocean_mask=load_ocean_mask(),
        tos_strategy=config["tos_strategy"],
    )
    model = build_model(config["model_name"], in_channels=8).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    output_dir = FIGURES_DIR / experiment_id / f"observed_vs_predicted_{args.year}"
    year_records = []
    for index in range(len(dataset)):
        sample = dataset[index]
        meta = sample["meta"]
        prediction = model(sample["x"][None, :, :, :].to(device)).cpu().numpy()[0, 0]
        prediction = np.clip(prediction, 0.0, 1.0)
        actual = sample["y"].numpy()[0]
        mask = sample["mask"].numpy()[0].astype(bool)

        actual_plot = np.where(mask, actual, np.nan)
        prediction_plot = np.where(mask, prediction, np.nan)
        year_records.append({"date": meta["target_date"], "actual": actual_plot, "predicted": prediction_plot})

        output_path = output_dir / f"observed_vs_predicted_{meta['target_date']}.png"
        draw_month_pair(
            actual,
            prediction,
            mask,
            output_path,
            title=f"{experiment_id} / input {meta['input_date']} -> target {meta['target_date']}",
        )
        print(f"Saved: {output_path}")

        observed_path = output_dir / "observed" / f"{meta['target_date']}_observed_sic.png"
        predicted_path = output_dir / "predicted" / f"{meta['target_date']}_predicted_sic.png"
        error_path = output_dir / "error" / f"{meta['target_date']}_prediction_error.png"
        draw_single_map(
            actual_plot,
            observed_path,
            title=f"Observed SIC / {meta['target_date']}",
            vmin=0.0,
            vmax=1.0,
            cmap="viridis",
            colorbar_label="SIC",
        )
        draw_single_map(
            prediction_plot,
            predicted_path,
            title=f"Predicted SIC / {meta['target_date']}",
            vmin=0.0,
            vmax=1.0,
            cmap="viridis",
            colorbar_label="SIC",
        )
        draw_single_map(
            np.where(mask, prediction - actual, np.nan),
            error_path,
            title=f"Prediction Error / {meta['target_date']}",
            vmin=-0.5,
            vmax=0.5,
            cmap="coolwarm",
            colorbar_label="Predicted - Observed",
        )
        print(f"Saved: {observed_path}")
        print(f"Saved: {predicted_path}")
        print(f"Saved: {error_path}")

    grid_path = output_dir / f"observed_vs_predicted_{args.year}_all_months.png"
    draw_year_grid(
        year_records,
        grid_path,
        title=f"{experiment_id} / Observed vs Predicted SIC / {args.year}",
    )
    print(f"Saved: {grid_path}")


if __name__ == "__main__":
    main()
