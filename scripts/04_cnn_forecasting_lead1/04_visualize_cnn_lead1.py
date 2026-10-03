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
    """시각화할 checkpoint와 target 날짜 목록을 command line 인자로 받는다."""
    parser = argparse.ArgumentParser(description="Visualize lead-1 CNN forecasts against persistence.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--dates", nargs="*", default=["2025-08", "2025-09", "2025-10"])
    parser.add_argument("--device", default="cuda:3")
    return parser.parse_args()


def experiment_id_from_checkpoint(path: Path, checkpoint: dict) -> str:
    """checkpoint config 또는 경로에서 experiment_id를 복원한다."""
    config = checkpoint.get("config", {})
    if "experiment_id" in config:
        return str(config["experiment_id"])
    return path.parent.name


def draw_forecast_figure(
    actual: np.ndarray,
    prediction: np.ndarray,
    persistence: np.ndarray,
    mask: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    """actual, CNN, persistence, error 비교 지도를 저장한다."""
    actual_plot = np.where(mask, actual, np.nan)
    prediction_plot = np.where(mask, prediction, np.nan)
    persistence_plot = np.where(mask, persistence, np.nan)
    cnn_error = np.where(mask, prediction - actual, np.nan)
    persistence_error = np.where(mask, persistence - actual, np.nan)
    abs_delta = np.where(mask, np.abs(prediction - actual) - np.abs(persistence - actual), np.nan)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    panels = [
        ("Actual SIC(t+1)", actual_plot, 0.0, 1.0, "viridis"),
        ("CNN Forecast", prediction_plot, 0.0, 1.0, "viridis"),
        ("Persistence SIC(t)", persistence_plot, 0.0, 1.0, "viridis"),
        ("CNN Error", cnn_error, -0.5, 0.5, "coolwarm"),
        ("Persistence Error", persistence_error, -0.5, 0.5, "coolwarm"),
        ("Abs Error Delta\nCNN - Persistence", abs_delta, -0.25, 0.25, "coolwarm"),
    ]

    for axis, (panel_title, data, vmin, vmax, cmap) in zip(axes.ravel(), panels):
        image = axis.imshow(data, origin="lower", vmin=vmin, vmax=vmax, cmap=cmap)
        axis.set_title(panel_title)
        axis.set_xticks([])
        axis.set_yticks([])
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    fig.suptitle(title)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


@torch.no_grad()
def main() -> None:
    """선택한 target 날짜에 대해 lead-1 예측 비교 지도를 저장한다."""
    ensure_output_dirs()
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint["config"]
    experiment_id = experiment_id_from_checkpoint(checkpoint_path, checkpoint)

    rows = [row for row in load_index_rows(split=args.split) if row["target_date"] in set(args.dates)]
    if not rows:
        raise ValueError(f"No rows found for target dates={args.dates} in split={args.split}.")

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

    output_dir = FIGURES_DIR / experiment_id
    for index in range(len(dataset)):
        sample = dataset[index]
        prediction = model(sample["x"][None, :, :, :].to(device)).cpu().numpy()[0, 0]
        prediction = np.clip(prediction, 0.0, 1.0)
        actual = sample["y"].numpy()[0]
        persistence = sample["persistence"].numpy()[0]
        mask = sample["mask"].numpy()[0].astype(bool)
        meta = sample["meta"]

        output_path = output_dir / f"forecast_{args.split}_{meta['target_date']}.png"
        draw_forecast_figure(
            actual,
            prediction,
            persistence,
            mask,
            output_path,
            title=f"{experiment_id} / input {meta['input_date']} -> target {meta['target_date']}",
        )
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
