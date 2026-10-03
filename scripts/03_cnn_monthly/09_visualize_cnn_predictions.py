from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from cnn_common import FIGURES_DIR, ensure_output_dirs
from cnn_torch_common import (
    MonthlyGridDataset,
    build_model,
    load_index_rows,
    load_normalization_stats,
    load_ocean_mask,
)


def parse_args() -> argparse.Namespace:
    """시각화할 checkpoint와 날짜 목록을 command line 인자로 받는다."""
    parser = argparse.ArgumentParser(description="Visualize CNN SIC predictions.")
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


def draw_prediction_figure(actual: np.ndarray, predicted: np.ndarray, output_path: Path, title: str) -> None:
    """actual, predicted, error SIC 지도를 한 장의 figure로 저장한다."""
    error = predicted - actual
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

    panels = [
        ("Actual SIC", actual, 0.0, 1.0, "viridis"),
        ("Predicted SIC", predicted, 0.0, 1.0, "viridis"),
        ("Error", error, -0.5, 0.5, "coolwarm"),
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


@torch.no_grad()
def main() -> None:
    """선택한 test 날짜의 실제 SIC, 예측 SIC, 오차 지도를 저장한다."""
    ensure_output_dirs()
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint["config"]
    experiment_id = experiment_id_from_checkpoint(checkpoint_path, checkpoint)

    rows = [row for row in load_index_rows(split=args.split) if row["target_date"] in set(args.dates)]
    if not rows:
        raise ValueError(f"No rows found for dates={args.dates} in split={args.split}.")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dataset = MonthlyGridDataset(
        rows=rows,
        stats=load_normalization_stats(),
        ocean_mask=load_ocean_mask(),
        tos_strategy=config["tos_strategy"],
        add_tos_missing_channel=True,
    )

    model = build_model(config["model_name"], in_channels=7).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    output_dir = FIGURES_DIR / experiment_id
    for index in range(len(dataset)):
        sample = dataset[index]
        x = sample["x"][None, :, :, :].to(device)
        prediction = model(x).cpu().numpy()[0, 0]
        prediction = np.clip(prediction, 0.0, 1.0)
        actual = sample["y"].numpy()[0]
        mask = sample["mask"].numpy()[0].astype(bool)

        actual_plot = np.where(mask, actual, np.nan)
        prediction_plot = np.where(mask, prediction, np.nan)
        date = sample["meta"]["target_date"]
        output_path = output_dir / f"prediction_{args.split}_{date}.png"
        draw_prediction_figure(
            actual_plot,
            prediction_plot,
            output_path,
            title=f"{experiment_id} / {date}",
        )
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
