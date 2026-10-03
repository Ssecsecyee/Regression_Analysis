from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from cnn_common import FIGURES_DIR, RESULTS_DIR, ensure_output_dirs, to_builtin, write_json
from cnn_torch_common import (
    MonthlyGridDataset,
    build_model,
    load_index_rows,
    load_normalization_stats,
    load_ocean_mask,
    masked_regression_metrics,
)


def parse_args() -> argparse.Namespace:
    """비교할 두 checkpoint와 날짜 목록을 command line 인자로 받는다."""
    parser = argparse.ArgumentParser(description="Compare two CNN model outputs as heatmaps.")
    parser.add_argument("--left-checkpoint", required=True)
    parser.add_argument("--right-checkpoint", required=True)
    parser.add_argument("--left-label", default="zero_fill")
    parser.add_argument("--right-label", default="median3x3_min3")
    parser.add_argument("--comparison-id", default="compare_zero_fill_vs_median3x3")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--dates", nargs="*", default=["2025-08", "2025-09", "2025-10"])
    parser.add_argument("--device", default="cuda:3")
    return parser.parse_args()


def load_checkpoint_model(path: Path, device: torch.device):
    """checkpoint에서 모델과 config를 복원한다."""
    checkpoint = torch.load(path, map_location="cpu")
    config = checkpoint["config"]
    model = build_model(config["model_name"], in_channels=7).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, config


def make_dataset(rows: list[dict], config: dict, ocean_mask: np.ndarray) -> MonthlyGridDataset:
    """checkpoint config에 맞는 Dataset을 만든다."""
    return MonthlyGridDataset(
        rows=rows,
        stats=load_normalization_stats(),
        ocean_mask=ocean_mask,
        tos_strategy=config["tos_strategy"],
        add_tos_missing_channel=True,
    )


@torch.no_grad()
def predict_one(model, sample: dict, device: torch.device) -> np.ndarray:
    """sample 하나에 대한 clipped SIC 예측을 반환한다."""
    prediction = model(sample["x"][None, :, :, :].to(device)).cpu().numpy()[0, 0]
    return np.clip(prediction, 0.0, 1.0)


def draw_comparison_figure(
    actual: np.ndarray,
    left_pred: np.ndarray,
    right_pred: np.ndarray,
    mask: np.ndarray,
    left_label: str,
    right_label: str,
    output_path: Path,
    title: str,
) -> None:
    """두 모델의 예측, 오차, 절대오차 차이를 비교하는 heatmap을 저장한다."""
    actual_plot = np.where(mask, actual, np.nan)
    left_plot = np.where(mask, left_pred, np.nan)
    right_plot = np.where(mask, right_pred, np.nan)
    left_error = np.where(mask, left_pred - actual, np.nan)
    right_error = np.where(mask, right_pred - actual, np.nan)

    # 양수면 right 모델의 절대오차가 더 큼, 음수면 right 모델이 더 나음.
    abs_error_delta = np.where(mask, np.abs(right_pred - actual) - np.abs(left_pred - actual), np.nan)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    panels = [
        ("Actual SIC", actual_plot, 0.0, 1.0, "viridis"),
        (f"{left_label} Pred", left_plot, 0.0, 1.0, "viridis"),
        (f"{right_label} Pred", right_plot, 0.0, 1.0, "viridis"),
        (f"{left_label} Error", left_error, -0.5, 0.5, "coolwarm"),
        (f"{right_label} Error", right_error, -0.5, 0.5, "coolwarm"),
        ("Abs Error Delta\nright - left", abs_error_delta, -0.25, 0.25, "coolwarm"),
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


def main() -> None:
    """두 CNN 모델의 같은 날짜 예측 결과를 지도와 수치로 비교한다."""
    ensure_output_dirs()
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    left_model, left_config = load_checkpoint_model(Path(args.left_checkpoint), device)
    right_model, right_config = load_checkpoint_model(Path(args.right_checkpoint), device)
    ocean_mask = load_ocean_mask()

    rows = [row for row in load_index_rows(split=args.split) if row["target_date"] in set(args.dates)]
    if not rows:
        raise ValueError(f"No rows found for dates={args.dates} in split={args.split}.")

    left_dataset = make_dataset(rows, left_config, ocean_mask)
    right_dataset = make_dataset(rows, right_config, ocean_mask)

    result_rows = []
    figure_dir = FIGURES_DIR / args.comparison_id
    for index in range(len(rows)):
        left_sample = left_dataset[index]
        right_sample = right_dataset[index]
        actual = left_sample["y"].numpy()[0]
        mask = left_sample["mask"].numpy()[0].astype(bool)
        meta = left_sample["meta"]

        left_pred = predict_one(left_model, left_sample, device)
        right_pred = predict_one(right_model, right_sample, device)

        left_metrics = masked_regression_metrics(left_pred, actual, mask)
        right_metrics = masked_regression_metrics(right_pred, actual, mask)
        result_rows.append(
            {
                "target_date": meta["target_date"],
                "left_label": args.left_label,
                "right_label": args.right_label,
                "left_rmse": left_metrics["rmse"],
                "right_rmse": right_metrics["rmse"],
                "rmse_delta_right_minus_left": right_metrics["rmse"] - left_metrics["rmse"],
                "left_mae": left_metrics["mae"],
                "right_mae": right_metrics["mae"],
                "mae_delta_right_minus_left": right_metrics["mae"] - left_metrics["mae"],
                "left_bias": left_metrics["bias"],
                "right_bias": right_metrics["bias"],
                "left_r2": left_metrics["r2"],
                "right_r2": right_metrics["r2"],
            }
        )

        output_path = figure_dir / f"compare_{args.split}_{meta['target_date']}.png"
        draw_comparison_figure(
            actual,
            left_pred,
            right_pred,
            mask,
            args.left_label,
            args.right_label,
            output_path,
            title=f"{args.comparison_id} / {meta['target_date']}",
        )
        print(f"Saved: {output_path}")

    result_dir = RESULTS_DIR / args.comparison_id
    result_dir.mkdir(parents=True, exist_ok=True)
    csv_path = result_dir / f"comparison_{args.split}_selected_dates.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(result_rows[0].keys()))
        writer.writeheader()
        writer.writerows(result_rows)

    json_path = result_dir / f"comparison_{args.split}_selected_dates.json"
    write_json(
        json_path,
        to_builtin(
            {
                "comparison_id": args.comparison_id,
                "split": args.split,
                "dates": args.dates,
                "left_checkpoint": args.left_checkpoint,
                "right_checkpoint": args.right_checkpoint,
                "left_label": args.left_label,
                "right_label": args.right_label,
                "csv": str(csv_path),
                "figures_dir": str(figure_dir),
                "rows": result_rows,
            }
        ),
    )
    print(f"Saved: {csv_path}")
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
