from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


THIS_DIR = Path(__file__).resolve().parent
FORECASTING_DIR = THIS_DIR.parent / "04_cnn_forecasting_lead1"
if str(FORECASTING_DIR) not in sys.path:
    sys.path.insert(0, str(FORECASTING_DIR))

from forecasting_common import (  # noqa: E402
    ROOT,
    Lead1ForecastDataset,
    build_model,
    forecast_metrics,
    load_index_rows,
    load_normalization_stats,
    load_ocean_mask,
    to_builtin,
    write_json,
)


EXPERIMENT_DIR = ROOT / "experiments" / "05_test_2015to2025"
GRIDS_DIR = EXPERIMENT_DIR / "grids_npz"
PIXEL_CSV_DIR = EXPERIMENT_DIR / "pixel_csv"
PNG_DIR = EXPERIMENT_DIR / "quicklook_png"
RESULTS_DIR = EXPERIMENT_DIR / "results"
PACKAGES_DIR = EXPERIMENT_DIR / "packages"


def parse_args() -> argparse.Namespace:
    """SIC 관측/예측 격자를 저장할 기간과 checkpoint를 받는다."""
    parser = argparse.ArgumentParser(description="Export observed and predicted SIC grids for 2015-2025.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--start-year", type=int, default=2015)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--save-png", action="store_true")
    parser.add_argument("--save-pixel-csv", action="store_true", default=True)
    return parser.parse_args()


def experiment_id_from_checkpoint(path: Path, checkpoint: dict) -> str:
    """checkpoint config 또는 경로에서 experiment_id를 복원한다."""
    config = checkpoint.get("config", {})
    if "experiment_id" in config:
        return str(config["experiment_id"])
    return path.parent.name


def ensure_output_dirs() -> None:
    """05 테스트 추출 산출물 폴더를 생성한다."""
    for path in [GRIDS_DIR, PIXEL_CSV_DIR, PNG_DIR, RESULTS_DIR, PACKAGES_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def draw_single_map(
    data: np.ndarray,
    output_path: Path,
    title: str,
    vmin: float,
    vmax: float,
    cmap: str,
    colorbar_label: str,
) -> None:
    """관측 SIC, 예측 SIC, 오차 SIC를 각각 독립 PNG로 저장한다."""
    fig, axis = plt.subplots(figsize=(5.5, 6.5), constrained_layout=True)
    image = axis.imshow(data, origin="lower", vmin=vmin, vmax=vmax, cmap=cmap)
    axis.set_title(title)
    axis.set_xticks([])
    axis.set_yticks([])
    colorbar = fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label(colorbar_label)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_pixel_csv(
    path: Path,
    observed: np.ndarray,
    predicted: np.ndarray,
    error: np.ndarray,
    persistence: np.ndarray,
    mask: np.ndarray,
    input_date: str,
    target_date: str,
) -> None:
    """월별 전체 격자의 관측/예측/오차 값을 long-form CSV로 저장한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    y_index, x_index = np.where(mask)

    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "target_date",
                "input_date",
                "y",
                "x",
                "observed_sic",
                "predicted_sic",
                "error_sic",
                "persistence_sic",
            ]
        )
        for y, x in zip(y_index, x_index):
            writer.writerow(
                [
                    target_date,
                    input_date,
                    int(y),
                    int(x),
                    float(observed[y, x]),
                    float(predicted[y, x]),
                    float(error[y, x]),
                    float(persistence[y, x]),
                ]
            )


@torch.no_grad()
def main() -> None:
    """2015~2025 관측 SIC, 예측 SIC, 오차, persistence를 월별 파일로 저장한다."""
    ensure_output_dirs()
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint["config"]
    experiment_id = experiment_id_from_checkpoint(checkpoint_path, checkpoint)

    rows = [
        row
        for row in load_index_rows(split=None)
        if args.start_year <= int(row["target_year"]) <= args.end_year
    ]
    rows = sorted(rows, key=lambda row: (row["target_year"], row["target_month"]))
    if not rows:
        raise ValueError(f"No rows found for {args.start_year}-{args.end_year}.")

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

    grid_dir = GRIDS_DIR / experiment_id / f"{args.start_year}_{args.end_year}"
    pixel_csv_dir = PIXEL_CSV_DIR / experiment_id / f"{args.start_year}_{args.end_year}"
    png_dir = PNG_DIR / experiment_id / f"{args.start_year}_{args.end_year}"
    grid_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for index in range(len(dataset)):
        sample = dataset[index]
        meta = sample["meta"]
        prediction = model(sample["x"][None, :, :, :].to(device)).cpu().numpy()[0, 0].astype("float32")
        prediction = np.clip(prediction, 0.0, 1.0).astype("float32")
        actual = sample["y"].numpy()[0].astype("float32")
        persistence = sample["persistence"].numpy()[0].astype("float32")
        mask = sample["mask"].numpy()[0].astype(bool)
        error = (prediction - actual).astype("float32")

        actual_masked = np.where(mask, actual, np.nan).astype("float32")
        prediction_masked = np.where(mask, prediction, np.nan).astype("float32")
        persistence_masked = np.where(mask, persistence, np.nan).astype("float32")
        error_masked = np.where(mask, error, np.nan).astype("float32")

        metrics = forecast_metrics(prediction, actual, persistence, mask)
        date = meta["target_date"]
        output_path = grid_dir / f"{date}_sic_observed_predicted.npz"
        np.savez_compressed(
            output_path,
            observed_sic=actual_masked,
            predicted_sic=prediction_masked,
            error_sic=error_masked,
            persistence_sic=persistence_masked,
            valid_mask=mask.astype("uint8"),
            input_date=np.array(meta["input_date"]),
            target_date=np.array(date),
        )

        summary_rows.append(
            {
                "target_date": date,
                "input_date": meta["input_date"],
                "target_year": meta["target_year"],
                "target_month": meta["target_month"],
                "npz_file": str(output_path),
                "pixel_csv_file": "",
                "cnn_rmse": metrics["cnn"]["rmse"],
                "cnn_mae": metrics["cnn"]["mae"],
                "cnn_bias": metrics["cnn"]["bias"],
                "cnn_r2": metrics["cnn"]["r2"],
                "persistence_rmse": metrics["persistence"]["rmse"],
                "persistence_mae": metrics["persistence"]["mae"],
                "rmse_improvement": metrics["rmse_improvement"],
                "mae_improvement": metrics["mae_improvement"],
            }
        )
        print(f"Saved: {output_path}")

        if args.save_pixel_csv:
            csv_grid_path = pixel_csv_dir / f"{date}_sic_values.csv"
            write_pixel_csv(
                csv_grid_path,
                actual_masked,
                prediction_masked,
                error_masked,
                persistence_masked,
                mask,
                input_date=meta["input_date"],
                target_date=date,
            )
            summary_rows[-1]["pixel_csv_file"] = str(csv_grid_path)
            print(f"Saved: {csv_grid_path}")

        if args.save_png:
            observed_path = png_dir / "observed" / f"{date}_observed_sic.png"
            predicted_path = png_dir / "predicted" / f"{date}_predicted_sic.png"
            error_path = png_dir / "error" / f"{date}_error_sic.png"
            draw_single_map(
                actual_masked,
                observed_path,
                title=f"{date} Observed SIC",
                vmin=0.0,
                vmax=1.0,
                cmap="viridis",
                colorbar_label="SIC",
            )
            draw_single_map(
                prediction_masked,
                predicted_path,
                title=f"{date} Predicted SIC",
                vmin=0.0,
                vmax=1.0,
                cmap="viridis",
                colorbar_label="SIC",
            )
            draw_single_map(
                error_masked,
                error_path,
                title=f"{date} Prediction Error",
                vmin=-0.5,
                vmax=0.5,
                cmap="coolwarm",
                colorbar_label="Predicted - Observed",
            )
            print(f"Saved: {observed_path}")
            print(f"Saved: {predicted_path}")
            print(f"Saved: {error_path}")

    csv_path = RESULTS_DIR / f"sic_prediction_export_summary_{args.start_year}_{args.end_year}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    payload = {
        "experiment_id": experiment_id,
        "checkpoint": str(checkpoint_path),
        "start_year": args.start_year,
        "end_year": args.end_year,
        "rows": len(summary_rows),
        "grid_dir": str(grid_dir),
        "pixel_csv_dir": str(pixel_csv_dir) if args.save_pixel_csv else None,
        "quicklook_png_dir": str(png_dir) if args.save_png else None,
        "summary_csv": str(csv_path),
        "pixel_csv_columns": [
            "target_date",
            "input_date",
            "y",
            "x",
            "observed_sic",
            "predicted_sic",
            "error_sic",
            "persistence_sic",
        ],
        "file_format": {
            "observed_sic": "float32 448x304, NaN outside valid mask",
            "predicted_sic": "float32 448x304, NaN outside valid mask",
            "error_sic": "predicted_sic - observed_sic",
            "persistence_sic": "SIC(t) baseline",
            "valid_mask": "uint8 448x304",
        },
    }
    json_path = RESULTS_DIR / f"sic_prediction_export_summary_{args.start_year}_{args.end_year}.json"
    write_json(json_path, to_builtin(payload))
    print(f"Saved: {csv_path}")
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
