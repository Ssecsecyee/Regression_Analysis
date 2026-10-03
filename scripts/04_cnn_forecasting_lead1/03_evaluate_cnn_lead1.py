from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from forecasting_common import (
    RESULTS_DIR,
    Lead1ForecastDataset,
    build_model,
    collate_grid_batch,
    ensure_output_dirs,
    forecast_metrics,
    load_active_union_mask,
    load_index_rows,
    load_monthly_active_mask,
    load_normalization_stats,
    load_ocean_mask,
    to_builtin,
    write_json,
)


def parse_args() -> argparse.Namespace:
    """평가할 checkpoint와 split을 command line 인자로 받는다."""
    parser = argparse.ArgumentParser(description="Evaluate lead-1 CNN against persistence baseline.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def experiment_id_from_checkpoint(path: Path, checkpoint: dict) -> str:
    """checkpoint config 또는 경로에서 experiment_id를 복원한다."""
    config = checkpoint.get("config", {})
    if "experiment_id" in config:
        return str(config["experiment_id"])
    return path.parent.name


def flatten_domain_metrics(prefix: str, metrics: dict) -> dict:
    """domain별 CNN/persistence metric을 CSV row에 넣기 쉬운 dict로 펼친다."""
    return {
        f"{prefix}_cnn_rmse": metrics["cnn"]["rmse"],
        f"{prefix}_cnn_mae": metrics["cnn"]["mae"],
        f"{prefix}_cnn_bias": metrics["cnn"]["bias"],
        f"{prefix}_cnn_r2": metrics["cnn"]["r2"],
        f"{prefix}_persistence_rmse": metrics["persistence"]["rmse"],
        f"{prefix}_persistence_mae": metrics["persistence"]["mae"],
        f"{prefix}_persistence_bias": metrics["persistence"]["bias"],
        f"{prefix}_persistence_r2": metrics["persistence"]["r2"],
        f"{prefix}_rmse_improvement": metrics["rmse_improvement"],
        f"{prefix}_mae_improvement": metrics["mae_improvement"],
    }


@torch.no_grad()
def main() -> None:
    """CNN lead-1 예측과 persistence baseline을 test split에서 비교한다."""
    ensure_output_dirs()
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint["config"]
    experiment_id = experiment_id_from_checkpoint(checkpoint_path, checkpoint)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ocean_mask = load_ocean_mask()
    active_union = load_active_union_mask()
    dataset = Lead1ForecastDataset(
        rows=load_index_rows(split=args.split),
        stats=load_normalization_stats(),
        ocean_mask=ocean_mask,
        tos_strategy=config["tos_strategy"],
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_grid_batch,
    )

    model = build_model(config["model_name"], in_channels=8).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    monthly_rows = []
    pooled = {"ocean": [], "active_union": [], "monthly_active": []}
    for batch in loader:
        prediction = model(batch["x"].to(device)).cpu().numpy()
        prediction = np.clip(prediction, 0.0, 1.0)
        target = batch["y"].numpy()
        persistence = np.clip(batch["persistence"].numpy(), 0.0, 1.0)
        target_mask = batch["mask"].numpy().astype(bool)

        for item_index, meta in enumerate(batch["meta"]):
            pred2d = prediction[item_index, 0]
            target2d = target[item_index, 0]
            persistence2d = persistence[item_index, 0]
            ocean_domain = target_mask[item_index, 0]
            active_union_domain = ocean_domain & active_union if active_union is not None else ocean_domain
            monthly_active = load_monthly_active_mask(int(meta["target_month"]))
            monthly_domain = ocean_domain & monthly_active if monthly_active is not None else ocean_domain

            domains = {
                "ocean": ocean_domain,
                "active_union": active_union_domain,
                "monthly_active": monthly_domain,
            }
            row = {
                "sample_id": meta["sample_id"],
                "input_date": meta["input_date"],
                "target_date": meta["target_date"],
                "target_year": meta["target_year"],
                "target_month": meta["target_month"],
            }
            for name, domain in domains.items():
                metrics = forecast_metrics(pred2d, target2d, persistence2d, domain)
                row.update(flatten_domain_metrics(name, metrics))
                pooled[name].append(metrics)
            monthly_rows.append(row)

    summary = {}
    for name, rows in pooled.items():
        summary[name] = {
            "cnn_mean_rmse": float(np.nanmean([row["cnn"]["rmse"] for row in rows])),
            "cnn_mean_mae": float(np.nanmean([row["cnn"]["mae"] for row in rows])),
            "cnn_mean_bias": float(np.nanmean([row["cnn"]["bias"] for row in rows])),
            "cnn_mean_r2": float(np.nanmean([row["cnn"]["r2"] for row in rows])),
            "persistence_mean_rmse": float(np.nanmean([row["persistence"]["rmse"] for row in rows])),
            "persistence_mean_mae": float(np.nanmean([row["persistence"]["mae"] for row in rows])),
            "persistence_mean_bias": float(np.nanmean([row["persistence"]["bias"] for row in rows])),
            "persistence_mean_r2": float(np.nanmean([row["persistence"]["r2"] for row in rows])),
            "mean_rmse_improvement": float(np.nanmean([row["rmse_improvement"] for row in rows])),
            "mean_mae_improvement": float(np.nanmean([row["mae_improvement"] for row in rows])),
        }

    output_dir = RESULTS_DIR / experiment_id
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"evaluation_{args.split}_monthly.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(monthly_rows[0].keys()))
        writer.writeheader()
        writer.writerows(monthly_rows)

    payload = {
        "experiment_id": experiment_id,
        "checkpoint": str(checkpoint_path),
        "split": args.split,
        "config": config,
        "baseline": "persistence: SIC(t+1)=SIC(t)",
        "summary": summary,
        "monthly_csv": str(csv_path),
    }
    json_path = output_dir / f"evaluation_{args.split}_summary.json"
    write_json(json_path, to_builtin(payload))
    print(f"Saved: {json_path}")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
