from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from cnn_common import RESULTS_DIR, ensure_output_dirs, to_builtin, write_json
from cnn_torch_common import (
    MonthlyGridDataset,
    build_model,
    collate_grid_batch,
    load_active_union_mask,
    load_index_rows,
    load_monthly_active_mask,
    load_normalization_stats,
    load_ocean_mask,
    masked_regression_metrics,
)


def parse_args() -> argparse.Namespace:
    """평가할 checkpoint와 split을 command line 인자로 받는다."""
    parser = argparse.ArgumentParser(description="Evaluate a trained monthly SIC CNN.")
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


@torch.no_grad()
def main() -> None:
    """test/val/train split에 대해 전체 및 영역별 회귀 지표를 계산한다."""
    ensure_output_dirs()
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint["config"]
    experiment_id = experiment_id_from_checkpoint(checkpoint_path, checkpoint)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ocean_mask = load_ocean_mask()
    active_union = load_active_union_mask()
    dataset = MonthlyGridDataset(
        rows=load_index_rows(split=args.split),
        stats=load_normalization_stats(),
        ocean_mask=ocean_mask,
        tos_strategy=config["tos_strategy"],
        add_tos_missing_channel=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_grid_batch,
    )

    model = build_model(config["model_name"], in_channels=7).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    monthly_rows = []
    pooled = {"ocean": [], "active_union": [], "monthly_active": []}

    for batch in loader:
        prediction = model(batch["x"].to(device)).cpu().numpy()
        prediction = np.clip(prediction, 0.0, 1.0)
        target = batch["y"].numpy()
        target_mask = batch["mask"].numpy().astype(bool)

        for item_index, meta in enumerate(batch["meta"]):
            pred2d = prediction[item_index, 0]
            target2d = target[item_index, 0]
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
                "target_date": meta["target_date"],
                "target_year": meta["target_year"],
                "target_month": meta["target_month"],
            }
            for name, domain in domains.items():
                metrics = masked_regression_metrics(pred2d, target2d, domain)
                row[f"{name}_rmse"] = metrics["rmse"]
                row[f"{name}_mae"] = metrics["mae"]
                row[f"{name}_bias"] = metrics["bias"]
                row[f"{name}_r2"] = metrics["r2"]
                pooled[name].append(metrics)
            monthly_rows.append(row)

    summary = {}
    for name, rows in pooled.items():
        summary[name] = {
            "mean_rmse": float(np.nanmean([row["rmse"] for row in rows])),
            "mean_mae": float(np.nanmean([row["mae"] for row in rows])),
            "mean_bias": float(np.nanmean([row["bias"] for row in rows])),
            "mean_r2": float(np.nanmean([row["r2"] for row in rows])),
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
        "summary": summary,
        "monthly_csv": str(csv_path),
    }
    json_path = output_dir / f"evaluation_{args.split}_summary.json"
    write_json(json_path, to_builtin(payload))
    print(f"Saved: {json_path}")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
