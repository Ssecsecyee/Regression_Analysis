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
    load_index_rows,
    load_normalization_stats,
    load_ocean_mask,
    masked_regression_metrics,
)


def parse_args() -> argparse.Namespace:
    """앙상블에 사용할 checkpoint 목록과 평가 split을 받는다."""
    parser = argparse.ArgumentParser(description="Evaluate an ensemble of trained CNN checkpoints.")
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--ensemble-id", default=None)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--batch-size", type=int, default=1)
    return parser.parse_args()


def load_member(path: Path, device: torch.device):
    """checkpoint 하나를 모델과 config로 복원한다."""
    checkpoint = torch.load(path, map_location="cpu")
    config = checkpoint["config"]
    model = build_model(config["model_name"], in_channels=7).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, config


@torch.no_grad()
def main() -> None:
    """여러 checkpoint 예측을 평균해 앙상블 test 지표를 저장한다."""
    ensure_output_dirs()
    args = parse_args()
    checkpoint_paths = [Path(path) for path in args.checkpoints]
    ensemble_id = args.ensemble_id or "ensemble_" + str(len(checkpoint_paths)) + "_members"

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    members = [load_member(path, device) for path in checkpoint_paths]
    first_config = members[0][1]

    dataset = MonthlyGridDataset(
        rows=load_index_rows(split=args.split),
        stats=load_normalization_stats(),
        ocean_mask=load_ocean_mask(),
        tos_strategy=first_config["tos_strategy"],
        add_tos_missing_channel=True,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_grid_batch)

    monthly_rows = []
    metrics_rows = []
    for batch in loader:
        x = batch["x"].to(device)
        member_predictions = []
        for model, _config in members:
            member_predictions.append(model(x).cpu().numpy())
        prediction = np.clip(np.mean(member_predictions, axis=0), 0.0, 1.0)
        target = batch["y"].numpy()
        mask = batch["mask"].numpy().astype(bool)

        for item_index, meta in enumerate(batch["meta"]):
            metrics = masked_regression_metrics(prediction[item_index, 0], target[item_index, 0], mask[item_index, 0])
            row = {
                "sample_id": meta["sample_id"],
                "target_date": meta["target_date"],
                "target_year": meta["target_year"],
                "target_month": meta["target_month"],
                **metrics,
            }
            monthly_rows.append(row)
            metrics_rows.append(metrics)

    summary = {
        "count": int(np.sum([row["count"] for row in metrics_rows])),
        "mean_rmse": float(np.nanmean([row["rmse"] for row in metrics_rows])),
        "mean_mae": float(np.nanmean([row["mae"] for row in metrics_rows])),
        "mean_bias": float(np.nanmean([row["bias"] for row in metrics_rows])),
        "mean_r2": float(np.nanmean([row["r2"] for row in metrics_rows])),
    }

    output_dir = RESULTS_DIR / ensemble_id
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"ensemble_{args.split}_monthly.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(monthly_rows[0].keys()))
        writer.writeheader()
        writer.writerows(monthly_rows)

    payload = {
        "ensemble_id": ensemble_id,
        "split": args.split,
        "checkpoints": [str(path) for path in checkpoint_paths],
        "summary": summary,
        "monthly_csv": str(csv_path),
    }
    json_path = output_dir / f"ensemble_{args.split}_summary.json"
    write_json(json_path, to_builtin(payload))
    print(f"Saved: {json_path}")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
