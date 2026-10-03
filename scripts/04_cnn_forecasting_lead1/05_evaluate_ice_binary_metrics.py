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
    load_active_union_mask,
    load_index_rows,
    load_monthly_active_mask,
    load_normalization_stats,
    load_ocean_mask,
    to_builtin,
    write_json,
)


ICE_THRESHOLD = 0.15


def parse_args() -> argparse.Namespace:
    """이진 해빙 평가에 사용할 checkpoint와 threshold를 받는다."""
    parser = argparse.ArgumentParser(description="Evaluate lead-1 CNN as binary ice/open-water classifier.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--threshold", type=float, default=ICE_THRESHOLD)
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


def binary_counts(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray, threshold: float) -> dict[str, int]:
    """threshold 기준 ice/open-water 이진 분류 confusion count를 계산한다."""
    valid = mask.astype(bool) & np.isfinite(prediction) & np.isfinite(target)
    pred_ice = prediction >= threshold
    true_ice = target >= threshold
    tp = int((valid & pred_ice & true_ice).sum())
    tn = int((valid & ~pred_ice & ~true_ice).sum())
    fp = int((valid & pred_ice & ~true_ice).sum())
    fn = int((valid & ~pred_ice & true_ice).sum())
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def metrics_from_counts(counts: dict[str, int]) -> dict[str, float | int]:
    """confusion count에서 accuracy, precision, recall, f1을 계산한다."""
    tp = counts["tp"]
    tn = counts["tn"]
    fp = counts["fp"]
    fn = counts["fn"]
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else np.nan
    precision = tp / (tp + fp) if (tp + fp) else np.nan
    recall = tp / (tp + fn) if (tp + fn) else np.nan
    f1 = 2 * precision * recall / (precision + recall) if np.isfinite(precision + recall) and (precision + recall) else np.nan
    return {
        **counts,
        "total": total,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def add_counts(left: dict[str, int], right: dict[str, int]) -> dict[str, int]:
    """두 confusion count dict를 더한다."""
    return {key: left.get(key, 0) + right.get(key, 0) for key in ["tp", "tn", "fp", "fn"]}


def flatten_metrics(prefix: str, metrics: dict[str, float | int]) -> dict:
    """metric dict를 CSV row용 column 이름으로 펼친다."""
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


@torch.no_grad()
def main() -> None:
    """CNN과 persistence baseline의 ice/open-water 이진 분류 성능을 평가한다."""
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
    pooled = {
        "ocean": {"cnn": {"tp": 0, "tn": 0, "fp": 0, "fn": 0}, "persistence": {"tp": 0, "tn": 0, "fp": 0, "fn": 0}},
        "active_union": {"cnn": {"tp": 0, "tn": 0, "fp": 0, "fn": 0}, "persistence": {"tp": 0, "tn": 0, "fp": 0, "fn": 0}},
        "monthly_active": {"cnn": {"tp": 0, "tn": 0, "fp": 0, "fn": 0}, "persistence": {"tp": 0, "tn": 0, "fp": 0, "fn": 0}},
    }

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
            for domain_name, domain_mask in domains.items():
                cnn_counts = binary_counts(pred2d, target2d, domain_mask, args.threshold)
                persistence_counts = binary_counts(persistence2d, target2d, domain_mask, args.threshold)
                cnn_metrics = metrics_from_counts(cnn_counts)
                persistence_metrics = metrics_from_counts(persistence_counts)
                row.update(flatten_metrics(f"{domain_name}_cnn", cnn_metrics))
                row.update(flatten_metrics(f"{domain_name}_persistence", persistence_metrics))
                row[f"{domain_name}_f1_improvement"] = cnn_metrics["f1"] - persistence_metrics["f1"]
                row[f"{domain_name}_accuracy_improvement"] = cnn_metrics["accuracy"] - persistence_metrics["accuracy"]
                pooled[domain_name]["cnn"] = add_counts(pooled[domain_name]["cnn"], cnn_counts)
                pooled[domain_name]["persistence"] = add_counts(pooled[domain_name]["persistence"], persistence_counts)
            monthly_rows.append(row)

    summary = {}
    for domain_name, domain_counts in pooled.items():
        cnn_metrics = metrics_from_counts(domain_counts["cnn"])
        persistence_metrics = metrics_from_counts(domain_counts["persistence"])
        summary[domain_name] = {
            "cnn": cnn_metrics,
            "persistence": persistence_metrics,
            "f1_improvement": cnn_metrics["f1"] - persistence_metrics["f1"],
            "accuracy_improvement": cnn_metrics["accuracy"] - persistence_metrics["accuracy"],
        }

    output_dir = RESULTS_DIR / experiment_id
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"binary_ice_metrics_{args.split}_monthly.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(monthly_rows[0].keys()))
        writer.writeheader()
        writer.writerows(monthly_rows)

    payload = {
        "experiment_id": experiment_id,
        "checkpoint": str(checkpoint_path),
        "split": args.split,
        "threshold": args.threshold,
        "positive_label": f"ice if SIC >= {args.threshold}",
        "baseline": "persistence: SIC(t+1)=SIC(t)",
        "summary": summary,
        "monthly_csv": str(csv_path),
    }
    json_path = output_dir / f"binary_ice_metrics_{args.split}_summary.json"
    write_json(json_path, to_builtin(payload))
    print(f"Saved: {json_path}")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
