from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from cnn_common import INPUT_VARIABLES, MODELS_DIR, RESULTS_DIR, ensure_output_dirs, to_builtin, write_json
from cnn_torch_common import (
    MonthlyGridDataset,
    build_model,
    collate_grid_batch,
    load_index_rows,
    load_normalization_stats,
    load_ocean_mask,
    masked_mse_loss,
)


def parse_args() -> argparse.Namespace:
    """학습 실행 옵션을 command line 인자로 받는다."""
    parser = argparse.ArgumentParser(description="Train a monthly grid SIC CNN baseline.")
    parser.add_argument("--experiment-id", default=None)
    parser.add_argument("--model-name", default="simple_cnn")
    parser.add_argument("--tos-strategy", choices=["zero_fill", "median3x3_min3"], default="zero_fill")
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def make_experiment_id(args: argparse.Namespace) -> str:
    """명시 ID가 없으면 시간과 TOS 전략을 포함한 실험 ID를 만든다."""
    if args.experiment_id:
        return args.experiment_id
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"cnn_{args.model_name}_{args.tos_strategy}_{stamp}"


def train_one_epoch(model, loader, optimizer, device: torch.device) -> float:
    """train split 한 epoch의 평균 masked MSE를 계산하며 학습한다."""
    model.train()
    total_loss = 0.0
    total_batches = 0
    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        mask = batch["mask"].to(device)

        optimizer.zero_grad(set_to_none=True)
        prediction = model(x)
        loss = masked_mse_loss(prediction, y, mask)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.detach().cpu())
        total_batches += 1
    return total_loss / max(total_batches, 1)


@torch.no_grad()
def validate_one_epoch(model, loader, device: torch.device) -> float:
    """validation split 한 epoch의 평균 masked MSE를 계산한다."""
    model.eval()
    total_loss = 0.0
    total_batches = 0
    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        mask = batch["mask"].to(device)
        prediction = model(x)
        loss = masked_mse_loss(prediction, y, mask)
        total_loss += float(loss.cpu())
        total_batches += 1
    return total_loss / max(total_batches, 1)


def save_checkpoint(path: Path, model, optimizer, epoch: int, val_loss: float, config: dict) -> None:
    """모델 가중치와 학습 상태를 checkpoint로 저장한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "val_loss": val_loss,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
        },
        path,
    )


def main() -> None:
    """CNN baseline을 학습하고 best/last checkpoint와 epoch 로그를 저장한다."""
    ensure_output_dirs()
    args = parse_args()
    torch.manual_seed(args.seed)

    experiment_id = make_experiment_id(args)
    model_dir = MODELS_DIR / experiment_id
    result_dir = RESULTS_DIR / experiment_id
    model_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    stats = load_normalization_stats()
    ocean_mask = load_ocean_mask()

    train_dataset = MonthlyGridDataset(
        rows=load_index_rows(split="train"),
        stats=stats,
        ocean_mask=ocean_mask,
        tos_strategy=args.tos_strategy,
        add_tos_missing_channel=True,
    )
    val_dataset = MonthlyGridDataset(
        rows=load_index_rows(split="val"),
        stats=stats,
        ocean_mask=ocean_mask,
        tos_strategy=args.tos_strategy,
        add_tos_missing_channel=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_grid_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_grid_batch,
    )

    in_channels = len(INPUT_VARIABLES) + 1
    model = build_model(args.model_name, in_channels=in_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    config = {
        "experiment_id": experiment_id,
        "model_name": args.model_name,
        "tos_strategy": args.tos_strategy,
        "input_variables": INPUT_VARIABLES,
        "extra_channels": ["tos_missing"],
        "target_variable": "sic",
        "task_type": "nowcasting",
        "lead_months": 0,
        "loss": "masked_mse_ocean_target_finite",
        "device_requested": args.device,
        "device_used": str(device),
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "num_workers": args.num_workers,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
    }
    write_json(model_dir / "config.json", to_builtin(config))

    log_path = result_dir / "training_log.csv"
    best_val = float("inf")
    with open(log_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["epoch", "train_loss", "val_loss"])
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, device)
            val_loss = validate_one_epoch(model, val_loader, device)
            writer.writerow({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
            file.flush()

            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(model_dir / "best.pt", model, optimizer, epoch, val_loss, config)
            save_checkpoint(model_dir / "last.pt", model, optimizer, epoch, val_loss, config)
            print(f"epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

    summary = {"experiment_id": experiment_id, "best_val_loss": best_val, "log_path": str(log_path)}
    write_json(result_dir / "training_summary.json", to_builtin(summary))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
