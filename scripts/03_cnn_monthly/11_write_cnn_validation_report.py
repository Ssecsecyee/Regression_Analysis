from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from cnn_common import REPORTS_DIR, RESULTS_DIR, ensure_output_dirs


def parse_args() -> argparse.Namespace:
    """보고서로 묶을 실험 ID와 평가 split을 command line 인자로 받는다."""
    parser = argparse.ArgumentParser(description="Write a short CNN validation report.")
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--split", default="test")
    return parser.parse_args()


def read_json_if_exists(path: Path) -> dict:
    """파일이 존재하면 JSON을 읽고 없으면 빈 dict를 반환한다."""
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def format_domain_summary(summary: dict) -> str:
    """평가 summary dict를 Markdown 표 행으로 변환한다."""
    if not summary:
        return "| - | - | - | - | - |\n"
    rows = []
    for domain, metrics in summary.items():
        rows.append(
            "| {domain} | {rmse:.6f} | {mae:.6f} | {bias:.6f} | {r2:.6f} |".format(
                domain=domain,
                rmse=float(metrics.get("mean_rmse", float("nan"))),
                mae=float(metrics.get("mean_mae", float("nan"))),
                bias=float(metrics.get("mean_bias", float("nan"))),
                r2=float(metrics.get("mean_r2", float("nan"))),
            )
        )
    return "\n".join(rows) + "\n"


def main() -> None:
    """학습/평가 산출물을 읽어 약식 Markdown 검증 보고서를 저장한다."""
    ensure_output_dirs()
    args = parse_args()
    result_dir = RESULTS_DIR / args.experiment_id

    training = read_json_if_exists(result_dir / "training_summary.json")
    evaluation = read_json_if_exists(result_dir / f"evaluation_{args.split}_summary.json")
    config = evaluation.get("config", {})

    report = f"""# CNN Validation Report

## Experiment

- Experiment ID: `{args.experiment_id}`
- Created at: `{datetime.now().isoformat(timespec="seconds")}`
- Split: `{args.split}`
- Model: `{config.get("model_name", "-")}`
- TOS strategy: `{config.get("tos_strategy", "-")}`
- Task: `{config.get("task_type", "-")}`, lead months `{config.get("lead_months", "-")}`
- Device used: `{config.get("device_used", "-")}`

## Training

- Best validation loss: `{training.get("best_val_loss", "-")}`
- Training log: `{training.get("log_path", "-")}`

## Test Metrics

| Domain | RMSE | MAE | Bias | R2 |
|---|---:|---:|---:|---:|
{format_domain_summary(evaluation.get("summary", {}))}
## Data Handling

- Input variables: `{", ".join(config.get("input_variables", []))}`
- Extra channels: `{", ".join(config.get("extra_channels", []))}`
- Target: `{config.get("target_variable", "sic")}`
- Loss: `{config.get("loss", "-")}`
- Output is clipped to `[0, 1]` only during evaluation and visualization.

## Notes

- This report summarizes the full-grid CNN model, not the earlier six-point regression baseline.
- `tos` missing values are never replaced with raw `0 K`.
- When `tos_strategy=zero_fill`, missing `tos` is filled with `0` after normalization, meaning train mean in z-score space.
- When `tos_strategy=median3x3_min3`, only missing `tos` cells with at least three valid ocean neighbors are interpolated before normalization; remaining missing cells are still filled after normalization.
"""

    output_path = REPORTS_DIR / args.experiment_id / f"validation_report_{args.split}.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
