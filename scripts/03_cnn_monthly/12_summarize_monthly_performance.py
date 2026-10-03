from __future__ import annotations

import argparse
import csv
import json
import zipfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from cnn_common import EXPERIMENT_DIR, FIGURES_DIR, REPORTS_DIR, RESULTS_DIR, ensure_output_dirs, to_builtin, write_json


DOMAINS = ["ocean", "active_union", "monthly_active"]
METRICS = ["rmse", "mae", "bias", "r2"]


def parse_args() -> argparse.Namespace:
    """월별 성능 요약 대상 실험과 출력 옵션을 command line 인자로 받는다."""
    parser = argparse.ArgumentParser(description="Summarize monthly CNN performance and package outputs.")
    parser.add_argument("--split", default="test")
    parser.add_argument("--experiment-ids", nargs="*", default=None)
    parser.add_argument("--summary-id", default="monthly_performance_summary")
    parser.add_argument("--make-zip", action="store_true", default=True)
    return parser.parse_args()


def discover_experiments(split: str) -> list[str]:
    """evaluation CSV가 존재하는 실험 ID를 results 폴더에서 자동 탐색한다."""
    experiments = []
    if not RESULTS_DIR.exists():
        return experiments

    for path in sorted(RESULTS_DIR.iterdir()):
        if not path.is_dir():
            continue
        csv_path = path / f"evaluation_{split}_monthly.csv"
        if csv_path.exists():
            experiments.append(path.name)
    return experiments


def read_monthly_csv(experiment_id: str, split: str) -> list[dict]:
    """08번 평가 스크립트가 만든 월별 CSV를 읽는다."""
    path = RESULTS_DIR / experiment_id / f"evaluation_{split}_monthly.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing monthly evaluation CSV: {path}")

    rows = []
    with open(path, newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            parsed = {
                "experiment_id": experiment_id,
                "sample_id": int(row["sample_id"]),
                "target_date": row["target_date"],
                "target_year": int(row["target_year"]),
                "target_month": int(row["target_month"]),
            }
            for domain in DOMAINS:
                for metric in METRICS:
                    key = f"{domain}_{metric}"
                    parsed[key] = float(row[key])
            rows.append(parsed)
    return rows


def summarize_by_month(rows: list[dict]) -> list[dict]:
    """실험별, 월별 평균 성능을 계산한다."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["experiment_id"], row["target_month"])].append(row)

    summary_rows = []
    for (experiment_id, month), group_rows in sorted(grouped.items()):
        summary = {
            "experiment_id": experiment_id,
            "month": month,
            "sample_count": len(group_rows),
        }
        for domain in DOMAINS:
            for metric in METRICS:
                key = f"{domain}_{metric}"
                values = [row[key] for row in group_rows]
                summary[f"{key}_mean"] = float(np.nanmean(values))
                summary[f"{key}_std"] = float(np.nanstd(values))
        summary_rows.append(summary)
    return summary_rows


def write_summary_csv(path: Path, rows: list[dict]) -> None:
    """월별 요약 결과를 CSV로 저장한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_metric(rows: list[dict], domain: str, metric: str, output_path: Path, title: str) -> None:
    """실험별 월별 metric line plot을 저장한다."""
    experiments = sorted({row["experiment_id"] for row in rows})
    fig, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)

    for experiment_id in experiments:
        exp_rows = sorted([row for row in rows if row["experiment_id"] == experiment_id], key=lambda item: item["month"])
        months = [row["month"] for row in exp_rows]
        values = [row[f"{domain}_{metric}_mean"] for row in exp_rows]
        axis.plot(months, values, marker="o", linewidth=2, label=experiment_id)

    axis.set_title(title)
    axis.set_xlabel("Month")
    axis.set_ylabel(metric.upper())
    axis.set_xticks(range(1, 13))
    axis.grid(True, alpha=0.3)
    axis.legend(fontsize=8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def best_experiment_by_domain(rows: list[dict], domain: str, metric: str = "rmse") -> dict:
    """특정 domain/metric 기준으로 전체 월 평균이 가장 낮은 실험을 찾는다."""
    grouped = defaultdict(list)
    key = f"{domain}_{metric}_mean"
    for row in rows:
        grouped[row["experiment_id"]].append(row[key])

    scores = {
        experiment_id: float(np.nanmean(values))
        for experiment_id, values in grouped.items()
    }
    best_id = min(scores, key=scores.get)
    return {"best_experiment_id": best_id, "score": scores[best_id], "all_scores": scores}


def write_report(path: Path, rows: list[dict], split: str, figure_paths: list[Path]) -> None:
    """월별 성능 요약 Markdown 보고서를 저장한다."""
    best_monthly_active = best_experiment_by_domain(rows, "monthly_active", "rmse")
    best_ocean = best_experiment_by_domain(rows, "ocean", "rmse")

    difficult_rows = sorted(rows, key=lambda row: row["monthly_active_rmse_mean"], reverse=True)[:8]
    difficult_table = "\n".join(
        [
            (
                f"| {row['experiment_id']} | {row['month']} | "
                f"{row['monthly_active_rmse_mean']:.6f} | {row['monthly_active_mae_mean']:.6f} | "
                f"{row['monthly_active_r2_mean']:.6f} |"
            )
            for row in difficult_rows
        ]
    )

    figure_lines = "\n".join([f"- `{path}`" for path in figure_paths])
    report = f"""# Monthly CNN Performance Summary

## Summary

- Created at: `{datetime.now().isoformat(timespec="seconds")}`
- Split: `{split}`
- Best ocean RMSE model: `{best_ocean['best_experiment_id']}` (`{best_ocean['score']:.6f}`)
- Best monthly active RMSE model: `{best_monthly_active['best_experiment_id']}` (`{best_monthly_active['score']:.6f}`)

## Most Difficult Monthly Cases

| Experiment | Month | Monthly Active RMSE | Monthly Active MAE | Monthly Active R2 |
|---|---:|---:|---:|---:|
{difficult_table}

## Figures

{figure_lines}

## Reading Notes

- `ocean` evaluates all valid ocean target pixels.
- `active_union` evaluates pixels that are seasonally active at least once.
- `monthly_active` is the strictest monthly sea-ice-active evaluation domain.
- High summer RMSE usually means the model struggles around fast-changing ice-edge transition zones.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")


def make_zip(zip_path: Path, files: list[Path]) -> None:
    """생성된 요약 파일과 원본 평가 CSV를 zip으로 묶는다."""
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_path in files:
            if file_path.exists():
                archive.write(file_path, arcname=str(file_path.relative_to(EXPERIMENT_DIR)))


def main() -> None:
    """월별 성능 요약 산출물과 로컬 확인용 zip 패키지를 만든다."""
    ensure_output_dirs()
    args = parse_args()
    experiment_ids = args.experiment_ids or discover_experiments(args.split)
    if not experiment_ids:
        raise ValueError(f"No experiments found with evaluation_{args.split}_monthly.csv.")

    all_rows = []
    source_files = []
    for experiment_id in experiment_ids:
        all_rows.extend(read_monthly_csv(experiment_id, args.split))
        source_files.append(RESULTS_DIR / experiment_id / f"evaluation_{args.split}_monthly.csv")

    summary_rows = summarize_by_month(all_rows)
    output_result_dir = RESULTS_DIR / args.summary_id
    output_figure_dir = FIGURES_DIR / args.summary_id
    output_report_dir = REPORTS_DIR / args.summary_id

    summary_csv = output_result_dir / f"monthly_metric_summary_{args.split}.csv"
    write_summary_csv(summary_csv, summary_rows)

    figure_paths = [
        output_figure_dir / f"monthly_active_rmse_{args.split}.png",
        output_figure_dir / f"monthly_active_mae_{args.split}.png",
        output_figure_dir / f"ocean_rmse_{args.split}.png",
        output_figure_dir / f"active_union_rmse_{args.split}.png",
    ]
    plot_metric(summary_rows, "monthly_active", "rmse", figure_paths[0], "Monthly Active RMSE by Month")
    plot_metric(summary_rows, "monthly_active", "mae", figure_paths[1], "Monthly Active MAE by Month")
    plot_metric(summary_rows, "ocean", "rmse", figure_paths[2], "Ocean RMSE by Month")
    plot_metric(summary_rows, "active_union", "rmse", figure_paths[3], "Active Union RMSE by Month")

    summary_json = output_result_dir / f"monthly_metric_summary_{args.split}.json"
    payload = {
        "summary_id": args.summary_id,
        "split": args.split,
        "experiment_ids": experiment_ids,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "best_ocean_rmse": best_experiment_by_domain(summary_rows, "ocean", "rmse"),
        "best_monthly_active_rmse": best_experiment_by_domain(summary_rows, "monthly_active", "rmse"),
        "summary_csv": str(summary_csv),
        "figures": [str(path) for path in figure_paths],
    }
    write_json(summary_json, to_builtin(payload))

    report_path = output_report_dir / f"monthly_performance_report_{args.split}.md"
    write_report(report_path, summary_rows, args.split, figure_paths)

    package_files = [summary_csv, summary_json, report_path, *figure_paths, *source_files]
    zip_path = EXPERIMENT_DIR / f"{args.summary_id}_{args.split}.zip"
    if args.make_zip:
        make_zip(zip_path, package_files)
        print(f"Saved: {zip_path}")

    print(f"Saved: {summary_csv}")
    print(f"Saved: {summary_json}")
    print(f"Saved: {report_path}")
    for path in figure_paths:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
