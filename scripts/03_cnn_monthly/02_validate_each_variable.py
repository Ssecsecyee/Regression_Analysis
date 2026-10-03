from __future__ import annotations

import csv

import h5py
import numpy as np

from cnn_common import (
    PREPROCESSING_DIR,
    VARIABLES,
    dataset_metadata,
    decode_packed_array,
    ensure_output_dirs,
    monthly_file,
    read_time_axis,
    split_name,
    to_builtin,
    write_json,
)


def summarize_array(values: np.ndarray) -> dict:
    """복원된 물리값 배열의 결측과 범위를 요약한다."""
    finite = np.isfinite(values)
    valid_count = int(finite.sum())
    total_count = int(values.size)
    missing_count = total_count - valid_count

    if valid_count == 0:
        return {
            "total_count": total_count,
            "valid_count": 0,
            "missing_count": missing_count,
            "missing_ratio": 1.0,
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
        }

    valid_values = values[finite]
    return {
        "total_count": total_count,
        "valid_count": valid_count,
        "missing_count": missing_count,
        "missing_ratio": missing_count / total_count,
        "min": float(np.nanmin(valid_values)),
        "max": float(np.nanmax(valid_values)),
        "mean": float(np.nanmean(valid_values)),
        "std": float(np.nanstd(valid_values)),
    }


def validate_variable(variable: str, time_axis: list[dict]) -> tuple[dict, list[dict]]:
    """변수 하나에 대해 전체, split별, 월별 결측/범위 정보를 계산한다."""
    path = monthly_file(variable)
    monthly_rows = []

    with h5py.File(path, "r") as file:
        dataset = file[variable]
        metadata = dataset_metadata(dataset, path)

        full_summary_accumulator = []
        split_accumulator: dict[str, list[np.ndarray]] = {"train": [], "val": [], "test": []}

        for time_row in time_axis:
            time_index = time_row["time_index"]
            raw = dataset[time_index, :, :]
            values = decode_packed_array(dataset, raw)
            month_summary = summarize_array(values)

            row = {
                "variable": variable,
                "time_index": time_index,
                "date": time_row["date"],
                "year": time_row["year"],
                "month": time_row["month"],
                "split": time_row["split"],
                **month_summary,
            }
            monthly_rows.append(row)

            full_summary_accumulator.append(values)
            if time_row["split"] in split_accumulator:
                split_accumulator[time_row["split"]].append(values)

        full_values = np.stack(full_summary_accumulator, axis=0)
        summary = {
            "variable": variable,
            "metadata": metadata,
            "overall": summarize_array(full_values),
            "by_split": {},
        }
        for split, arrays in split_accumulator.items():
            if arrays:
                summary["by_split"][split] = summarize_array(np.stack(arrays, axis=0))
            else:
                summary["by_split"][split] = None

    return summary, monthly_rows


def write_monthly_csv(rows: list[dict]) -> None:
    """변수별 월별 검증 요약을 CSV로 저장한다."""
    output_path = PREPROCESSING_DIR / "02_variable_monthly_validation.csv"
    fieldnames = [
        "variable",
        "time_index",
        "date",
        "year",
        "month",
        "split",
        "total_count",
        "valid_count",
        "missing_count",
        "missing_ratio",
        "min",
        "max",
        "mean",
        "std",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """각 변수별 복원값, 결측, 유효 범위를 검증한다."""
    ensure_output_dirs()
    time_axis = read_time_axis()

    summaries = []
    monthly_rows = []
    for variable in VARIABLES:
        summary, rows = validate_variable(variable, time_axis)
        summaries.append(summary)
        monthly_rows.extend(rows)

    write_monthly_csv(monthly_rows)

    output_path = PREPROCESSING_DIR / "02_variable_validation_summary.json"
    write_json(
        output_path,
        to_builtin(
            {
                "purpose": "Validate decoded physical values and missing ratios for CNN inputs.",
                "variables": VARIABLES,
                "summaries": summaries,
            }
        ),
    )

    print(f"Saved: {output_path}")
    print(f"Saved: {PREPROCESSING_DIR / '02_variable_monthly_validation.csv'}")


if __name__ == "__main__":
    main()
