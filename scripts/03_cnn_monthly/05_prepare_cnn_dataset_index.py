from __future__ import annotations

import csv

from cnn_common import (
    DATASETS_DIR,
    INPUT_VARIABLES,
    TARGET_VARIABLE,
    ensure_output_dirs,
    monthly_file,
    read_time_axis,
    to_builtin,
    write_json,
)


DEFAULT_LEAD_MONTHS = 0
DEFAULT_TASK_TYPE = "nowcasting"


def build_dataset_rows(lead_months: int) -> list[dict]:
    """lead_months에 따라 CNN sample index를 생성한다.

    lead_months=0이면 X(t) -> SIC(t) nowcasting이다.
    lead_months=1이면 X(t) -> SIC(t+1) forecasting이다.
    """
    time_axis = read_time_axis()
    max_input_index = len(time_axis) - lead_months
    rows = []

    for input_index in range(max_input_index):
        target_index = input_index + lead_months
        input_row = time_axis[input_index]
        target_row = time_axis[target_index]

        rows.append(
            {
                "sample_id": len(rows),
                "task_type": "nowcasting" if lead_months == 0 else "forecasting",
                "lead_months": lead_months,
                "split": target_row["split"],
                "input_time_index": input_row["time_index"],
                "input_date": input_row["date"],
                "input_year": input_row["year"],
                "input_month": input_row["month"],
                "target_time_index": target_row["time_index"],
                "target_date": target_row["date"],
                "target_year": target_row["year"],
                "target_month": target_row["month"],
            }
        )

    return rows


def write_index_csv(rows: list[dict], lead_months: int) -> None:
    """CNN sample index를 CSV로 저장한다."""
    output_path = DATASETS_DIR / f"05_cnn_dataset_index_lead{lead_months}.csv"
    fieldnames = [
        "sample_id",
        "task_type",
        "lead_months",
        "split",
        "input_time_index",
        "input_date",
        "input_year",
        "input_month",
        "target_time_index",
        "target_date",
        "target_year",
        "target_month",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """CNN Dataset이 사용할 월별 sample index와 metadata를 저장한다."""
    ensure_output_dirs()
    lead_months = DEFAULT_LEAD_MONTHS
    rows = build_dataset_rows(lead_months)
    write_index_csv(rows, lead_months)

    split_counts = {}
    for row in rows:
        split_counts[row["split"]] = split_counts.get(row["split"], 0) + 1

    payload = {
        "purpose": "CNN dataset index for monthly grid samples.",
        "task_type": DEFAULT_TASK_TYPE,
        "lead_months": lead_months,
        "input_variables": INPUT_VARIABLES,
        "target_variable": TARGET_VARIABLE,
        "input_files": {variable: str(monthly_file(variable)) for variable in INPUT_VARIABLES},
        "target_file": str(monthly_file(TARGET_VARIABLE)),
        "tensor_shape": {
            "X": "B x 6 x 448 x 304",
            "y": "B x 1 x 448 x 304",
        },
        "split_counts": split_counts,
        "index_csv": str(DATASETS_DIR / f"05_cnn_dataset_index_lead{lead_months}.csv"),
    }
    output_path = DATASETS_DIR / f"05_cnn_dataset_index_lead{lead_months}.json"
    write_json(output_path, to_builtin(payload))
    print(f"Saved: {output_path}")
    print(f"Saved: {DATASETS_DIR / f'05_cnn_dataset_index_lead{lead_months}.csv'}")


if __name__ == "__main__":
    main()
