from __future__ import annotations

import csv

from forecasting_common import (
    DATASETS_DIR,
    INPUT_VARIABLES,
    LEAD1_INDEX_PATH,
    TARGET_VARIABLE,
    build_lead1_rows,
    ensure_output_dirs,
    to_builtin,
    write_json,
)


def write_index_csv(rows: list[dict]) -> None:
    """lead-1 sample index를 CSV로 저장한다."""
    LEAD1_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
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
    with open(LEAD1_INDEX_PATH, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """SIC(t)+X(t) -> SIC(t+1) 학습용 sample index를 생성한다."""
    ensure_output_dirs()
    rows = build_lead1_rows()
    write_index_csv(rows)

    split_counts = {}
    for row in rows:
        split_counts[row["split"]] = split_counts.get(row["split"], 0) + 1

    payload = {
        "purpose": "Lead-1 forecasting dataset index.",
        "task_type": "forecasting",
        "lead_months": 1,
        "input_channels": ["sic_t", *[f"{variable}_t" for variable in INPUT_VARIABLES], "tos_missing_t"],
        "target_variable": f"{TARGET_VARIABLE}_t_plus_1",
        "tensor_shape": {
            "X": "B x 8 x 448 x 304",
            "y": "B x 1 x 448 x 304",
        },
        "split_rule": "Split is assigned by target date.",
        "split_counts": split_counts,
        "index_csv": str(LEAD1_INDEX_PATH),
    }
    output_path = DATASETS_DIR / "01_lead1_dataset_index.json"
    write_json(output_path, to_builtin(payload))
    print(f"Saved: {output_path}")
    print(f"Saved: {LEAD1_INDEX_PATH}")


if __name__ == "__main__":
    main()
