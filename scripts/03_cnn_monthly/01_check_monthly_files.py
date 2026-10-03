from __future__ import annotations

import h5py

from cnn_common import (
    INPUT_VARIABLES,
    PREPROCESSING_DIR,
    TARGET_VARIABLE,
    VARIABLES,
    dataset_metadata,
    ensure_output_dirs,
    monthly_file,
    read_time_axis,
    to_builtin,
    write_json,
)


def inspect_file(variable: str) -> dict:
    """월별 NetCDF 파일 하나의 존재 여부와 핵심 구조를 확인한다."""
    path = monthly_file(variable)
    record = {
        "variable": variable,
        "path": str(path),
        "exists": path.exists(),
        "has_variable": False,
        "has_year": False,
        "has_month": False,
        "has_y": False,
        "has_x": False,
        "metadata": None,
        "status": "missing_file",
    }

    if not path.exists():
        return record

    with h5py.File(path, "r") as file:
        record["has_variable"] = variable in file
        record["has_year"] = "year" in file
        record["has_month"] = "month" in file
        record["has_y"] = "y" in file
        record["has_x"] = "x" in file
        if variable in file:
            record["metadata"] = dataset_metadata(file[variable], path)

    required_ok = all(
        [
            record["has_variable"],
            record["has_year"],
            record["has_month"],
            record["has_y"],
            record["has_x"],
        ]
    )
    record["status"] = "ok" if required_ok else "invalid_structure"
    return record


def main() -> None:
    """CNN 입력에 필요한 월별 NetCDF 파일들의 기본 구조를 검증한다."""
    ensure_output_dirs()

    file_records = [inspect_file(variable) for variable in VARIABLES]
    time_axis = read_time_axis()

    shape_set = {
        tuple(record["metadata"]["shape"])
        for record in file_records
        if record["metadata"] is not None
    }
    all_shapes_match = len(shape_set) == 1
    all_files_ok = all(record["status"] == "ok" for record in file_records)

    payload = {
        "purpose": "Validate monthly NetCDF files before CNN preprocessing.",
        "input_variables": INPUT_VARIABLES,
        "target_variable": TARGET_VARIABLE,
        "time_axis_length": len(time_axis),
        "date_start": time_axis[0]["date"],
        "date_end": time_axis[-1]["date"],
        "all_files_ok": all_files_ok,
        "all_shapes_match": all_shapes_match,
        "shape_set": [list(shape) for shape in sorted(shape_set)],
        "files": file_records,
    }

    output_path = PREPROCESSING_DIR / "01_monthly_file_check.json"
    write_json(output_path, to_builtin(payload))

    print(f"Saved: {output_path}")
    print(f"All files ok: {all_files_ok}")
    print(f"All shapes match: {all_shapes_match}")


if __name__ == "__main__":
    main()
