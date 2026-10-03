from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MONTHLY_DIR = ROOT / "data" / "monthly"
MASK_DIR = ROOT / "data" / "masks"
OUT_DIR = ROOT / "processed_data"

TABLE_PATH = OUT_DIR / "monthly_point_regression_table.csv"
METADATA_PATH = OUT_DIR / "monthly_point_regression_metadata.json"

VARIABLES = ["sic", "tos", "tas", "rlds", "rsds", "uas", "vas"]
INPUT_VARIABLES = ["tos", "tas", "rlds", "rsds", "uas", "vas"]
TARGET_VARIABLE = "sic"

POINTS = {
    1: {"name": "Vilkitsky Strait", "y_idx": 259, "x_idx": 183},
    2: {"name": "NW Laptev Sea", "y_idx": 262, "x_idx": 171},
    3: {"name": "Sannikov Strait", "y_idx": 281, "x_idx": 150},
    4: {"name": "Dmitry Laptev Strait", "y_idx": 286, "x_idx": 146},
    5: {"name": "Central East Siberian Sea", "y_idx": 273, "x_idx": 126},
    6: {"name": "Long Strait / Wrangel", "y_idx": 277, "x_idx": 94},
}


def monthly_file(variable: str) -> Path:
    return MONTHLY_DIR / f"{variable}_arctic25km_monthly_1988-2025.nc"


def decode_variable_series(dataset: h5py.Dataset, y_idx: int, x_idx: int) -> np.ndarray:
    raw = dataset[:, y_idx, x_idx].astype("float64")

    fill_value = dataset.attrs.get("_FillValue")
    if fill_value is not None:
        raw[raw == float(fill_value[0])] = np.nan

    scale_factor = dataset.attrs.get("scale_factor")
    if scale_factor is not None:
        raw *= float(scale_factor[0])

    add_offset = dataset.attrs.get("add_offset")
    if add_offset is not None:
        raw += float(add_offset[0])

    return raw


def split_name(year: int) -> str:
    if year <= 2013:
        return "train"
    if year <= 2017:
        return "val"
    return "test"


def to_builtin(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    required_files = [monthly_file(variable) for variable in VARIABLES]
    required_files.extend([MASK_DIR / "land_mask.nc", MASK_DIR / "active_mask.nc"])
    missing = [path for path in required_files if not path.exists()]
    if missing:
        missing_list = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"Required files are missing:\n{missing_list}")

    with h5py.File(monthly_file(TARGET_VARIABLE), "r") as sic_file:
        year = sic_file["year"][:].astype(int)
        month = sic_file["month"][:].astype(int)
        n_time = len(year)

    with h5py.File(MASK_DIR / "land_mask.nc", "r") as land_file:
        lat = land_file["lat"][:]
        lon = land_file["lon"][:]
        ocean = land_file["ocean"][:].astype(bool)
        land = land_file["land"][:].astype(bool)
        coast = land_file["coast"][:].astype(bool)

    with h5py.File(MASK_DIR / "active_mask.nc", "r") as active_file:
        active_union = active_file["active_union"][:].astype(bool)
        tos_valid = active_file["tos_valid"][:].astype(bool)
        monthly_active = {
            mm: active_file[f"active_{mm:02d}"][:].astype(bool)
            for mm in range(1, 13)
        }

    variable_data: dict[str, dict[int, np.ndarray]] = {variable: {} for variable in VARIABLES}
    for variable in VARIABLES:
        with h5py.File(monthly_file(variable), "r") as variable_file:
            dataset = variable_file[variable]
            for point_id, point in POINTS.items():
                variable_data[variable][point_id] = decode_variable_series(
                    dataset,
                    point["y_idx"],
                    point["x_idx"],
                )

    rows = []
    for point_id, point in POINTS.items():
        y_idx = point["y_idx"]
        x_idx = point["x_idx"]
        for t in range(n_time):
            row = {
                "time_index": t,
                "year": int(year[t]),
                "month": int(month[t]),
                "date": f"{int(year[t]):04d}-{int(month[t]):02d}",
                "split": split_name(int(year[t])),
                "point_id": point_id,
                "point_name": point["name"],
                "y_idx": y_idx,
                "x_idx": x_idx,
                "lat": float(lat[y_idx, x_idx]),
                "lon": float(lon[y_idx, x_idx]),
                "ocean": bool(ocean[y_idx, x_idx]),
                "land": bool(land[y_idx, x_idx]),
                "coast": bool(coast[y_idx, x_idx]),
                "active_union": bool(active_union[y_idx, x_idx]),
                "active_month": bool(monthly_active[int(month[t])][y_idx, x_idx]),
                "tos_valid": bool(tos_valid[y_idx, x_idx]),
            }
            for variable in VARIABLES:
                row[variable] = variable_data[variable][point_id][t]
            rows.append(row)

    table = pd.DataFrame(rows)
    table.to_csv(TABLE_PATH, index=False, encoding="utf-8")

    missing_counts = (
        table.groupby(["point_id", "point_name"])[VARIABLES]
        .apply(lambda frame: frame.isna().sum())
        .reset_index()
        .to_dict(orient="records")
    )
    split_counts = (
        table.groupby(["point_id", "point_name", "split"])
        .size()
        .reset_index(name="n_rows")
        .to_dict(orient="records")
    )

    metadata = {
        "source": "monthly NetCDF files under data/monthly",
        "target_variable": TARGET_VARIABLE,
        "input_variables": INPUT_VARIABLES,
        "n_points": len(POINTS),
        "n_months": int(n_time),
        "n_rows": int(len(table)),
        "split_definition": {
            "train": "1988-02 .. 2013-12",
            "val": "2014-01 .. 2017-12",
            "test": "2018-01 .. 2025-12",
        },
        "points": {
            str(point_id): {
                key: to_builtin(value)
                for key, value in point.items()
            }
            for point_id, point in POINTS.items()
        },
        "missing_counts": missing_counts,
        "split_counts": split_counts,
        "notes": [
            "Each row is one point-month sample.",
            "Point 4 has tos missing for every month in the current source data.",
            "Generated files in processed_data are ignored by Git.",
        ],
    }
    with METADATA_PATH.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"Saved table: {TABLE_PATH}")
    print(f"Saved metadata: {METADATA_PATH}")
    print(f"Rows: {len(table)}")


if __name__ == "__main__":
    main()
