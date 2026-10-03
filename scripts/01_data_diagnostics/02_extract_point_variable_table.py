from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "experiments" / "02_data_diagnostics" / "analysis_data"

RESOLUTION = "monthly"
VARIABLES = ["sic", "tos", "tas", "rlds", "rsds", "uas", "vas"]

TABLE_PATH = OUTPUT_DIR / "point_monthly_physical_values.csv"
METADATA_PATH = OUTPUT_DIR / "point_monthly_decode_metadata.json"

POINTS = {
    1: {"name": "Vilkitsky Strait", "y_idx": 259, "x_idx": 183},
    2: {"name": "NW Laptev Sea", "y_idx": 262, "x_idx": 171},
    3: {"name": "Sannikov Strait", "y_idx": 281, "x_idx": 150},
    4: {"name": "Dmitry Laptev Strait", "y_idx": 286, "x_idx": 146},
    5: {"name": "Central East Siberian Sea", "y_idx": 273, "x_idx": 126},
    6: {"name": "Long Strait / Wrangel", "y_idx": 277, "x_idx": 94},
}


def variable_file(variable: str) -> Path:
    """분석할 월별 NetCDF 파일 경로를 만든다.

    변수별로 파일이 나뉘어 있으므로, 변수명만 넣으면 해당 파일 경로가 나오게 한다.
    예: sic -> data/monthly/sic_arctic25km_monthly_1988-2025.nc
    """
    return DATA_DIR / RESOLUTION / f"{variable}_arctic25km_{RESOLUTION}_1988-2025.nc"


def attr_scalar(attrs: h5py.AttributeManager, name: str, default: float | None = None) -> float | None:
    """NetCDF attribute에서 숫자 스칼라 값을 안전하게 꺼낸다.

    h5py attribute는 값 하나도 numpy array처럼 들어올 수 있다.
    회귀용 디코딩에서는 _FillValue, scale_factor, add_offset을 숫자 하나로 써야 하므로
    여기서 Python float 또는 None으로 정리한다.
    """
    value = attrs.get(name)
    if value is None:
        return default
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list):
        if not value:
            return default
        value = value[0]
    return float(value)


def attr_text(attrs: h5py.AttributeManager, name: str) -> str | None:
    """NetCDF attribute에서 문자열 값을 읽는다.

    units 같은 attribute는 bytes로 저장되어 있을 수 있으므로 사람이 읽을 수 있는 문자열로
    바꾼다. 값이 없으면 None을 반환한다.
    """
    value = attrs.get(name)
    if value is None:
        return None
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list):
        if not value:
            return None
        value = value[0]
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def decode_point_series(dataset: h5py.Dataset, y_idx: int, x_idx: int) -> np.ndarray:
    """단일 포인트의 패킹된 시계열을 실제 물리값으로 복원한다.

    원본 NetCDF 값은 디스크 용량을 줄이기 위해 정수로 패킹되어 있다.
    분석에 쓰려면 다음 순서가 필요하다.

    1. raw 값 중 _FillValue와 같은 값은 NaN으로 바꾼다.
    2. scale_factor를 곱한다.
    3. add_offset이 있으면 더한다.

    공식:
        physical_value = raw_value * scale_factor + add_offset

    단, _FillValue는 물리값이 아니므로 scale/offset을 적용하지 않고 NaN으로 둔다.
    """
    raw = dataset[:, y_idx, x_idx].astype("float64")

    fill_value = attr_scalar(dataset.attrs, "_FillValue")
    if fill_value is not None:
        raw[raw == fill_value] = np.nan

    scale = attr_scalar(dataset.attrs, "scale_factor", 1.0)
    offset = attr_scalar(dataset.attrs, "add_offset", 0.0)

    return raw * float(scale) + float(offset)


def read_time_axis() -> pd.DataFrame:
    """sic 월별 파일에서 공통 시간축을 읽어 DataFrame으로 만든다.

    모든 월별 변수 파일은 같은 time/year/month 축을 공유한다. 따라서 하나의 대표 파일,
    여기서는 sic 파일에서 시간축을 한 번만 읽으면 된다.
    """
    path = variable_file("sic")
    with h5py.File(path, "r") as file:
        year = file["year"][:].astype(int)
        month = file["month"][:].astype(int)

    return pd.DataFrame(
        {
            "time_index": np.arange(len(year), dtype=int),
            "year": year,
            "month": month,
            "date": [f"{y:04d}-{m:02d}" for y, m in zip(year, month)],
            "split": [split_name(y) for y in year],
        }
    )


def split_name(year: int) -> str:
    """연도 기준으로 train/validation/test 구간을 붙인다.

    README의 권장 분할을 그대로 따른다.
    train: 1988-2013
    val:   2014-2017
    test:  2018-2025
    """
    if year <= 2013:
        return "train"
    if year <= 2017:
        return "val"
    return "test"


def read_point_coordinates() -> dict[int, dict[str, Any]]:
    """마스크 파일에서 각 포인트의 실제 lat/lon 및 마스크 정보를 읽는다.

    포인트는 y/x 인덱스로 정의되어 있지만, 결과 테이블에는 사람이 확인하기 쉬운 위경도와
    ocean/land/active/tos_valid 같은 마스크 정보도 같이 넣는다.
    """
    land_path = DATA_DIR / "masks" / "land_mask.nc"
    active_path = DATA_DIR / "masks" / "active_mask.nc"

    with h5py.File(land_path, "r") as land_file, h5py.File(active_path, "r") as active_file:
        lat = land_file["lat"][:]
        lon = land_file["lon"][:]
        ocean = land_file["ocean"][:].astype(bool)
        land = land_file["land"][:].astype(bool)
        coast = land_file["coast"][:].astype(bool)
        active_union = active_file["active_union"][:].astype(bool)
        tos_valid = active_file["tos_valid"][:].astype(bool)

        point_info = {}
        for point_id, point in POINTS.items():
            y_idx = point["y_idx"]
            x_idx = point["x_idx"]
            point_info[point_id] = {
                **point,
                "lat": float(lat[y_idx, x_idx]),
                "lon": float(lon[y_idx, x_idx]),
                "ocean": bool(ocean[y_idx, x_idx]),
                "land": bool(land[y_idx, x_idx]),
                "coast": bool(coast[y_idx, x_idx]),
                "active_union": bool(active_union[y_idx, x_idx]),
                "tos_valid": bool(tos_valid[y_idx, x_idx]),
            }

    return point_info


def collect_decode_metadata() -> dict[str, dict[str, Any]]:
    """변수별 디코딩 규칙을 metadata로 정리한다.

    이 값들이 바로 사용자가 말한 fill_val, scale, offset이다.
    나중에 CSV 값을 검증할 때 어떤 규칙으로 복원했는지 추적할 수 있도록 JSON으로 저장한다.
    """
    metadata = {}
    for variable in VARIABLES:
        path = variable_file(variable)
        with h5py.File(path, "r") as file:
            dataset = file[variable]
            metadata[variable] = {
                "file": str(path.relative_to(ROOT)),
                "shape": tuple(int(v) for v in dataset.shape),
                "dtype": str(dataset.dtype),
                "fill_value": attr_scalar(dataset.attrs, "_FillValue"),
                "scale_factor": attr_scalar(dataset.attrs, "scale_factor", 1.0),
                "add_offset": attr_scalar(dataset.attrs, "add_offset", 0.0),
                "units": attr_text(dataset.attrs, "units"),
            }
    return metadata


def to_jsonable(value: Any) -> Any:
    """json.dump가 처리할 수 있도록 numpy 타입을 Python 기본 타입으로 바꾼다."""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        if np.isnan(value):
            return None
        return float(value)
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    return value


def main() -> None:
    """6개 포인트의 월별 변수값을 물리값으로 복원해 테이블로 저장한다.

    이 스크립트는 아직 이상치 보정이나 결측 대체를 하지 않는다.
    하는 일은 오직 원본 NetCDF 패킹값을 물리값으로 복원하고, 6개 포인트의 시계열을
    사람이 보기 쉬운 테이블로 만드는 것이다.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    time_table = read_time_axis()
    point_info = read_point_coordinates()
    decode_metadata = collect_decode_metadata()

    variable_series: dict[str, dict[int, np.ndarray]] = {
        variable: {} for variable in VARIABLES
    }
    for variable in VARIABLES:
        path = variable_file(variable)
        with h5py.File(path, "r") as file:
            dataset = file[variable]
            for point_id, point in POINTS.items():
                variable_series[variable][point_id] = decode_point_series(
                    dataset,
                    point["y_idx"],
                    point["x_idx"],
                )

    rows = []
    for point_id, point in point_info.items():
        for _, time_row in time_table.iterrows():
            time_index = int(time_row["time_index"])
            row = {
                "time_index": time_index,
                "year": int(time_row["year"]),
                "month": int(time_row["month"]),
                "date": time_row["date"],
                "split": time_row["split"],
                "point_id": point_id,
                "point_name": point["name"],
                "y_idx": point["y_idx"],
                "x_idx": point["x_idx"],
                "lat": point["lat"],
                "lon": point["lon"],
                "ocean": point["ocean"],
                "land": point["land"],
                "coast": point["coast"],
                "active_union": point["active_union"],
                "tos_valid": point["tos_valid"],
            }
            for variable in VARIABLES:
                row[variable] = variable_series[variable][point_id][time_index]
            rows.append(row)

    table = pd.DataFrame(rows)
    table.to_csv(TABLE_PATH, index=False, encoding="utf-8")

    missing_counts = (
        table.groupby(["point_id", "point_name"])[VARIABLES]
        .apply(lambda frame: frame.isna().sum())
        .reset_index()
        .to_dict(orient="records")
    )

    metadata = {
        "purpose": "Decode packed NetCDF point time series into physical values.",
        "resolution": RESOLUTION,
        "variables": VARIABLES,
        "n_points": len(POINTS),
        "n_rows": int(len(table)),
        "decode_rules": decode_metadata,
        "points": point_info,
        "missing_counts": missing_counts,
        "notes": [
            "This table applies _FillValue, scale_factor, and add_offset only.",
            "No statistical outlier correction is applied here.",
            "No missing-value imputation is applied here.",
        ],
    }
    with open(str(METADATA_PATH), "w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2, default=to_jsonable)

    print(f"Saved table: {TABLE_PATH}")
    print(f"Saved metadata: {METADATA_PATH}")
    print(f"Rows: {len(table)}")
    print("\nMissing counts by point:")
    print(
        table.groupby(["point_id", "point_name"])[VARIABLES]
        .apply(lambda frame: frame.isna().sum())
        .to_string()
    )
    print("\nPreview:")
    print(table.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
