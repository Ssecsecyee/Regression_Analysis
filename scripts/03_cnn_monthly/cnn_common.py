from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
MONTHLY_DIR = DATA_DIR / "monthly"
MASK_DIR = DATA_DIR / "masks"
EXPERIMENT_DIR = ROOT / "experiments" / "03_cnn_monthly"

PREPROCESSING_DIR = EXPERIMENT_DIR / "preprocessing"
DATASETS_DIR = EXPERIMENT_DIR / "datasets"
MODELS_DIR = EXPERIMENT_DIR / "models"
FIGURES_DIR = EXPERIMENT_DIR / "figures"
RESULTS_DIR = EXPERIMENT_DIR / "results"
REPORTS_DIR = EXPERIMENT_DIR / "reports"
METADATA_DIR = EXPERIMENT_DIR / "metadata"

VARIABLES = ["sic", "tos", "tas", "rlds", "rsds", "uas", "vas"]
INPUT_VARIABLES = ["tos", "tas", "rlds", "rsds", "uas", "vas"]
TARGET_VARIABLE = "sic"

SPLITS = {
    "train": (1988, 2013),
    "val": (2014, 2017),
    "test": (2018, 2025),
}


def monthly_file(variable: str) -> Path:
    """변수명에 해당하는 월별 NetCDF 파일 경로를 반환한다."""
    return MONTHLY_DIR / f"{variable}_arctic25km_monthly_1988-2025.nc"


def ensure_output_dirs() -> None:
    """CNN 실험 산출물 폴더를 생성한다."""
    for path in [
        PREPROCESSING_DIR,
        DATASETS_DIR,
        MODELS_DIR,
        FIGURES_DIR,
        RESULTS_DIR,
        REPORTS_DIR,
        METADATA_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def attr_scalar(attrs: h5py.AttributeManager, name: str, default: float | None = None) -> float | None:
    """h5py attribute에서 숫자 스칼라 값을 안전하게 읽는다."""
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
    """h5py attribute에서 문자열 값을 읽는다."""
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


def split_name(year: int) -> str:
    """연도를 train/val/test split 이름으로 변환한다."""
    if SPLITS["train"][0] <= year <= SPLITS["train"][1]:
        return "train"
    if SPLITS["val"][0] <= year <= SPLITS["val"][1]:
        return "val"
    if SPLITS["test"][0] <= year <= SPLITS["test"][1]:
        return "test"
    return "unknown"


def read_time_axis() -> list[dict[str, Any]]:
    """대표 sic 파일에서 월별 시간축을 읽는다."""
    path = monthly_file(TARGET_VARIABLE)
    with h5py.File(path, "r") as file:
        year = file["year"][:].astype(int)
        month = file["month"][:].astype(int)

    rows = []
    for index, (yyyy, mm) in enumerate(zip(year, month)):
        rows.append(
            {
                "time_index": int(index),
                "year": int(yyyy),
                "month": int(mm),
                "date": f"{int(yyyy):04d}-{int(mm):02d}",
                "split": split_name(int(yyyy)),
            }
        )
    return rows


def decode_packed_array(dataset: h5py.Dataset, raw: np.ndarray) -> np.ndarray:
    """NetCDF packed 값을 실제 물리값으로 복원한다.

    _FillValue는 물리값이 아니므로 scale/offset 적용 전에 NaN으로 바꾼다.
    """
    array = raw.astype("float64", copy=False)
    fill_value = attr_scalar(dataset.attrs, "_FillValue")
    if fill_value is not None:
        array = np.where(array == fill_value, np.nan, array)

    scale = attr_scalar(dataset.attrs, "scale_factor", 1.0)
    offset = attr_scalar(dataset.attrs, "add_offset", 0.0)
    return array * float(scale) + float(offset)


def dataset_metadata(dataset: h5py.Dataset, file_path: Path) -> dict[str, Any]:
    """NetCDF 변수 dataset의 기본 metadata를 dict로 정리한다."""
    return {
        "file": str(file_path.relative_to(ROOT)),
        "shape": [int(v) for v in dataset.shape],
        "dtype": str(dataset.dtype),
        "fill_value": attr_scalar(dataset.attrs, "_FillValue"),
        "scale_factor": attr_scalar(dataset.attrs, "scale_factor", 1.0),
        "add_offset": attr_scalar(dataset.attrs, "add_offset", 0.0),
        "units": attr_text(dataset.attrs, "units"),
    }


def write_json(path: Path, payload: Any) -> None:
    """JSON 산출물을 UTF-8로 저장한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def to_builtin(value: Any) -> Any:
    """numpy 타입을 JSON 직렬화 가능한 Python 기본 타입으로 변환한다."""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        if np.isnan(value):
            return None
        return float(value)
    if isinstance(value, np.ndarray):
        return [to_builtin(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {key: to_builtin(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_builtin(item) for item in value]
    return value
