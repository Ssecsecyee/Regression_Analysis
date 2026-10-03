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


def normalize_array(values: np.ndarray, mean: float, std: float) -> np.ndarray:
    """CNN 입력용 z-score normalization을 적용한다.

    NaN은 여기서 채우지 않는다. 결측 처리 전략은 normalization 이후에 적용한다.
    """
    if std == 0:
        raise ValueError("Cannot normalize with std=0.")
    return (values.astype("float64", copy=False) - float(mean)) / float(std)


def fill_nan_after_normalization(values: np.ndarray, fill_value: float = 0.0) -> np.ndarray:
    """정규화된 CNN 입력에서 NaN을 지정값으로 채운다.

    fill_value=0은 z-score 공간에서 train 평균값을 의미한다.
    raw 물리값 0과는 다른 의미다.
    """
    return np.where(np.isfinite(values), values, fill_value)


def fill_nan_with_3x3_stat(
    values: np.ndarray,
    valid_domain: np.ndarray | None = None,
    min_valid_count: int = 3,
    method: str = "median",
) -> np.ndarray:
    """2-D 배열의 NaN을 주변 3x3 valid 통계값으로 1회 보간한다.

    이 함수는 tos 결측 보간 후보 실험용이다.
    - finite 값만 이웃 후보로 사용한다.
    - valid_domain이 주어지면 해당 영역 안의 값만 이웃 후보로 사용한다.
    - 주변 valid 후보 수가 min_valid_count보다 적으면 NaN으로 남긴다.
    - method는 "median" 또는 "mean"을 지원한다.

    반환값은 원본 배열을 수정하지 않는 새 배열이다.
    """
    if values.ndim != 2:
        raise ValueError("fill_nan_with_3x3_stat expects a 2-D array.")
    if method not in {"median", "mean"}:
        raise ValueError("method must be 'median' or 'mean'.")

    source = values.astype("float64", copy=True)
    if valid_domain is None:
        domain = np.ones(source.shape, dtype=bool)
    else:
        domain = valid_domain.astype(bool)
        if domain.shape != source.shape:
            raise ValueError("valid_domain shape must match values shape.")

    finite = np.isfinite(source) & domain
    filled = source.copy()

    # 448x304 월별 격자에서 3x3 창만 다루므로 명시적 루프가 충분히 단순하고 안전하다.
    candidate_y, candidate_x = np.where(np.isnan(filled) & domain)
    for y_idx, x_idx in zip(candidate_y, candidate_x):
        y0 = max(0, y_idx - 1)
        y1 = min(source.shape[0], y_idx + 2)
        x0 = max(0, x_idx - 1)
        x1 = min(source.shape[1], x_idx + 2)

        window = source[y0:y1, x0:x1]
        window_domain = domain[y0:y1, x0:x1]
        candidates = window[np.isfinite(window) & window_domain]
        if candidates.size < min_valid_count:
            continue
        if method == "median":
            filled[y_idx, x_idx] = float(np.median(candidates))
        else:
            filled[y_idx, x_idx] = float(np.mean(candidates))

    return filled


def make_missing_mask(values: np.ndarray) -> np.ndarray:
    """입력 변수의 결측 위치를 0/1 mask로 반환한다."""
    return np.isnan(values).astype("float32")


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
