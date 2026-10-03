from __future__ import annotations

import json

import h5py
import numpy as np

from cnn_common import (
    DATASETS_DIR,
    MASK_DIR,
    PREPROCESSING_DIR,
    decode_packed_array,
    ensure_output_dirs,
    fill_nan_after_normalization,
    fill_nan_with_3x3_stat,
    make_missing_mask,
    monthly_file,
    normalize_array,
    read_time_axis,
    to_builtin,
    write_json,
)


NORMALIZATION_PATH = DATASETS_DIR / "03_cnn_normalization_stats.json"


def load_tos_stats() -> dict:
    """03번 스크립트가 만든 normalization 통계에서 tos 통계를 읽는다."""
    if not NORMALIZATION_PATH.exists():
        raise FileNotFoundError(
            f"Missing normalization stats: {NORMALIZATION_PATH}\n"
            "Run 03_build_cnn_normalization_stats.py first."
        )

    with open(NORMALIZATION_PATH, "r", encoding="utf-8") as file:
        payload = json.load(file)

    for stat in payload["stats"]:
        if stat["variable"] == "tos":
            return stat
    raise KeyError("tos stats were not found in normalization stats.")


def load_ocean_mask() -> np.ndarray:
    """land_mask.nc에서 ocean mask를 읽는다."""
    path = MASK_DIR / "land_mask.nc"
    with h5py.File(path, "r") as file:
        return file["ocean"][:].astype(bool)


MIN_VALID_NEIGHBORS = 3
FILL_METHOD = "median"


def summarize_strategy(raw_tos: np.ndarray, ocean_mask: np.ndarray, mean: float, std: float) -> dict:
    """한 시점의 tos에 대해 두 결측 처리 후보를 비교 요약한다."""
    missing_mask = make_missing_mask(raw_tos)
    normalized = normalize_array(raw_tos, mean=mean, std=std)

    zero_filled = fill_nan_after_normalization(normalized, fill_value=0.0)
    neighbor_physical = fill_nan_with_3x3_stat(
        raw_tos,
        valid_domain=ocean_mask,
        min_valid_count=MIN_VALID_NEIGHBORS,
        method=FILL_METHOD,
    )
    neighbor_normalized = normalize_array(neighbor_physical, mean=mean, std=std)
    neighbor_filled = fill_nan_after_normalization(neighbor_normalized, fill_value=0.0)

    original_missing = np.isnan(raw_tos)
    neighbor_remaining_missing = np.isnan(neighbor_physical)

    ocean_total = int(ocean_mask.sum())
    ocean_missing = int((original_missing & ocean_mask).sum())
    neighbor_filled_count = int((original_missing & ocean_mask & ~neighbor_remaining_missing).sum())
    neighbor_remaining_count = int((neighbor_remaining_missing & ocean_mask).sum())

    return {
        "ocean_total": ocean_total,
        "ocean_missing": ocean_missing,
        "ocean_missing_ratio": ocean_missing / ocean_total if ocean_total else None,
        "zero_fill_value_after_normalization": 0.0,
        "neighbor_fill_method": FILL_METHOD,
        "neighbor_min_valid_count": MIN_VALID_NEIGHBORS,
        "missing_mask_sum": float(missing_mask.sum()),
        "neighbor_filled_count": neighbor_filled_count,
        "neighbor_remaining_missing_count": neighbor_remaining_count,
        "neighbor_fill_success_ratio_among_ocean_missing": (
            neighbor_filled_count / ocean_missing if ocean_missing else None
        ),
        "zero_filled_min": float(np.nanmin(zero_filled)),
        "zero_filled_max": float(np.nanmax(zero_filled)),
        "neighbor_filled_min": float(np.nanmin(neighbor_filled)),
        "neighbor_filled_max": float(np.nanmax(neighbor_filled)),
    }


def main() -> None:
    """tos 결측 처리 후보를 검증한다.

    이 스크립트는 학습용 tensor를 만들지 않는다.
    1차 전략인 normalization 후 0 채움과 2차 전략인 3x3 valid ocean 보간이
    얼마나 다른지 월별 요약으로 확인하기 위한 검증 스크립트다.
    """
    ensure_output_dirs()
    time_axis = read_time_axis()
    stats = load_tos_stats()
    ocean_mask = load_ocean_mask()

    mean = float(stats["mean"])
    std = float(stats["std"])

    monthly_rows = []
    with h5py.File(monthly_file("tos"), "r") as file:
        dataset = file["tos"]
        for time_row in time_axis:
            time_index = time_row["time_index"]
            raw_tos = decode_packed_array(dataset, dataset[time_index, :, :])
            row = {
                "time_index": time_index,
                "date": time_row["date"],
                "year": time_row["year"],
                "month": time_row["month"],
                "split": time_row["split"],
                **summarize_strategy(raw_tos, ocean_mask, mean=mean, std=std),
            }
            monthly_rows.append(row)

    split_summary: dict[str, dict[str, float]] = {}
    for split in ["train", "val", "test"]:
        rows = [row for row in monthly_rows if row["split"] == split]
        if not rows:
            continue
        split_summary[split] = {
            "mean_ocean_missing_ratio": float(np.mean([row["ocean_missing_ratio"] for row in rows])),
            "mean_neighbor_fill_success_ratio": float(
                np.mean([row["neighbor_fill_success_ratio_among_ocean_missing"] for row in rows])
            ),
            "mean_neighbor_remaining_missing_count": float(
                np.mean([row["neighbor_remaining_missing_count"] for row in rows])
            ),
        }

    payload = {
        "purpose": "Validate tos missing-value handling strategies before CNN training.",
        "strategies": {
            "baseline": "Normalize tos with train stats, then fill NaN with 0 and keep missing mask.",
            "candidate": (
                "Fill physical tos NaN using 3x3 valid ocean median only when at least "
                f"{MIN_VALID_NEIGHBORS} valid neighbors exist; normalize; then fill remaining NaN with 0."
            ),
        },
        "neighbor_fill_method": FILL_METHOD,
        "neighbor_min_valid_count": MIN_VALID_NEIGHBORS,
        "normalization_stats": stats,
        "split_summary": split_summary,
        "monthly_rows": monthly_rows,
        "recommendation": [
            "Do not replace raw tos NaN with raw 0 K.",
            "Keep tos_missing or tos_valid information even when interpolation is applied.",
            "Compare baseline zero-fill and 3x3-fill models on the same test split.",
        ],
    }

    output_path = PREPROCESSING_DIR / "06_tos_fill_strategy_validation.json"
    write_json(output_path, to_builtin(payload))
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
