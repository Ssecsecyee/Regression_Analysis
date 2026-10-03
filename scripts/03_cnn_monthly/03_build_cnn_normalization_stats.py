from __future__ import annotations

import h5py
import numpy as np

from cnn_common import (
    DATASETS_DIR,
    INPUT_VARIABLES,
    decode_packed_array,
    ensure_output_dirs,
    monthly_file,
    read_time_axis,
    to_builtin,
    write_json,
)


def running_stats_update(count: int, mean: float, m2: float, values: np.ndarray) -> tuple[int, float, float]:
    """NaN이 제거된 값 배열을 이용해 Welford 방식 running mean/std를 갱신한다."""
    flat = values[np.isfinite(values)].astype("float64", copy=False)
    for value in flat:
        count += 1
        delta = value - mean
        mean += delta / count
        delta2 = value - mean
        m2 += delta * delta2
    return count, mean, m2


def compute_train_stats(variable: str, train_indices: list[int]) -> dict:
    """train 기간 valid pixel만 사용해 변수별 normalization 통계를 계산한다."""
    path = monthly_file(variable)
    count = 0
    mean = 0.0
    m2 = 0.0
    min_value = np.inf
    max_value = -np.inf
    missing_count = 0
    total_count = 0

    with h5py.File(path, "r") as file:
        dataset = file[variable]
        for time_index in train_indices:
            values = decode_packed_array(dataset, dataset[time_index, :, :])
            finite = np.isfinite(values)
            total_count += int(values.size)
            missing_count += int(values.size - finite.sum())
            if finite.any():
                valid_values = values[finite]
                min_value = min(min_value, float(valid_values.min()))
                max_value = max(max_value, float(valid_values.max()))
                count, mean, m2 = running_stats_update(count, mean, m2, values)

    variance = m2 / count if count > 0 else np.nan
    std = float(np.sqrt(variance)) if count > 0 else np.nan
    return {
        "variable": variable,
        "split": "train",
        "count": count,
        "mean": mean if count > 0 else None,
        "std": std if count > 0 else None,
        "min": min_value if count > 0 else None,
        "max": max_value if count > 0 else None,
        "total_count": total_count,
        "missing_count": missing_count,
        "missing_ratio": missing_count / total_count if total_count else None,
        "nan_fill_after_normalization": 0.0,
        "note": "Stats are computed from train split valid pixels only.",
    }


def main() -> None:
    """CNN 입력 변수 normalization 통계를 train 기간 기준으로 생성한다."""
    ensure_output_dirs()
    time_axis = read_time_axis()
    train_indices = [row["time_index"] for row in time_axis if row["split"] == "train"]

    stats = [compute_train_stats(variable, train_indices) for variable in INPUT_VARIABLES]
    payload = {
        "purpose": "Train-split normalization stats for CNN input variables.",
        "input_variables": INPUT_VARIABLES,
        "train_indices_count": len(train_indices),
        "normalization": "z_score",
        "nan_policy": "Fill NaN with 0 after normalization; preserve missing masks in metadata.",
        "stats": stats,
    }

    output_path = DATASETS_DIR / "03_cnn_normalization_stats.json"
    write_json(output_path, to_builtin(payload))
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
