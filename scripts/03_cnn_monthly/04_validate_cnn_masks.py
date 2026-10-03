from __future__ import annotations

import h5py
import numpy as np

from cnn_common import MASK_DIR, PREPROCESSING_DIR, ensure_output_dirs, to_builtin, write_json


MASK_FILES = {
    "land_mask": MASK_DIR / "land_mask.nc",
    "active_mask": MASK_DIR / "active_mask.nc",
}


def summarize_mask(array: np.ndarray) -> dict:
    """boolean mask의 픽셀 수와 비율을 요약한다."""
    bool_array = array.astype(bool)
    total = int(bool_array.size)
    true_count = int(bool_array.sum())
    return {
        "shape": [int(v) for v in bool_array.shape],
        "total_count": total,
        "true_count": true_count,
        "false_count": total - true_count,
        "true_ratio": true_count / total if total else None,
    }


def inspect_land_mask() -> dict:
    """land_mask.nc의 주요 mask와 좌표 배열을 검증한다."""
    path = MASK_FILES["land_mask"]
    record = {"file": str(path), "exists": path.exists(), "datasets": {}}
    if not path.exists():
        return record

    with h5py.File(path, "r") as file:
        for name in ["land", "ocean", "coast"]:
            if name in file:
                record["datasets"][name] = summarize_mask(file[name][:])
        for name in ["lat", "lon", "x", "y"]:
            if name in file:
                data = file[name][:]
                record["datasets"][name] = {
                    "shape": [int(v) for v in data.shape],
                    "min": float(np.nanmin(data)),
                    "max": float(np.nanmax(data)),
                }
    return record


def inspect_active_mask() -> dict:
    """active_mask.nc의 monthly/union mask를 검증한다."""
    path = MASK_FILES["active_mask"]
    record = {"file": str(path), "exists": path.exists(), "datasets": {}}
    if not path.exists():
        return record

    with h5py.File(path, "r") as file:
        for name in file.keys():
            if name.startswith("active_") or name in ["active_union", "tos_valid"]:
                data = file[name][:]
                if data.ndim == 2:
                    record["datasets"][name] = summarize_mask(data)
        for name in ["x", "y"]:
            if name in file:
                data = file[name][:]
                record["datasets"][name] = {
                    "shape": [int(v) for v in data.shape],
                    "min": float(np.nanmin(data)),
                    "max": float(np.nanmax(data)),
                }
    return record


def main() -> None:
    """CNN 학습/평가에 사용할 mask 구조를 검증한다."""
    ensure_output_dirs()
    payload = {
        "purpose": "Validate masks for CNN loss and evaluation domains.",
        "recommended_training_loss_domain": "ocean mask with target missing pixels excluded.",
        "recommended_evaluation_domains": [
            "ocean",
            "active_union",
            "monthly_active_mask",
        ],
        "land_mask": inspect_land_mask(),
        "active_mask": inspect_active_mask(),
    }

    output_path = PREPROCESSING_DIR / "04_cnn_mask_validation.json"
    write_json(output_path, to_builtin(payload))
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
