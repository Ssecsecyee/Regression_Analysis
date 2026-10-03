from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset


THIS_DIR = Path(__file__).resolve().parent
CNN_MONTHLY_DIR = THIS_DIR.parent / "03_cnn_monthly"
if str(CNN_MONTHLY_DIR) not in sys.path:
    sys.path.insert(0, str(CNN_MONTHLY_DIR))

from cnn_common import (  # noqa: E402
    INPUT_VARIABLES,
    MASK_DIR,
    ROOT,
    TARGET_VARIABLE,
    decode_packed_array,
    fill_nan_after_normalization,
    fill_nan_with_3x3_stat,
    make_missing_mask,
    monthly_file,
    normalize_array,
    read_time_axis,
    split_name,
    to_builtin,
    write_json,
)
from cnn_torch_common import SimpleSICCNN, masked_mse_loss, masked_regression_metrics  # noqa: E402


EXPERIMENT_DIR = ROOT / "experiments" / "04_cnn_forecasting_lead1"
DATASETS_DIR = EXPERIMENT_DIR / "datasets"
MODELS_DIR = EXPERIMENT_DIR / "models"
RESULTS_DIR = EXPERIMENT_DIR / "results"
FIGURES_DIR = EXPERIMENT_DIR / "figures"
REPORTS_DIR = EXPERIMENT_DIR / "reports"
METADATA_DIR = EXPERIMENT_DIR / "metadata"

NOWCASTING_STATS_PATH = ROOT / "experiments" / "03_cnn_monthly" / "datasets" / "03_cnn_normalization_stats.json"
LEAD1_INDEX_PATH = DATASETS_DIR / "01_lead1_dataset_index.csv"


def ensure_output_dirs() -> None:
    """lead-1 forecasting 실험 산출물 폴더를 생성한다."""
    for path in [DATASETS_DIR, MODELS_DIR, RESULTS_DIR, FIGURES_DIR, REPORTS_DIR, METADATA_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Any:
    """JSON 파일을 읽는다."""
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_normalization_stats(path: Path = NOWCASTING_STATS_PATH) -> dict[str, dict[str, float]]:
    """03번 nowcasting 실험에서 만든 train normalization 통계를 재사용한다."""
    payload = load_json(path)
    stats = {}
    for row in payload["stats"]:
        stats[row["variable"]] = {"mean": float(row["mean"]), "std": float(row["std"])}
    return stats


def load_ocean_mask() -> np.ndarray:
    """land_mask.nc에서 ocean mask를 읽는다."""
    with h5py.File(MASK_DIR / "land_mask.nc", "r") as file:
        return file["ocean"][:].astype(bool)


def load_active_union_mask() -> np.ndarray | None:
    """active_mask.nc에서 active_union mask를 읽는다."""
    path = MASK_DIR / "active_mask.nc"
    if not path.exists():
        return None
    with h5py.File(path, "r") as file:
        if "active_union" not in file:
            return None
        return file["active_union"][:].astype(bool)


def load_monthly_active_mask(month: int) -> np.ndarray | None:
    """active_mask.nc에서 특정 월 active mask를 읽는다."""
    path = MASK_DIR / "active_mask.nc"
    if not path.exists():
        return None
    candidates = [f"active_{month:02d}", f"active_{month}"]
    with h5py.File(path, "r") as file:
        for name in candidates:
            if name in file:
                return file[name][:].astype(bool)
    return None


def build_lead1_rows() -> list[dict[str, Any]]:
    """SIC(t)+X(t) -> SIC(t+1) sample index row를 만든다."""
    time_axis = read_time_axis()
    rows = []
    for input_index in range(len(time_axis) - 1):
        target_index = input_index + 1
        input_row = time_axis[input_index]
        target_row = time_axis[target_index]
        rows.append(
            {
                "sample_id": len(rows),
                "task_type": "forecasting",
                "lead_months": 1,
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


def load_index_rows(path: Path = LEAD1_INDEX_PATH, split: str | None = None) -> list[dict[str, Any]]:
    """lead-1 sample index CSV를 읽는다."""
    with open(path, newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    parsed = []
    for row in rows:
        item = dict(row)
        for key in [
            "sample_id",
            "lead_months",
            "input_time_index",
            "input_year",
            "input_month",
            "target_time_index",
            "target_year",
            "target_month",
        ]:
            item[key] = int(item[key])
        if split is None or item["split"] == split:
            parsed.append(item)
    return parsed


class Lead1ForecastDataset(Dataset):
    """SIC(t)와 X(t)를 입력으로 SIC(t+1)을 반환하는 Dataset이다."""

    def __init__(
        self,
        rows: list[dict[str, Any]],
        stats: dict[str, dict[str, float]],
        ocean_mask: np.ndarray,
        tos_strategy: str = "zero_fill",
    ) -> None:
        """sample index와 전처리 설정을 저장한다."""
        if tos_strategy not in {"zero_fill", "median3x3_min3"}:
            raise ValueError("tos_strategy must be 'zero_fill' or 'median3x3_min3'.")
        self.rows = rows
        self.stats = stats
        self.ocean_mask = ocean_mask
        self.tos_strategy = tos_strategy

    def __len__(self) -> int:
        """sample 개수를 반환한다."""
        return len(self.rows)

    def _read_variable(self, variable: str, time_index: int) -> np.ndarray:
        """한 변수의 특정 월 2-D 격자를 physical value로 복원한다."""
        with h5py.File(monthly_file(variable), "r") as file:
            dataset = file[variable]
            return decode_packed_array(dataset, dataset[time_index, :, :])

    def _prepare_climate_channel(self, variable: str, values: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        """기후 변수 채널을 정규화하고 tos 결측 mask를 선택적으로 반환한다."""
        prepared = values
        missing_mask = None
        if variable == "tos":
            missing_mask = make_missing_mask(values)
            if self.tos_strategy == "median3x3_min3":
                prepared = fill_nan_with_3x3_stat(
                    values,
                    valid_domain=self.ocean_mask,
                    min_valid_count=3,
                    method="median",
                )

        stat = self.stats[variable]
        normalized = normalize_array(prepared, mean=stat["mean"], std=stat["std"])
        filled = fill_nan_after_normalization(normalized, fill_value=0.0)
        return filled.astype("float32"), missing_mask

    def __getitem__(self, index: int) -> dict[str, Any]:
        """입력 tensor, target tensor, mask, persistence baseline을 반환한다."""
        row = self.rows[index]
        input_sic = self._read_variable(TARGET_VARIABLE, row["input_time_index"]).astype("float32")
        target_sic = self._read_variable(TARGET_VARIABLE, row["target_time_index"]).astype("float32")

        channels = [np.where(np.isfinite(input_sic), input_sic, 0.0).astype("float32")]
        tos_missing = None
        for variable in INPUT_VARIABLES:
            values = self._read_variable(variable, row["input_time_index"])
            channel, missing_mask = self._prepare_climate_channel(variable, values)
            channels.append(channel)
            if variable == "tos":
                tos_missing = missing_mask
        channels.append(tos_missing.astype("float32"))

        loss_mask = (self.ocean_mask & np.isfinite(target_sic)).astype("float32")
        target_filled = np.where(np.isfinite(target_sic), target_sic, 0.0).astype("float32")
        persistence = np.where(np.isfinite(input_sic), input_sic, 0.0).astype("float32")

        return {
            "x": torch.from_numpy(np.stack(channels, axis=0)),
            "y": torch.from_numpy(target_filled[None, :, :]),
            "mask": torch.from_numpy(loss_mask[None, :, :]),
            "persistence": torch.from_numpy(persistence[None, :, :]),
            "meta": row,
        }


def collate_grid_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Dataset sample 목록을 batch tensor로 묶는다."""
    return {
        "x": torch.stack([item["x"] for item in batch], dim=0),
        "y": torch.stack([item["y"] for item in batch], dim=0),
        "mask": torch.stack([item["mask"] for item in batch], dim=0),
        "persistence": torch.stack([item["persistence"] for item in batch], dim=0),
        "meta": [item["meta"] for item in batch],
    }


def build_model(model_name: str, in_channels: int = 8) -> nn.Module:
    """lead-1 forecasting 모델을 생성한다."""
    if model_name == "simple_cnn":
        return SimpleSICCNN(in_channels=in_channels)
    raise ValueError(f"Unsupported model_name: {model_name}")


def forecast_metrics(prediction: np.ndarray, target: np.ndarray, persistence: np.ndarray, mask: np.ndarray) -> dict:
    """CNN과 persistence baseline metric을 함께 계산한다."""
    cnn = masked_regression_metrics(prediction, target, mask)
    base = masked_regression_metrics(persistence, target, mask)
    return {
        "cnn": cnn,
        "persistence": base,
        "rmse_improvement": base["rmse"] - cnn["rmse"],
        "mae_improvement": base["mae"] - cnn["mae"],
    }


__all__ = [
    "DATASETS_DIR",
    "EXPERIMENT_DIR",
    "FIGURES_DIR",
    "INPUT_VARIABLES",
    "LEAD1_INDEX_PATH",
    "METADATA_DIR",
    "MODELS_DIR",
    "REPORTS_DIR",
    "RESULTS_DIR",
    "TARGET_VARIABLE",
    "Lead1ForecastDataset",
    "build_lead1_rows",
    "build_model",
    "collate_grid_batch",
    "ensure_output_dirs",
    "forecast_metrics",
    "load_active_union_mask",
    "load_index_rows",
    "load_monthly_active_mask",
    "load_normalization_stats",
    "load_ocean_mask",
    "masked_mse_loss",
    "to_builtin",
    "write_json",
]
