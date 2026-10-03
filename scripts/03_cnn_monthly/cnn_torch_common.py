from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from cnn_common import (
    DATASETS_DIR,
    INPUT_VARIABLES,
    MASK_DIR,
    TARGET_VARIABLE,
    decode_packed_array,
    fill_nan_after_normalization,
    fill_nan_with_3x3_stat,
    make_missing_mask,
    monthly_file,
    normalize_array,
)


NORMALIZATION_PATH = DATASETS_DIR / "03_cnn_normalization_stats.json"
INDEX_PATH = DATASETS_DIR / "05_cnn_dataset_index_lead0.csv"


def load_json(path: Path) -> Any:
    """JSON 파일을 읽어 Python 객체로 반환한다."""
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_normalization_stats(path: Path = NORMALIZATION_PATH) -> dict[str, dict[str, float]]:
    """03번 스크립트가 만든 변수별 train normalization 통계를 읽는다."""
    payload = load_json(path)
    stats = {}
    for row in payload["stats"]:
        stats[row["variable"]] = {
            "mean": float(row["mean"]),
            "std": float(row["std"]),
        }
    return stats


def load_index_rows(path: Path = INDEX_PATH, split: str | None = None) -> list[dict[str, Any]]:
    """05번 스크립트가 만든 sample index CSV를 읽는다."""
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


def load_ocean_mask() -> np.ndarray:
    """land_mask.nc에서 ocean mask를 boolean 배열로 읽는다."""
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


def masked_regression_metrics(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """mask 영역에서 RMSE, MAE, Bias, R2를 계산한다."""
    valid = mask.astype(bool) & np.isfinite(prediction) & np.isfinite(target)
    count = int(valid.sum())
    if count == 0:
        return {"count": 0, "rmse": np.nan, "mae": np.nan, "bias": np.nan, "r2": np.nan}

    pred = prediction[valid].astype("float64")
    true = target[valid].astype("float64")
    error = pred - true
    rmse = float(np.sqrt(np.mean(error**2)))
    mae = float(np.mean(np.abs(error)))
    bias = float(np.mean(error))
    denom = float(np.sum((true - np.mean(true)) ** 2))
    r2 = float(1.0 - np.sum(error**2) / denom) if denom > 0 else np.nan
    return {"count": count, "rmse": rmse, "mae": mae, "bias": bias, "r2": r2}


class MonthlyGridDataset(Dataset):
    """월별 전체 격자를 CNN 입력/타깃 tensor로 제공한다."""

    def __init__(
        self,
        rows: list[dict[str, Any]],
        stats: dict[str, dict[str, float]],
        ocean_mask: np.ndarray,
        tos_strategy: str = "zero_fill",
        add_tos_missing_channel: bool = True,
    ) -> None:
        """Dataset 설정값과 sample index를 저장한다."""
        if tos_strategy not in {"zero_fill", "median3x3_min3"}:
            raise ValueError("tos_strategy must be 'zero_fill' or 'median3x3_min3'.")
        self.rows = rows
        self.stats = stats
        self.ocean_mask = ocean_mask
        self.tos_strategy = tos_strategy
        self.add_tos_missing_channel = add_tos_missing_channel

    def __len__(self) -> int:
        """sample 개수를 반환한다."""
        return len(self.rows)

    def _read_variable(self, variable: str, time_index: int) -> np.ndarray:
        """한 변수의 특정 월 2-D 격자를 physical value로 복원한다."""
        with h5py.File(monthly_file(variable), "r") as file:
            dataset = file[variable]
            return decode_packed_array(dataset, dataset[time_index, :, :])

    def _prepare_input_variable(self, variable: str, values: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        """입력 변수 하나를 정규화하고 필요한 경우 missing mask를 함께 만든다."""
        missing_mask = None
        prepared = values

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
        """한 sample의 입력 tensor, 타깃 tensor, loss mask, metadata를 반환한다."""
        row = self.rows[index]
        channels = []
        tos_missing = None

        for variable in INPUT_VARIABLES:
            values = self._read_variable(variable, row["input_time_index"])
            channel, missing_mask = self._prepare_input_variable(variable, values)
            channels.append(channel)
            if variable == "tos":
                tos_missing = missing_mask

        if self.add_tos_missing_channel:
            channels.append(tos_missing.astype("float32"))

        target = self._read_variable(TARGET_VARIABLE, row["target_time_index"]).astype("float32")
        loss_mask = (self.ocean_mask & np.isfinite(target)).astype("float32")
        target_filled = np.where(np.isfinite(target), target, 0.0).astype("float32")

        return {
            "x": torch.from_numpy(np.stack(channels, axis=0)),
            "y": torch.from_numpy(target_filled[None, :, :]),
            "mask": torch.from_numpy(loss_mask[None, :, :]),
            "meta": row,
        }


class SimpleSICCNN(nn.Module):
    """전체 격자 SIC 예측을 위한 작은 baseline CNN이다."""

    def __init__(self, in_channels: int, hidden_channels: int = 48) -> None:
        """채널 수를 받아 convolution block을 구성한다."""
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """입력 격자를 SIC 예측 격자로 변환한다."""
        return self.net(x)


def masked_mse_loss(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """mask 영역에 대해서만 MSE loss를 계산한다."""
    squared_error = (prediction - target) ** 2
    masked_error = squared_error * mask
    denominator = mask.sum().clamp_min(1.0)
    return masked_error.sum() / denominator


def build_model(model_name: str, in_channels: int) -> nn.Module:
    """문자열 이름에 맞는 CNN 모델을 생성한다."""
    if model_name == "simple_cnn":
        return SimpleSICCNN(in_channels=in_channels)
    raise ValueError(f"Unsupported model_name: {model_name}")


def collate_grid_batch(batch: list[dict]) -> dict:
    """Dataset sample 목록을 학습/평가 가능한 batch tensor로 묶는다."""
    return {
        "x": torch.stack([item["x"] for item in batch], dim=0),
        "y": torch.stack([item["y"] for item in batch], dim=0),
        "mask": torch.stack([item["mask"] for item in batch], dim=0),
        "meta": [item["meta"] for item in batch],
    }
