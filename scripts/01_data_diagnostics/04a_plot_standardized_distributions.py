from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
TABLE_PATH = ROOT / "experiments" / "02_data_diagnostics" / "analysis_data" / "point_monthly_physical_values.csv"
FIGURE_DIR = ROOT / "experiments" / "02_data_diagnostics" / "figures"

VARIABLES = ["sic", "tos", "tas", "rlds", "rsds", "uas", "vas"]
VARIABLE_LABELS = {
    "sic": "Sea ice concentration (SIC)",
    "tos": "Sea surface temperature",
    "tas": "Near-surface air temperature",
    "rlds": "Downwelling longwave radiation",
    "rsds": "Downwelling shortwave radiation",
    "uas": "Eastward wind",
    "vas": "Northward wind",
}


def load_point_table() -> pd.DataFrame:
    """02번에서 만든 포인트별 월별 물리값 테이블을 읽는다."""
    if not TABLE_PATH.exists():
        raise FileNotFoundError(f"Missing input table: {TABLE_PATH}")
    return pd.read_csv(TABLE_PATH)


def standardize(values: pd.Series) -> pd.Series:
    """변수 값을 평균 0, 표준편차 1로 표준화한다."""
    values = values.dropna()
    std = values.std(ddof=0)
    if values.empty or pd.isna(std) or std == 0:
        return pd.Series(dtype=float)
    return (values - values.mean()) / std


def normal_pdf(x: np.ndarray) -> np.ndarray:
    """표준정규분포 확률밀도값을 계산한다."""
    return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)


def main() -> None:
    """변수별 표준화 히스토그램과 정규분포 기준선을 비교한다."""
    table = load_point_table()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(4, 2, figsize=(15, 18))
    axes = axes.ravel()
    x_grid = np.linspace(-4, 4, 300)

    for ax, variable in zip(axes, VARIABLES):
        values = standardize(table[variable])
        if values.empty:
            ax.text(0.5, 0.5, "No valid values", ha="center", va="center")
            ax.set_axis_off()
            continue

        ax.hist(values, bins=35, density=True, color="#60a5fa", alpha=0.75)
        ax.plot(x_grid, normal_pdf(x_grid), color="#111827", linewidth=1.5)
        ax.set_title(f"{VARIABLE_LABELS[variable]} (standardized)", fontsize=10, pad=10)
        ax.set_xlabel("z-score")
        ax.set_ylabel("density")
        ax.grid(True, alpha=0.25)

    axes[-1].axis("off")
    fig.suptitle("Standardized variable distributions vs normal curve", fontsize=14, y=0.995)
    fig.subplots_adjust(hspace=0.55, wspace=0.28, top=0.95)

    output_path = FIGURE_DIR / "04a_standardized_distribution_histograms.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {output_path}")


if __name__ == "__main__":
    main()
