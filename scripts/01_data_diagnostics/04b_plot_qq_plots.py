from __future__ import annotations

from pathlib import Path
from statistics import NormalDist

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
    """Q-Q plot 비교를 위해 변수 값을 표준화한다."""
    values = values.dropna()
    std = values.std(ddof=0)
    if values.empty or pd.isna(std) or std == 0:
        return pd.Series(dtype=float)
    return (values - values.mean()) / std


def normal_quantiles(n: int) -> np.ndarray:
    """표준정규분포의 이론 분위수를 계산한다."""
    normal = NormalDist()
    probs = (np.arange(1, n + 1) - 0.5) / n
    return np.array([normal.inv_cdf(float(prob)) for prob in probs])


def main() -> None:
    """변수별 Q-Q plot으로 정규분포와의 차이를 확인한다."""
    table = load_point_table()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(4, 2, figsize=(15, 18))
    axes = axes.ravel()

    for ax, variable in zip(axes, VARIABLES):
        values = np.sort(standardize(table[variable]).to_numpy())
        if len(values) == 0:
            ax.text(0.5, 0.5, "No valid values", ha="center", va="center")
            ax.set_axis_off()
            continue

        theoretical = normal_quantiles(len(values))
        ax.scatter(theoretical, values, s=8, alpha=0.45, color="#2563eb")
        lim_min = float(min(theoretical.min(), values.min()))
        lim_max = float(max(theoretical.max(), values.max()))
        ax.plot([lim_min, lim_max], [lim_min, lim_max], color="#111827", linewidth=1.0)
        ax.set_title(f"{VARIABLE_LABELS[variable]} Q-Q plot", fontsize=10, pad=10)
        ax.set_xlabel("Normal theoretical quantile")
        ax.set_ylabel("Observed quantile")
        ax.grid(True, alpha=0.25)

    axes[-1].axis("off")
    fig.suptitle("Q-Q plots for normality check", fontsize=14, y=0.995)
    fig.subplots_adjust(hspace=0.55, wspace=0.28, top=0.95)

    output_path = FIGURE_DIR / "04b_qq_plots_by_variable.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {output_path}")


if __name__ == "__main__":
    main()
