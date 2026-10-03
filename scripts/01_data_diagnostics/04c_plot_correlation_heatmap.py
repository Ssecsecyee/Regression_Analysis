from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
TABLE_PATH = ROOT / "experiments" / "02_data_diagnostics" / "analysis_data" / "point_monthly_physical_values.csv"
FIGURE_DIR = ROOT / "experiments" / "02_data_diagnostics" / "figures"

VARIABLES = ["sic", "tos", "tas", "rlds", "rsds", "uas", "vas"]


def load_point_table() -> pd.DataFrame:
    """02번에서 만든 포인트별 월별 물리값 테이블을 읽는다."""
    if not TABLE_PATH.exists():
        raise FileNotFoundError(f"Missing input table: {TABLE_PATH}")
    return pd.read_csv(TABLE_PATH)


def main() -> None:
    """SIC와 입력 변수들 사이의 피어슨 상관계수를 heatmap으로 저장한다."""
    table = load_point_table()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    corr = table[VARIABLES].corr(method="pearson")

    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    image = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(np.arange(len(VARIABLES)))
    ax.set_yticks(np.arange(len(VARIABLES)))
    ax.set_xticklabels(VARIABLES)
    ax.set_yticklabels(VARIABLES)
    ax.set_title("Pearson correlation matrix", fontsize=13, pad=12)

    for i in range(len(VARIABLES)):
        for j in range(len(VARIABLES)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=9)

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="correlation")
    output_path = FIGURE_DIR / "04c_correlation_heatmap.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {output_path}")


if __name__ == "__main__":
    main()
