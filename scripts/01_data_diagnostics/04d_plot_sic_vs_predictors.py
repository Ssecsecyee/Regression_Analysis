from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
TABLE_PATH = ROOT / "experiments" / "02_data_diagnostics" / "analysis_data" / "point_monthly_physical_values.csv"
FIGURE_DIR = ROOT / "experiments" / "02_data_diagnostics" / "figures"

TARGET = "sic"
PREDICTORS = ["tos", "tas", "rlds", "rsds", "uas", "vas"]


def load_point_table() -> pd.DataFrame:
    """02번에서 만든 포인트별 월별 물리값 테이블을 읽는다."""
    if not TABLE_PATH.exists():
        raise FileNotFoundError(f"Missing input table: {TABLE_PATH}")
    return pd.read_csv(TABLE_PATH)


def main() -> None:
    """목표 변수 SIC와 각 입력 변수의 산점도를 저장한다."""
    table = load_point_table()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 2, figsize=(14, 16), sharey=True)
    axes = axes.ravel()

    for ax, variable in zip(axes, PREDICTORS):
        for point_id, group in table.groupby("point_id"):
            valid = group[[TARGET, variable]].dropna()
            if valid.empty:
                continue
            ax.scatter(valid[variable], valid[TARGET], s=10, alpha=0.45, label=f"Point {point_id}")

        valid_all = table[[TARGET, variable]].dropna()
        corr = valid_all[TARGET].corr(valid_all[variable]) if len(valid_all) > 1 else np.nan
        ax.set_title(f"SIC vs {variable} (r={corr:.2f})", fontsize=10, pad=10)
        ax.set_xlabel(variable)
        ax.set_ylabel("sic")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", frameon=False)
    fig.suptitle("SIC vs predictors before regression", fontsize=14, y=0.995)
    fig.subplots_adjust(hspace=0.42, wspace=0.25, right=0.84, top=0.94)

    output_path = FIGURE_DIR / "04d_sic_vs_predictor_scatter.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {output_path}")


if __name__ == "__main__":
    main()
