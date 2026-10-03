from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
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


def main() -> None:
    """월별 분포를 boxplot으로 저장해 계절성을 확인한다."""
    table = load_point_table()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(4, 2, figsize=(15, 18))
    axes = axes.ravel()

    for ax, variable in zip(axes, VARIABLES):
        data_by_month = [table.loc[table["month"] == month, variable].dropna() for month in range(1, 13)]
        ax.boxplot(data_by_month, showfliers=True)
        ax.set_title(f"{VARIABLE_LABELS[variable]} by month", fontsize=10, pad=10)
        ax.set_xlabel("month")
        ax.set_ylabel(variable)
        ax.grid(True, axis="y", alpha=0.25)

    axes[-1].axis("off")
    fig.suptitle("Monthly distributions for seasonality check", fontsize=14, y=0.995)
    fig.subplots_adjust(hspace=0.55, wspace=0.28, top=0.95)

    output_path = FIGURE_DIR / "04e_monthly_distributions_by_variable.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {output_path}")


if __name__ == "__main__":
    main()
