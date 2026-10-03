from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
TABLE_PATH = (
    ROOT
    / "experiments"
    / "02_data_diagnostics"
    / "analysis_data"
    / "point_monthly_physical_values.csv"
)

VARIABLES = ["sic", "tos", "tas", "rlds", "rsds", "uas", "vas"]


def load_point_table() -> pd.DataFrame:
    """02번 스크립트에서 만든 포인트별 월별 물리값 테이블을 읽는다.

    이 테이블은 이미 _FillValue, scale_factor, add_offset을 적용한 값이다.
    여기서는 값을 수정하지 않고 결측 여부만 확인한다.
    """
    if not TABLE_PATH.exists():
        raise FileNotFoundError(
            f"Missing input table: {TABLE_PATH}\n"
            "Run 02_extract_point_variable_table.py first."
        )
    return pd.read_csv(TABLE_PATH)


def print_table_overview(table: pd.DataFrame) -> None:
    """전체 행 개수, 포인트 개수, 기간 범위를 출력한다."""
    print("=== Table overview ===")
    print(f"Input table: {TABLE_PATH}")
    print(f"Rows: {len(table):,}")
    print(f"Points: {table['point_id'].nunique()}")
    print(f"Date range: {table['date'].min()} ~ {table['date'].max()}")
    print(f"Splits: {', '.join(table['split'].drop_duplicates().astype(str))}")


def print_missing_by_point_variable(table: pd.DataFrame) -> None:
    """포인트별, 변수별 결측 개수를 확인한다.

    사용자가 가장 먼저 확인하고 싶은 표다.
    예를 들어 point 4의 tos가 455로 나오면, 455개월 전체가 결측이라는 뜻이다.
    """
    print("\n=== Missing count by point and variable ===")
    summary = (
        table.groupby(["point_id", "point_name"])[VARIABLES]
        .apply(lambda frame: frame.isna().sum())
        .reset_index()
    )
    print(summary.to_string(index=False))


def print_missing_by_split(table: pd.DataFrame) -> None:
    """train/val/test 구간별 결측 개수를 확인한다.

    결측이 특정 기간에만 몰려 있는지, 전체 기간에 걸쳐 있는지 판단하기 위한 표다.
    """
    print("\n=== Missing count by split ===")
    summary = (
        table.groupby(["split", "point_id", "point_name"])[VARIABLES]
        .apply(lambda frame: frame.isna().sum())
        .reset_index()
    )
    summary = summary[summary[VARIABLES].sum(axis=1) > 0]
    if summary.empty:
        print("No missing values by split.")
        return
    print(summary.to_string(index=False))


def print_missing_rows(table: pd.DataFrame) -> None:
    """결측이 실제로 들어 있는 행을 출력한다.

    전체 행을 다 보여주면 길어질 수 있으므로 결측이 있는 행만 필터링한다.
    지금 데이터에서는 point 4의 tos 행만 나오는지 확인하는 용도다.
    """
    print("\n=== Rows containing at least one missing value ===")
    missing_mask = table[VARIABLES].isna().any(axis=1)
    missing_rows = table.loc[
        missing_mask,
        ["date", "split", "point_id", "point_name", *VARIABLES],
    ]
    if missing_rows.empty:
        print("No rows contain missing values.")
        return

    print(f"Rows with missing values: {len(missing_rows):,}")
    print("\nFirst 20 missing rows:")
    print(missing_rows.head(20).to_string(index=False))
    print("\nLast 20 missing rows:")
    print(missing_rows.tail(20).to_string(index=False))


def print_missing_ratio(table: pd.DataFrame) -> None:
    """포인트별, 변수별 결측 비율을 퍼센트로 출력한다."""
    print("\n=== Missing ratio by point and variable (%) ===")
    summary = (
        table.groupby(["point_id", "point_name"])[VARIABLES]
        .apply(lambda frame: frame.isna().mean() * 100)
        .reset_index()
    )
    print(summary.round(2).to_string(index=False))


def main() -> None:
    """결측치 상태를 표로 확인한다.

    이 스크립트는 결측을 채우거나 삭제하지 않는다.
    다음 단계에서 보정 여부를 결정하기 전에, 원본 복원 테이블의 결측 구조를 확인한다.
    """
    table = load_point_table()
    print_table_overview(table)
    print_missing_by_point_variable(table)
    print_missing_by_split(table)
    print_missing_ratio(table)
    print_missing_rows(table)


if __name__ == "__main__":
    main()
