from __future__ import annotations

from pathlib import Path

import h5py


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"

# 회귀분석에서 사용할 7개 핵심 변수.
# sic은 타깃이고, 나머지 6개는 입력 후보 변수다.
VARIABLES = ["sic", "tos", "tas", "rlds", "rsds", "uas", "vas"]

# 현재 데이터는 월별과 주별 두 해상도로 제공된다.
# 첫 회귀 실험은 월별을 쓰지만, 구조 비교를 위해 둘 다 확인한다.
RESOLUTIONS = ["monthly", "weekly"]


def format_attr(value: object) -> str:
    """NetCDF/HDF5 attribute 값을 화면 출력용 문자열로 변환한다.

    h5py로 읽은 attribute는 bytes, numpy scalar, numpy array 등으로 섞여 나온다.
    표 형태로 보기 좋게 출력하려면 Python 기본 문자열로 정리하는 과정이 필요하다.
    값이 없으면 "-"로 표시한다.
    """
    if value is None:
        return "-"
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    return str(value)


def inspect_main_variable(resolution: str, variable: str) -> dict[str, str]:
    """월별 또는 주별 NetCDF 파일에서 핵심 변수의 구조 정보를 읽는다.

    여기서 확인하는 값은 분석 전에 반드시 알아야 하는 메타데이터다.

    - shape: 실제 배열 구조. 이 데이터는 (time, y, x) 순서다.
    - dtype: 디스크에 저장된 자료형. 물리값이 아니라 패킹된 정수일 수 있다.
    - chunks: 파일을 읽을 때의 저장 단위. 현재는 한 시점 전체 필드가 한 chunk다.
    - _FillValue: 결측값 코드. 분석 전 NaN으로 바꿔야 한다.
    - scale_factor/add_offset: 패킹된 값을 실제 물리값으로 복원할 때 필요하다.
    """
    path = DATA_DIR / resolution / f"{variable}_arctic25km_{resolution}_1988-2025.nc"
    if not path.exists():
        raise FileNotFoundError(path)

    with h5py.File(path, "r") as file:
        dataset = file[variable]
        time_name = "month" if resolution == "monthly" else "week"

        # row 하나가 출력 테이블의 한 줄이 된다.
        # 전부 문자열로 맞춰 두면 print_table에서 폭 계산이 단순해진다.
        row = {
            "resolution": resolution,
            "variable": variable,
            "file": str(path.relative_to(ROOT)),
            "shape": str(tuple(dataset.shape)),
            "dtype": str(dataset.dtype),
            "chunks": str(dataset.chunks),
            "time_count": str(len(file["time"])),
            time_name: f"{int(file[time_name][:].min())}-{int(file[time_name][:].max())}",
            "year": f"{int(file['year'][:].min())}-{int(file['year'][:].max())}",
            "fill": format_attr(dataset.attrs.get("_FillValue")),
            "scale": format_attr(dataset.attrs.get("scale_factor")),
            "offset": format_attr(dataset.attrs.get("add_offset")),
            "units": format_attr(dataset.attrs.get("units")),
        }
    return row


def inspect_mask_file(path: Path) -> list[dict[str, str]]:
    """마스크 NetCDF 파일 안에 들어 있는 모든 Dataset의 구조를 읽는다.

    마스크 파일에는 land/ocean/active mask뿐 아니라 x, y, lat, lon 같은 좌표 변수도
    함께 들어 있다. 회귀 대상 픽셀이 실제 해양인지, active 영역인지 확인하려면 이
    구조를 먼저 알아야 한다.
    """
    rows = []
    with h5py.File(path, "r") as file:
        for name, obj in file.items():
            # h5py 파일 안에는 Dataset 외에 Group이 있을 수 있다.
            # 여기서는 shape/dtype이 있는 Dataset만 표에 포함한다.
            if not isinstance(obj, h5py.Dataset):
                continue
            rows.append(
                {
                    "file": str(path.relative_to(ROOT)),
                    "variable": name,
                    "shape": str(tuple(obj.shape)),
                    "dtype": str(obj.dtype),
                    "fill": format_attr(obj.attrs.get("_FillValue")),
                    "units": format_attr(obj.attrs.get("units")),
                }
            )
    return rows


def print_table(rows: list[dict[str, str]], columns: list[str]) -> None:
    """dict row 목록을 고정폭 텍스트 표로 출력한다.

    pandas 없이도 터미널에서 바로 읽기 좋게 보려고 만든 작은 출력 함수다.
    각 column의 최대 글자 길이를 계산한 뒤, 모든 row를 같은 폭으로 맞춰 출력한다.
    """
    if not rows:
        print("(no rows)")
        return

    widths = {
        column: max(len(column), *(len(str(row.get(column, ""))) for row in rows))
        for column in columns
    }
    header = "  ".join(column.ljust(widths[column]) for column in columns)
    print(header)
    print("-" * len(header))
    for row in rows:
        print("  ".join(str(row.get(column, "")).ljust(widths[column]) for column in columns))


def main() -> None:
    """데이터 구조 진단을 실행한다.

    이 스크립트는 파일을 생성하지 않고, 원본 NetCDF의 구조를 읽어서 터미널에 출력한다.
    첫 단계 목적은 "어떤 배열을 어떤 방식으로 읽어야 하는지"를 확인하는 것이다.
    """
    print("\nMAIN VARIABLE FILES")

    # 7개 변수 x 2개 시간해상도 = 14개 파일의 구조를 확인한다.
    main_rows = [
        inspect_main_variable(resolution, variable)
        for resolution in RESOLUTIONS
        for variable in VARIABLES
    ]
    print_table(
        main_rows,
        [
            "resolution",
            "variable",
            "shape",
            "dtype",
            "chunks",
            "time_count",
            "year",
            "fill",
            "scale",
            "offset",
            "units",
        ],
    )

    print("\nMASK FILES")
    mask_rows = []

    # 개별 마스크 파일과 합본 마스크 파일 모두 구조를 확인한다.
    # 실제 분석에서는 active_mask.nc와 land_mask.nc를 주로 사용한다.
    for mask_path in [
        DATA_DIR / "masks" / "land_mask.nc",
        DATA_DIR / "masks" / "active_mask.nc",
        DATA_DIR / "masks" / "masks_arctic25km.nc",
    ]:
        if mask_path.exists():
            mask_rows.extend(inspect_mask_file(mask_path))

    print_table(mask_rows, ["file", "variable", "shape", "dtype", "fill", "units"])

    print("\nNOTES")
    print("- Main variables use shape (time, y, x).")
    print("- The grid size is y=448, x=304 for monthly and weekly files.")
    print("- Values are packed on disk; use scale_factor/add_offset before analysis.")
    print("- _FillValue should be converted to NaN before regression.")


if __name__ == "__main__":
    main()
