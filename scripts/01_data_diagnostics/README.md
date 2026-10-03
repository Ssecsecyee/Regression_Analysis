# 01 Data Diagnostics

이 폴더는 원본 데이터 구조를 확인하고, 회귀분석 전에 6개 포인트 데이터의 상태를 점검하기
위한 스크립트를 담는다.

## Purpose

이 단계의 목적은 모델을 바로 학습하는 것이 아니라, 원본 데이터가 어떤 구조인지 확인하고
회귀분석에 들어가기 전 데이터 상태를 검증하는 것이다.

확인 대상은 다음과 같다.

```text
변수 shape
_FillValue
scale_factor
add_offset
단위
결측값
월별/변수별 분포
SIC와 입력 변수의 관계
```

## Scripts

| File | Purpose | Output |
|---|---|---|
| `01_inspect_variable_shapes.py` | 원본 NetCDF 변수 shape, dtype, fill/scale/offset 확인 | 콘솔 출력 |
| `02_extract_point_variable_table.py` | 6개 포인트 x 7개 변수 월별 물리값 복원 | `experiments/02_data_diagnostics/analysis_data` |
| `03_check_missing_values.py` | 포인트별, 변수별, split별 결측값 확인 | 콘솔 출력 |
| `04a_plot_standardized_distributions.py` | 변수별 표준화 분포와 정규곡선 비교 | `04a_*.png` |
| `04b_plot_qq_plots.py` | 변수별 Q-Q plot 생성 | `04b_*.png` |
| `04c_plot_correlation_heatmap.py` | 변수 간 피어슨 상관관계 heatmap 생성 | `04c_*.png` |
| `04d_plot_sic_vs_predictors.py` | `SIC`와 입력 변수 산점도 생성 | `04d_*.png` |
| `04e_plot_monthly_distributions.py` | 변수별 월별 분포 boxplot 생성 | `04e_*.png` |

## Main Outputs

복원된 포인트 데이터는 다음 위치에 저장된다.

```text
experiments/02_data_diagnostics/analysis_data
├─ point_monthly_physical_values.csv
└─ point_monthly_decode_metadata.json
```

회귀 전 진단 그림은 다음 위치에 저장된다.

```text
experiments/02_data_diagnostics/figures
├─ 04a_standardized_distribution_histograms.png
├─ 04b_qq_plots_by_variable.png
├─ 04c_correlation_heatmap.png
├─ 04d_sic_vs_predictor_scatter.png
└─ 04e_monthly_distributions_by_variable.png
```

## Current Findings

- 월별 변수 shape는 기본적으로 `(455, 448, 304)`이다.
- 총 6개 포인트와 7개 변수를 사용한다.
- 4번 지점 `Dmitry Laptev Strait`의 `tos`는 전체 455개월에서 결측이다.
- 다른 포인트와 변수에서는 현재 단계 기준 명확한 전체 결측 문제는 확인되지 않았다.
- `SIC`는 0~1 범위에 갇힌 변수라 일반적인 정규분포를 기대하기 어렵다.
- `rlds`, `rsds`, `tas`, `tos` 등은 계절성이 강해 전체 월을 한꺼번에 보면 분포가 쌍봉 또는 비대칭으로 나타날 수 있다.

## Important Rule

이 단계에서는 값을 보정하지 않는다.

```text
하는 일: 원본 packed 값 복원, 결측 확인, 분포 시각화
하지 않는 일: 결측 대체, 이상치 제거, 회귀모델 학습
```
