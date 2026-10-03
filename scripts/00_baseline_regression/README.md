# 00 Baseline Regression

이 폴더는 처음 수행한 1차 회귀분석 코드를 보관하는 곳이다. 현재 새 분석의 입력 데이터로
사용하지 않고, 기존 결과를 비교하거나 과거 실험 흐름을 확인하는 용도로만 둔다.

## Purpose

1차 실험의 목적은 6개 북극항로 후보 지점에서 월별 `SIC`를 선형 회귀와 Ridge 회귀로
예측해 보는 것이었다.

```text
입력 변수: tos, tas, rlds, rsds, uas, vas
출력 변수: sic
모델: Linear Regression, Ridge Regression
단위: point별 독립 모델
```

## Scripts

| File | Purpose |
|---|---|
| `plot_nsr_bottlenecks.py` | 6개 NSR 후보 지점을 active mask 위에 표시 |
| `01_extract_monthly_point_data.py` | 원본 월별 NetCDF에서 6개 포인트 데이터 추출 |
| `02_run_point_regression.py` | 포인트별 Linear/Ridge 회귀분석 실행 |
| `03_visualize_point_regression.py` | 회귀 결과 전체 시각화 |
| `04_visualize_navigation_season.py` | 7~10월 항해 시기 예측 결과만 별도 시각화 |

## Archived Outputs

이 단계에서 생성된 결과는 다음 위치에 보관한다.

```text
experiments/01_baseline_linear_ridge
├─ processed_data
└─ figures
```

주요 산출물은 다음과 같다.

```text
monthly_point_regression_table.csv
monthly_point_regression_metadata.json
point_regression_metrics.csv
point_regression_coefficients.csv
point_regression_predictions.csv
```

## Notes

- 이 단계의 결과는 참고용 baseline이다.
- 이 단계에서 만든 CSV를 새 분석의 출발점으로 사용하지 않는다.
- 새 분석은 원본 NetCDF에서 다시 복원한 `experiments/02_data_diagnostics/analysis_data`를 기준으로 진행한다.
- 기존 실험에서는 학습된 모델 파일을 저장하지 않았고, 결과 CSV와 그림만 남겼다.
