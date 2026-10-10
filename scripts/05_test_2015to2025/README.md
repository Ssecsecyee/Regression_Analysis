# 05 Test 2015 to 2025

이 폴더는 `04_cnn_forecasting_lead1`에서 학습한 lead-1 모델을 사용해 2015~2025년 target 월의
관측 `SIC`, 예측 `SIC`, 오차 격자를 추출하는 후처리 섹션이다.

## Purpose

`04` 폴더는 모델 학습/평가용이고, 이 `05` 폴더는 학습된 모델 결과를 공부하고 재시각화하기
좋은 형태로 따로 뽑아내기 위한 공간이다.

## Input

학습된 lead-1 checkpoint:

```text
experiments/04_cnn_forecasting_lead1/models/lead1_simple_cnn_zero_fill_20261003_044342/best.pt
```

모델 입력 구조:

```text
SIC(t) + tos/tas/rlds/rsds/uas/vas(t) + tos_missing(t) -> SIC(t+1)
```

## Output

기본 저장 위치:

```text
experiments/05_test_2015to2025
├─ grids_npz
├─ pixel_csv
├─ quicklook_png
├─ results
└─ packages
```

월별 `.npz` 파일에는 다음 배열이 들어간다.

```text
observed_sic
predicted_sic
error_sic
persistence_sic
valid_mask
input_date
target_date
```

월별 pixel CSV에는 다음 열이 들어간다.

```text
target_date
input_date
y
x
observed_sic
predicted_sic
error_sic
persistence_sic
```

이미지는 다음처럼 독립 저장한다.

```text
quicklook_png/<experiment_id>/2015_2025/observed/YYYY-MM_observed_sic.png
quicklook_png/<experiment_id>/2015_2025/predicted/YYYY-MM_predicted_sic.png
quicklook_png/<experiment_id>/2015_2025/error/YYYY-MM_error_sic.png
```

## Scripts

| File | Purpose |
|---|---|
| `01_export_2015_2025_sic_predictions.py` | 2015~2025 관측/예측 SIC 격자와 요약 CSV 추출 |

## Run

```bash
cd /root/Regression_Analysis/scripts/05_test_2015to2025

python 01_export_2015_2025_sic_predictions.py \
  --checkpoint /root/Regression_Analysis/experiments/04_cnn_forecasting_lead1/models/lead1_simple_cnn_zero_fill_20261003_044342/best.pt \
  --save-png
```

기본으로 월별 pixel CSV가 저장된다. `--save-png`를 붙이면 월별 관측/예측/오차 이미지를 각각
독립 PNG로 저장한다.
