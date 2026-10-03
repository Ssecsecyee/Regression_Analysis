# 04 CNN Forecasting Lead-1

이 폴더는 `03_cnn_monthly`에서 검증한 전체 격자 CNN 파이프라인을 미래예측 문제로 확장한다.
목표는 한 달 전 해빙 상태와 한 달 전 기후 조건을 사용해 다음 달 `SIC` 격자를 예측하는 것이다.

## 1. Task Definition

현재 단계의 예측 문제는 다음과 같다.

```text
SIC(t) + X(t) -> SIC(t+1)
```

여기서 `X(t)`는 다음 6개 기후 변수이다.

```text
tos, tas, rlds, rsds, uas, vas
```

입력 채널은 다음 8개이다.

```text
sic_t
tos_t
tas_t
rlds_t
rsds_t
uas_t
vas_t
tos_missing_t
```

출력은 다음 1개이다.

```text
sic_t_plus_1
```

Tensor shape은 다음과 같다.

```text
X = B x 8 x 448 x 304
y = B x 1 x 448 x 304
```

## 2. Why This Comes After 03

`03_cnn_monthly`는 같은 달 기후 변수로 같은 달 `SIC`를 맞추는 nowcasting/diagnostic 모델이었다.
그 실험의 목적은 전체 격자 데이터 파이프라인, `tos` 결측 처리, 마스크 기반 loss, 평가/시각화
체계를 검증하는 것이었다.

이번 `04`는 실제 미래예측에 더 가까운 모델이다. 특히 이전 달 `SIC`를 입력에 넣기 때문에,
모델은 이미 존재하는 해빙 분포가 다음 달 기후 조건 아래에서 어떻게 이동하거나 약화되는지
학습할 수 있다.

## 3. Split Rule

시간 분할은 target 날짜 기준으로 유지한다.

```text
train target: 1988-02 ~ 2013-12
val target:   2014-01 ~ 2017-12
test target:  2018-01 ~ 2025-12
```

예시는 다음과 같다.

```text
input: 2018-01 SIC + 2018-01 climate variables
target: 2018-02 SIC
split: test
```

첫 시점은 `t+1` target을 만들 수 없는 경우 제외한다.

## 4. Baseline To Beat

미래예측 모델은 반드시 persistence baseline과 비교한다.

```text
SIC(t+1) = SIC(t)
```

즉, "다음 달도 이번 달과 같을 것"이라고 보는 단순 기준선이다. CNN이 이 기준선을 이겨야 실제
예측 가치가 있다고 말할 수 있다.

## 5. Scripts

```text
01_prepare_lead1_dataset_index.py
02_train_cnn_lead1.py
03_evaluate_cnn_lead1.py
04_visualize_cnn_lead1.py
05_evaluate_ice_binary_metrics.py
06_visualize_2025_forecast_series.py
```

| File | Purpose |
|---|---|
| `forecasting_common.py` | lead-1 Dataset, 모델, metric, persistence baseline 공통 함수 |
| `01_prepare_lead1_dataset_index.py` | `SIC(t)+X(t) -> SIC(t+1)` sample index 생성 |
| `02_train_cnn_lead1.py` | lead-1 Simple CNN 학습 및 checkpoint 저장 |
| `03_evaluate_cnn_lead1.py` | CNN과 persistence baseline test 성능 비교 |
| `04_visualize_cnn_lead1.py` | 실제값, CNN 예측, persistence, 오차 비교 지도 생성 |
| `05_evaluate_ice_binary_metrics.py` | `SIC >= 0.15` 기준 ice/open-water Accuracy, Precision, Recall, F1 평가 |
| `06_visualize_2025_forecast_series.py` | 2025년 1~12월 관측 SIC와 예측 SIC 월별 비교 그림 생성 |

## 6. Output Storage

산출물은 다음 위치에 저장한다.

```text
experiments/04_cnn_forecasting_lead1
├─ datasets
├─ models
├─ results
├─ figures
├─ reports
└─ metadata
```

`experiments/`는 Git에 올리지 않는다.

## 7. First Run Order

서버에서는 다음 순서로 실행한다.

```bash
python 01_prepare_lead1_dataset_index.py
python 02_train_cnn_lead1.py --tos-strategy zero_fill
python 03_evaluate_cnn_lead1.py --checkpoint /root/Regression_Analysis/experiments/04_cnn_forecasting_lead1/models/실험ID/best.pt
python 04_visualize_cnn_lead1.py --checkpoint /root/Regression_Analysis/experiments/04_cnn_forecasting_lead1/models/실험ID/best.pt
python 05_evaluate_ice_binary_metrics.py --checkpoint /root/Regression_Analysis/experiments/04_cnn_forecasting_lead1/models/실험ID/best.pt
python 06_visualize_2025_forecast_series.py --checkpoint /root/Regression_Analysis/experiments/04_cnn_forecasting_lead1/models/실험ID/best.pt
```

현재 `03_cnn_monthly` 결과상 `zero_fill + tos_missing mask`가 `median3x3_min3`보다 안정적이었으므로,
lead-1 첫 baseline은 `zero_fill`로 시작한다.
