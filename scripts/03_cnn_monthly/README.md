# 03 CNN Monthly

이 폴더는 월별 격자 데이터를 이용해 CNN 기반 `SIC` 예측 실험을 준비하기 위한 설계 문서이다.
아직 학습 코드는 작성하지 않았으며, 먼저 데이터 형태와 저장 구조, 실험 순서를 고정한다.

## Purpose

기존 6개 포인트 회귀분석은 단일 픽셀의 시계열만 사용한다. CNN 실험은 월별 격자 이미지를
입력으로 사용해 공간 패턴을 함께 학습하는 방향이다.

목표는 다음과 같다.

```text
월별 기후 변수 격자 -> CNN -> 월별 SIC 격자 또는 관심 영역 SIC
```

현재 로컬 장비 성능을 고려해, 처음부터 전체 북극 격자 학습을 크게 돌리기보다 작은 입력
범위와 단순한 CNN 구조부터 시작한다.

## Input Data

월별 NetCDF 파일을 사용한다.

```text
data/monthly
├─ sic_arctic25km_monthly_1988-2025.nc
├─ tos_arctic25km_monthly_1988-2025.nc
├─ tas_arctic25km_monthly_1988-2025.nc
├─ rlds_arctic25km_monthly_1988-2025.nc
├─ rsds_arctic25km_monthly_1988-2025.nc
├─ uas_arctic25km_monthly_1988-2025.nc
└─ vas_arctic25km_monthly_1988-2025.nc
```

각 변수의 기본 shape는 다음과 같다.

```text
time x y x x = 455 x 448 x 304
```

CNN 입력 후보 변수는 다음 6개이다.

```text
tos, tas, rlds, rsds, uas, vas
```

타깃은 다음과 같다.

```text
sic
```

## Basic Tensor Design

CNN 입력 텐서는 월별 한 시점을 하나의 샘플로 보고, 변수를 채널로 쌓는 형태를 기본으로 한다.

```text
X[t] shape = channels x height x width
            = 6 x 448 x 304

y[t] shape = 1 x 448 x 304
```

PyTorch를 사용할 경우 기본 형태는 다음과 같다.

```text
X_batch = batch x channels x height x width
y_batch = batch x 1 x height x width
```

예상 shape:

```text
X_batch = B x 6 x 448 x 304
y_batch = B x 1 x 448 x 304
```

단, 전체 격자를 그대로 쓰면 메모리 부담이 크므로 1차 CNN 실험에서는 다음 중 하나를 선택한다.

```text
1. 전체 격자 대신 active mask 주변 crop 사용
2. 6개 포인트 주변 patch 사용
3. 해상도를 낮춘 downsampled grid 사용
```

## First Experiment Scope

처음 CNN은 큰 모델이 아니라 데이터 흐름 검증용 작은 모델로 시작한다.

추천 1차 범위:

```text
resolution: monthly
input variables: tos, tas, rlds, rsds, uas, vas
target: sic
period: 1988-02 ~ 2025-12
split:
  train: 1988-02 ~ 2013-12
  val:   2014-01 ~ 2017-12
  test:  2018-01 ~ 2025-12
```

1차 목표는 높은 성능이 아니라 다음을 확인하는 것이다.

```text
1. NetCDF -> tensor 변환이 맞는가
2. fill/scale/offset 복원이 맞는가
3. mask 적용 방식이 맞는가
4. train/val/test 시간 분리가 맞는가
5. CNN 출력이 SIC 범위 0~1 안에서 안정적인가
```

## Output Storage

CNN 실험 산출물은 별도 실험 폴더에 저장한다.

```text
experiments/04_cnn_monthly
├─ datasets
├─ models
├─ figures
├─ results
└─ metadata
```

각 폴더 목적은 다음과 같다.

| Folder | Purpose |
|---|---|
| `datasets` | CNN용 tensor index, crop/patch metadata 저장 |
| `models` | 학습된 CNN model checkpoint 저장 |
| `figures` | 예측 SIC 지도, 오차 지도, 학습 곡선 저장 |
| `results` | metrics, predictions summary 저장 |
| `metadata` | 변수 목록, mask, crop 범위, split, normalization 정보 저장 |

## Proposed Scripts

아직 생성하지 않았지만, 구현 시 다음 순서를 권장한다.

```text
01_prepare_monthly_cnn_index.py
02_build_monthly_cnn_dataset.py
03_train_monthly_cnn.py
04_evaluate_monthly_cnn.py
05_visualize_monthly_cnn_predictions.py
```

각 스크립트의 역할은 다음과 같다.

| File | Purpose |
|---|---|
| `01_prepare_monthly_cnn_index.py` | 월별 시간축, split, 입력/타깃 파일 목록 정리 |
| `02_build_monthly_cnn_dataset.py` | NetCDF에서 CNN 입력 tensor를 읽는 Dataset 준비 |
| `03_train_monthly_cnn.py` | 작은 CNN baseline 학습 및 checkpoint 저장 |
| `04_evaluate_monthly_cnn.py` | test RMSE, MAE, Bias, 공간 오차 평가 |
| `05_visualize_monthly_cnn_predictions.py` | 실제 SIC, 예측 SIC, 오차 지도 시각화 |

## Important Design Choices

CNN으로 넘어가기 전에 먼저 결정해야 할 항목은 다음과 같다.

```text
1. 전체 격자를 쓸지, crop/patch를 쓸지
2. land/ocean/active mask를 loss에 어떻게 반영할지
3. target을 전체 SIC grid로 둘지, 특정 포인트/patch SIC로 둘지
4. normalization을 전체 train grid 기준으로 할지, 변수별 제공 통계를 쓸지
5. 출력 activation을 sigmoid로 제한할지, 예측 후 0~1 clip을 할지
```

현재 추천은 다음과 같다.

```text
1차 CNN:
  - 작은 crop 또는 포인트 주변 patch 사용
  - 입력 6채널
  - 타깃 SIC patch
  - train 기간 기준 normalization
  - ocean/active mask 영역 중심 loss
```

## Relation To Regression Workflow

CNN은 기존 선형/Ridge 회귀모델을 대체하기보다, 공간 패턴을 반영하는 비교 모델로 둔다.

```text
Linear/Ridge:
  단일 포인트 시계열 모델

CNN:
  월별 공간 격자 또는 patch 모델
```

따라서 같은 test 기간에서 다음을 비교할 수 있다.

```text
1. 포인트별 SIC 예측 성능
2. 월별 공간 오차 패턴
3. 7~10월 또는 저해빙 시즌 후보 월에서의 예측 안정성
```

## Current Status

현재 상태는 설계 단계이다.

```text
완료:
  - 월별 원본 데이터 구조 확인
  - 6개 포인트 진단 데이터 생성
  - 회귀 전 분포/상관 진단
  - CNN 실험 하네스 README 작성

다음:
  - CNN 입력 범위 결정
  - crop/patch 방식 결정
  - Dataset 스크립트 작성
```
