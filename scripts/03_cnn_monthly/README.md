# 03 CNN Monthly

이 폴더는 월별 NetCDF 격자 전체를 사용해 CNN 기반 `SIC` 예측 모델을 학습하기 위한 실험
하네스이다. 기존 6개 포인트 회귀분석과 달리, CNN 실험은 월별 데이터의 모든 픽셀을 사용해
공간 패턴을 학습하는 독립적인 예측 모델을 만드는 것을 목표로 한다.

현재 단계에서는 코드 작성 전에 전체 파이프라인 구조, 전처리 검증 항목, 모델 저장 방식,
활성함수/앙상블 실험, 검증 보고서 산출 방식을 먼저 고정한다.

## 1. Project Summary

- 목표: 6개 기후 변수(`tos`, `tas`, `rlds`, `rsds`, `uas`, `vas`)의 월별 격자 데이터를 입력받아
  동일 월 또는 지정 lead time의 해빙 농도(`SIC`) 격자를 산출하는 CNN 모델 개발
- 데이터: 1988-2025년 월별 NetCDF
- 기본 입력 shape: `B x 6 x 448 x 304`
- 기본 타깃 shape: `B x 1 x 448 x 304`
- 실행 환경: RTX 6000 Ada 4대 중 `cuda:3` 사용
- 산출물 저장: `experiments/03_cnn_monthly/`

핵심 파이프라인은 다음과 같다.

```text
1. 전처리 검증
   결측치, scale/offset 복원, mask 검증

2. CNN 모델링
   Baseline CNN -> U-Net -> ensemble

3. 평가와 보고
   전체 공간 평가, 월별 평가, 지도 시각화, 약식 검증 보고서 작성
```

## 2. Prediction Task Definition

CNN 실험에서는 두 가지 문제 설정을 구분한다.

### 2.1 Diagnostic / Nowcasting

같은 월의 기후 변수로 같은 월의 `SIC`를 예측한다.

```text
X(t) -> SIC(t)
```

이 설정은 미래 예측이라기보다, 기후 변수와 `SIC`의 동시적 공간 관계를 학습하는 진단 모델이다.
첫 CNN baseline은 데이터 파이프라인 검증과 공간 패턴 학습 가능성 확인을 위해 이 설정으로
시작할 수 있다.

### 2.2 Forecasting

현재 또는 과거 기후 변수로 미래 `SIC`를 예측한다.

```text
X(t) -> SIC(t + lead)
```

예:

```text
lead = 1 month
X(2020-10) -> SIC(2020-11)
```

미래 해빙 상태를 직접 예측하려면 이 설정이 필요하다. 따라서 README 기준으로는
`nowcasting`과 `forecasting`을 명확히 분리하고, 각 실험의 `lead_months`를 metadata에 반드시
저장한다.

## 3. Compute Environment

현재 로컬 장비 대신 터널을 통해 GPU 서버를 사용할 수 있다. 확인된 PyTorch/CUDA 환경은 다음과
같다.

```text
PyTorch version: 2.6.0+cu124
PyTorch CUDA build: 12.4
CUDA available to PyTorch: True
GPU count: 4
GPU 0: NVIDIA RTX 6000 Ada Generation
GPU 1: NVIDIA RTX 6000 Ada Generation
GPU 2: NVIDIA RTX 6000 Ada Generation
GPU 3: NVIDIA RTX 6000 Ada Generation
CUDA tensor check: [2.0, 2.0, 2.0]
```

기본 학습 GPU는 다음으로 고정한다.

```text
CUDA device: 3
device = cuda:3
```

모델 저장 metadata에는 실제 실행 당시의 PyTorch 버전, CUDA 버전, GPU 이름, 사용 device를
반드시 함께 저장한다.

## 4. Input Data

월별 NetCDF 파일을 사용한다.

새 작업 환경에서는 원본 데이터가 루트의 `data.zip`으로 압축되어 있을 수 있다. CNN 전처리
스크립트를 실행하기 전에 `data.zip`을 풀어 아래와 같은 `data/` 폴더 구조가 존재해야 한다.

```text
C:\Regression_Analysis\해빙
├─ data.zip
└─ data
   ├─ monthly
   ├─ weekly
   └─ masks
```

학습 스크립트는 압축 파일을 직접 읽지 않고 압축 해제된 `data/` 폴더를 입력으로 사용한다.

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

각 NetCDF 파일은 변수 하나를 담고 있으며, 공통적으로 `(time, y, x)` 구조를 가진다.

```text
time = 455
y    = 448
x    = 304
```

분석 기간은 다음과 같다.

```text
1988-02 ~ 2025-12
```

시간 분할은 기존 회귀분석과 동일하게 유지한다.

```text
train: 1988-02 ~ 2013-12
val:   2014-01 ~ 2017-12
test:  2018-01 ~ 2025-12
```

시간자료이므로 무작위 분할은 사용하지 않는다.

## 5. Tensor Design

월별 한 시점을 하나의 샘플로 보고, 입력 변수를 채널로 쌓는다.

```text
X_batch = batch x channels x height x width
        = B x 6 x 448 x 304

y_batch = batch x 1 x height x width
        = B x 1 x 448 x 304
```

기본 입력 변수는 다음 6개이다.

```text
tos, tas, rlds, rsds, uas, vas
```

타깃 변수는 다음과 같다.

```text
sic
```

CNN 실험은 6개 포인트 patch가 아니라 전체 `448 x 304` 격자 픽셀을 사용한다.

## 6. Output Storage

CNN 실험 산출물은 `03_cnn_monthly` 섹션 번호와 맞춰 다음 폴더에 저장한다.

```text
experiments/03_cnn_monthly
├─ preprocessing
├─ datasets
├─ models
├─ figures
├─ results
├─ reports
└─ metadata
```

각 폴더 목적은 다음과 같다.

| Folder | Purpose |
|---|---|
| `preprocessing` | 변수별 전처리 검증 결과, 결측 검증표, scale/offset 확인 결과 |
| `datasets` | CNN용 index, normalization 통계, mask 정보, tensor cache 정보 |
| `models` | 학습된 CNN checkpoint, ensemble checkpoint, best model |
| `figures` | 학습 곡선, 예측 SIC 지도, 오차 지도, 월별 비교 그림 |
| `results` | metrics CSV, split별 예측 요약, grid 평가 결과 |
| `reports` | 약식 검증 보고서 Markdown 또는 HTML |
| `metadata` | 실험 설정, 변수 목록, GPU 환경, seed, model config |

`experiments/`는 Git에 올리지 않는 산출물 영역이다.

## 7. Preprocessing Harness

CNN 학습 전에 각 변수 파일별 전처리 검증 스크립트를 분리한다. 목적은 학습 전에 데이터 shape,
결측, packed value 복원, 마스크 적용 가능성을 명확히 검증하는 것이다.

제안 스크립트는 다음과 같다.

```text
01_check_monthly_files.py
02_validate_each_variable.py
03_build_cnn_normalization_stats.py
04_validate_cnn_masks.py
05_prepare_cnn_dataset_index.py
06_validate_tos_fill_strategies.py
```

각 파일 목적은 다음과 같다.

| File | Purpose |
|---|---|
| `01_check_monthly_files.py` | 월별 NetCDF 7개 파일 존재 여부, 변수명, shape, time axis 일치 확인 |
| `02_validate_each_variable.py` | 각 변수별 `_FillValue`, `scale_factor`, `add_offset`, 결측 수, 유효 범위 검증 |
| `03_build_cnn_normalization_stats.py` | train 기간 기준 변수별 normalization 통계 계산 |
| `04_validate_cnn_masks.py` | land/ocean/active mask shape, target/loss 적용 가능 영역 확인 |
| `05_prepare_cnn_dataset_index.py` | train/val/test 월별 sample index와 입력/타깃 파일 매핑 생성 |
| `06_validate_tos_fill_strategies.py` | `tos` 결측 처리 후보인 정규화 후 0 채움과 3x3 valid ocean 보간 비교 |

각 전처리 스크립트는 반드시 다음을 산출해야 한다.

```text
1. 콘솔 요약
2. CSV 또는 JSON 검증 결과
3. 실패 조건 명시
4. metadata 기록
```

전처리 단계에서는 모델 학습을 하지 않는다.

## 8. Variable Restoration And Missing Validation

NetCDF 값은 packed value일 수 있으므로 CNN 입력 전에 반드시 다음 순서를 적용한다.

```text
1. raw value 읽기
2. _FillValue를 NaN으로 변환
3. scale_factor 적용
4. add_offset 적용
5. 유효 범위 확인
```

복원 공식은 다음과 같다.

```text
physical_value = raw_value * scale_factor + add_offset
```

단, `_FillValue`는 실제 물리값이 아니므로 scale/offset을 적용하지 않고 결측으로 처리한다.

검증해야 할 항목은 다음과 같다.

```text
shape
dtype
units
_FillValue
scale_factor
add_offset
valid pixel count
missing pixel count
missing ratio by month
min / max after restoration
train / val / test별 missing ratio
```

## 9. Missing Value Strategy

CNN 입력 텐서에는 `NaN`이 들어갈 수 없다. 따라서 결측 처리 전략을 명시적으로 고정해야 한다.

특히 `tos`는 해빙, 육지, 마스크 조건에 따라 결측이 광범위하게 나타날 수 있으므로 단순
`np.nanmean` 기반 통계와 단순 0 채움은 편향을 만들 수 있다.

중요한 원칙은 다음과 같다.

```text
raw tos NaN -> raw 0 K 대체는 금지한다.
```

0 K는 물리적으로 불가능한 해수면 온도이며, 모델에 강한 가짜 신호를 넣는다. 다만 z-score
정규화 이후의 0은 train 평균을 의미하므로, raw 0과 의미가 다르다.

전처리 검증 후 다음 후보를 비교한다.

```text
1. normalization 전 valid pixel만으로 train mean/std 계산
2. normalization 후 NaN을 0으로 채움
3. 입력 채널별 missing mask를 추가 채널로 제공
4. 특정 변수의 결측 영역을 loss/mask와 분리해서 추적
```

`tos`는 모델 영향도가 큰 변수이므로 별도 전략을 둔다.

### 9.1 1차 CNN: zero-fill baseline

```text
1. tos를 train 통계로 normalization
2. normalization 후 NaN을 0으로 채움
3. tos_missing 또는 tos_valid 정보를 별도 mask로 보관
4. 보고서에 tos 결측 비율 명시
```

이때 0은 raw 0 K가 아니라 normalized 평균값이다.

### 9.2 2차 CNN: 3x3 valid ocean median interpolation

```text
1. raw tos NaN 위치 확인
2. 같은 시점의 3x3 주변 valid ocean tos median으로 결측 보간
3. 3x3 창 안에 valid ocean 픽셀이 최소 3개 이상일 때만 보간
4. valid ocean 픽셀이 3개 미만이면 보간하지 않고 NaN 유지
5. 보간 후 train 통계로 normalization
6. 그래도 남은 NaN은 normalization 후 0으로 채움
7. tos_missing mask는 계속 유지
8. 1차 zero-fill baseline과 같은 test split에서 성능 비교
```

3x3 보간은 육지나 invalid pixel을 통계값 계산에 넣지 않는다. 같은 시점에서 `tos`가 실제로
존재하는 주변 ocean pixel만 사용한다. 평균이 아니라 median을 기본값으로 둔다. 결측 비율이
높은 변수에서 mean은 경계값을 과도하게 확산시킬 수 있고, max는 따뜻한 값으로 과대 보간될
위험이 크기 때문이다.

### 9.3 3차 CNN: strategy comparison

```text
1. zero-fill baseline
2. 3x3 valid ocean median interpolation
3. 필요 시 active/ocean 영역별 성능 비교
```

비교 대상은 전체 test 성능뿐 아니라 다음 영역별 성능이다.

```text
ocean
active_union
monthly active mask
tos missing area
tos valid area
```

현재 1차 원칙은 다음과 같다.

```text
normalization 통계:
  train 기간의 valid pixel만 사용

CNN 입력:
  normalization 후 NaN을 0으로 채움
  단, missing mask를 metadata와 검증 결과에 저장

검증:
  변수별 missing ratio가 높은 영역과 월을 보고서에 기록
```

이 방식에서 0은 normalized space의 평균값을 의미하므로, raw value 0과 혼동하지 않는다.

## 10. Mask And Loss Design

전체 픽셀을 사용하더라도 모든 픽셀이 동일한 의미를 갖지는 않는다. 육지, 바다, active sea-ice
영역을 구분해야 한다.

검토 대상 mask는 다음과 같다.

```text
land mask
ocean mask
coast mask
active monthly mask
active union mask
tos valid mask
```

중요한 원칙은 다음과 같다.

```text
loss mask는 target이 이미 얼음인 픽셀만 뜻하지 않는다.
loss mask는 모델이 예측해야 하는 유효 도메인을 뜻해야 한다.
```

따라서 `valid target pixel`만 사용하는 loss는 false positive를 놓칠 수 있다. 예를 들어 원래
해빙이 없어야 할 바다에 모델이 얼음을 예측했는데 그 픽셀이 loss에서 제외되면 오답에 대한
페널티가 사라진다.

1차 기준은 다음 후보를 비교한다.

```text
1. ocean mask 전체 기준 loss
2. active_union + ocean 기준 loss
3. month별 active mask + ocean 기준 loss
4. target 결측만 제외하고 ocean 전체를 평가하는 masked loss
```

기본 추천은 다음과 같다.

```text
training loss:
  ocean 영역 전체를 기본 도메인으로 사용
  target 결측과 land만 제외

evaluation:
  ocean 전체
  active_union
  month별 active mask
  세 기준을 모두 저장
```

## 11. CNN Model Plan

처음부터 복잡한 모델을 사용하지 않고 baseline CNN부터 시작한다.

추천 모델 단계는 다음과 같다.

```text
1. Simple CNN baseline
2. U-Net small
3. U-Net with residual blocks
4. Ensemble of selected models
```

기본 출력 범위는 `SIC` 특성상 0~1이다. 출력 전략은 다음 후보를 비교한다.

```text
1. linear output, loss는 raw prediction 기준
2. linear output, 평가/저장 시 clip to [0, 1]
3. sigmoid output
4. bounded ReLU 또는 clamp 계열 후처리
```

현재 1차 추천은 다음과 같다.

```text
학습:
  linear output

평가/저장:
  prediction_clipped = clip(prediction, 0, 1)

비교 실험:
  sigmoid output을 별도 실험으로 수행
```

`sigmoid`는 출력 범위를 직접 제한할 수 있지만, 0과 1에 값이 몰린 `SIC` 특성에서 극단값 학습이
둔해질 수 있다. `linear + clip`은 학습 안정성과 물리 범위 후처리를 분리하는 baseline으로
둔다.

## 12. Ensemble Plan

단일 CNN 결과가 불안정할 수 있으므로 앙상블을 고려한다. 단, 앙상블은 baseline CNN 검증 후
진행한다.

앙상블 후보는 다음과 같다.

```text
1. seed ensemble
2. architecture ensemble
3. activation strategy ensemble
4. checkpoint averaging
```

우선순위는 다음과 같다.

```text
1차: 단일 Simple CNN
2차: Simple CNN seed ensemble
3차: Simple CNN + U-Net small ensemble
```

앙상블 결과는 단일 모델 결과와 반드시 같은 test split에서 비교한다.

## 13. Training Plan

학습 스크립트는 다음 정보를 명시적으로 받아야 한다.

```text
device
seed
model_name
task_type
lead_months
output_strategy
batch_size
learning_rate
epochs
loss_mask_type
input_variables
target_variable
normalization_file
```

기본 device는 다음과 같다.

```text
cuda:3
```

학습 중 저장해야 할 항목은 다음과 같다.

```text
best checkpoint
last checkpoint
training log
validation metrics per epoch
model config
normalization stats
mask config
environment metadata
```

## 14. Proposed Scripts

구현 시 권장 스크립트 구조는 다음과 같다.

```text
01_check_monthly_files.py
02_validate_each_variable.py
03_build_cnn_normalization_stats.py
04_validate_cnn_masks.py
05_prepare_cnn_dataset_index.py
06_validate_tos_fill_strategies.py
cnn_torch_common.py
07_train_cnn_baseline.py
08_evaluate_cnn_model.py
09_visualize_cnn_predictions.py
10_train_cnn_ensemble.py
11_write_cnn_validation_report.py
```

각 스크립트 목적은 다음과 같다.

| File | Purpose |
|---|---|
| `01_check_monthly_files.py` | 입력 파일과 시간축 일치 검증 |
| `02_validate_each_variable.py` | 변수별 결측, 복원값, 유효 범위 검증 |
| `03_build_cnn_normalization_stats.py` | train 기준 normalization 통계 생성 |
| `04_validate_cnn_masks.py` | 학습/평가 mask 검증 |
| `05_prepare_cnn_dataset_index.py` | CNN sample index와 split metadata 생성 |
| `06_validate_tos_fill_strategies.py` | `tos` 결측 처리 후보 검증 |
| `cnn_torch_common.py` | PyTorch Dataset, CNN 모델, masked loss, 평가 metric 공통 함수 |
| `07_train_cnn_baseline.py` | 단일 CNN baseline 학습 및 checkpoint 저장 |
| `08_evaluate_cnn_model.py` | test 성능, 월별 성능, mask별 성능 평가 |
| `09_visualize_cnn_predictions.py` | 실제/예측/오차 SIC 지도 생성 |
| `10_train_cnn_ensemble.py` | 여러 checkpoint 예측 평균 기반 ensemble 평가 |
| `11_write_cnn_validation_report.py` | 약식 검증 보고서 작성 |

## 15. Evaluation

CNN 평가는 전체 공간 모델의 성능을 중심으로 수행한다.

기본 지표는 다음과 같다.

```text
RMSE
MAE
Bias
R2
masked RMSE
masked MAE
```

추가로 다음 단위별 평가를 저장한다.

```text
split별 성능
month별 성능
year별 성능
ocean 영역 성능
active_union 영역 성능
month별 active mask 영역 성능
7~10월 성능
저해빙 기준 후보 월 성능
```

이 CNN 실험의 목적은 6개 포인트 회귀모델과 직접 비교하는 것이 아니라, 전체 격자 `SIC`
예측 모델을 구축하는 것이다. 6개 NSR 포인트 값은 필요할 경우 해석용 샘플로 추출할 수 있지만,
주 평가축으로 두지 않는다.

## 16. Point Sampling Note

CNN 출력은 `448 x 304` 이산 격자이다. 만약 특정 지점의 예측값을 추출해야 한다면 기존
포인트 회귀분석과 동일한 격자 인덱스를 사용한다.

```text
Point 1: y=259, x=183
Point 2: y=262, x=171
Point 3: y=281, x=150
Point 4: y=286, x=146
Point 5: y=273, x=126
Point 6: y=277, x=94
```

이 경우 기본 방식은 `nearest grid index`이다. 위경도 기반 bilinear interpolation은 별도
분석 목적이 있을 때만 추가한다. 기존 포인트 회귀와 비교해야 할 경우에는 동일 인덱스 추출을
사용해야 한다.

## 17. Figures

시각화 산출물은 다음을 기본으로 한다.

```text
training_loss_curve.png
validation_metric_curve.png
test_actual_sic_YYYY_MM.png
test_predicted_sic_YYYY_MM.png
test_error_sic_YYYY_MM.png
monthly_rmse_heatmap.png
ensemble_vs_single_model_metrics.png
```

그림은 다음 위치에 저장한다.

```text
experiments/03_cnn_monthly/figures
```

## 18. Validation Report

모델 학습 후 약식 검증 보고서를 반드시 작성한다.

보고서 위치는 다음과 같다.

```text
experiments/03_cnn_monthly/reports
```

보고서에는 다음 항목을 포함한다.

```text
1. 실험 ID
2. 실행 날짜
3. GPU 환경
4. 입력 변수
5. target 변수
6. task_type과 lead_months
7. train/val/test 기간
8. normalization 방식
9. NaN 처리 방식
10. mask/loss 설정
11. 모델 구조
12. output strategy
13. best epoch
14. test metrics
15. 월별 성능 요약
16. 대표 예측 지도
17. 오차가 큰 월/영역
18. 단일 모델 vs ensemble 비교
19. 다음 실험에서 수정할 점
```

보고서는 모델 검증 결과를 빠르게 확인하기 위한 문서이며, 논문용 최종 보고서가 아니다.

## 19. Relation To Other Sections

기존 단계와의 관계는 다음과 같다.

```text
00_baseline_regression:
  6개 포인트 기반 Linear/Ridge baseline

01_data_diagnostics:
  원본 데이터 구조, 결측, 분포 진단

02_modeling:
  포인트 회귀모델 저장, 미래 SIC*, 로지스틱 확률화 계획

03_cnn_monthly:
  전체 월별 격자 픽셀 기반 CNN 예측 모델
```

CNN은 6개 포인트 회귀분석의 부속 평가가 아니라, 전체 격자 공간 패턴을 학습하는 별도 예측
모델이다.

## 20. Current Status

현재 상태는 README 하네스 작성 단계이다.

```text
완료:
  - CNN 전체 격자 실험 방향 정리
  - GPU 사용 환경 기록
  - nowcasting / forecasting 구분
  - 전처리 검증 스크립트 목적 정의
  - NaN 처리, loss mask, output activation 취약점 반영
  - CNN 모델/앙상블/보고서 구조 정의
  - 전처리 검증 스크립트 01~06 작성
  - 터널 GPU 서버에서 전처리 검증 스크립트 01~06 실행 확인
  - tos 3x3 valid ocean median 보간 검증 반영
  - CNN 학습/평가/시각화/앙상블/보고서 스크립트 07~11 작성

아직 하지 않음:
  - 터널 GPU 서버에서 07 학습 실행
  - checkpoint 생성
  - 08~11 평가, 시각화, ensemble, 검증 보고서 실행
```

다음 작업은 터널 GPU 서버에서 `07_train_cnn_baseline.py`를 실행해 1차 zero-fill baseline
checkpoint를 만든 뒤, 같은 split에서 `median3x3_min3` 전략을 별도 실험으로 학습해 비교하는
것이다.
