# 02 Modeling

이 폴더는 앞으로 회귀모델 저장, 미래 `SIC*` 예측, 로지스틱 회귀 기반 저해빙 확률 계산을
진행하기 위한 모델링 단계이다.

현재는 폴더 구조와 계획만 준비되어 있으며, 모델 학습 스크립트는 아직 작성하지 않았다.

## Model Storage

모델링 산출물은 다음 위치에 저장한다.

```text
experiments/03_modeling
├─ models
├─ figures
├─ results
└─ metadata
```

| Folder | Purpose |
|---|---|
| `models` | 학습된 회귀모델과 로지스틱 회귀모델 저장 |
| `figures` | 모델 예측 결과와 확률 결과 시각화 |
| `results` | metrics, predictions, labels 같은 표 결과 |
| `metadata` | feature 목록, 제외 변수, 기준점 정의, split 정보 저장 |

## Regression Plan

기존 00번 실험처럼 먼저 해석 가능한 선형 계열 모델부터 시작한다.

```text
Linear Regression
Ridge Regression
```

6개 지점에 대해 2개 모델을 학습하면 총 12개 회귀모델이 된다.

```text
6 points x 2 models = 12 regression models
```

회귀모델의 역할은 미래 입력 변수로부터 연속형 `SIC*` 값을 예측하는 것이다.

```text
기후 변수 X -> Linear/Ridge model -> 미래 SIC*
```

4번 지점은 `tos`가 전체 결측이므로, 해당 포인트 모델에서는 `tos` 제외 여부를 반드시
metadata에 기록한다.

## Logistic Regression Plan

로지스틱 회귀는 `SIC` 농도 자체를 맞추는 모델이 아니다. 회귀모델에서 나온 미래 `SIC*` 또는
`SIC` 기반 입력을 사용해, 특정 기준을 만족하는 저해빙 상태의 확률을 계산하는 모델이다.

개념 흐름은 다음과 같다.

```text
1. Linear/Ridge regression
   기후 변수 -> 미래 SIC*

2. Logistic regression
   미래 SIC* 또는 SIC 기반 입력 -> 저해빙 상태 확률
```

회귀모델과 로지스틱 회귀모델은 평가 기준이 다르다.

| Model | Output | Main Metrics |
|---|---|---|
| Linear/Ridge regression | continuous `SIC*` | RMSE, MAE, R2, Bias |
| Logistic regression | probability of low-ice state | Accuracy, Precision, Recall, F1, ROC-AUC, Brier score |

## Next Step

바로 로지스틱 회귀를 학습하지 않는다. 먼저 로지스틱 회귀에 사용할 저해빙 상태의 기준점을
찾아야 한다.

현재 다음 순서를 계획한다.

```text
1. Linear/Ridge 회귀모델 저장 구조 확정
2. 지점별 회귀모델 학습 및 저장
3. 미래 SIC* 예측값 생성
4. 저해빙 상태 판정을 위한 SIC 기준점 탐색
5. 기준점 기반 0/1 라벨 정의
6. 로지스틱 회귀모델 학습
7. 미래 SIC*에 대한 저해빙 상태 확률 계산
8. 특정 지점과 월에서 저해빙 상태가 나타나는 시점 탐색
```

현재 단계의 핵심 결론은 다음과 같다.

```text
기준점 탐색이 먼저이고,
그 다음에 로지스틱 회귀 라벨과 모델을 확정한다.
```
