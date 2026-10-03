# Northern Sea Route SIC Modeling

북극항로 주요 지점의 월별 해빙 농도(`SIC`)를 예측하고, 이후 예측된 미래 `SIC*`를 이용해
저해빙 상태 발생 확률을 평가하기 위한 실험 프로젝트이다.

현재는 전체 격자를 한 번에 학습하지 않고, 북극항로 병목 후보 6개 지점의 단일 25 km 픽셀을
대상으로 단계별 실험을 진행한다.

## Project Structure

```text
C:\Regression_Analysis\해빙
├─ data                         # 원본 NetCDF 데이터, Git 제외
├─ experiments                  # 생성 데이터, 모델, 그림, 결과, Git 제외
├─ scripts
│  ├─ 00_baseline_regression    # 기존 1차 회귀분석 보관
│  ├─ 01_data_diagnostics       # 원본 데이터 점검 및 회귀 전 진단
│  ├─ 02_modeling               # 회귀모델 저장 및 로지스틱 확률화 계획
│  └─ 03_cnn_monthly            # 월별 격자 CNN 실험 설계
├─ .gitignore
└─ README.md
```

## Data Summary

- 공간 격자: 북극 극지투영 25 km 격자
- 배열 크기: `448 x 304`
- 월별 기간: `1988-02` ~ `2025-12`
- 월별 샘플 수: `455`
- 타깃 변수: `sic`
- 입력 변수 후보: `tos`, `tas`, `rlds`, `rsds`, `uas`, `vas`

## Target Points

| Point | Name | Grid `(y, x)` | Lat, Lon |
|---:|---|---:|---|
| 1 | Vilkitsky Strait | `(259, 183)` | `77.5332, 102.0426` |
| 2 | NW Laptev Sea | `(262, 171)` | `78.1418, 115.1593` |
| 3 | Sannikov Strait | `(281, 150)` | `74.4931, 137.9682` |
| 4 | Dmitry Laptev Strait | `(286, 146)` | `73.2940, 140.9061` |
| 5 | Central East Siberian Sea | `(273, 126)` | `74.9566, 159.8057` |
| 6 | Long Strait / Wrangel | `(277, 94)` | `70.1118, 178.1374` |

## Sections

자세한 내용은 각 단계 README에 분리했다.

- [00 Baseline Regression](scripts/00_baseline_regression/README.md)
- [01 Data Diagnostics](scripts/01_data_diagnostics/README.md)
- [02 Modeling](scripts/02_modeling/README.md)
- [03 CNN Monthly](scripts/03_cnn_monthly/README.md)

## Current Direction

현재 핵심 흐름은 다음과 같다.

```text
1. 원본 데이터 구조와 결측 상태 확인
2. 6개 포인트의 월별 물리값 테이블 생성
3. 회귀 전 분포, 정규성, 상관관계 진단
4. Linear/Ridge 회귀모델을 지점별로 저장
5. 회귀모델로 미래 SIC* 예측
6. 저해빙 상태 기준점 탐색
7. 미래 SIC*를 로지스틱 회귀로 확률화
8. 월별 격자 또는 patch 기반 CNN 비교 실험 설계
```

다음으로 먼저 해야 할 일은 로지스틱 회귀에 사용할 저해빙 상태의 기준점을 찾는 것이다.
기준점이 정해진 뒤에야 `0/1` 라벨, 로지스틱 입력값, 평가 지표를 확정할 수 있다.

## Git Policy

`data/`, `experiments/`, 그림 파일, 결과 파일, 모델 파일은 Git에 올리지 않는다. 코드와 문서만
추적한다.
