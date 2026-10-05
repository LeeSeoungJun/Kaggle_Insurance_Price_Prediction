# Insurance Premium Prediction

Kaggle **Playground Series - Season 4, Episode 12** 데이터를 활용해 고객 정보를 기반으로 보험료(`Premium Amount`)를 예측한 회귀 프로젝트입니다.

🔗 [Kaggle Competition](https://www.kaggle.com/competitions/playground-series-s4e12/overview)

## 프로젝트 목표

고객의 나이, 소득, 건강 상태, 신용 점수, 보험 이력 등의 데이터를 활용해 보험료를 예측하는 머신러닝 모델을 구축했습니다.

## 분석 과정

### 1. 데이터 전처리
- 결측치 처리
  - 수치형: 평균값 대체
  - 범주형: 최빈값 대체
- `Policy Start Date`에서 연도, 월, 일 등 날짜 파생변수 생성
- `StandardScaler`를 활용한 수치형 변수 표준화
- `OrdinalEncoder`를 활용한 범주형 변수 인코딩

### 2. 데이터 분석
- 주요 수치형 변수 간 상관관계 분석
- 직업 등 범주형 변수와 보험료 간 관계 시각화
- OLS 회귀 분석을 활용한 변수 영향 확인

### 3. 모델링
- Train / Validation 데이터를 8:2 비율로 분리
- `XGBRegressor`를 활용해 보험료 예측 모델 학습

### 4. 평가

Kaggle 공식 평가 지표는 **RMSLE**이며, 모델 학습 과정에서는 Validation 데이터 기준 RMSE도 확인했습니다.

```text
Validation RMSE: 843.78
```

## 사용 기술

`Python` `Pandas` `NumPy` `Matplotlib` `Seaborn`  
`Scikit-learn` `XGBoost` `Statsmodels`

## Notebook

전체 EDA, 전처리 및 모델링 과정은  
`Insurance_regression.ipynb`에서 확인할 수 있습니다.
