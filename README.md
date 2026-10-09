# 보험료 예측

## 분석 질문과 방법

RMSLE에 맞춰 log1p 보험료 모델과 로그 평균 기준 모델을 비교합니다. 학습 80% 내부 선택용 분리로 모델을 고르고 미사용 20% 홀드아웃을 평가합니다. 결측·범주형 처리는 Pipeline 안에서 학습하며 테스트에 같은 기준을 적용합니다. 명목 범주의 임의 순서를 OLS 계수로 해석하지 않습니다. 원본 전체 120만 행을 사용합니다. 이 자료의 관계를 실제 보험료의 인과 효과로 해석하지 않습니다.

## 실행

Python 3.11 이상에서 저장소 폴더를 작업 디렉터리로 사용합니다.

```bash
python -m pip install -r requirements.txt
python analysis.py
```

[Insurance_regression.ipynb](Insurance_regression.ipynb)에서 실행 결과와 그래프를 확인할 수 있습니다.
원본 데이터 경로는 기존 저장소와 동일합니다. 주가 프로젝트만 최초 실행 시 Yahoo Finance 연결이 필요합니다.

## 결과와 한계

실제 실행 결과는 `outputs/metrics.json`과 `outputs/`의 비교표·그래프에 저장됩니다.
수정 전 저장된 점수는 새 검증 결과와 혼용하지 않습니다. 검증 점수는 대회 리더보드 점수가 아닙니다.
모델을 정한 뒤 제출 데이터 전체를 예측하며, 제출 파일을 만들었다는 사실이 대회에 제출했다는 의미는 아닙니다.
분석에서 확인한 관계와 제안은 실제 업무 개선 효과를 증명하지 않습니다.

## 재실행 결과 (2026-10-10)

```json
{
  "selected_model": "Log-LightGBM",
  "holdout_rmsle": 1.0487463495642597,
  "baseline_rmsle": 1.097964437014376,
  "holdout_rmse": 921.4959586817513,
  "split_seed": 2026
}
```
