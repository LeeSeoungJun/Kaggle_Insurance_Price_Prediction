from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
from IPython.display import display
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
fonts = {f.name for f in font_manager.fontManager.ttflist}
for font in ['Malgun Gothic', 'AppleGothic', 'NanumGothic']:
    if font in fonts:
        plt.rcParams['font.family'] = font
        break
plt.rcParams['axes.unicode_minus'] = False
OUT = Path('outputs')
OUT.mkdir(exist_ok=True)
SEED = 2026

def preprocessing(frame):
    categorical = frame.select_dtypes(include=['object', 'string', 'category']).columns.tolist()
    numerical = [c for c in frame if c not in categorical]
    return ColumnTransformer([
        ('num', SimpleImputer(strategy='median', add_indicator=True), numerical),
        ('cat', Pipeline([('fill', SimpleImputer(strategy='constant', fill_value='Unknown')),
                          ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), categorical)
    ])

def save_submission(sample, ids, target, prediction):
    assert list(sample.columns) == [ids.name, target], '제출 열 확인 필요'
    assert len(ids) == len(prediction) == len(sample)
    assert sample[ids.name].astype(str).tolist() == ids.astype(str).tolist(), '제출 ID 순서 불일치'
    assert np.isfinite(prediction).all()
    result = sample.copy()
    result[target] = prediction
    result.to_csv(OUT / 'submission.csv', index=False)
    return result

# %% 데이터와 날짜 파생변수
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import root_mean_squared_log_error
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
sample = pd.read_csv('sample_submission.csv')
TARGET = 'Premium Amount'
assert train['id'].is_unique and test['id'].is_unique
assert train[TARGET].notna().all() and train[TARGET].ge(0).all()

def features(frame):
    result = frame.drop(columns=['id', TARGET], errors='ignore').copy()
    dates = pd.to_datetime(result.pop('Policy Start Date'), errors='coerce')
    result['policy_year'] = dates.dt.year
    result['policy_month'] = dates.dt.month
    result['policy_day'] = dates.dt.day
    return result

X, X_test = features(train), features(test)
assert X.columns.tolist() == X_test.columns.tolist()
y = train[TARGET]
display(train.isna().sum().rename('missing').to_frame())
print('학습 / 제출:', X.shape, X_test.shape)
X_fit, X_hold, y_fit, y_hold = train_test_split(X, y, test_size=.2, random_state=SEED)
# %% 학습 데이터 내부 선택용 분리. 최종 홀드아웃은 모델 선택에 사용하지 않는다.
X_sub, X_select, y_sub, y_select = train_test_split(X_fit, y_fit, test_size=.2, random_state=SEED)
models = {
    'Log-mean baseline': TransformedTargetRegressor(regressor=DummyRegressor(strategy='mean'), func=np.log1p, inverse_func=np.expm1),
    'Log-XGBoost': TransformedTargetRegressor(regressor=XGBRegressor(n_estimators=250, max_depth=5,
        learning_rate=.05, tree_method='hist', random_state=SEED, n_jobs=4), func=np.log1p, inverse_func=np.expm1),
    'Log-LightGBM': TransformedTargetRegressor(regressor=LGBMRegressor(n_estimators=250, learning_rate=.05,
        random_state=SEED, n_jobs=4, verbosity=-1), func=np.log1p, inverse_func=np.expm1),
}
rows = []
for name, model in models.items():
    pipe = Pipeline([('prep', preprocessing(X)), ('model', model)])
    pipe.fit(X_sub, y_sub)
    pred = np.maximum(pipe.predict(X_select), 0)
    rows.append({'model': name, 'selection_rmsle': root_mean_squared_log_error(y_select, pred)})
scores = pd.DataFrame(rows).sort_values('selection_rmsle')
display(scores)
scores.to_csv(OUT / 'model_comparison.csv', index=False)
# %% 미사용 홀드아웃 최종 평가
best_name = scores.iloc[0]['model']
best = Pipeline([('prep', preprocessing(X)), ('model', models[best_name])])
best.fit(X_fit, y_fit)
prediction = np.maximum(best.predict(X_hold), 0)
baseline = np.full(len(y_hold), np.expm1(np.log1p(y_fit).mean()))
metrics = {'selected_model': best_name, 'holdout_rmsle': root_mean_squared_log_error(y_hold, prediction),
           'baseline_rmsle': root_mean_squared_log_error(y_hold, baseline),
           'holdout_rmse': root_mean_squared_error(y_hold, prediction), 'split_seed': SEED}
print(json.dumps(metrics, ensure_ascii=False, indent=2))
(OUT / 'metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
errors = pd.DataFrame({'actual': y_hold, 'prediction': prediction})
errors['absolute_log_error'] = abs(np.log1p(errors.actual) - np.log1p(errors.prediction))
errors['premium_band'] = pd.qcut(y_hold, 5, duplicates='drop')
summary = errors.groupby('premium_band', observed=True).agg(n=('actual', 'size'), mean_log_error=('absolute_log_error', 'mean'))
display(summary)
summary.to_csv(OUT / 'errors_by_premium.csv')
errors.to_csv(OUT / 'holdout_predictions.csv', index=False)
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(np.log1p(y_hold), np.log1p(prediction) - np.log1p(y_hold), s=2, alpha=.05)
ax.axhline(0, color='black'); ax.set(xlabel='log1p(실제 보험료)', ylabel='로그 잔차', title='보험료 구간별 오차')
fig.tight_layout(); fig.savefig(OUT / 'residuals.png'); plt.show(); plt.close(fig)
# %% 전체 데이터 학습과 제출
best.fit(X, y)
submission = save_submission(sample, test['id'], TARGET, np.maximum(best.predict(X_test), 0))
display(submission.head())
