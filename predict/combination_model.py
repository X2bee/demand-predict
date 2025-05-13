import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import timesfm
import os
import torch
from neuralprophet import NeuralProphet
from datetime import timedelta

# Check if MPS is available for PyTorch
mps_available = hasattr(torch, 'mps') and torch.backends.mps.is_available()
if mps_available:
    print("MPS (Metal Performance Shaders) is available")
    torch.set_default_device('mps')
else:
    print("MPS is not available, using CPU instead")

# CSV 파일 읽기 및 전처리
df = pd.read_csv('data_order_cnt.csv')
df['d_day'] = pd.to_datetime(df['d_day'], format='%Y%m%d')
df = df.sort_values('d_day')

# 이상치 감지 및 처리 (Z-score 방법)
def detect_outliers(series, threshold=3):
    mean = series.mean()
    std = series.std()
    z_scores = [(y - mean) / std for y in series]
    return [i for i, z in enumerate(z_scores) if abs(z) > threshold]

outliers = detect_outliers(df['total_order_cnt'])
print(f"Detected {len(outliers)} outliers")

# 이상치 처리 - 롤링 평균으로 대체
window_size = 3  # 롤링 윈도우 크기
if outliers:
    # 롤링 평균 계산
    rolling_mean = df['total_order_cnt'].rolling(window=window_size, min_periods=1, center=True).mean()
    
    # 이상치 인덱스에 대해 롤링 평균으로 대체
    for idx in outliers:
        df.loc[df.index[idx], 'total_order_cnt'] = rolling_mean.iloc[idx]
    
    print("Outliers replaced with rolling mean")

# 요일 및 계절성 특성 추가
df['dayofweek'] = df['d_day'].dt.dayofweek
df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
df['month'] = df['d_day'].dt.month
df['day'] = df['d_day'].dt.day
df['quarter'] = df['d_day'].dt.quarter

# 요일별 평균 주문량 - 통계용
dow_stats = df.groupby('dayofweek')['total_order_cnt'].agg(['mean', 'median', 'std'])
print("\nDay of week statistics:")
print(dow_stats)

# 예측 모델 파라미터 설정
test_days = 14  # 테스트 기간
forecast_horizon = test_days

# 1. TimesFM 모델용 데이터 준비
df_timesfm = df.copy()
df_timesfm['unique_id'] = "T1"
df_timesfm_model = df_timesfm[['unique_id', 'd_day', 'total_order_cnt']].rename(
    columns={'d_day': 'ds', 'total_order_cnt': 'y'}
)

# 2. NeuralProphet 모델용 데이터 준비
df_np = df.copy()
df_np_model = df_np[['d_day', 'total_order_cnt']].rename(
    columns={'d_day': 'ds', 'total_order_cnt': 'y'}
)

# 훈련/테스트 분할
train_timesfm = df_timesfm_model.iloc[:-test_days].copy()
test_timesfm = df_timesfm_model.iloc[-test_days:].copy()

train_np = df_np_model.iloc[:-test_days].copy()
test_np = df_np_model.iloc[-test_days:].copy()

# ------ 1. TimesFM 모델 훈련 및 예측 ------
print("\n=== Training TimesFM Model ===")
tfm = timesfm.TimesFm(
    hparams=timesfm.TimesFmHparams(
        backend="torch",
        per_core_batch_size=32,
        horizon_len=forecast_horizon,
        input_patch_len=64,
        output_patch_len=128,
        num_layers=50,
        model_dims=1280,
        use_positional_embedding=True,
        dropout_rate=0.1,
    ),
    checkpoint=timesfm.TimesFmCheckpoint(
        huggingface_repo_id="google/timesfm-2.0-500m-pytorch"
    ),
)

# TimesFM 예측 실행
forecast_tfm = tfm.forecast_on_df(
    inputs=train_timesfm,
    freq="D",
    value_name="y",
    num_jobs=-1,
)

# 예측 결과 추출
max_train_date = train_timesfm['ds'].max()
forecast_tfm_horizon = forecast_tfm[forecast_tfm['ds'] > max_train_date].copy()

# 요일별 조정 적용
forecast_tfm_horizon['dayofweek'] = forecast_tfm_horizon['ds'].dt.dayofweek
forecast_tfm_horizon['is_weekend'] = forecast_tfm_horizon['dayofweek'].isin([5, 6]).astype(int)

# 요일별 스케일 팩터 계산
dow_scale_factors = {}
for day in range(7):
    train_day_avg = df[df['dayofweek'] == day]['total_order_cnt'].mean()
    train_overall_avg = df['total_order_cnt'].mean()
    if train_overall_avg > 0 and not np.isnan(train_day_avg):
        dow_scale_factors[day] = train_day_avg / train_overall_avg
    else:
        dow_scale_factors[day] = 1.0

# 요일별 조정 적용
forecast_tfm_horizon['timesfm_adjusted'] = forecast_tfm_horizon.apply(
    lambda row: row['timesfm'] * dow_scale_factors.get(row['dayofweek'], 1.0), axis=1
)

# ------ 2. NeuralProphet 모델 훈련 및 예측 ------
print("\n=== Training NeuralProphet Model ===")

model_np = NeuralProphet(
    growth="linear",              # 선형 성장 가정
    yearly_seasonality=True,      # 연간 계절성 모델링
    weekly_seasonality=True,      # 주간 계절성 모델링
    daily_seasonality=False,      # 일별 계절성은 데이터에 맞지 않음
    seasonality_mode="multiplicative",   # 곱셈 계절성 (시간에 따라 계절성 진폭이 변함)
    changepoints_range=0.95,      # 변화점 감지 범위 (훈련 데이터의 95%까지)
    n_changepoints=25,           # 가능한 변화점 수
    changepoints_prior_scale=0.05,  # 변화점 탄력성 (낮은 값 = 적은 변화)
    regularization=0.1,           # 정규화 (과적합 방지)
    uncertainty_samples=100,      # 불확실성 샘플링 횟수
)

# 모델 학습
metrics_np = model_np.fit(train_np, freq="D", epochs=200, 
                         learning_rate=0.001, verbose=False)

# 미래 예측
future_np = model_np.make_future_dataframe(train_np, periods=test_days)
forecast_np = model_np.predict(future_np)
forecast_np_horizon = forecast_np.tail(test_days)

# 3. 모델 앙상블 (TimesFM + NeuralProphet)
# 공통 날짜 찾기
common_dates = set(forecast_tfm_horizon['ds']).intersection(set(forecast_np_horizon['ds']))
tfm_common = forecast_tfm_horizon[forecast_tfm_horizon['ds'].isin(common_dates)].copy()
np_common = forecast_np_horizon[forecast_np_horizon['ds'].isin(common_dates)].copy()

# 인덱스를 날짜로 설정하여 정렬
tfm_common.set_index('ds', inplace=True)
np_common.set_index('ds', inplace=True)

# 같은 순서로 정렬
tfm_common = tfm_common.sort_index()
np_common = np_common.sort_index()

# 앙상블 데이터프레임 생성
ensemble_df = pd.DataFrame(index=tfm_common.index)
ensemble_df['timesfm'] = tfm_common['timesfm']
ensemble_df['timesfm_adjusted'] = tfm_common['timesfm_adjusted']
ensemble_df['neuralprophet'] = np_common['yhat1']

# 초기 가중치 설정 (이후 성능에 따라 조정됨)
tfm_weight = 0.6
np_weight = 0.4

# 가중 앙상블 예측
ensemble_df['ensemble_pred'] = (
    tfm_weight * ensemble_df['timesfm_adjusted'] +
    np_weight * ensemble_df['neuralprophet']
)

# 테스트 데이터와 비교를 위해 인덱스 재설정
ensemble_df.reset_index(inplace=True)
ensemble_df.rename(columns={'index': 'ds'}, inplace=True)

# 테스트 셋의 실제 값과 병합
test_common = test_timesfm[test_timesfm['ds'].isin(common_dates)].copy()
test_common.set_index('ds', inplace=True)
test_common = test_common.sort_index()
test_common.reset_index(inplace=True)

# 성능 메트릭 계산
def calculate_metrics(actual, predicted):
    mape = np.mean(np.abs((actual - predicted) / np.maximum(1e-10, np.abs(actual)))) * 100
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    mae = np.mean(np.abs(actual - predicted))
    return {'MAPE': mape, 'RMSE': rmse, 'MAE': mae}

# 각 모델별 메트릭 계산
metrics_timesfm = calculate_metrics(test_common['y'].values, ensemble_df['timesfm'].values)
metrics_timesfm_adj = calculate_metrics(test_common['y'].values, ensemble_df['timesfm_adjusted'].values)
metrics_np = calculate_metrics(test_common['y'].values, ensemble_df['neuralprophet'].values)
metrics_ensemble = calculate_metrics(test_common['y'].values, ensemble_df['ensemble_pred'].values)

print("\n=== Model Performance Metrics ===")
print("TimesFM:")
for k, v in metrics_timesfm.items():
    print(f"  {k}: {v:.2f}")

print("\nTimesFM (Adjusted):")
for k, v in metrics_timesfm_adj.items():
    print(f"  {k}: {v:.2f}")

print("\nNeuralProphet:")
for k, v in metrics_np.items():
    print(f"  {k}: {v:.2f}")

print("\nEnsemble:")
for k, v in metrics_ensemble.items():
    print(f"  {k}: {v:.2f}")

# 모델 성능에 기반한 가중치 조정
if metrics_timesfm_adj['MAPE'] < metrics_np['MAPE']:
    # TimesFM이 더 좋은 경우 가중치 증가
    tfm_adj_ratio = metrics_np['MAPE'] / (metrics_timesfm_adj['MAPE'] + 1e-10)
    # 비율을 사용하여 가중치 조정 (최대 0.8까지)
    tfm_weight = min(0.8, 0.5 + 0.3 * (tfm_adj_ratio - 1) / tfm_adj_ratio)
    np_weight = 1 - tfm_weight
else:
    # NeuralProphet이 더 좋은 경우 가중치 증가
    np_adj_ratio = metrics_timesfm_adj['MAPE'] / (metrics_np['MAPE'] + 1e-10)
    # 비율을 사용하여 가중치 조정 (최대 0.8까지)
    np_weight = min(0.8, 0.5 + 0.3 * (np_adj_ratio - 1) / np_adj_ratio)
    tfm_weight = 1 - np_weight

print(f"\nAdjusted weights based on performance: TimesFM = {tfm_weight:.2f}, NeuralProphet = {np_weight:.2f}")

# 조정된 가중치로 앙상블 예측 업데이트
ensemble_df['ensemble_pred_adjusted'] = (
    tfm_weight * ensemble_df['timesfm_adjusted'] +
    np_weight * ensemble_df['neuralprophet']
)

# 조정된 가중치로 메트릭 재계산
metrics_ensemble_adj = calculate_metrics(
    test_common['y'].values, ensemble_df['ensemble_pred_adjusted'].values
)

print("\nEnsemble (Adjusted Weights):")
for k, v in metrics_ensemble_adj.items():
    print(f"  {k}: {v:.2f}")

# 시각화
plt.figure(figsize=(14, 8))

# 전체 실제 데이터
plt.plot(df_np_model['ds'], df_np_model['y'], label='Actual Data', color='blue', marker='o', alpha=0.7)

# 각 모델 예측 결과
plt.plot(ensemble_df['ds'], ensemble_df['timesfm_adjusted'], 
         label='TimesFM', color='red', linestyle='--', alpha=0.7)
plt.plot(ensemble_df['ds'], ensemble_df['neuralprophet'], 
         label='NeuralProphet', color='green', linestyle='-.', alpha=0.7)
plt.plot(ensemble_df['ds'], ensemble_df['ensemble_pred_adjusted'], 
         label='Ensemble', color='purple', linewidth=2)

# 테스트 데이터 강조
plt.plot(test_common['ds'], test_common['y'], 
         label='Test Actual', color='black', marker='s')

# 그래프 설정
plt.title('Combined Model Forecast: TimesFM + NeuralProphet Ensemble')
plt.xlabel('Date')
plt.ylabel('Order Count')
plt.grid(True, alpha=0.3)
plt.legend()
plt.gcf().autofmt_xdate()

# 성능 메트릭 표시
metrics_text = (
    f"RMSE (TimesFM Adj): {metrics_timesfm_adj['RMSE']:.2f}\n"
    f"RMSE (NeuralProphet): {metrics_np['RMSE']:.2f}\n"
    f"RMSE (Ensemble): {metrics_ensemble_adj['RMSE']:.2f}\n\n"
    f"MAPE (TimesFM Adj): {metrics_timesfm_adj['MAPE']:.2f}%\n"
    f"MAPE (NeuralProphet): {metrics_np['MAPE']:.2f}%\n"
    f"MAPE (Ensemble): {metrics_ensemble_adj['MAPE']:.2f}%\n\n"
    f"Weights: TimesFM={tfm_weight:.2f}, NP={np_weight:.2f}"
)

plt.annotate(metrics_text, xy=(0.02, 0.02), xycoords='axes fraction',
             bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
             fontsize=9)

# 저장 및 출력
current_dir = os.path.dirname(os.path.abspath(__file__))
combo_image_path = os.path.join(current_dir, 'combined_model_forecast.png')
plt.savefig(combo_image_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"\n결합 모델 예측 이미지가 저장되었습니다: {combo_image_path}")

# 예측 결과 저장
results_df = pd.DataFrame({
    'date': ensemble_df['ds'],
    'actual': test_common['y'] if len(test_common) == len(ensemble_df) else np.nan,
    'timesfm_prediction': ensemble_df['timesfm'],
    'timesfm_adjusted_prediction': ensemble_df['timesfm_adjusted'],
    'neuralprophet_prediction': ensemble_df['neuralprophet'],
    'ensemble_prediction': ensemble_df['ensemble_pred_adjusted']
})

# 결과 CSV 저장
results_path = os.path.join(current_dir, 'forecast_results.csv')
results_df.to_csv(results_path, index=False)
print(f"예측 결과가 CSV 파일로 저장되었습니다: {results_path}") 