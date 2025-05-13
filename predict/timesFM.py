import pandas as pd
import matplotlib.pyplot as plt
import timesfm
import os
import torch
import numpy as np
from datetime import timedelta

# Check for available hardware acceleration
cuda_available = torch.cuda.is_available()
mps_available = hasattr(torch, 'mps') and torch.backends.mps.is_available()

if cuda_available:
    print("CUDA is available, using GPU for acceleration")
    device = 'cuda'
    # Set PyTorch to use CUDA
    torch.set_default_device('cuda')
elif mps_available:
    print("MPS (Metal Performance Shaders) is available, using Apple Silicon GPU")
    device = 'mps'
    # Set PyTorch to use MPS
    torch.set_default_device('mps')
else:
    print("No GPU acceleration available, using CPU instead")
    device = 'cpu'

print(f"Using device: {device}")

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
    # 롤링 평균 계산 (이상치가 롤링 평균에 영향을 주지 않도록 제외)
    rolling_mean = df['total_order_cnt'].rolling(window=window_size, min_periods=1, center=True).mean()
    
    # 이상치 인덱스에 대해 롤링 평균으로 대체
    for idx in outliers:
        df.loc[df.index[idx], 'total_order_cnt'] = rolling_mean.iloc[idx]
    
    print("Outliers replaced with rolling mean")

# 추가 특성 엔지니어링
df['dayofweek'] = df['d_day'].dt.dayofweek
df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
df['month'] = df['d_day'].dt.month
df['day'] = df['d_day'].dt.day

# 요일 및 월별 평균 주문량 계산 (참조용)
dayofweek_avg = df.groupby('dayofweek')['total_order_cnt'].mean()
month_avg = df.groupby('month')['total_order_cnt'].mean()

print("Day of week average order counts:")
print(dayofweek_avg)
print("\nMonthly average order counts:")
print(month_avg)

# TimesFM이 요구하는 형식: unique_id, ds (날짜), y (타겟값)
df['unique_id'] = "T1"
df_model = df[['unique_id', 'd_day', 'total_order_cnt']].rename(
    columns={'d_day': 'ds', 'total_order_cnt': 'y'}
)

# Train/Test split: 마지막 14일은 테스트셋으로 사용 (기존 7일에서 증가)
test_days = 14
train_df = df_model.iloc[:-test_days].copy()
test_df = df_model.iloc[-test_days:].copy()

# TimesFM 모델 초기화 (지원되는 파라미터만 사용)
tfm = timesfm.TimesFm(
    hparams=timesfm.TimesFmHparams(
        backend="torch",             # PyTorch backend 사용
        per_core_batch_size=32,
        horizon_len=test_days,       # 예측 기간 14일로 증가
        input_patch_len=64,          # 더 긴 입력 패치 사용
        output_patch_len=128,
        num_layers=50,
        model_dims=1280,
        use_positional_embedding=True,  # 포지셔널 임베딩 활성화
        # 지원되지 않는 매개변수는 제거: learning_rate, dropout_rate
    ),
    checkpoint=timesfm.TimesFmCheckpoint(
        huggingface_repo_id="google/timesfm-2.0-500m-pytorch"
    ),
)

# 모델이 학습 시 사용할 수 있는 과거 예시 기간 설정 (더 길게 설정)
context_length = min(365, len(train_df) - 1)  # 최대 1년 또는 사용 가능한 데이터

# forecast_on_df API를 사용하여 예측 수행
forecast_df = tfm.forecast_on_df(
    inputs=train_df,
    freq="D",
    value_name="y",
    num_jobs=-1,
    # 지원되지 않는 매개변수는 제거: training_months
)

# 앙상블 예측 시도 (다양한 horizon_len 결과 평균화)
horizon_lens = [test_days, test_days + 7]
ensemble_forecasts = []

for horizon in horizon_lens:
    print(f"Forecasting with horizon_len={horizon}")
    tfm_horizon = timesfm.TimesFm(
        hparams=timesfm.TimesFmHparams(
            backend="torch",
            per_core_batch_size=32,
            horizon_len=horizon,
            input_patch_len=64,
            output_patch_len=128,
            num_layers=50,
            model_dims=1280,
            use_positional_embedding=True,
        ),
        checkpoint=timesfm.TimesFmCheckpoint(
            huggingface_repo_id="google/timesfm-2.0-500m-pytorch"
        ),
    )
    
    fcst = tfm_horizon.forecast_on_df(
        inputs=train_df,
        freq="D",
        value_name="y",
        num_jobs=-1,
    )
    
    # 학습 데이터의 마지막 날짜 이후의 예측 결과만 추출
    max_train_date = train_df['ds'].max()
    horizon_forecast = fcst[fcst['ds'] > max_train_date]
    ensemble_forecasts.append(horizon_forecast)

# 학습 데이터의 마지막 날짜 이후의 예측 결과만 추출합니다.
max_train_date = train_df['ds'].max()
forecast_horizon = forecast_df[forecast_df['ds'] > max_train_date]

# 앙상블 결과 계산 (사용 가능한 경우)
if len(ensemble_forecasts) > 1:
    # 모든 예측의 공통 날짜 찾기
    common_dates = set(forecast_horizon['ds'])
    for ef in ensemble_forecasts:
        common_dates = common_dates.intersection(set(ef['ds']))
    
    # 공통 날짜에 대한 앙상블 예측값 계산
    forecast_horizon_ensemble = forecast_horizon[forecast_horizon['ds'].isin(common_dates)].copy()
    
    # 각 모델의 예측값을 합산하여 평균 계산
    for i, ef in enumerate(ensemble_forecasts):
        ef_common = ef[ef['ds'].isin(common_dates)]
        if i == 0:
            forecast_horizon_ensemble['ensemble'] = ef_common['timesfm']
        else:
            forecast_horizon_ensemble['ensemble'] += ef_common['timesfm']
    
    # 평균 계산
    forecast_horizon_ensemble['ensemble'] /= (len(ensemble_forecasts) + 1)  # 원래 예측 포함
    forecast_horizon_ensemble['ensemble'] = (forecast_horizon_ensemble['ensemble'] + 
                                            forecast_horizon_ensemble['timesfm']) / 2
else:
    forecast_horizon_ensemble = forecast_horizon
    forecast_horizon_ensemble['ensemble'] = forecast_horizon_ensemble['timesfm']

# 요일 패턴 활용 - 예측 값 조정
all_dates = pd.date_range(start=max_train_date + timedelta(days=1), 
                         periods=len(forecast_horizon_ensemble))
forecast_horizon_ensemble['dayofweek'] = [d.dayofweek for d in forecast_horizon_ensemble['ds']]
forecast_horizon_ensemble['is_weekend'] = forecast_horizon_ensemble['dayofweek'].isin([5, 6]).astype(int)

# 요일별 스케일 팩터 계산 (학습 데이터 기반)
dow_scale_factors = {}
for day in range(7):
    train_day_avg = df[df['dayofweek'] == day]['total_order_cnt'].mean()
    train_overall_avg = df['total_order_cnt'].mean()
    if train_overall_avg > 0 and not np.isnan(train_day_avg):
        dow_scale_factors[day] = train_day_avg / train_overall_avg
    else:
        dow_scale_factors[day] = 1.0

# 요일별 스케일 팩터 적용
forecast_horizon_ensemble['timesfm_adjusted'] = forecast_horizon_ensemble.apply(
    lambda row: row['timesfm'] * dow_scale_factors.get(row['dayofweek'], 1.0), axis=1
)

forecast_horizon_ensemble['ensemble_adjusted'] = forecast_horizon_ensemble.apply(
    lambda row: row['ensemble'] * dow_scale_factors.get(row['dayofweek'], 1.0), axis=1
)

# 시각화
plt.figure(figsize=(14, 7))

# 전체 실제 데이터 (학습 + 테스트)
plt.plot(df_model['ds'], df_model['y'], label='Actual', marker='o', color='blue')

# TimesFM 예측 결과 (학습 데이터 이후 예측)
plt.plot(forecast_horizon_ensemble['ds'], forecast_horizon_ensemble['timesfm_adjusted'], 
         label='Forecast (Adjusted)', marker='x', linestyle='--', color='red')

# 앙상블 결과도 표시
plt.plot(forecast_horizon_ensemble['ds'], forecast_horizon_ensemble['ensemble_adjusted'], 
         label='Ensemble Forecast', marker='+', linestyle='-.', color='purple')

# 테스트셋의 실제값
plt.plot(test_df['ds'], test_df['y'], label='Test Actual', marker='s', linestyle='-', color='green')

plt.xlabel('Date')
plt.ylabel('Total Order Count')
plt.title('TimesFM Forecast vs Actual (Last {} Days Test)'.format(test_days))
plt.legend()
plt.gcf().autofmt_xdate()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

# 예측 성능 평가
def calculate_metrics(actual, predicted):
    mape = np.mean(np.abs((actual - predicted) / np.maximum(1e-10, np.abs(actual)))) * 100
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    mae = np.mean(np.abs(actual - predicted))
    return {'MAPE': mape, 'RMSE': rmse, 'MAE': mae}

# 테스트 데이터와 예측의 공통 날짜 찾기
common_dates = set(test_df['ds']).intersection(set(forecast_horizon_ensemble['ds']))
test_common = test_df[test_df['ds'].isin(common_dates)]
forecast_common = forecast_horizon_ensemble[forecast_horizon_ensemble['ds'].isin(common_dates)]

if not test_common.empty and not forecast_common.empty:
    # 원래 예측에 대한 측정
    metrics_original = calculate_metrics(
        test_common['y'].values, 
        forecast_common['timesfm'].values
    )
    
    # 조정된 예측에 대한 측정
    metrics_adjusted = calculate_metrics(
        test_common['y'].values, 
        forecast_common['timesfm_adjusted'].values
    )
    
    # 앙상블 예측에 대한 측정
    metrics_ensemble = calculate_metrics(
        test_common['y'].values, 
        forecast_common['ensemble_adjusted'].values
    )
    
    print("\nPerformance Metrics (Original):")
    for k, v in metrics_original.items():
        print(f"{k}: {v:.2f}")
    
    print("\nPerformance Metrics (Adjusted):")
    for k, v in metrics_adjusted.items():
        print(f"{k}: {v:.2f}")
    
    print("\nPerformance Metrics (Ensemble):")
    for k, v in metrics_ensemble.items():
        print(f"{k}: {v:.2f}")
    
    # 텍스트로 그래프에 메트릭스 표시
    metrics_text = (f"MAPE (Adjusted): {metrics_adjusted['MAPE']:.2f}%\n"
                   f"RMSE (Adjusted): {metrics_adjusted['RMSE']:.2f}\n"
                   f"MAE (Adjusted): {metrics_adjusted['MAE']:.2f}")
    
    plt.annotate(metrics_text, xy=(0.02, 0.02), xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
                fontsize=9)

# 현재 스크립트의 디렉토리 경로
current_dir = os.path.dirname(os.path.abspath(__file__))
# 이미지 파일 저장 경로
image_path = os.path.join(current_dir, 'timesfm_forecast_enhanced.png')
# 이미지 저장
plt.savefig(image_path, dpi=300, bbox_inches='tight')
# 그래프 창 닫기
plt.close()

print(f"향상된 모델 이미지가 저장되었습니다: {image_path}")