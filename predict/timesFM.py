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

# 사용 가능한 모델 체크포인트 목록
checkpoint_options = [
    "google/timesfm-1.0-base",
    "google/timesfm-1.0-1b", 
    "google/timesfm-2.0-500m",
    "google/timesfm-2.0-500m-pytorch"
]

# TimesFM 모델 초기화
for checkpoint_id in checkpoint_options:
    try:
        print(f"Trying checkpoint: {checkpoint_id}")
        
        # 체크포인트와 일치하는 파라미터로 설정
        hparams = timesfm.TimesFmHparams(
            backend="torch",
            per_core_batch_size=32,
            horizon_len=test_days,
            input_patch_len=64,
            output_patch_len=64,
            model_dims=1280,
            num_layers=20
        )
        
        tfm = timesfm.TimesFm(
            hparams=hparams,
            checkpoint=timesfm.TimesFmCheckpoint(
                huggingface_repo_id=checkpoint_id
            ),
        )
        
        print(f"Successfully initialized model with checkpoint: {checkpoint_id}")
        break
    except Exception as e:
        print(f"Error with checkpoint {checkpoint_id}: {e}")
else:
    # 모든 체크포인트가 실패한 경우 가장 기본적인 구성 시도
    try:
        print("Trying with minimal configuration...")
        hparams = timesfm.TimesFmHparams(
            backend="torch",
            horizon_len=test_days
        )
        
        tfm = timesfm.TimesFm(
            hparams=hparams,
            checkpoint=timesfm.TimesFmCheckpoint(
                huggingface_repo_id="google/timesfm-1.0-base"
            )
        )
    except Exception as e:
        print(f"Failed to initialize model: {e}")
        # 스크립트 종료 전 예외 정보 출력
        import traceback
        traceback.print_exc()
        # 스크립트 종료
        exit(1)

# 모델이 학습 시 사용할 수 있는 과거 예시 기간 설정
context_length = min(365, len(train_df) - 1)  # 최대 1년 또는 사용 가능한 데이터

# forecast_on_df API를 사용하여 예측 수행
try:
    forecast_df = tfm.forecast_on_df(
        inputs=train_df,
        freq="D",
        value_name="y",
        num_jobs=-1
    )

    # 학습 데이터의 마지막 날짜 이후의 예측 결과만 추출합니다.
    max_train_date = train_df['ds'].max()
    forecast_horizon = forecast_df[forecast_df['ds'] > max_train_date]

    # 요일 패턴 활용 - 예측 값 조정
    forecast_horizon['dayofweek'] = [d.dayofweek for d in forecast_horizon['ds']]
    forecast_horizon['is_weekend'] = forecast_horizon['dayofweek'].isin([5, 6]).astype(int)

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
    forecast_horizon['timesfm_adjusted'] = forecast_horizon.apply(
        lambda row: row['timesfm'] * dow_scale_factors.get(row['dayofweek'], 1.0), axis=1
    )

    # 시각화
    plt.figure(figsize=(14, 7))

    # 전체 실제 데이터 (학습 + 테스트)
    plt.plot(df_model['ds'], df_model['y'], label='Actual', marker='o', color='blue')

    # TimesFM 예측 결과 (학습 데이터 이후 예측)
    plt.plot(forecast_horizon['ds'], forecast_horizon['timesfm_adjusted'], 
             label='Forecast (Adjusted)', marker='x', linestyle='--', color='red')

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
    common_dates = set(test_df['ds']).intersection(set(forecast_horizon['ds']))
    test_common = test_df[test_df['ds'].isin(common_dates)]
    forecast_common = forecast_horizon[forecast_horizon['ds'].isin(common_dates)]

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
        
        print("\nPerformance Metrics (Original):")
        for k, v in metrics_original.items():
            print(f"{k}: {v:.2f}")
        
        print("\nPerformance Metrics (Adjusted):")
        for k, v in metrics_adjusted.items():
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
    image_path = os.path.join(current_dir, 'timesfm_forecast.png')
    # 이미지 저장
    plt.savefig(image_path, dpi=300, bbox_inches='tight')
    # 그래프 창 닫기
    plt.close()

    print(f"모델 예측 이미지가 저장되었습니다: {image_path}")

except Exception as e:
    print(f"Error during forecasting: {e}")
    import traceback
    traceback.print_exc()