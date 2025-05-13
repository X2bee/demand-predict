import pandas as pd
import matplotlib.pyplot as plt
import timesfm
import os
import torch

# 디바이스 설정: GPU 대신 CPU로 고정 (TimesFM 인덱싱 오류 방지)
device = 'cpu'
torch.set_default_device('cpu')
print(f"Using device: {device}")

# CSV 파일 읽기 및 전처리
df = pd.read_csv('data_order_cnt.csv')
df['d_day'] = pd.to_datetime(df['d_day'], format='%Y%m%d')
df = df.sort_values('d_day')

# TimesFM이 요구하는 형식: unique_id, ds (날짜), y (타겟값)
df['unique_id'] = "T1"
df_model = df[['unique_id', 'd_day', 'total_order_cnt']].rename(
    columns={'d_day': 'ds', 'total_order_cnt': 'y'}
)

# Train/Test split: 마지막 7일은 테스트셋으로 사용
train_df = df_model.iloc[:-7].copy()
test_df = df_model.iloc[-7:].copy()

# TimesFM 모델 초기화 (500m 모델에 필요한 고정 파라미터 사용)
tfm = timesfm.TimesFm(
    hparams=timesfm.TimesFmHparams(
        backend="torch",             # PyTorch backend 사용
        per_core_batch_size=2,
        horizon_len=7,             # 예측 기간 7일
        input_patch_len=32,
        output_patch_len=128,
        num_layers=50,
        model_dims=1280,
        use_positional_embedding=True,
    ),
    checkpoint=timesfm.TimesFmCheckpoint(
        huggingface_repo_id="google/timesfm-2.0-500m-pytorch"
    ),
)

# forecast_on_df API를 사용하여 예측 수행
forecast_df = tfm.forecast_on_df(
    inputs=train_df,
    freq="D",
    value_name="y",
    num_jobs=-1,
)

# 학습 데이터의 마지막 날짜 이후의 예측 결과만 추출합니다.
max_train_date = train_df['ds'].max()
forecast_horizon = forecast_df[forecast_df['ds'] > max_train_date]

# 시각화
plt.figure(figsize=(12, 6))
plt.plot(df_model['ds'], df_model['y'], label='Actual', marker='o', color='blue')
plt.plot(forecast_horizon['ds'], forecast_horizon['timesfm'], label='Forecast', marker='x', linestyle='--', color='red')
plt.plot(test_df['ds'], test_df['y'], label='Test Actual', marker='s', linestyle='-', color='green')

plt.xlabel('Date')
plt.ylabel('Total Order Count')
plt.title('TimesFM Forecast vs Actual (Last 7 Days Test)')
plt.legend()
plt.gcf().autofmt_xdate()
plt.tight_layout()

# 이미지 저장
current_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(current_dir, 'timesfm_forecast.png')
plt.savefig(image_path, dpi=300, bbox_inches='tight')
print(f"이미지가 저장되었습니다: {image_path}")

# 그래프 보여주기
plt.show()
plt.close()
