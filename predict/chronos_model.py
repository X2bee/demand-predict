# This file has been renamed to chronos_model.py to avoid module name conflicts
# Please use chronos_model.py instead of this file

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from chronos import ChronosPipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 설정
forecast_length = 7  # 예측 길이(7일)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Chronos 모델 로드
print("Chronos 모델 로드 중...")
pipeline = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-large",
    device_map=device,
    torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
)

# 1. CSV 불러오기 및 전처리
print("데이터 로드 중...")
df = pd.read_csv("data_order_cnt.csv")
df['d_day'] = pd.to_datetime(df['d_day'], format='%Y%m%d')
df = df.sort_values('d_day')
print(f"원본 데이터 크기: {len(df)}")

# 마지막 7일을 테스트셋으로 분리
test_data = df.iloc[-forecast_length:].copy()
train_data = df.iloc[:-forecast_length].copy()

print(f"훈련 데이터 크기: {len(train_data)}")
print(f"테스트 데이터 크기: {len(test_data)}")

# Chronos는 텐서 입력을 받음
context = torch.tensor(train_data["total_order_cnt"].values, dtype=torch.float32)

# 2. 예측 수행
print("7일 예측 수행 중...")
forecast = pipeline.predict(context, forecast_length, num_samples=100)  # [num_series, num_samples, prediction_length]
print(f"예측 결과 shape: {forecast.shape}")

# 3. 결과 시각화를 위한 처리
forecast_index = range(len(train_data), len(train_data) + forecast_length)
low, median, high = np.quantile(forecast[0].cpu().numpy(), [0.1, 0.5, 0.9], axis=0)

# 4. 날짜 범위 생성
last_train_date = train_data['d_day'].iloc[-1]
forecast_dates = pd.date_range(start=last_train_date + pd.Timedelta(days=1), periods=forecast_length, freq='D')
test_actual = test_data['total_order_cnt'].values

# 5. 시각화
plt.figure(figsize=(15, 7))

# 전체 실제 데이터
plt.plot(df['d_day'], df['total_order_cnt'], marker='o', markersize=4, 
         color='royalblue', label='Historical Data', alpha=0.7)

# 테스트 데이터 강조 (실제값)
plt.plot(test_data['d_day'], test_actual, 
         marker='s', markersize=6, color='green', label='Test Actual')

# 예측값 (중앙값)
plt.plot(forecast_dates, median, 
         marker='x', markersize=8, linestyle='--', color='tomato', label='Median Forecast')

# 신뢰구간 표시
plt.fill_between(forecast_dates, low, high, 
                 color='tomato', alpha=0.3, label='80% Prediction Interval')

plt.title('Chronos Time Series Forecast (7-Day Prediction)')
plt.xlabel('Date')
plt.ylabel('Order Count')
plt.legend()
plt.grid(True, alpha=0.3)
plt.gcf().autofmt_xdate()  # 날짜 레이블 포맷 조정
plt.tight_layout()

# 이미지 저장
current_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(current_dir, 'chronos_forecast.png')
plt.savefig(image_path, dpi=300, bbox_inches='tight')
print(f"이미지가 저장되었습니다: {image_path}")

# 그래프 표시
plt.show()

# 6. 성능 평가 지표 계산
rmse = np.sqrt(mean_squared_error(test_actual, median))
mae = mean_absolute_error(test_actual, median)
mape = np.mean(np.abs((test_actual - median) / test_actual)) * 100
r2 = r2_score(test_actual, median)

print("\n성능 평가 지표:")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"MAPE: {mape:.2f}%")
print(f"R²: {r2:.4f}")

# 7. 예측값과 실제값 비교 테이블 출력
comparison_df = pd.DataFrame({
    'Date': forecast_dates,
    'Actual': test_actual,
    'Predicted (Median)': median,
    'Lower Bound (10%)': low,
    'Upper Bound (90%)': high,
    'Error': test_actual - median,
    'Error(%)': np.abs((test_actual - median) / test_actual) * 100
})
print("\n예측값과 실제값 비교:")
print(comparison_df.to_string())

print("예측 완료.")
