import pandas as pd
from neuralprophet import NeuralProphet
import matplotlib.pyplot as plt
import logging
import os

# 로그 레벨 설정: DEBUG로 설정하면 상세 로그를 출력합니다.
logging.basicConfig(level=logging.DEBUG)

# CSV 파일 읽기
df = pd.read_csv('/data/data_order_cnt.csv')

# d_day 컬럼을 datetime 형식으로 변환 (YYYYMMDD 형식)
df['d_day'] = pd.to_datetime(df['d_day'], format='%Y%m%d')

# 날짜 기준 정렬
df = df.sort_values('d_day')

# NeuralProphet에서 사용하는 형식으로 컬럼명 변경: ds (날짜), y (대상값)
df_model = df[['d_day', 'total_order_cnt']].rename(columns={'d_day': 'ds', 'total_order_cnt': 'y'})

# train/test split: 마지막 7일은 테스트셋으로 사용
train_df = df_model.iloc[:-7]
test_df = df_model.iloc[-7:]

# NeuralProphet 모델 생성
model = NeuralProphet()

# 모델 학습 (verbose 인자는 제거합니다)
metrics = model.fit(train_df, freq='D', epochs=100)

# 미래 7일에 대한 예측 데이터프레임 생성 (훈련 구간 + 7일 예측)
future = model.make_future_dataframe(train_df, periods=7)
forecast = model.predict(future)

# forecast 결과는 전체 기간(훈련+예측)을 포함하므로, 테스트 기간(최근 7일)만 추출
forecast_test = forecast.tail(7)

# 전체 실제 데이터와 NeuralProphet 예측 결과(전체 예측값)를 함께 시각화
plt.figure(figsize=(12, 6))

# 전체 실제 데이터 (train + test)
plt.plot(df_model['ds'], df_model['y'], label='Actual', color='blue', marker='o')

# NeuralProphet 예측 (전체 기간, 점선)
plt.plot(forecast['ds'], forecast['yhat1'], label='Forecast', color='orange', linestyle='--')

# 테스트 영역의 예측값만 별도로 강조 (마커)
plt.plot(forecast_test['ds'], forecast_test['yhat1'], label='Forecast (Test)', color='red', 
         marker='x', linestyle='None', markersize=10)

# 테스트 구간 시작일에 수직선 추가
plt.axvline(x=test_df['ds'].iloc[0], color='gray', linestyle='--', label='Test Period Start')

plt.xlabel('Date')
plt.ylabel('Total Order Count')
plt.title('Actual vs Forecast (Entire Data and Last 7 Days Test)')
plt.legend()
plt.gcf().autofmt_xdate()
plt.tight_layout()
plt.show()

# 현재 스크립트의 디렉토리 경로
current_dir = os.path.dirname(os.path.abspath(__file__))
# 이미지 파일 저장 경로
image_path = os.path.join(current_dir, 'neuralProphet_forecast.png')
# 이미지 저장
plt.savefig(image_path, dpi=300, bbox_inches='tight')
# 그래프 창 닫기