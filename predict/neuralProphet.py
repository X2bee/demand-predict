import pandas as pd
from neuralprophet import NeuralProphet
import matplotlib.pyplot as plt
import logging
import os
import numpy as np
import requests
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

logging.basicConfig(level=logging.INFO)

def get_korean_holidays(year):
    try:
        api_key = "pBXJzQHZG14qbLav3sMIVFfAv3//XZ0+NlcS3k9pXDELFpz0HiJTGATZkTlIoAHRhnTjJXW8eyfsMQ9y3kTe9Q==" 
        url = f'http://apis.data.go.kr/B090041/openapi/service/SpcdeInfoService/getRestDeInfo'
        params = {
            'serviceKey': api_key,
            'solYear': year,
            'numOfRows': 100,
            '_type': 'json'
        }
        
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            json_data = response.json()
            
            if 'response' in json_data and 'body' in json_data['response'] and 'items' in json_data['response']['body']:
                items = json_data['response']['body']['items']
                
                if 'item' in items:
                    holidays = []
                    
                    if isinstance(items['item'], list):
                        for item in items['item']:
                            holidays.append(str(item['locdate']))
                    else:
                        holidays.append(str(items['item']['locdate']))
                    
                    return holidays
            
            print(f"API에서 휴일 데이터를 찾을 수 없습니다: {json_data}")
            return manual_holiday_list(year)
        
        else:
            print(f"API 요청 실패: {response.status_code}")
            return manual_holiday_list(year)
            
    except Exception as e:
        print(f"공휴일 데이터 가져오기 오류: {e}")
        return manual_holiday_list(year)

def manual_holiday_list(year):
    print(f"{year}년 공휴일 데이터를 수동으로 생성합니다.")
    
    holidays = []
    
    holidays.append(f"{year}0101")
    holidays.append(f"{year}0301")
    holidays.append(f"{year}0505")
    holidays.append(f"{year}0606")
    holidays.append(f"{year}0815")
    holidays.append(f"{year}1003")
    holidays.append(f"{year}1009")
    holidays.append(f"{year}1225")
    
    return holidays

df = pd.read_csv('../data_order_cnt.csv', index_col=0)

df['d_day'] = pd.to_datetime(df['d_day'], format='%Y%m%d')

df['dayofweek'] = df['d_day'].dt.dayofweek
df['is_weekend'] = df['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)

start_year = df['d_day'].min().year
end_year = df['d_day'].max().year + 1

print(f"데이터 기간: {start_year}년 ~ {end_year-1}년")

all_holidays = []
for year in range(start_year, end_year):
    holidays = get_korean_holidays(year)
    all_holidays.extend(holidays)

print(f"수집된 공휴일 수: {len(all_holidays)}")
print(f"공휴일 목록: {all_holidays}")

df['date_str'] = df['d_day'].dt.strftime('%Y%m%d')
df['is_holiday'] = df['date_str'].apply(lambda x: 1 if x in all_holidays else 0)
df.drop('date_str', axis=1, inplace=True)

df['is_offday'] = ((df['is_holiday'] == 1) | (df['is_weekend'] == 1)).astype(int)

df = df.sort_values('d_day')

weekday_avg = df[df['is_offday'] == 0]['total_order_cnt'].mean()
offday_avg = df[df['is_offday'] == 1]['total_order_cnt'].mean()
print(f"평일 평균 주문량: {weekday_avg:.2f}")
print(f"주말/휴일 평균 주문량: {offday_avg:.2f}")
print(f"주말/휴일 주문량 비율: {offday_avg/weekday_avg:.2%}")

# 데이터 스케일링을 위한 스케일러 생성
scaler = MinMaxScaler(feature_range=(0, 1))
# 주문량 데이터 스케일링
df['scaled_order_cnt'] = scaler.fit_transform(df[['total_order_cnt']])

# 원래 값 별도로 저장 (모델 데이터프레임에는 포함시키지 않음)
df_original = df.rename(columns={'d_day': 'ds', 'total_order_cnt': 'y_original'})
df_original = df_original[['ds', 'y_original']]

# 모델에 사용할 데이터 (스케일 변환된 데이터)
df_model = df.rename(columns={'d_day': 'ds', 'scaled_order_cnt': 'y'})
df_model = df_model[['ds', 'y']]

df_features = df_model.copy()
df_features['dayofweek'] = df['dayofweek']
df_features['is_weekend'] = df['is_weekend']
df_features['is_holiday'] = df['is_holiday']
df_features['is_offday'] = df['is_offday']

test_size = 7
train_df = df_model.iloc[:-test_size].copy()
test_df = df_model.iloc[-test_size:].copy()

train_features = df_features.iloc[:-test_size].copy()
test_features = df_features.iloc[-test_size:].copy()

# 원본 데이터도 같은 방식으로 분할
train_original = df_original.iloc[:-test_size].copy()
test_original = df_original.iloc[-test_size:].copy()

print(f"훈련 데이터 크기: {len(train_df)}")
print(f"테스트 데이터 크기: {len(test_df)}")
print(f"테스트 데이터 주말/휴일 수: {test_features['is_offday'].sum()}")

# 개선된 NeuralProphet 모델 생성 - 호환성 문제 해결
model = NeuralProphet(
    n_forecasts=test_size,
    n_lags=14,  # 2주 데이터로 감소 (과거 패턴에 덜 의존)
    daily_seasonality=False,  # 일별 계절성 패턴 비활성화
    weekly_seasonality=10,  # 주간 계절성 패턴 유지
    yearly_seasonality=False,  # 연간 계절성 비활성화
    learning_rate=0.01,  # 학습률 증가
    batch_size=8,  # 더 작은 배치로 변경
    normalize='minmax',  # minmax 정규화 유지
    n_changepoints=3,  # 변화점 줄임
    changepoints_range=0.8,  # 변화점 범위 줄임
    trend_reg=0.5,  # 추세 정규화 강화
    newer_samples_weight=5.0,  # 최근 데이터 가중치 대폭 증가
    loss_func='MAE',  # MAE 손실함수로 변경
    impute_missing=True,
    impute_rolling=3,  # 롤링 윈도우 감소
)

# 주말/휴일 정보 추가 (중요 외부 변수)
model.add_future_regressor(name='is_offday', regularization=0.05)
model.add_future_regressor(name='is_weekend', regularization=0.05)
model.add_future_regressor(name='is_holiday', regularization=0.05)

# regressor 변수 추가 (모든 변수 포함)
train_df_with_regressors = train_df.copy()
train_df_with_regressors['is_offday'] = train_features['is_offday']
train_df_with_regressors['is_weekend'] = train_features['is_weekend'] 
train_df_with_regressors['is_holiday'] = train_features['is_holiday']

# 모델 학습
metrics = model.fit(
    train_df_with_regressors,
    freq='D',
    epochs=500,  # 에포크 수 조정
    early_stopping=True,  # 조기 종료 유지
)

# 테스트 데이터 기간의 regressor 값 준비 (미래 예측을 위한 regressor 데이터프레임)
future_regressors = pd.DataFrame({
    'ds': test_df['ds'].values,
    'is_offday': test_features['is_offday'].values,
    'is_weekend': test_features['is_weekend'].values,
    'is_holiday': test_features['is_holiday'].values
})

# 예측할 미래 기간 생성 및 regressor 값 포함
future = model.make_future_dataframe(
    df=train_df_with_regressors,
    periods=test_size,
    n_historic_predictions=len(train_df),
    regressors_df=future_regressors
)

# 예측 수행
forecast = model.predict(future)

# 올바른 예측값 접근을 위해 수정
# 1-step 예측값만 사용하는 대신, 모든 forecast steps 사용
forecast_test = forecast[forecast['ds'].isin(test_df['ds'])]

# yhat1, yhat2, ... 컬럼들을 사용하여 각 날짜별 예측값 추출
predicted_values = []
predicted_dates = []

for i, date in enumerate(test_df['ds']):
    # 해당 날짜의 예측값 찾기 (여러 컬럼 중 해당 스텝의 예측값 사용)
    yhat_col = f'yhat{i+1}'
    if yhat_col in forecast.columns:
        # 첫 번째 행의 해당 컬럼값 가져오기 (첫 번째 예측 기간의 예측값)
        day_prediction = forecast[forecast['ds'] == date.to_pydatetime()][yhat_col].values
        if len(day_prediction) > 0:
            # 스케일 원복 및 음수 값 방지
            scaled_pred = max(0, day_prediction[0])  # 음수 값을 0으로 대체
            original_pred = scaler.inverse_transform([[scaled_pred]])[0][0]
            predicted_values.append(original_pred)
            predicted_dates.append(date)
        else:
            predicted_values.append(np.nan)
            predicted_dates.append(date)
    else:
        predicted_values.append(np.nan)
        predicted_dates.append(date)

# 원래 스케일로 변환된 실제 값
actual_values = test_original['y_original'].values

comparison_df = pd.DataFrame({
    'Date': predicted_dates,
    'Actual': actual_values,
    'Predicted': predicted_values,
    'Error': actual_values - np.array(predicted_values),
    'Error(%)': np.abs((actual_values - np.array(predicted_values)) / np.maximum(actual_values, 1)) * 100,
    'DayOfWeek': test_features['dayofweek'].values,
    'DayName': [d.day_name() for d in predicted_dates],
    'IsWeekend': test_features['is_weekend'].values,
    'IsHoliday': test_features['is_holiday'].values,
    'IsOffday': test_features['is_offday'].values
})

print(comparison_df)

plt.figure(figsize=(15, 7))

# 전체 실제 데이터 (train + test)
plt.plot(df_original['ds'], df_original['y_original'], label='Actual', color='blue', marker='o', markersize=4)

# 훈련 데이터 예측 (얇은 점선)
train_dates = forecast.iloc[:-test_size]['ds']
# 훈련 예측값 역변환
train_preds_scaled = forecast.iloc[:-test_size]['yhat1'].values
train_preds = []
for val in train_preds_scaled:
    # 음수 값 처리 및 역변환
    scaled_val = max(0, val)
    original_val = scaler.inverse_transform([[scaled_val]])[0][0]
    train_preds.append(original_val)

plt.plot(train_dates, train_preds, label='Historical Fit', color='green', linestyle=':', alpha=0.5)

# 테스트 데이터 (실제값) 강조
plt.plot(test_df['ds'], actual_values, label='Test Actual', color='forestgreen', marker='s',linestyle=':', linewidth=1, markersize=3)

# 예측값으로 올바른 값 사용
plt.plot(predicted_dates, predicted_values, 
         label='NeuralProphet Forecast', 
         color='magenta', marker='D', linestyle=':', linewidth=2, markersize=3)

# 주말/휴일 표시
holiday_dates = test_features[test_features['is_holiday'] == 1]['ds'].values
weekend_dates = test_features[test_features['is_weekend'] == 1]['ds'].values

if len(holiday_dates) > 0:
    plt.scatter(holiday_dates, [df_original['y_original'].max() * 1.05] * len(holiday_dates), 
               color='red', marker='v', s=100, label='Holidays')

if len(weekend_dates) > 0:
    plt.scatter(weekend_dates, [df_original['y_original'].max() * 1.02] * len(weekend_dates), 
               color='orange', marker='^', s=100, label='Weekends')

# 테스트 기간 시작 표시
plt.axvline(x=test_df['ds'].iloc[0], color='darkgray', linestyle='--', linewidth=2, label='Test Period Start')

# 그리드 설정
plt.grid(True, which='major', linestyle='-', linewidth=0.5, alpha=0.7)
plt.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.4)

# 축 레이블과 제목
plt.xlabel('Date', fontsize=12, fontweight='bold')
plt.ylabel('Total Order Count', fontsize=12, fontweight='bold')
plt.title('NeuralProphet Forecast', fontsize=16, fontweight='bold')

# 범례 설정
plt.legend(loc='upper right', fontsize=10, framealpha=0.9)

# 그래프 스타일 설정
plt.xticks(fontsize=10, rotation=45)
plt.yticks(fontsize=10)
plt.gcf().autofmt_xdate()
plt.tight_layout()

# 저장 경로
current_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(current_dir, 'neuralProphet_forecast.png')
plt.savefig(image_path, dpi=300, bbox_inches='tight')
print(f"이미지가 저장되었습니다: {image_path}")

# 예측 결과 추가 테이블 시각화
print("\n=== 예측 결과 상세 ===")
result_table = comparison_df.copy()
result_table['Actual'] = result_table['Actual'].round(1)
result_table['Predicted'] = result_table['Predicted'].round(1)
result_table['Error(%)'] = result_table['Error(%)'].round(1)
print(result_table[['Date', 'DayName', 'Actual', 'Predicted', 'Error(%)', 'IsWeekend', 'IsHoliday']])

plt.show()

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# NaN 제외하고 성능 평가
valid_indices = ~np.isnan(predicted_values)
rmse = np.sqrt(mean_squared_error(actual_values[valid_indices], np.array(predicted_values)[valid_indices])) if any(valid_indices) else np.nan
mae = mean_absolute_error(actual_values[valid_indices], np.array(predicted_values)[valid_indices]) if any(valid_indices) else np.nan
mape = np.mean(np.abs((actual_values[valid_indices] - np.array(predicted_values)[valid_indices]) / np.maximum(actual_values[valid_indices], 1))) * 100 if any(valid_indices) else np.nan
r2 = r2_score(actual_values[valid_indices], np.array(predicted_values)[valid_indices]) if any(valid_indices) else np.nan

print("\n예측 성능 평가 지표:")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"MAPE: {mape:.2f}%")
print(f"R²: {r2:.4f}")

print("\n요일별 오차 분석:")
for day in range(7):
    day_error = comparison_df[comparison_df['DayOfWeek'] == day]['Error(%)'].mean() if len(comparison_df[comparison_df['DayOfWeek'] == day]) > 0 else "N/A"
    day_name = comparison_df[comparison_df['DayOfWeek'] == day]['DayName'].iloc[0] if len(comparison_df[comparison_df['DayOfWeek'] == day]) > 0 else f"Day {day}"
    if day_error != "N/A" and not np.isnan(day_error):
        print(f"{day_name} 평균 오차율: {day_error:.2f}%")
    else:
        print(f"{day_name} 평균 오차율: N/A")

print("\n주말/휴일에 따른 오차 분석:")
weekday_errors = comparison_df[comparison_df['IsOffday'] == 0]['Error(%)'].mean() if len(comparison_df[comparison_df['IsOffday'] == 0]) > 0 else "N/A"
offday_errors = comparison_df[comparison_df['IsOffday'] == 1]['Error(%)'].mean() if len(comparison_df[comparison_df['IsOffday'] == 1]) > 0 else "N/A"

if weekday_errors != "N/A" and not np.isnan(weekday_errors):
    print(f"평일 평균 오차율: {weekday_errors:.2f}%")
else:
    print("평일 평균 오차율: N/A")

if offday_errors != "N/A" and not np.isnan(offday_errors):
    print(f"주말/휴일 평균 오차율: {offday_errors:.2f}%")
else:
    print("주말/휴일 평균 오차율: N/A")

print("\n예측값과 실제값 비교:")
print(comparison_df.to_string())

print("\n예측 완료!")