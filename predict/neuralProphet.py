import pandas as pd
from neuralprophet import NeuralProphet
import matplotlib.pyplot as plt
import logging
import os
import numpy as np
import requests
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_korean_holidays(year):
    """
    공공 API를 통해 특정 연도의 한국 공휴일 목록을 가져옵니다.
    API 오류 발생 시 수동으로 생성된 공휴일 목록을 반환합니다.
    """
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
            
            logger.warning(f"API에서 휴일 데이터를 찾을 수 없습니다: {json_data}")
            return manual_holiday_list(year)
        
        else:
            logger.warning(f"API 요청 실패: {response.status_code}")
            return manual_holiday_list(year)
            
    except Exception as e:
        logger.error(f"공휴일 데이터 가져오기 오류: {e}")
        return manual_holiday_list(year)

def manual_holiday_list(year):
    """
    API 호출 실패 시 사용할 수동 공휴일 목록을 생성합니다.
    """
    logger.info(f"{year}년 공휴일 데이터를 수동으로 생성합니다.")
    
    holidays = [
        f"{year}0101",  # 신정
        f"{year}0301",  # 삼일절
        f"{year}0505",  # 어린이날
        f"{year}0606",  # 현충일
        f"{year}0815",  # 광복절
        f"{year}1003",  # 개천절
        f"{year}1009",  # 한글날
        f"{year}1225",  # 크리스마스
    ]
    
    return holidays

def load_data(file_path):
    """
    데이터를 로드하고 기본 전처리를 수행합니다.
    """
    df = pd.read_csv(file_path, index_col=0)
    df['d_day'] = pd.to_datetime(df['d_day'], format='%Y%m%d')
    df['dayofweek'] = df['d_day'].dt.dayofweek
    df['is_weekend'] = df['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)
    
    return df

def add_holiday_features(df):
    """
    데이터프레임에 공휴일 관련 특성을 추가합니다.
    """
    start_year = df['d_day'].min().year
    end_year = df['d_day'].max().year + 1

    logger.info(f"데이터 기간: {start_year}년 ~ {end_year-1}년")

    all_holidays = []
    for year in range(start_year, end_year):
        holidays = get_korean_holidays(year)
        all_holidays.extend(holidays)

    logger.info(f"수집된 공휴일 수: {len(all_holidays)}")
    
    df['date_str'] = df['d_day'].dt.strftime('%Y%m%d')
    df['is_holiday'] = df['date_str'].apply(lambda x: 1 if x in all_holidays else 0)
    df.drop('date_str', axis=1, inplace=True)
    
    # 주말 또는 휴일이면 1(off day), 아니면 0(근무일)
    df['is_offday'] = ((df['is_holiday'] == 1) | (df['is_weekend'] == 1)).astype(int)
    
    return df, all_holidays

def scale_data(df):
    """
    주문량 데이터를 스케일링합니다.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    df['scaled_order_cnt'] = scaler.fit_transform(df[['total_order_cnt']])
    
    return df, scaler

def prepare_model_data(df, test_size=14):
    """
    모델 학습을 위한 데이터를 준비합니다.
    """
    # 원래 값 별도로 저장
    df_original = df.rename(columns={'d_day': 'ds', 'total_order_cnt': 'y_original'})
    df_original = df_original[['ds', 'y_original']]

    # 모델에 사용할 데이터 (스케일 변환된 데이터)
    df_model = df.rename(columns={'d_day': 'ds', 'scaled_order_cnt': 'y'})
    df_model = df_model[['ds', 'y']]

    # 특성 데이터
    df_features = df_model.copy()
    df_features['dayofweek'] = df['dayofweek']
    df_features['is_weekend'] = df['is_weekend']
    df_features['is_holiday'] = df['is_holiday']
    df_features['is_offday'] = df['is_offday']

    # 훈련/테스트 데이터 분할
    df = df.sort_values('d_day')
    train_df = df_model.iloc[:-test_size].copy()
    test_df = df_model.iloc[-test_size:].copy()
    train_features = df_features.iloc[:-test_size].copy()
    test_features = df_features.iloc[-test_size:].copy()
    train_original = df_original.iloc[:-test_size].copy()
    test_original = df_original.iloc[-test_size:].copy()
    
    logger.info(f"훈련 데이터 크기: {len(train_df)}")
    logger.info(f"테스트 데이터 크기: {len(test_df)}")
    logger.info(f"테스트 데이터 주말/휴일 수: {test_features['is_offday'].sum()}")
    
    return train_df, test_df, train_features, test_features, train_original, test_original, df_original

def create_model(test_size):
    """
    NeuralProphet 모델을 생성합니다.
    """
    model = NeuralProphet(
        n_forecasts=test_size,
        n_lags=14,  # 2주 데이터
        daily_seasonality=False,  # 일별 계절성 패턴 비활성화
        weekly_seasonality=14,  # 주간 계절성
        yearly_seasonality=False,  # 연간 계절성 비활성화
        learning_rate=0.003,  # 학습률
        batch_size=8,  # 배치 사이즈
    )
    
    # 오직 is_offday만 외부 변수로 사용
    model.add_future_regressor(name='is_offday', regularization=0.1)
    
    return model

def train_model(model, train_df, train_features):
    """
    모델을 학습합니다.
    """
    # 모든 필요한 외부 변수를 포함하는 학습 데이터프레임 생성
    train_df_with_regressors = train_df.copy()
    train_df_with_regressors['is_offday'] = train_features['is_offday']
    
    metrics = model.fit(
        train_df_with_regressors,
        freq='D',
        epochs=300,
        early_stopping=True,
    )
    
    return model, metrics, train_df_with_regressors  # 외부 변수가 포함된 데이터프레임도 반환

def predict(model, train_df, test_df, test_features):
    """
    학습된 모델로 예측을 수행합니다.
    """
    # 테스트 데이터 기간의 regressor 값 준비
    future_regressors = pd.DataFrame({
        'ds': test_df['ds'].values,
        'is_offday': test_features['is_offday'].values
    })
    
    # 예측할 미래 기간 생성
    # train_df에 외부 변수 추가 (is_offday가 포함된 데이터프레임 사용)
    train_df_with_regressors = train_df.copy()
    if 'is_offday' not in train_df.columns and 'is_offday' in test_features.columns:
        train_df_with_regressors['is_offday'] = 0  # 기본값으로 채움 (실제 값은 future_regressors에서 제공)
    
    future = model.make_future_dataframe(
        df=train_df_with_regressors,
        periods=len(test_df),
        n_historic_predictions=len(train_df),
        regressors_df=future_regressors
    )
    
    # 예측 수행
    forecast = model.predict(future)
    
    return forecast, future_regressors

def process_predictions(forecast, test_df, test_original, scaler):
    """
    예측 결과를 처리하고 원래 스케일로 변환합니다.
    """
    predicted_values = []
    predicted_dates = []

    for i, date in enumerate(test_df['ds']):
        # 해당 날짜의 예측값 찾기
        yhat_col = f'yhat{i+1}'
        if yhat_col in forecast.columns:
            day_prediction = forecast[forecast['ds'] == date.to_pydatetime()][yhat_col].values
            if len(day_prediction) > 0:
                # 스케일 원복 및 음수 값 방지
                scaled_pred = max(0, day_prediction[0])
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
    
    return predicted_values, predicted_dates, actual_values

def create_comparison_df(predicted_dates, actual_values, predicted_values, test_features):
    """
    예측값과 실제값을 비교하는 데이터프레임을 생성합니다.
    """
    comparison_df = pd.DataFrame({
        'Date': predicted_dates,
        'Actual': actual_values,
        'Predicted': predicted_values,
        'Error': actual_values - np.array(predicted_values),
        'Error(%)': np.abs((actual_values - np.array(predicted_values)) / np.maximum(actual_values, 1)) * 100,
        'DayOfWeek': test_features['dayofweek'].values,
        'DayName': [d.day_name() for d in predicted_dates],
        'IsOffday': test_features['is_offday'].values
    })
    
    return comparison_df

def plot_forecast(df_original, forecast, test_df, test_features, actual_values, predicted_dates, predicted_values, test_size, scaler):
    """
    예측 결과를 시각화합니다.
    """
    plt.figure(figsize=(15, 7))

    # 전체 실제 데이터 (train + test)
    plt.plot(df_original['ds'], df_original['y_original'], label='Actual', color='blue', marker='o', markersize=4)

    # 훈련 데이터 예측 (얇은 점선)
    train_dates = forecast.iloc[:-test_size]['ds']
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

    # 주말/휴일(off days) 표시
    offday_dates = test_features[test_features['is_offday'] == 1]['ds'].values

    if len(offday_dates) > 0:
        plt.scatter(offday_dates, [df_original['y_original'].max() * 1.05] * len(offday_dates), 
                   color='red', marker='v', s=100, label='Off Days (Weekend/Holiday)')

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
    logger.info(f"이미지가 저장되었습니다: {image_path}")
    
    return image_path

def evaluate_model(actual_values, predicted_values, comparison_df):
    """
    모델 성능을 평가합니다.
    """
    # NaN 제외하고 성능 평가
    valid_indices = ~np.isnan(predicted_values)
    rmse = np.sqrt(mean_squared_error(actual_values[valid_indices], np.array(predicted_values)[valid_indices])) if any(valid_indices) else np.nan
    mae = mean_absolute_error(actual_values[valid_indices], np.array(predicted_values)[valid_indices]) if any(valid_indices) else np.nan
    mape = np.mean(np.abs((actual_values[valid_indices] - np.array(predicted_values)[valid_indices]) / np.maximum(actual_values[valid_indices], 1))) * 100 if any(valid_indices) else np.nan
    r2 = r2_score(actual_values[valid_indices], np.array(predicted_values)[valid_indices]) if any(valid_indices) else np.nan

    logger.info("\n예측 성능 평가 지표:")
    logger.info(f"RMSE: {rmse:.2f}")
    logger.info(f"MAE: {mae:.2f}")
    logger.info(f"MAPE: {mape:.2f}%")
    logger.info(f"R²: {r2:.4f}")

    logger.info("\n요일별 오차 분석:")
    for day in range(7):
        day_error = comparison_df[comparison_df['DayOfWeek'] == day]['Error(%)'].mean() if len(comparison_df[comparison_df['DayOfWeek'] == day]) > 0 else "N/A"
        day_name = comparison_df[comparison_df['DayOfWeek'] == day]['DayName'].iloc[0] if len(comparison_df[comparison_df['DayOfWeek'] == day]) > 0 else f"Day {day}"
        if day_error != "N/A" and not np.isnan(day_error):
            logger.info(f"{day_name} 평균 오차율: {day_error:.2f}%")
        else:
            logger.info(f"{day_name} 평균 오차율: N/A")

    logger.info("\n근무일/휴무일에 따른 오차 분석:")
    weekday_errors = comparison_df[comparison_df['IsOffday'] == 0]['Error(%)'].mean() if len(comparison_df[comparison_df['IsOffday'] == 0]) > 0 else "N/A"
    offday_errors = comparison_df[comparison_df['IsOffday'] == 1]['Error(%)'].mean() if len(comparison_df[comparison_df['IsOffday'] == 1]) > 0 else "N/A"

    if weekday_errors != "N/A" and not np.isnan(weekday_errors):
        logger.info(f"근무일 평균 오차율: {weekday_errors:.2f}%")
    else:
        logger.info("근무일 평균 오차율: N/A")

    if offday_errors != "N/A" and not np.isnan(offday_errors):
        logger.info(f"휴무일(주말/휴일) 평균 오차율: {offday_errors:.2f}%")
    else:
        logger.info("휴무일(주말/휴일) 평균 오차율: N/A")
    
    return rmse, mae, mape, r2

def print_results(comparison_df):
    """
    예측 결과를 출력합니다.
    """
    # 예측 결과 추가 테이블 시각화
    logger.info("\n=== 예측 결과 상세 ===")
    result_table = comparison_df.copy()
    result_table['Actual'] = result_table['Actual'].round(1)
    result_table['Predicted'] = result_table['Predicted'].round(1)
    result_table['Error(%)'] = result_table['Error(%)'].round(1)
    print(result_table[['Date', 'DayName', 'Actual', 'Predicted', 'Error(%)', 'IsOffday']])
    
    return result_table

# 메인 실행 부분
if __name__ == "__main__":
    # 1. 데이터 로드
    df = load_data('../data_order_cnt.csv')
    
    # 2. 공휴일 정보 추가
    df, all_holidays = add_holiday_features(df)
    
    # 3. 데이터 통계 출력
    weekday_avg = df[df['is_offday'] == 0]['total_order_cnt'].mean()
    offday_avg = df[df['is_offday'] == 1]['total_order_cnt'].mean()
    logger.info(f"평일 평균 주문량: {weekday_avg:.2f}")
    logger.info(f"주말/휴일 평균 주문량: {offday_avg:.2f}")
    logger.info(f"주말/휴일 주문량 비율: {offday_avg/weekday_avg:.2%}")
    
    # 4. 데이터 스케일링
    df, scaler = scale_data(df)
    
    # 5. 모델 데이터 준비 - df_original 추가 반환 받음
    test_size = 14
    train_df, test_df, train_features, test_features, train_original, test_original, df_original = prepare_model_data(df, test_size)
    
    # 6. 모델 생성
    model = create_model(test_size)
    
    # 7. 모델 학습
    model, metrics, train_df_with_regressors = train_model(model, train_df, train_features)
    
    # 8. 예측 수행 - train_df 대신 train_df_with_regressors 사용
    forecast, future_regressors = predict(model, train_df_with_regressors, test_df, test_features)
    
    # 9. 예측 결과 처리
    predicted_values, predicted_dates, actual_values = process_predictions(forecast, test_df, test_original, scaler)
    
    # 10. 비교 데이터프레임 생성
    comparison_df = create_comparison_df(predicted_dates, actual_values, predicted_values, test_features)
    print(comparison_df)
    
    # 11. 결과 시각화 - scaler 추가 전달
    image_path = plot_forecast(df_original, forecast, test_df, test_features, actual_values, predicted_dates, 
                             predicted_values, test_size, scaler)
    
    # 12. 모델 평가
    rmse, mae, mape, r2 = evaluate_model(actual_values, predicted_values, comparison_df)
    
    # 13. 결과 출력
    result_table = print_results(comparison_df)
    
    # 14. 그래프 표시
    plt.show()
    
    # 15. 예측값과 실제값 비교 출력
    logger.info("\n예측값과 실제값 비교:")
    print(comparison_df.to_string())
    
    logger.info("\n예측 완료!")