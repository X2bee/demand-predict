import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from neuralprophet import NeuralProphet
import logging
import os
import torch
from datetime import timedelta, datetime

# 로그 레벨 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('neuralprophet')
logger.setLevel(logging.WARNING)  # 경고 메시지만 표시

# 하드웨어 가속 확인
cuda_available = torch.cuda.is_available()
if cuda_available:
    print("CUDA is available, using GPU for acceleration")
    device = torch.device('cuda')
else:
    print("GPU not available, using CPU")
    device = 'cpu'

print(f"Using device: {device}")

# CSV 파일 읽기
df = pd.read_csv('data_order_cnt.csv')

# d_day 컬럼을 datetime 형식으로 변환 (YYYYMMDD 형식)
df['d_day'] = pd.to_datetime(df['d_day'], format='%Y%m%d')

# 날짜 기준 정렬
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

# 요일별 통계 출력
dow_stats = df.groupby('dayofweek')['total_order_cnt'].agg(['mean', 'median', 'std'])
print("\nDay of week statistics:")
print(dow_stats)

month_stats = df.groupby('month')['total_order_cnt'].agg(['mean', 'median', 'std'])
print("\nMonthly statistics:")
print(month_stats)

# NeuralProphet에서 사용하는 형식으로 컬럼명 변경: ds (날짜), y (대상값)
df_model = df[['d_day', 'total_order_cnt']].rename(columns={'d_day': 'ds', 'total_order_cnt': 'y'})

# 요일별 스케일 팩터 계산 (나중에 사용)
dow_scale_factors = {}
for day in range(7):
    day_avg = df[df['dayofweek'] == day]['total_order_cnt'].mean()
    overall_avg = df['total_order_cnt'].mean()
    if overall_avg > 0 and not np.isnan(day_avg):
        dow_scale_factors[day] = day_avg / overall_avg
    else:
        dow_scale_factors[day] = 1.0

print("\nDay of week scale factors:")
for day, factor in dow_scale_factors.items():
    day_name = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][day]
    print(f"{day_name}: {factor:.4f}")

# train/test split: 마지막 14일은 테스트셋으로 사용 (확장된 범위)
test_days = 14
train_df = df_model.iloc[:-test_days].copy()
test_df = df_model.iloc[-test_days:].copy()

print(f"\nTraining data: {len(train_df)} days")
print(f"Test data: {len(test_df)} days")

# 통합 모델: 앙상블 NeuralProphet
models = []
forecasts = []

# 모델 하이퍼파라미터 설정들 - 다양한 설정으로 앙상블 구성
configs = [
    # 기본 모델 - 주간, 연간 계절성
    {
        "name": "Base Model",
        "params": {
            "growth": "linear",
            "yearly_seasonality": True,
            "weekly_seasonality": True,
            "daily_seasonality": False,
            "seasonality_mode": "multiplicative",
            "n_changepoints": 25,
            "changepoints_range": 0.95
        },
        "epochs": 200
    },
    # 빠른 변화 감지 모델 - 더 많은 변화점
    {
        "name": "Change-sensitive Model",
        "params": {
            "growth": "linear",
            "yearly_seasonality": True,
            "weekly_seasonality": True,
            "daily_seasonality": False,
            "seasonality_mode": "multiplicative",
            "n_changepoints": 35,
            "changepoints_range": 0.9,
            "changepoint_prior_scale": 0.1
        },
        "epochs": 200
    },
    # 규제가 강한 모델 - 오버피팅 방지
    {
        "name": "Regularized Model",
        "params": {
            "growth": "linear",
            "yearly_seasonality": True,
            "weekly_seasonality": True,
            "seasonality_mode": "multiplicative",
            "n_changepoints": 20,
            "changepoints_range": 0.8,
            "regularization": 0.5
        },
        "epochs": 300
    }
]

print("\n=== Training Ensemble of NeuralProphet Models ===")

try:
    # 각 설정으로 모델 학습
    for i, config in enumerate(configs):
        print(f"\nTraining Model {i+1}: {config['name']}")
        
        try:
            if cuda_available:
                model = NeuralProphet(**config["params"], device=device)
            else:
                model = NeuralProphet(**config["params"])
                
            # 모델 학습
            metrics = model.fit(train_df, freq="D", epochs=config["epochs"], 
                               learning_rate=0.001, verbose=False)
            
            # 모델 및 설정 저장
            models.append({"model": model, "config": config})
            
            # 미래 예측 (훈련 + 테스트 기간)
            future = model.make_future_dataframe(train_df, periods=test_days)
            forecast = model.predict(future)
            
            # 저장
            forecasts.append(forecast)
            
            print(f"Successfully trained {config['name']}")
            
        except Exception as e:
            print(f"Error training {config['name']}: {e}")
            print("Trying with simplified parameters...")
            
            try:
                # 간소화된 모델로 재시도
                simplified_model = NeuralProphet(
                    yearly_seasonality=True,
                    weekly_seasonality=True
                )
                
                metrics = simplified_model.fit(train_df, freq="D", epochs=100, 
                                             learning_rate=0.005, verbose=False)
                
                models.append({"model": simplified_model, "config": {"name": f"Simplified {config['name']}"}})
                
                future = simplified_model.make_future_dataframe(train_df, periods=test_days)
                forecast = simplified_model.predict(future)
                forecasts.append(forecast)
                
                print(f"Successfully trained simplified model instead of {config['name']}")
                
            except Exception as e2:
                print(f"Could not train even simplified model: {e2}")
    
    # 앙상블 결과 계산
    if len(forecasts) > 0:
        # 테스트 기간에 대한 예측값만 추출
        ensemble_results = []
        
        for i, forecast in enumerate(forecasts):
            # 테스트 기간만 추출
            test_forecast = forecast.tail(test_days).copy()
            test_forecast['model'] = models[i]['config']['name']
            ensemble_results.append(test_forecast)
        
        # 앙상블 결과를 데이터프레임으로 변환
        ensemble_df = pd.concat(ensemble_results)
        
        # 날짜별로 그룹화하여 각 모델의 예측 평균 계산
        ensemble_avg = ensemble_df.groupby('ds')['yhat1'].mean().reset_index()
        ensemble_avg.rename(columns={'yhat1': 'ensemble_pred'}, inplace=True)
        
        # 요일 정보 추가
        ensemble_avg['dayofweek'] = ensemble_avg['ds'].dt.dayofweek
        
        # 요일 패턴 조정
        ensemble_avg['ensemble_adjusted'] = ensemble_avg.apply(
            lambda row: row['ensemble_pred'] * dow_scale_factors.get(row['dayofweek'], 1.0), 
            axis=1
        )
        
        # 성능 평가
        # 테스트 데이터와 예측 결과 병합
        eval_df = pd.merge(test_df, ensemble_avg, on='ds')
        
        # 메트릭 계산
        def calculate_metrics(actual, predicted):
            mape = np.mean(np.abs((actual - predicted) / np.maximum(1e-10, np.abs(actual)))) * 100
            rmse = np.sqrt(np.mean((actual - predicted) ** 2))
            mae = np.mean(np.abs(actual - predicted))
            return {'MAPE': mape, 'RMSE': rmse, 'MAE': mae}
        
        metrics_original = calculate_metrics(eval_df['y'], eval_df['ensemble_pred'])
        metrics_adjusted = calculate_metrics(eval_df['y'], eval_df['ensemble_adjusted'])
        
        print("\n=== Performance Metrics ===")
        print("Original Ensemble:")
        for k, v in metrics_original.items():
            print(f"  {k}: {v:.2f}")
        
        print("\nAdjusted Ensemble (Day of Week):")
        for k, v in metrics_adjusted.items():
            print(f"  {k}: {v:.2f}")
        
        # 시각화
        plt.figure(figsize=(14, 8))
        
        # 전체 실제 데이터
        plt.plot(df_model['ds'], df_model['y'], label='Actual Data', marker='o', 
                color='blue', alpha=0.5, markersize=4)
        
        # 각 개별 모델 예측 (투명도 낮게)
        for i, forecast in enumerate(forecasts):
            test_forecast = forecast.tail(test_days)
            plt.plot(test_forecast['ds'], test_forecast['yhat1'], 
                    linestyle='--', alpha=0.3, 
                    label=f"{models[i]['config']['name']}")
        
        # 앙상블 예측 결과
        plt.plot(ensemble_avg['ds'], ensemble_avg['ensemble_pred'], 
                label='Ensemble Prediction', color='green', 
                linestyle='--', linewidth=2)
        
        # 요일 조정된 앙상블 결과
        plt.plot(ensemble_avg['ds'], ensemble_avg['ensemble_adjusted'], 
                label='Day-Adjusted Ensemble', color='red', 
                linewidth=3)
        
        # 테스트 데이터 (실제값) 강조
        plt.plot(test_df['ds'], test_df['y'], label='Test Actual', 
                color='black', marker='s', linewidth=2)
        
        # 그래프 설정
        plt.title('Enhanced NeuralProphet Ensemble Forecast', fontsize=16)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Order Count', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best')
        plt.gcf().autofmt_xdate()
        
        # 성능 메트릭 표시
        metrics_text = (
            f"RMSE (Original): {metrics_original['RMSE']:.2f}\n"
            f"RMSE (Adjusted): {metrics_adjusted['RMSE']:.2f}\n\n"
            f"MAPE (Original): {metrics_original['MAPE']:.2f}%\n"
            f"MAPE (Adjusted): {metrics_adjusted['MAPE']:.2f}%\n\n"
            f"MAE (Original): {metrics_original['MAE']:.2f}\n"
            f"MAE (Adjusted): {metrics_adjusted['MAE']:.2f}\n"
        )
        
        plt.annotate(metrics_text, xy=(0.02, 0.02), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
                    fontsize=9)
        
        # 저장 및 출력
        current_dir = os.path.dirname(os.path.abspath(__file__))
        enhanced_image_path = os.path.join(current_dir, 'neuralprophet_enhanced_forecast.png')
        plt.savefig(enhanced_image_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n향상된 NeuralProphet 앙상블 예측 이미지가 저장되었습니다: {enhanced_image_path}")
        
        # 예측 결과 저장
        results_df = pd.DataFrame({
            'date': ensemble_avg['ds'],
            'actual': eval_df['y'] if len(eval_df) == len(ensemble_avg) else np.nan,
            'ensemble_prediction': ensemble_avg['ensemble_pred'],
            'day_adjusted_prediction': ensemble_avg['ensemble_adjusted'],
        })
        
        # 미래 예측 (다음 14일)
        last_date = df_model['ds'].max()
        next_dates = pd.date_range(start=last_date + timedelta(days=1), periods=14)
        
        future_forecasts = []
        for model_info in models:
            try:
                model = model_info['model']
                future = model.make_future_dataframe(df_model, periods=14)
                forecast = model.predict(future)
                future_forecast = forecast.tail(14).copy()
                future_forecast['model'] = model_info['config']['name']
                future_forecasts.append(future_forecast)
            except Exception as e:
                print(f"Error generating future forecast for {model_info['config']['name']}: {e}")
        
        if future_forecasts:
            future_ensemble = pd.concat(future_forecasts)
            future_avg = future_ensemble.groupby('ds')['yhat1'].mean().reset_index()
            future_avg.rename(columns={'yhat1': 'ensemble_pred'}, inplace=True)
            
            # 요일 정보 추가
            future_avg['dayofweek'] = future_avg['ds'].dt.dayofweek
            
            # 요일 패턴 조정
            future_avg['ensemble_adjusted'] = future_avg.apply(
                lambda row: row['ensemble_pred'] * dow_scale_factors.get(row['dayofweek'], 1.0), 
                axis=1
            )
            
            # 미래 예측 결과 저장
            future_results = pd.DataFrame({
                'date': future_avg['ds'],
                'ensemble_prediction': future_avg['ensemble_pred'],
                'day_adjusted_prediction': future_avg['ensemble_adjusted'],
            })
            
            # 결과 CSV 저장
            results_path = os.path.join(current_dir, 'neuralprophet_test_results.csv')
            results_df.to_csv(results_path, index=False)
            
            future_path = os.path.join(current_dir, 'neuralprophet_future_forecast.csv')
            future_results.to_csv(future_path, index=False)
            
            print(f"테스트 예측 결과가 CSV 파일로 저장되었습니다: {results_path}")
            print(f"미래 14일 예측 결과가 CSV 파일로 저장되었습니다: {future_path}")
            
            # 미래 예측 시각화
            plt.figure(figsize=(14, 8))
            
            # 전체 실제 데이터
            plt.plot(df_model['ds'], df_model['y'], label='Historical Data', 
                    color='blue', alpha=0.5, marker='o', markersize=4)
            
            # 앙상블 예측 결과
            plt.plot(future_avg['ds'], future_avg['ensemble_pred'], 
                    label='Ensemble Forecast', color='green', 
                    linestyle='--', linewidth=2)
            
            # 요일 조정된 앙상블 결과
            plt.plot(future_avg['ds'], future_avg['ensemble_adjusted'], 
                    label='Day-Adjusted Forecast', color='red', 
                    linewidth=3)
            
            # 그래프 설정
            plt.title('Future 14-Day Forecast', fontsize=16)
            plt.xlabel('Date', fontsize=12)
            plt.ylabel('Order Count', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.legend(loc='best')
            plt.gcf().autofmt_xdate()
            
            # 저장 및 출력
            future_image_path = os.path.join(current_dir, 'neuralprophet_future_forecast.png')
            plt.savefig(future_image_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"미래 예측 이미지가 저장되었습니다: {future_image_path}")
            
except Exception as e:
    print(f"Error in ensemble process: {e}")
    import traceback
    traceback.print_exc() 