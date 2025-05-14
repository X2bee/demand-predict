import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from transformers import set_seed, EarlyStoppingCallback, Trainer, TrainingArguments
import requests
import math
from datetime import datetime
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import Subset
from tsfm_public import (
    TimeSeriesPreprocessor, TinyTimeMixerForPrediction,
    TimeSeriesForecastingPipeline, ForecastDFDataset,
    TrackingCallback, count_parameters
)
from tsfm_public.toolkit.lr_finder import optimal_lr_finder
from tsfm_public.toolkit.time_series_preprocessor import prepare_data_splits
from tsfm_public.toolkit.util import select_by_timestamp
from tsfm_public.toolkit.visualization import plot_predictions

# 설정
context_length = 90  # 더 긴 컨텍스트 사용 (원래 40)
forecast_length = 7  # 예측 길이(7일)
device = "cuda" if torch.cuda.is_available() else "cpu"
set_seed(1234)

# 한국 공휴일 API 호출 함수
def get_korean_holidays(year):
    """
    공공데이터포털 한국천문연구원 특일 정보 API를 사용하여 지정된 연도의 공휴일 목록을 가져옵니다.
    API 키는 https://www.data.go.kr/data/15012690/openapi.do 에서 발급받을 수 있습니다.
    """
    try:
        # API 키를 발급받아 아래에 입력해야 합니다
        api_key = "pBXJzQHZG14qbLav3sMIVFfAv3//XZ0+NlcS3k9pXDELFpz0HiJTGATZkTlIoAHRhnTjJXW8eyfsMQ9y3kTe9Q==" 
        # API 호출
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
            
            # API 응답 구조에 따른 데이터 추출
            if 'response' in json_data and 'body' in json_data['response'] and 'items' in json_data['response']['body']:
                items = json_data['response']['body']['items']
                
                if 'item' in items:
                    holidays = []
                    
                    # 여러 항목인 경우 리스트, 단일 항목인 경우 딕셔너리로 반환될 수 있음
                    if isinstance(items['item'], list):
                        for item in items['item']:
                            holidays.append(str(item['locdate']))
                    else:
                        # 단일 항목인 경우
                        holidays.append(str(items['item']['locdate']))
                    
                    return holidays
        
        else:
            print(f"API 요청 실패: {response.status_code}")
            
    except Exception as e:
        print(f"공휴일 데이터 가져오기 오류: {e}")

# 1. CSV 불러오기 및 전처리
df = pd.read_csv("../data_order_cnt.csv")
df['d_day'] = pd.to_datetime(df['d_day'], format='%Y%m%d')
df['state_id'] = 'T1'

# 요일 정보 추가 (주말/평일 패턴을 학습하기 위함)
df['dayofweek'] = df['d_day'].dt.dayofweek  # 0=월요일, 6=일요일
df['is_weekend'] = df['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)  # 주말 여부

# 요일별 원-핫 인코딩 추가 (요일 패턴을 더 명확하게 잡기 위함)
for i in range(7):
    df[f'day_{i}'] = (df['dayofweek'] == i).astype(int)

# 공휴일 정보 추가
# 데이터의 시작 연도와 끝 연도 구하기
start_year = df['d_day'].min().year
end_year = df['d_day'].max().year

# 각 연도별 공휴일 목록 가져오기
all_holidays = []
for year in range(start_year, end_year + 1):
    holidays = get_korean_holidays(year)
    all_holidays.extend(holidays)

print(f"수집된 공휴일 수: {len(all_holidays)}")
print(f"공휴일 목록: {all_holidays}")

# 날짜를 yyyymmdd 형식의 문자열로 변환하여 공휴일 여부 확인
df['date_str'] = df['d_day'].dt.strftime('%Y%m%d')
df['is_holiday'] = df['date_str'].apply(lambda x: 1 if x in all_holidays else 0)
df.drop('date_str', axis=1, inplace=True)

# 해당 날짜가 공휴일이거나 주말인 경우 1, 아닌 경우 0
df['is_offday'] = ((df['is_holiday'] == 1) | (df['is_weekend'] == 1)).astype(int)

# 주문량의 로그 변환 (큰 변동성을 줄이기 위함)
df['log_sales'] = np.log1p(df['total_order_cnt'])

# 주말/휴일의 주문량 감소를 강조하기 위한 특성 추가
df['sales_x_offday'] = df['total_order_cnt'] * (1 - df['is_offday']*0.8)

print("공휴일 데이터 처리 완료")
df = df.rename(columns={"d_day": "date", "total_order_cnt": "sales"})

# 필수 컬럼과 추가 특성 컬럼 선택
select_columns = ['date', 'state_id', 'sales', 'log_sales', 'sales_x_offday', 'dayofweek', 
                 'is_weekend', 'is_holiday', 'is_offday']
# 요일별 원-핫 인코딩 컬럼 추가
for i in range(7):
    select_columns.append(f'day_{i}')

df = df[select_columns]
df = df.sort_values('date')  # 날짜순으로 정렬

print(f"원본 데이터 크기: {len(df)}")

# 데이터 시각화 - 원본 데이터의 주말/휴일 패턴 확인
plt.figure(figsize=(15, 7))
plt.plot(df['date'], df['sales'], label='Sales')

# 주말 표시
weekend_dates = df[df['is_weekend'] == 1]['date']
if len(weekend_dates) > 0:
    plt.scatter(weekend_dates, df[df['is_weekend'] == 1]['sales'], 
               color='orange', marker='^', s=50, label='Weekends')

# 공휴일 표시
holiday_dates = df[df['is_holiday'] == 1]['date']
if len(holiday_dates) > 0:
    plt.scatter(holiday_dates, df[df['is_holiday'] == 1]['sales'], 
               color='red', marker='v', s=80, label='Holidays')

plt.title('Sales Data with Weekend and Holiday Patterns')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('sales_patterns.png', dpi=300)
plt.close()

# 주말/휴일과 평일의 평균 주문량 비교
weekday_avg = df[df['is_offday'] == 0]['sales'].mean()
offday_avg = df[df['is_offday'] == 1]['sales'].mean()
print(f"평일 평균 주문량: {weekday_avg:.2f}")
print(f"주말/휴일 평균 주문량: {offday_avg:.2f}")
print(f"주말/휴일 주문량 비율: {offday_avg/weekday_avg:.2%}")

# TimesFM 스타일의 분할: 마지막 7일을 테스트셋으로 사용
train_data = df.iloc[:-forecast_length].copy()
test_data = df.iloc[-forecast_length:].copy()

print(f"훈련 데이터 크기: {len(train_data)}")
print(f"테스트 데이터 크기: {len(test_data)}")
print(f"테스트 데이터 주말/휴일 수: {test_data['is_offday'].sum()}")

print("train_data.head()", train_data.head())

# 2. Preprocessor 정의
tsp = TimeSeriesPreprocessor(
    timestamp_column="date",
    id_columns=["state_id"],
    target_columns=["sales", "log_sales", "sales_x_offday"],  # 여러 타겟 변수 사용
    static_categorical_columns=[],
    static_real_columns=[],
    time_varying_known_categorical_columns=["dayofweek"],
    time_varying_known_real_columns=[
        "is_weekend", "is_holiday", "is_offday", 
        "day_0", "day_1", "day_2", "day_3", "day_4", "day_5", "day_6"
    ],
    time_varying_unknown_categorical_columns=[],
    time_varying_unknown_real_columns=[],
    context_length=context_length,
    prediction_length=forecast_length,
    scaling=True,
    scaler_type="standard",
)

# TimeSeriesPreprocessor 학습 (훈련 데이터만 사용)
trained_tsp = tsp.train(train_data)

# 3. 모델 로드 (Zero-shot 방식)
print("Granite 모델 로딩 중...")
TTM_MODEL_PATH = "ibm-granite/granite-timeseries-ttm-r2"
REVISION = "90-30-ft-l1-r2.1"  # 모델의 context_length=90, forecast_length=30 형태

model = TinyTimeMixerForPrediction.from_pretrained(
    TTM_MODEL_PATH,
    revision=REVISION,
    num_input_channels=tsp.num_input_channels,
    prediction_channel_indices=tsp.prediction_channel_indices,
    exogenous_channel_indices=tsp.exogenous_channel_indices,
)

# 4. Zero-shot 예측 파이프라인 설정
pipeline = TimeSeriesForecastingPipeline(
    model=model,
    feature_extractor=tsp,
    device=device,
    pad_context=True  # 중요: context를 패딩합니다
)

# 5. 예측 수행 (훈련 데이터만 사용하여 예측)
print("테스트 기간에 대한 예측 수행 중...")
forecast = pipeline(train_data)

# 예측 결과에서 7일치 예측값 추출
forecast_test = forecast[forecast.state_id == "T1"]
sample_prediction = forecast_test['sales_prediction'].iloc[0]
print(f"예측 값 타입: {type(sample_prediction)}")

# numpy 배열에서 7일치 예측값 추출
if isinstance(sample_prediction, (list, np.ndarray)) and len(sample_prediction) >= forecast_length:
    # forecast_length일 치 예측을 사용
    predictions = sample_prediction[:forecast_length]
    
    # 마지막 날짜 이후의 7일 예측 날짜 생성
    max_train_date = train_data['date'].max()
    forecast_dates = pd.date_range(start=max_train_date + pd.Timedelta(days=1), periods=forecast_length, freq='D')
    
    # 주말/휴일에 대한 보정 (주말/휴일의 경우 예측값을 로그 변환된 예측으로 대체)
    log_predictions = forecast_test['log_sales_prediction'].iloc[0][:forecast_length]
    sales_x_offday_predictions = forecast_test['sales_x_offday_prediction'].iloc[0][:forecast_length]
    
    # 주말/휴일 특성 가져오기
    test_is_offday = test_data[test_data.state_id == "T1"]['is_offday'].values
    
    # 보정된 예측값 생성
    corrected_predictions = predictions.copy()
    for i, is_off in enumerate(test_is_offday):
        if is_off == 1:  # 주말/휴일인 경우
            # 로그 예측에서 변환
            log_pred = np.expm1(log_predictions[i])
            # 가중치 부여 합성
            corrected_predictions[i] = (log_pred * 0.7 + sales_x_offday_predictions[i] * 0.3)
            # 주말/휴일 평균 비율 적용
            weekday_ratio = weekday_avg / (offday_avg if offday_avg > 0 else 1)
            corrected_predictions[i] = corrected_predictions[i] / weekday_ratio
    
    # 테스트 데이터와 예측값 준비
    test_dates = test_data[test_data.state_id == "T1"]['date']
    test_actual = test_data[test_data.state_id == "T1"]['sales'].values
    
    # 훈련 데이터 정보
    train_dates = train_data[train_data.state_id == "T1"]['date']
    train_actual = train_data[train_data.state_id == "T1"]['sales'].values
    
    # 테스트 날짜의 휴일/주말 정보
    test_holidays = test_data[test_data.state_id == "T1"]['is_holiday'].values
    test_weekends = test_data[test_data.state_id == "T1"]['is_weekend'].values
    test_offdays = test_data[test_data.state_id == "T1"]['is_offday'].values
    
    # 전체 데이터 (훈련 + 테스트)
    all_dates = df[df.state_id == "T1"]['date']
    all_actual = df[df.state_id == "T1"]['sales'].values
    
    # 예측 결과 요약
    print(f"예측 날짜: {forecast_dates}")
    print(f"모델의 원본 예측값: {predictions}")
    print(f"휴일/주말 보정된 예측값: {corrected_predictions}")
    
    # 6. TimesFM 스타일의 시각화
    plt.figure(figsize=(15, 7))
    
    # 전체 실제 데이터
    plt.plot(all_dates, all_actual, label='Actual', marker='o', markersize=4, color='blue')
    
    # 테스트 데이터 (실제값) 강조
    plt.plot(test_dates, test_actual, label='Test Actual', marker='s', linestyle='-', color='green')
    
    # 보정된 예측값만 표시
    plt.plot(forecast_dates, corrected_predictions, label='Granite Forecast with Holiday Adjustment', 
             marker='*', linestyle='-.', color='purple', linewidth=2)
    
    # 휴일과 주말 표시
    holiday_dates = [date for i, date in enumerate(test_dates) if test_holidays[i] == 1]
    weekend_dates = [date for i, date in enumerate(test_dates) if test_weekends[i] == 1]
    
    if holiday_dates:
        plt.scatter(holiday_dates, [all_actual.max() * 1.05] * len(holiday_dates), 
                   color='red', marker='v', s=80, label='Holidays')
    
    if weekend_dates:
        plt.scatter(weekend_dates, [all_actual.max() * 1.02] * len(weekend_dates), 
                   color='orange', marker='^', s=80, label='Weekends')
    
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.title('Granite TimeSeries Model: 7-Day Forecast with Holiday Adjustment')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    
    # 이미지 저장
    current_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(current_dir, 'granite_forecast.png')
    plt.savefig(image_path, dpi=300, bbox_inches='tight')
    print(f"이미지가 저장되었습니다: {image_path}")
    
    # 그래프 표시
    plt.show()
    
    # 7. 성능 평가 지표 계산
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    # 보정된 예측에 대한 지표만 계산
    rmse = np.sqrt(mean_squared_error(test_actual, corrected_predictions))
    mae = mean_absolute_error(test_actual, corrected_predictions)
    mape = np.mean(np.abs((test_actual - corrected_predictions) / np.maximum(test_actual, 1))) * 100
    r2 = r2_score(test_actual, corrected_predictions)
    
    print("\n휴일/주말 보정된 예측 성능 평가 지표:")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"R²: {r2:.4f}")
    
    # 8. 예측값과 실제값 비교 테이블 출력
    comparison_df = pd.DataFrame({
        'Date': forecast_dates,
        'Actual': test_actual,
        'Predicted': corrected_predictions,
        'Error': test_actual - corrected_predictions,
        'Error(%)': np.abs((test_actual - corrected_predictions) / np.maximum(test_actual, 1)) * 100,
        'DayOfWeek': [d.dayofweek for d in forecast_dates],
        'DayName': [d.day_name() for d in forecast_dates],  # 요일 이름 추가
        'IsWeekend': test_weekends,
        'IsHoliday': test_holidays,
        'IsOffday': test_offdays
    })
    print("\n예측값과 실제값 비교:")
    print(comparison_df.to_string())
    
    # 9. 주말/휴일에 따른 오차 분석
    print("\n요일별 오차 분석:")
    for day in range(7):
        day_error = comparison_df[comparison_df['DayOfWeek'] == day]['Error(%)'].mean()
        day_name = comparison_df[comparison_df['DayOfWeek'] == day]['DayName'].iloc[0] if len(comparison_df[comparison_df['DayOfWeek'] == day]) > 0 else f"Day {day}"
        print(f"{day_name} 평균 오차율: {day_error:.2f}%")
    
    print("\n주말/휴일에 따른 오차 분석:")
    weekday_errors = comparison_df[comparison_df['IsOffday'] == 0]['Error(%)'].mean()
    offday_errors = comparison_df[comparison_df['IsOffday'] == 1]['Error(%)'].mean()
    
    print(f"평일 평균 오차율: {weekday_errors:.2f}%")
    print(f"주말/휴일 평균 오차율: {offday_errors:.2f}%")
else:
    print("예측값 추출 실패: 예측 결과가 유효한 배열이 아니거나 길이가 충분하지 않습니다.")

print("예측 완료.")

# 10. 모델 파인튜닝 (Few-shot Fine-tuning)
print("\n\n======== 모델 파인튜닝 시작 ========\n")

# 파인튜닝 설정
do_finetune = True  # 파인튜닝 여부
fewshot_fraction = 0.20  # 데이터의 20%만 사용 (Few-shot 설정)
num_epochs = 50
batch_size = 64
OUT_DIR = "ttm_finetuned_models/"
os.makedirs(OUT_DIR, exist_ok=True)

if do_finetune:
    # 데이터 분할 비율 조정 (train 비율 낮추기)
    split_params = {"train": 0.5, "test": 0.2}  # 나머지 0.3은 검증 세트로
    
    print("데이터 분할 준비 중...")
    train_data_full, valid_data, test_data_ft = prepare_data_splits(
        df, id_columns=["state_id"], split_config=split_params, context_length=context_length
    )
    
    # 분할 후 데이터셋 길이 확인 및 로깅 추가
    print(f"분할된 데이터 크기 - 훈련: {len(train_data_full)}, 검증: {len(valid_data)}, 테스트: {len(test_data_ft)}")

    # 데이터 유효성 검사 추가
    if len(valid_data) == 0:
        print("검증 데이터가 비어있습니다. 분할 비율을 조정합니다.")
        # 검증 데이터가 없는 경우 훈련 데이터에서 일부 가져오기
        valid_data = train_data_full.sample(frac=0.3)
        train_data_full = train_data_full.drop(valid_data.index)
    
    # 모델이 사용하는 주파수 토큰 가져오기
    frequency_token = trained_tsp.get_frequency_token(trained_tsp.freq)
    
    # 데이터셋 파라미터 설정
    dataset_params = {
        "timestamp_column": "date",
        "id_columns": ["state_id"],
        "target_columns": ["sales"],
        "frequency_token": frequency_token,
        "context_length": context_length,
        "prediction_length": forecast_length,
    }
    
    # 전처리된 데이터로 토치 데이터셋 생성
    print("데이터셋 생성 중...")
    train_dataset = ForecastDFDataset(trained_tsp.preprocess(train_data_full), **dataset_params)
    valid_dataset = ForecastDFDataset(trained_tsp.preprocess(valid_data), **dataset_params)
    test_dataset_ft = ForecastDFDataset(trained_tsp.preprocess(test_data_ft), **dataset_params)
    
    # Few-shot 학습을 위한 데이터 샘플링
    print(f"Few-shot 학습을 위한 데이터 샘플링 ({fewshot_fraction*100}%)...")
    n_train_all = len(train_dataset)
    train_index = np.random.permutation(n_train_all)[:int(fewshot_fraction * n_train_all)]
    train_dataset = Subset(train_dataset, train_index)
    
    n_valid_all = len(valid_dataset)
    valid_index = np.random.permutation(n_valid_all)[:int(fewshot_fraction * n_valid_all)]
    valid_dataset = Subset(valid_dataset, valid_index)
    
    print(f"전체 학습 데이터: {n_train_all}, Few-shot 학습 데이터: {len(train_dataset)}")
    print(f"전체 검증 데이터: {n_valid_all}, Few-shot 검증 데이터: {len(valid_dataset)}")
    
    # 파인튜닝을 위한 모델 로드
    print("파인튜닝을 위한 모델 로드 중...")
    finetune_model = TinyTimeMixerForPrediction.from_pretrained(
        TTM_MODEL_PATH,
        revision=REVISION,
        context_length=context_length,
        prediction_filter_length=forecast_length,
        num_input_channels=trained_tsp.num_input_channels,
        decoder_mode="mix_channel",  # 채널 믹싱 사용
        prediction_channel_indices=trained_tsp.prediction_channel_indices,
        exogenous_channel_indices=trained_tsp.exogenous_channel_indices,
        fcm_context_length=1,  # 외부 변수 융합에 사용할 시차 길이
        fcm_use_mixer=True,  # 외부 변수 믹싱 활성화
        fcm_mix_layers=2,  # 외부 변수 믹싱 레이어 수
        enable_forecast_channel_mixing=True,  # 예측 채널 믹싱 활성화
        fcm_prepend_past=True,  # 외부 변수 주입시 과거 데이터 포함
    )
    
    # 백본 동결 여부 설정 (선택적)
    freeze_backbone = False
    if freeze_backbone:
        print(f"백본 동결 전 파라미터 수: {count_parameters(finetune_model)}")
        # 모델의 백본 부분을 고정
        for param in finetune_model.backbone.parameters():
            param.requires_grad = False
        print(f"백본 동결 후 파라미터 수: {count_parameters(finetune_model)}")
    
    # 최적의 학습률 찾기
    print("최적의 학습률 찾는 중...")
    learning_rate, finetune_model = optimal_lr_finder(
        finetune_model,
        train_dataset,
        batch_size=batch_size,
        enable_prefix_tuning=True,
    )
    print(f"최적 학습률: {learning_rate}")
    
    # 트레이너 설정
    print("트레이너 설정 중...")
    finetune_args = TrainingArguments(
        output_dir=os.path.join(OUT_DIR, "output"),
        overwrite_output_dir=True,
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        do_eval=True,
        eval_strategy="epoch",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=2 * batch_size,
        dataloader_num_workers=1,
        report_to="none",
        save_strategy="epoch",
        logging_strategy="epoch",
        save_total_limit=1,
        logging_dir=os.path.join(OUT_DIR, "logs"),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        use_cpu=device == "cpu",
    )
    
    # 조기 종료 콜백 및 트래킹 콜백 설정
    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=10,
        early_stopping_threshold=0.0,
    )
    tracking_callback = TrackingCallback()
    
    # 옵티마이저 및 스케줄러 설정
    optimizer = AdamW(finetune_model.parameters(), lr=learning_rate)
    scheduler = OneCycleLR(
        optimizer,
        learning_rate,
        epochs=num_epochs,
        steps_per_epoch=math.ceil(len(train_dataset) / batch_size),
    )
    
    # 트레이너 생성
    finetune_trainer = Trainer(
        model=finetune_model,
        args=finetune_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        callbacks=[early_stopping, tracking_callback],
        optimizers=(optimizer, scheduler),
    )
    
    # 모델 파인튜닝 시작
    print("파인튜닝 시작...")
    finetune_trainer.train()
    
    # 파인튜닝된 모델 평가
    print("파인튜닝된 모델 평가 중...")
    eval_results = finetune_trainer.evaluate(test_dataset_ft)
    print(f"평가 결과: {eval_results}")
    
    # 파인튜닝된 모델로 예측 수행
    print("파인튜닝된 모델로 예측 수행 중...")
    finetune_pipeline = TimeSeriesForecastingPipeline(
        model=finetune_model,
        feature_extractor=trained_tsp,
        device=device,
        batch_size=batch_size,
        pad_context=True
    )
    
    # 테스트 데이터에 대한 예측
    finetune_forecast = finetune_pipeline(test_data_ft)
    
    # 성능 평가 함수 정의
    def custom_metric(actual, prediction, column_header="results"):
        """MSE, RMSE, MAE를 계산하는 함수"""
        a = np.asarray(actual.tolist())
        p = np.asarray(prediction.tolist())
        if p.shape[1] < a.shape[1]:
            a = a[:, : p.shape[1]]
        
        mask = ~np.any(np.isnan(a), axis=1)
        
        mse = np.mean(np.square(a[mask, :] - p[mask, :]))
        mae = np.mean(np.abs(a[mask, :] - p[mask, :]))
        return pd.DataFrame(
            {
                column_header: {
                    "mean_squared_error": mse,
                    "root_mean_squared_error": np.sqrt(mse),
                    "mean_absolute_error": mae,
                }
            }
        )
    
    # 파인튜닝된 모델의 성능 지표 계산
    metrics = custom_metric(finetune_forecast["sales"], finetune_forecast["sales_prediction"], "파인튜닝된 모델 예측")
    print("파인튜닝된 모델 성능 지표:")
    print(metrics)
    
    # 예측 결과 시각화
    print("파인튜닝된 모델 예측 결과 시각화...")
    # 시각화 수행 (첫 번째 state_id에 대해서만)
    first_state = finetune_forecast["state_id"].unique()[0]
    plot_predictions(
        input_df=test_data_ft[test_data_ft.state_id == first_state],
        predictions_df=finetune_forecast[finetune_forecast.state_id == first_state],
        freq="d",
        timestamp_column="date",
        channel="sales",
    )
    
    # 이미지 저장
    current_dir = os.path.dirname(os.path.abspath(__file__))
    ft_image_path = os.path.join(current_dir, 'granite_finetuned_forecast.png')
    plt.savefig(ft_image_path, dpi=300, bbox_inches='tight')
    print(f"파인튜닝된 모델 예측 이미지가 저장되었습니다: {ft_image_path}")
    
    # 그래프 표시
    plt.show()
    
    print("\n======== 모델 파인튜닝 완료 ========\n")
