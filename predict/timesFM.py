import pandas as pd
import matplotlib.pyplot as plt
import timesfm
import os
import torch
import numpy as np
import requests
from datetime import datetime
from collections import defaultdict
import time
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 디바이스 설정: GPU 대신 CPU로 고정 (TimesFM 인덱싱 오류 방지)
device = 'cpu'  # CUDA에서 인덱싱 오류가 발생하므로 CPU로 강제 변경
torch.set_default_device('cpu')
print(f"Using device: {device} (TimesFM는 GPU 인덱싱 이슈로 CPU만 지원)")

# 한국 공휴일 API 함수
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

# 평가 메트릭 함수 정의
def mse(y_pred, y_true):
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    return np.mean(np.square(y_pred - y_true), axis=1, keepdims=True)

def mae(y_pred, y_true):
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    return np.mean(np.abs(y_pred - y_true), axis=1, keepdims=True)

# CSV 파일 읽기 및 전처리
df = pd.read_csv('data_order_cnt.csv')
df['d_day'] = pd.to_datetime(df['d_day'], format='%Y%m%d')
df = df.sort_values('d_day')

# 요일 및 주말/휴일 정보 추가
df['dayofweek'] = df['d_day'].dt.dayofweek
df['week_day'] = df['dayofweek']  # TimesFM의 covariate 형식 맞추기
df['is_weekend'] = df['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)

# 공휴일 정보 추가
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

# 주말 또는 공휴일인 경우 'is_offday' = 1
df['is_offday'] = ((df['is_holiday'] == 1) | (df['is_weekend'] == 1)).astype(int)

# 평일/주말 주문량 통계
weekday_avg = df[df['is_offday'] == 0]['total_order_cnt'].mean()
offday_avg = df[df['is_offday'] == 1]['total_order_cnt'].mean()
print(f"평일 평균 주문량: {weekday_avg:.2f}")
print(f"주말/휴일 평균 주문량: {offday_avg:.2f}")
print(f"주말/휴일 주문량 비율: {offday_avg/weekday_avg:.2%}")

# TimesFM이 요구하는 형식: unique_id, ds (날짜), y (타겟값)
df['unique_id'] = "T1"
df_model = df[['unique_id', 'd_day', 'total_order_cnt']].rename(
    columns={'d_day': 'ds', 'total_order_cnt': 'y'}
)

# Covariates 데이터 추가
df_model['week_day'] = df['week_day']
df_model['is_weekend'] = df['is_weekend']
df_model['is_holiday'] = df['is_holiday']
df_model['is_offday'] = df['is_offday']

# Train/Test split: 마지막 7일은 테스트셋으로 사용
horizon_len = 14
train_df = df_model.iloc[:-horizon_len].copy()
test_df = df_model.iloc[-horizon_len:].copy()

# TimesFM 모델 초기화 (500m 모델에 필요한 고정 파라미터 사용)
tfm = timesfm.TimesFm(
    hparams=timesfm.TimesFmHparams(
        backend="torch",             # PyTorch backend 사용
        per_core_batch_size=32,
        horizon_len=horizon_len,     # 예측 기간 7일
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

print("df_model", df_model)

def get_batched_data_fn(batch_size=32, context_len=32, horizon_len=7):
    examples = defaultdict(list)
    
    num_examples = 0
    for unique_id in df_model['unique_id'].unique():
        sub_df = df_model[df_model["unique_id"] == unique_id]
        sub_df = sub_df.sort_values('ds')
        
        # 마지막 배치만 생성 (테스트 기간에 해당하는 배치)
        start = len(sub_df) - (context_len + horizon_len)
        if start >= 0:  # 충분한 데이터가 있는지 확인
            num_examples += 1
            context_end = start + context_len
            
            examples["unique_id"].append(unique_id)
            examples["inputs"].append(sub_df["y"][start:context_end].tolist())
            examples["is_offday"].append(sub_df["is_offday"][start:context_end + horizon_len].tolist())
            examples["outputs"].append(sub_df["y"][context_end:context_end + horizon_len].tolist())
            
            print(f"생성된 예측 배치: 학습 구간 {sub_df['ds'][start]} ~ {sub_df['ds'][context_end-1]}, 예측 구간 {sub_df['ds'][context_end]} ~ {sub_df['ds'][context_end+horizon_len-1]}")
    
    def data_fn():
        for i in range(1 + (num_examples - 1) // batch_size):
            batch = {k: v[(i * batch_size) : ((i + 1) * batch_size)] for k, v in examples.items()}
            if batch["inputs"]:  # 빈 배치가 아닌 경우에만 반환
                yield batch
    
    return data_fn

# 배치 데이터 생성
batch_size = 32
context_len = 32
input_data = get_batched_data_fn(
    batch_size=batch_size, 
    context_len=context_len, 
    horizon_len=horizon_len
)

# 3. Covariates를 활용한 예측 실행
metrics = defaultdict(list)
cov_forecasts = []
raw_forecasts = []
has_valid_covariates = False

for i, example in enumerate(input_data()):
    if len(example["inputs"]) == 0:
        continue
    
    print(f"\n배치 {i+1} 처리 중 - 입력 데이터 {len(example['inputs'])}개...")
    
    start_time = time.time()
    
    # 기본 TimesFM 예측
    raw_forecast, _ = tfm.forecast(
        inputs=example["inputs"], 
        freq=[0] * len(example["inputs"])
    )
    
    # 예측값이 음수인 경우 0으로 보정
    raw_forecast_clipped = np.maximum(raw_forecast[:, :horizon_len], 0)
    raw_forecasts.append(raw_forecast_clipped)
    
    try:
        # Covariates를 활용한 예측 - is_offday만 사용
        cov_forecast, ols_forecast = tfm.forecast_with_covariates(  
            inputs=example["inputs"],
            dynamic_numerical_covariates={},
            dynamic_categorical_covariates={
                "is_offday": example["is_offday"],
            },
            static_numerical_covariates={},
            static_categorical_covariates={
                "unique_id": example["unique_id"]
            },
            freq=[0] * len(example["inputs"]),
            xreg_mode="xreg + timesfm",           # default
            ridge=0.0,
            force_on_cpu=True,                   # CPU 강제 사용으로 변경
            normalize_xreg_target_per_input=True,  # default
        )
        
        # NaN 검사 및 처리
        if isinstance(cov_forecast, list):
            contains_nan = any(np.isnan(np.array(cov_forecast)).any() for cf in cov_forecast)
        else:
            contains_nan = np.isnan(np.array(cov_forecast)).any()
            
        if not contains_nan:
            # 예측값이 음수인 경우 0으로 보정
            if isinstance(cov_forecast, list):
                cov_forecast = [np.maximum(cf, 0) for cf in cov_forecast]
            else:
                cov_forecast = np.maximum(cov_forecast, 0)
                
            has_valid_covariates = True
            cov_forecasts.append(cov_forecast)
            
            metrics["eval_mae_xreg_timesfm"].extend(
                mae(cov_forecast, example["outputs"])
            )
            metrics["eval_mae_xreg"].extend(
                mae(ols_forecast, example["outputs"])
            )
            metrics["eval_mse_xreg_timesfm"].extend(
                mse(cov_forecast, example["outputs"])
            )
            metrics["eval_mse_xreg"].extend(
                mse(ols_forecast, example["outputs"])
            )
        else:
            print(f"\n주의: 배치 {i}에서 NaN 값 발견, 이 배치는 건너뜁니다.")
    except Exception as e:
        print(f"\n오류 발생: {e}")
    
    # TimesFM 기본 예측은 항상 평가
    metrics["eval_mae_timesfm"].extend(
        mae(raw_forecast[:, :horizon_len], example["outputs"])
    )
    metrics["eval_mse_timesfm"].extend(
        mse(raw_forecast[:, :horizon_len], example["outputs"])
    )
    
    print(
        f"배치 {i+1} 처리 완료, 소요 시간: {time.time() - start_time:.2f}초"
    )

print("\n\n모델 평가 지표:")
for k, v in metrics.items():
    if len(v) > 0:
        print(f"{k}: {np.mean(v)}")
    else:
        print(f"{k}: 유효한 데이터 없음")

# 4. 시각화
plt.figure(figsize=(15, 7))

# 전체 실제 데이터
plt.plot(df_model['ds'], df_model['y'], label='Actual', marker='o', color='blue', markersize=4)

# 테스트 기간 데이터 (실제값) 강조
plt.plot(test_df['ds'], test_df['y'], label='Test Actual', color='forestgreen', marker='s', linewidth=2, markersize=6)

# Covariates 적용 예측 결과
if cov_forecasts and has_valid_covariates:
    try:
        # 첫 번째 배치의 첫 번째 예측 결과
        if isinstance(cov_forecasts[0], list):
            cov_pred = cov_forecasts[0][0]  
        else:
            cov_pred = cov_forecasts[0][0]
        
        # NaN 값 확인 및 처리
        if not np.isnan(np.array(cov_pred)).any():
            plt.plot(test_df['ds'], cov_pred, 
                     label='TimesFM+Offday', marker='D', linestyle=':', color='magenta', linewidth=2, markersize=6)
        else:
            print("경고: Covariates 예측 결과에 NaN 값이 포함되어 있어 그래프에 표시하지 않습니다.")
    except Exception as e:
        print(f"Covariates 그래프 그리기 오류: {e}")

# 주말/휴일 표시 (Offday로 통합해서 표시)
offday_dates = df[df['is_offday'] == 1]['d_day'].values

if len(offday_dates) > 0:
    offday_in_range = [date for date in offday_dates if date >= df_model['ds'].min() and date <= df_model['ds'].max()]
    if offday_in_range:
        plt.scatter(offday_in_range, [df_model['y'].max() * 1.05] * len(offday_in_range), 
                  color='purple', marker='v', s=100, label='Off Days')

# 테스트 기간 시작 표시
plt.axvline(x=test_df['ds'].iloc[0], color='darkgray', linestyle='--', linewidth=2, label='Test Period Start')

# 그리드 설정
plt.grid(True, which='major', linestyle='-', linewidth=0.5, alpha=0.7)
plt.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.4)

# 축 레이블과 제목
plt.xlabel('Date', fontsize=12, fontweight='bold')
plt.ylabel('Total Order Count', fontsize=12, fontweight='bold')
plt.title('TimesFM Forecast with Offday Covariate', fontsize=16, fontweight='bold')

# 범례 설정 - 중복 제거 및 원하는 항목만 표시
handles, labels = plt.gca().get_legend_handles_labels()
by_label = {}
# 원하는 레이블만 선택
for handle, label in zip(handles, labels):
    if label not in ['TimesFM']:  # TimesFM 범례 제외
        by_label[label] = handle
plt.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=10, framealpha=0.9)

# 그래프 스타일 설정
plt.xticks(fontsize=10, rotation=45)
plt.yticks(fontsize=10)
plt.gcf().autofmt_xdate()
plt.tight_layout()

# 이미지 저장
current_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(current_dir, 'timesfm_offday_forecast.png')
plt.savefig(image_path, dpi=300, bbox_inches='tight')
print(f"이미지가 저장되었습니다: {image_path}")

# 그래프 보여주기
plt.show()
plt.close()
