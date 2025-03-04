import os
import tempfile
import pandas as pd
import matplotlib.pyplot as plt
from transformers import Trainer, TrainingArguments
# Local
from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction
from tsfm_public.toolkit.callbacks import TrackingCallback
# === 1. 데이터셋 설정 및 전처리 ===
from tsfm_public.toolkit.time_series_preprocessor import TimeSeriesPreprocessor

# Dataset 설정
TARGET_DATASET = "my_dataset"
# 여러분의 데이터 파일 경로 (예: CSV 파일)
dataset_path = "../data/data_order_cnt.csv"
timestamp_column = "d_day"  # 날짜 컬럼
id_columns = []             # 유니크 아이디 컬럼이 없으면 빈 리스트로
target_columns = ["total_order_cnt", "total_order_amt"]

# CSV 파일 읽기 (날짜 컬럼을 파싱)
data = pd.read_csv(dataset_path, parse_dates=[timestamp_column])
data = data.sort_values(timestamp_column)

# 데이터 분할 (예시: 70% 학습, 15% 검증, 15% 테스트)
n = len(data)
split_config = {
    "train": [0, int(0.7 * n)],
    "valid": [int(0.7 * n), int(0.85 * n)],
    "test":  [int(0.85 * n), n],
}

# 컬럼 스펙 지정 (TimeSeriesPreprocessor 등에 사용)
column_specifiers = {
    "timestamp_column": timestamp_column,
    "id_columns": id_columns,
    "target_columns": target_columns,
    "control_columns": [],
}

# === 2. 모델 및 평가 관련 상수 ===

# 모델 경로 또는 Hugging Face 모델 식별자 (실제 경로로 수정하세요)
TTM_MODEL_PATH = "ibm-granite/granite-timeseries-ttm-r2"
SEED = 42
OUT_DIR = "./output"

# === 3. Zeroshot 평가 및 예측 함수 ===

tsp = TimeSeriesPreprocessor(
    **column_specifiers,
    context_length=512,
    prediction_length=96,
    scaling=True,
    encode_categorical=False,
    scaler_type="standard",
)

train_dataset, valid_dataset, test_dataset = tsp.get_datasets(
    data, split_config, fewshot_fraction=0.0, fewshot_location="first"
)
print(f"Data lengths: train = {len(train_dataset)}, val = {len(valid_dataset)}, test = {len(test_dataset)}")
    