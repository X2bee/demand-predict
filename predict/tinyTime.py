import os
import tempfile
import pandas as pd
import matplotlib.pyplot as plt
from transformers import Trainer, TrainingArguments

# === 1. 데이터셋 설정 및 전처리 ===

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
TTM_MODEL_PATH = "path/to/your/model"
SEED = 42
OUT_DIR = "./output"

# === 3. Zeroshot 평가 및 예측 함수 ===

def zeroshot_eval(dataset_name, batch_size, context_length=512, forecast_length=96):
    """
    데이터 전처리, 모델 로드, zero-shot 평가 및 예측 후 결과 플롯 저장까지 수행합니다.
    """
    # 아래 함수들은 여러분이 사용하는 타임시리즈 모듈에서 제공해야 합니다.
    # from my_timeseries_module import TimeSeriesPreprocessor, get_datasets, get_model, plot_predictions
    # 예시로 임포트하는 코드:
    try:
        from my_timeseries_module import TimeSeriesPreprocessor, get_datasets, get_model, plot_predictions
    except ImportError:
        raise ImportError("TimeSeries 관련 함수(TimeSeriesPreprocessor, get_datasets, get_model, plot_predictions)를 불러올 수 없습니다. 경로를 확인하세요.")

    # 데이터 전처리: TimeSeriesPreprocessor 인스턴스 생성
    tsp = TimeSeriesPreprocessor(
        **column_specifiers,
        context_length=context_length,
        prediction_length=forecast_length,
        scaling=True,
        encode_categorical=False,
        scaler_type="standard",
    )

    # 학습/검증/테스트 데이터셋 생성 (여기서는 여러분의 get_datasets 함수 사용)
    dset_train, dset_valid, dset_test = get_datasets(tsp, data, split_config)

    # 모델 로드: TTM_MODEL_PATH와 함께 context, prediction 길이 전달
    zeroshot_model = get_model(TTM_MODEL_PATH, context_length=context_length, prediction_length=forecast_length)

    # 임시 출력 디렉토리 생성
    temp_dir = tempfile.mkdtemp()

    # Trainer 설정 (zero-shot 평가 전용)
    zeroshot_trainer = Trainer(
        model=zeroshot_model,
        args=TrainingArguments(
            output_dir=temp_dir,
            per_device_eval_batch_size=batch_size,
            seed=SEED,
            report_to="none",
        ),
    )

    # Zero-shot 평가 진행 (예: MSE 등 지표 출력)
    print("++++++++++++++++++++ Test MSE zero-shot ++++++++++++++++++++")
    zeroshot_output = zeroshot_trainer.evaluate(dset_test)
    print("Zero-shot evaluation output:", zeroshot_output)

    # 테스트 데이터셋에 대해 예측 진행
    predictions_dict = zeroshot_trainer.predict(dset_test)
    # predictions_dict.predictions가 튜플로 (예측값, backbone embedding) 등을 반환한다고 가정
    predictions_np = predictions_dict.predictions[0]
    print("Predictions shape:", predictions_np.shape)

    # backbone embedding (분석용)
    backbone_embedding = predictions_dict.predictions[1]
    print("Backbone embedding shape:", backbone_embedding.shape)

    # 예측 결과 플롯 저장 (여러 인덱스 예시, channel은 첫 번째 타겟 컬럼 기준)
    plot_dir = os.path.join(OUT_DIR, dataset_name)
    os.makedirs(plot_dir, exist_ok=True)
    plot_predictions(
        model=zeroshot_trainer.model,
        dset=dset_test,
        plot_dir=plot_dir,
        plot_prefix="test_zeroshot",
        indices=[0, 1, 2, 3, 4],  # 원하는 인덱스로 수정하세요.
        channel=0,
    )
    print(f"예측 결과 플롯이 {plot_dir}에 저장되었습니다.")

# === 4. 메인 실행부 ===

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    # 배치 사이즈, context 길이, forecast 길이는 필요에 따라 조정하세요.
    zeroshot_eval(TARGET_DATASET, batch_size=16, context_length=512, forecast_length=96)
