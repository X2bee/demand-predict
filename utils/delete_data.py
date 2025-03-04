import pandas as pd

# CSV 파일 읽기
df = pd.read_csv("5years_data.csv")

# 날짜 컬럼을 datetime으로 변환 (YYYYMMDD 가정)
df['d_day'] = pd.to_datetime(df['d_day'], format='%Y%m%d')

# 2025년인 행만 필터
df_2025 = df[df['d_day'].dt.year == 2025]

# 새로운 CSV로 저장
df_2025.to_csv("5years_data_2025_only.csv", index=True)
