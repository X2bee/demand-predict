from pyspark.sql import SparkSession
from pyspark.sql.functions import sum, col
import matplotlib.pyplot as plt
from setuptools._distutils import *
import pandas as pd
from matplotlib.ticker import MaxNLocator
from neuralprophet import NeuralProphet

# SparkSession 생성
spark = SparkSession.builder \
    .appName("Aggregate Daily Orders and Sales") \
    .getOrCreate()

# CSV 파일 읽기
df = spark.read.option("header", "true") \
               .option("inferSchema", "true") \
               .csv("5years_data.csv")

# 날짜별 전체 주문 건수와 주문 금액 집계
df_ord = df.groupBy("d_day").agg(
    sum("order_cnt").alias("total_order_cnt"),
    sum("order_amt").alias("total_order_amt")
).orderBy(col("d_day").asc()).filter((col("total_order_cnt").isNotNull()) & (col("total_order_cnt") > 0))

# 상위 10개 행 출력
df_ord.show(10)
# 집계 결과를 Pandas DataFrame으로 변환
pdf = df_ord.toPandas()

#pdf csv 저장
pdf.to_csv("data_order_cnt.csv", index=True)