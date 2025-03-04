import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import concurrent.futures
from db.db_connection import engine  # SQLAlchemy 엔진 객체

def fetch_month_data(start_date, month_end):
    """
    start_date ~ month_end 범위의 데이터를 조회하는 쿼리를 실행하여 DataFrame 반환
    """
    query = f"""
    WITH 
    basket_data AS (
        SELECT TO_CHAR(sys_reg_dtm, 'YYYYMMDD') AS d_day,
               goods_no,
               itm_no,
               COUNT(DISTINCT bsket_no) AS bsket_cnt
          FROM op_bsket_info
         GROUP BY TO_CHAR(sys_reg_dtm, 'YYYYMMDD'), goods_no, itm_no
    ),
    order_data AS (
        SELECT agrt_dt AS d_day,
               goods_no,
               itm_no,
               goods_nm,
               SUM(ord_qty - cncl_qty) AS order_cnt,
               SUM(ord_amt) AS order_amt
          FROM SM_DAYCL_ORD_AGRT
         WHERE agrt_gb = '01'
         GROUP BY agrt_dt, goods_no, itm_no, goods_nm
    ),
    order_non_mem_data AS (
        SELECT agrt_dt AS d_day,
               goods_no,
               itm_no,
               SUM(ord_qty - cncl_qty) AS order_non_mem
          FROM SM_DAYCL_ORD_AGRT
         WHERE agrt_gb = '01'
           AND mbr_no = '999999999'
         GROUP BY agrt_dt, goods_no, itm_no
    ),
    return_data AS (
        SELECT TO_CHAR(sys_reg_dtm, 'YYYYMMDD') AS d_day,
               goods_no,
               itm_no,
               COUNT(DISTINCT claim_no) AS return_cnt
          FROM op_ord_dtl
         WHERE ord_dtl_gb_cd = '20'
         GROUP BY TO_CHAR(sys_reg_dtm, 'YYYYMMDD'), goods_no, itm_no
    ),
    exchange_data AS (
        SELECT TO_CHAR(sys_reg_dtm, 'YYYYMMDD') AS d_day,
               goods_no,
               itm_no,
               COUNT(DISTINCT claim_no) AS exchange_cnt
          FROM op_ord_dtl
         WHERE ord_dtl_gb_cd = '30'
         GROUP BY TO_CHAR(sys_reg_dtm, 'YYYYMMDD'), goods_no, itm_no
    ),
    category_info AS (
        SELECT pgb.GOODS_NO,
               psc.STD_CTG_NM
          FROM PR_GOODS_BASE pgb
          LEFT JOIN PR_STD_CTG psc ON pgb.STD_CTG_NO = psc.STD_CTG_NO
         WHERE pgb.STD_CTG_NO IS NOT NULL
           AND pgb.SALE_STAT_CD = '10'
           AND pgb.DISP_YN = 'Y'
           AND NOW() < pgb.SALE_END_DTM
    ),
    keys AS (
        SELECT d_day, goods_no, itm_no FROM basket_data
        UNION
        SELECT d_day, goods_no, itm_no FROM order_data
        UNION
        SELECT d_day, goods_no, itm_no FROM order_non_mem_data
        UNION
        SELECT d_day, goods_no, itm_no FROM return_data
        UNION
        SELECT d_day, goods_no, itm_no FROM exchange_data
    ),
    first_result AS (
        SELECT
          k.d_day,
          k.goods_no,
          k.itm_no,
          COALESCE(b.bsket_cnt, 0)       AS bsket_cnt,
          COALESCE(o.order_cnt, 0)       AS order_cnt,
          COALESCE(o.order_amt, 0)       AS order_amt,
          COALESCE(o.goods_nm, '')       AS goods_nm,
          COALESCE(n.order_non_mem, 0)   AS order_non_mem,
          COALESCE(r.return_cnt, 0)      AS return_cnt,
          COALESCE(e.exchange_cnt, 0)    AS exchange_cnt
        FROM keys k
        LEFT JOIN basket_data b ON k.d_day = b.d_day AND k.goods_no = b.goods_no AND k.itm_no = b.itm_no
        LEFT JOIN order_data o ON k.d_day = o.d_day AND k.goods_no = o.goods_no AND k.itm_no = o.itm_no
        LEFT JOIN order_non_mem_data n ON k.d_day = n.d_day AND k.goods_no = n.goods_no AND k.itm_no = n.itm_no
        LEFT JOIN return_data r ON k.d_day = r.d_day AND k.goods_no = r.goods_no AND k.itm_no = r.itm_no
        LEFT JOIN exchange_data e ON k.d_day = e.d_day AND k.goods_no = e.goods_no AND k.itm_no = e.itm_no
    ),
    calendar AS (
        SELECT TO_CHAR(d, 'YYYYMMDD') AS d_day
        FROM generate_series(
             '{start_date.strftime("%Y-%m-%d")}',
             '{month_end.strftime("%Y-%m-%d")}',
             INTERVAL '1 day'
        ) d
    ),
    products AS (
        SELECT DISTINCT goods_no
        FROM (
          SELECT fvr_tgt_no AS goods_no FROM CC_PROMO_APLY_INFO
          UNION
          SELECT goods_no FROM PR_MKDP_GOODS_INFO
          UNION
          SELECT goods_no FROM PR_GOODS_REV_INFO
        ) t
    ),
    promo AS (
        SELECT c.d_day, cpai.fvr_tgt_no AS goods_no
        FROM CC_PROMO_APLY_INFO cpai
        JOIN CC_PROM_BASE cpb ON cpai.promo_no = cpb.promo_no
        CROSS JOIN calendar c
        WHERE cpai.FVR_APLY_GB_CD = '01'
          AND cpai.FVR_APLY_TYP_CD = '02'
          AND cpb.PROMO_STAT_CD = '10'
          AND TO_DATE(c.d_day, 'YYYYMMDD') BETWEEN cpb.promo_str_dtm AND cpb.promo_end_dtm
        GROUP BY c.d_day, cpai.fvr_tgt_no
    ),
    display AS (
        SELECT c.d_day, pmgi.goods_no
        FROM PR_MKDP_BASE pmb
        JOIN PR_MKDP_DIVOBJ_INFO pmdi ON pmb.mkdp_no = pmdi.mkdp_no
        JOIN PR_MKDP_GOODS_INFO pmgi ON pmdi.mkdp_no = pmgi.mkdp_no AND pmdi.divobj_no = pmgi.divobj_no
        CROSS JOIN calendar c
        WHERE pmb.disp_yn = 'Y'
          AND pmb.del_yn = 'N'
          AND pmdi.disp_yn = 'Y'
          AND TO_DATE(c.d_day, 'YYYYMMDD') BETWEEN pmb.disp_str_dtm AND pmb.disp_end_dtm
        GROUP BY c.d_day, pmgi.goods_no
    ),
    reviews AS (
        SELECT TO_CHAR(sys_reg_dtm, 'YYYYMMDD') AS d_day,
               goods_no,
               COUNT(*) AS review_count
        FROM PR_GOODS_REV_INFO
        GROUP BY TO_CHAR(sys_reg_dtm, 'YYYYMMDD'), goods_no
    ),
    second_result AS (
        SELECT 
          cal.d_day,
          prod.goods_no,
          CASE WHEN p.goods_no IS NOT NULL THEN 'Y' ELSE 'N' END AS promo_active,
          CASE WHEN d.goods_no IS NOT NULL THEN 'Y' ELSE 'N' END AS display_active,
          COALESCE(r.review_count, 0) AS review_count
        FROM calendar cal
        CROSS JOIN products prod
        LEFT JOIN promo p ON cal.d_day = p.d_day AND prod.goods_no = p.goods_no
        LEFT JOIN display d ON cal.d_day = d.d_day AND prod.goods_no = d.goods_no
        LEFT JOIN reviews r ON cal.d_day = r.d_day AND prod.goods_no = r.goods_no
    )
    SELECT 
        f.d_day,
        f.goods_no,
        f.goods_nm,
        ci.STD_CTG_NM,
        f.itm_no,
        f.bsket_cnt,
        f.order_cnt,
        f.order_amt,
        f.order_non_mem,
        f.return_cnt,
        f.exchange_cnt,
        s.promo_active,
        s.display_active,
        s.review_count
    FROM first_result f
    LEFT JOIN second_result s ON f.d_day = s.d_day AND f.goods_no = s.goods_no
    LEFT JOIN category_info ci ON f.goods_no = ci.goods_no
    ORDER BY f.goods_no, f.d_day, f.itm_no
    """
    
    df_chunk = pd.read_sql(query, engine)
    print(f"Loaded data from {start_date.strftime('%Y-%m-%d')} to {month_end.strftime('%Y-%m-%d')} - Shape: {df_chunk.shape}")
    return df_chunk

# 오늘 날짜 기준으로 5년 전부터 오늘까지의 데이터를 조회
end_date = datetime.today()
start_date = end_date - relativedelta(years=1)
# 월별 날짜 범위를 생성하여 concurrent.futures로 병렬 조회
all_chunks = []
date_ranges = []
current_start = start_date

while current_start <= end_date:
    current_end = current_start + relativedelta(months=1) - timedelta(days=1)
    # 마지막 월인 경우 end_date로 맞춤
    if current_end > end_date:
        current_end = end_date
    date_ranges.append((current_start, current_end))
    current_start = current_end + timedelta(days=1)

# 동시에 여러 월의 데이터를 조회 (스레드 풀 사용)
with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
    future_to_range = {executor.submit(fetch_month_data, sr, er): (sr, er) for sr, er in date_ranges}
    for future in concurrent.futures.as_completed(future_to_range):
        sr, er = future_to_range[future]
        try:
            chunk_df = future.result()
            all_chunks.append(chunk_df)
        except Exception as exc:
            print(f"Error fetching data from {sr.strftime('%Y-%m-%d')} to {er.strftime('%Y-%m-%d')}: {exc}")

# 모든 월별 데이터를 하나의 DataFrame으로 결합
df_all = pd.concat(all_chunks, ignore_index=True)

# 최종 CSV 파일로 저장
df_all.to_csv("5years_data.csv", index=False)
print("5년치 데이터 저장 완료. CSV 파일: 5years_data.csv")
