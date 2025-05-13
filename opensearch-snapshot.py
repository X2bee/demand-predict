#!/usr/bin/env python3
import requests
import json
import sys
import copy
import datetime
import pytz

# 클러스터 엔드포인트 설정
SOURCE_CLUSTER = "http://10.100.0.8:9200"
TARGET_CLUSTER = "http://10.100.3.5:9200"

# 한국 시간대 설정 및 전날 계산
kst = pytz.timezone("Asia/Seoul")
today_kst = datetime.datetime.now(kst)
yesterday = today_kst - datetime.timedelta(days=1)

# 전날 날짜 기반 logstash 인덱스 이름 생성
SOURCE_INDEX = "logstash-" + yesterday.strftime("%Y%m%d")
DEST_INDEX = SOURCE_INDEX + "-reindexed"

# 요청 헤더 설정
headers = {'Content-Type': 'application/json'}

def get_source_index_info():
    """소스 인덱스의 settings와 mappings 가져오기"""
    url = f"{SOURCE_CLUSTER}/{SOURCE_INDEX}"
    print(f"소스 인덱스 정보를 가져옵니다: {url}")
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    data = response.json()
    if SOURCE_INDEX not in data:
        raise Exception("소스 인덱스 정보를 찾을 수 없습니다.")
    index_info = data[SOURCE_INDEX]
    settings = index_info.get("settings", {}).get("index", {})
    mappings = index_info.get("mappings", {})
    return settings, mappings

def clean_settings(settings):
    """인덱스 생성 시 불필요한 settings 제거하기"""
    cleaned = copy.deepcopy(settings)
    # 제거할 키 목록 (필요에 따라 추가)
    remove_keys = [
        "creation_date", "uuid", "version", "provided_name"
    ]
    for key in remove_keys:
        cleaned.pop(key, None)
    return cleaned

def create_target_index(settings, mappings):
    """대상 클러스터에 인덱스 생성"""
    url = f"{TARGET_CLUSTER}/{DEST_INDEX}"
    payload = {
        "settings": settings,
        "mappings": mappings
    }
    print(f"대상 인덱스 생성: {url}")
    response = requests.put(url, headers=headers, data=json.dumps(payload))
    response.raise_for_status()
    print("대상 인덱스 생성 완료:")
    print(json.dumps(response.json(), indent=2))

def reindex_data():
    """Reindex API를 사용해 데이터를 복사"""
    url = f"{TARGET_CLUSTER}/_reindex?wait_for_completion=true"
    reindex_payload = {
        "source": {
            "remote": {
                "host": SOURCE_CLUSTER
            },
            "index": SOURCE_INDEX
        },
        "dest": {
            "index": DEST_INDEX
        }
    }
    print("대상 클러스터에서 Reindex 작업 시작...")
    response = requests.post(url, headers=headers, data=json.dumps(reindex_payload))
    response.raise_for_status()
    result = response.json()
    print("Reindex 작업 결과:")
    print(json.dumps(result, indent=2))

def main():
    print(f"오늘(한국시간): {today_kst.strftime('%Y-%m-%d')}")
    print(f"전날 날짜: {yesterday.strftime('%Y-%m-%d')}")
    print(f"소스 인덱스: {SOURCE_INDEX}")
    print(f"대상 인덱스: {DEST_INDEX}")
    try:
        source_settings, source_mappings = get_source_index_info()
        cleaned_settings = clean_settings(source_settings)
        create_target_index(cleaned_settings, source_mappings)
        reindex_data()
        print("\n작업 완료")
    except Exception as e:
        print("작업 실패:", e)
        sys.exit(1)

if __name__ == "__main__":
    main()
