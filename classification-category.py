from transformers import AutoTokenizer, AutoModel
from transformers import AutoProcessor, AutoModelForImageTextToText
import torch
from torch.nn.functional import cosine_similarity
import requests
from PIL import Image
from io import BytesIO
import numpy as np

# GPU 사용 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 텍스트 임베딩 모델 로드
text_tokenizer = AutoTokenizer.from_pretrained("BM-K/KoSimCSE-roberta-multitask")
text_model = AutoModel.from_pretrained("BM-K/KoSimCSE-roberta-multitask").to(device)

# 이미지 캡셔닝 모델 로드
image_processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
image_model = AutoModelForImageTextToText.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

category_list = {
    1: {"label": "치마", "url": "https://d3ha2047wt6x28.cloudfront.net/KYuSCraubzs/pr:GOODS_DETAIL/czM6Ly9hYmx5LWltYWdlLWxlZ2FjeS9kYXRhL2dvb2RzLzIwMjIwMzMwXzE2NDg2MDY3ODY1MjY3MTVtLmpwZw"},
    2: {"label": "신발", "url": "https://sitem.ssgcdn.com/09/97/17/item/1000442179709_i1_750.jpg"},
    3: {"label": "가방", "url": "https://bagstay.co.kr/web/product/big/202304/7575fd546cbc4527cddfd5f663aeb919.jpg"},
    4: {"label": "액세서리", "url": "https://digitalchosun.dizzo.com/site/data/img_dir/2016/11/29/2016112911276_0.jpg"},
    5: {"label": "전자기기", "url": "https://cdn.hkbs.co.kr/news/photo/202007/579805_333006_5759.jpg"},
    6: {"label": "뷰티", "url": "https://image.ajunews.com/content/image/2020/08/30/20200830104026237326.jpg"},
    7: {"label": "식품", "url": "https://cdn.mindgil.com/news/photo/202110/72631_10632_3251.jpg"},
    11: {"label": "도서/DVD", "url": "https://media.bunjang.co.kr/product/253999766_5_1708439692_w360.jpg"},
}

def get_text_embedding(text):
    """텍스트를 임베딩 벡터로 변환"""
    inputs = text_tokenizer(text, return_tensors='pt', padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = text_model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :].to(device)  # CLS 토큰 임베딩 사용
    return embeddings

def get_image_caption(image_url):
    """이미지 URL에서 이미지를 다운로드하고 캡션 생성"""
    try:
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content)).convert('RGB')
        
        # 이미지 전처리
        inputs = image_processor(images=image, return_tensors="pt").to(device)
        
        # 캡션 생성
        with torch.no_grad():
            outputs = image_model.generate(**inputs, max_length=30)
            caption = image_processor.batch_decode(outputs, skip_special_tokens=True)[0]
        
        return caption
    except Exception as e:
        print(f"이미지 캡션 생성 중 오류 발생: {e}")
        return "이미지를 처리할 수 없습니다."

def calculate_similarity(query_embedding, category_embedding):
    """두 임베딩 간의 코사인 유사도 계산"""
    return cosine_similarity(query_embedding, category_embedding).item()

def classify_query(query, weight_text=0.7, weight_image=0.3):
    """
    검색어를 입력받아 카테고리별 유사도 계산 및 순위 매기기
    
    Args:
        query: 검색어
        weight_text: 텍스트 유사도 가중치
        weight_image: 이미지 유사도 가중치
    
    Returns:
        카테고리 ID와 유사도 점수를 포함한 정렬된 리스트
    """
    # 검색어 임베딩
    query_embedding = get_text_embedding(query)
    
    results = []
    
    for category_id, category_info in category_list.items():
        # 카테고리 레이블 임베딩
        label_embedding = get_text_embedding(category_info["label"])
        
        # 이미지 캡션 생성 및 임베딩
        image_caption = get_image_caption(category_info["url"])
        image_caption_embedding = get_text_embedding(image_caption)
        
        # 텍스트 유사도 계산
        text_similarity = calculate_similarity(query_embedding, label_embedding)
        
        # 이미지 캡션 유사도 계산
        image_similarity = calculate_similarity(query_embedding, image_caption_embedding)
        
        # 가중 평균 유사도 계산
        combined_similarity = (weight_text * text_similarity) + (weight_image * image_similarity)
        
        results.append({
            "category_id": category_id,
            "label": category_info["label"],
            "image_caption": image_caption,
            "text_similarity": text_similarity,
            "image_similarity": image_similarity,
            "combined_similarity": combined_similarity
        })
    
    # 유사도 점수에 따라 내림차순 정렬
    sorted_results = sorted(results, key=lambda x: x["combined_similarity"], reverse=True)
    
    return sorted_results

# 사용 예시
if __name__ == "__main__":
    # 검색어 입력
    search_query = input("검색어를 입력하세요: ")
    
    # 분류 실행
    classification_results = classify_query(search_query)
    
    # 결과 출력
    print("\n=== 검색 결과 ===")
    for i, result in enumerate(classification_results):
        print(f"{i+1}. 카테고리: {result['label']} (ID: {result['category_id']})")
        print(f"   이미지 캡션: {result['image_caption']}")
        print(f"   텍스트 유사도: {result['text_similarity']:.4f}")
        print(f"   이미지 유사도: {result['image_similarity']:.4f}")
        print(f"   종합 유사도: {result['combined_similarity']:.4f}")
        print()