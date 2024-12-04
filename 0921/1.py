# 키워드 전체에 대한 유사도를 기반으로, 파인튜닝된 모델에서 그와 유사한 문장을 생성하는 방법을 구현하겠습니다. 여기서는 다음의 두 가지 단계를 수행합니다:

# BM25 유사도 계산: 주어진 키워드 리스트와 데이터셋에 있는 각 문서 간의 BM25 유사도를 계산하여, 유사도가 높은 문서를 찾습니다.
# 파인튜닝된 모델에서 문장 생성: 유사도가 높은 문서와 관련된 프롬프트를 사용해 파인튜닝된 모델에서 문장을 생성합니다.abs

from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 모델 로드
model_path = "/teamspace/studios/this_studio/outputv2/checkpoint-13800"  # 실제 경로로 변경
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# BM25용 토크나이저 함수
def tokenize_text(text):
    return text.split()

# BM25 유사도 계산 함수 정의
def get_similar_documents(keywords, corpus, top_n=3):
    tokenized_corpus = [tokenize_text(doc["text"]) for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    
    # 키워드를 하나의 문자열로 결합하여 쿼리 생성
    query = tokenize_text(' '.join(keywords))
    
    # BM25 유사도를 이용해 가장 유사한 문서 top_n개 추출
    scores = bm25.get_scores(query)
    top_n_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]
    
    # 유사도가 높은 문서들 반환
    return [corpus[i] for i in top_n_indices]

# 모델에서 문장 생성 함수 정의
def generate_sentence(prompt, model, tokenizer, max_length=150):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_length=max_length, do_sample=True, top_p=0.9)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# 입력 데이터 (question과 keyword 리스트)
input_data = [
    {
        "question": "입사 후 포부",
        "keyword": ["기술 혁신", "인공지능", "데이터 분석", "첨단 기술 개발", "혁신적인 프로젝트"]
    }
]

# 기존에 학습된 데이터셋에서 문장들과 키워드가 포함된 예시 데이터셋
corpus = [
    {
        "text": "저는 데이터 분석 전문가로서 성장하기 위해...",
        "keyword": ["데이터 분석"]
    },
    {
        "text": "기술 혁신을 통해 인공지능 기술을 연구하며...",
        "keyword": ["기술 혁신", "인공지능"]
    },
    {
        "text": "첨단 기술 개발 프로젝트에서 혁신적인 결과를...",
        "keyword": ["첨단 기술 개발", "혁신적인 프로젝트"]
    }
]

# 모델을 통해 자기소개서를 재생성할 때 프롬프트에 question과 keyword를 포함
finetuned_docs = []
for entry in input_data:
    question = entry["question"]
    keywords = entry["keyword"]
    
    # 키워드 전체를 사용해 유사한 문서 찾기
    similar_docs = get_similar_documents(keywords, corpus)
    
    # 유사한 문서 기반으로 프롬프트 생성 및 문장 생성
    for doc in similar_docs:
        prompt = f"자기소개서에서 '{question}'와 관련된 내용을 작성하세요. 특히 '{', '.join(doc['keyword'])}'와 관련된 내용을 추가하세요."
        
        # 모델에서 문장 생성
        generated_text = generate_sentence(prompt, model, tokenizer)
        
        # 생성된 문서와 키워드를 저장
        finetuned_docs.append({"question": question, "text": generated_text, "keyword": doc["keyword"]})

# 생성된 문서 출력
for idx, doc in enumerate(finetuned_docs, 1):
    print(f"{idx}. Question: {doc['question']}")
    print(f"   문서 내용: {doc['text']}")
    print(f"   키워드: {doc['keyword']}")
