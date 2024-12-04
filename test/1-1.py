import json
import random
from pprint import pprint
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import torch
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import warnings

# 경고 메시지 무시
warnings.filterwarnings("ignore", category=FutureWarning)

# 임베딩 모델 정의
model_name = 'BAAI/bge-m3'
tokenizer = AutoTokenizer.from_pretrained(model_name, force_download=True)
model = AutoModel.from_pretrained(model_name, force_download=True).eval()

def embed_text(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
    return embeddings

client = QdrantClient(url="http://localhost:6333")

def load_data():
    collection_name = "cover_letter_collection"
    
    # 컬렉션 생성
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE),
    )

    # 데이터 로드
    with open("/Users/choiyoungmi/Desktop/code/essay/test/data/final_information.json", "r") as f:
        rows = json.load(f)

    data = []
    count = 1
    for row in tqdm(rows, position=0):
        embeddings = embed_text(row["text"])
        data.append(
            PointStruct(id=row["id"], vector=embeddings[0], payload=row),
        )
        count += 1

    # 컬렉션에 데이터 삽입
    operation_info = client.upsert(
        collection_name=collection_name,
        wait=True,
        points=data,
    )
    print(operation_info)

def search_similar(text):
    embeddings = embed_text(text)
    
    search_result = client.search(
        collection_name="cover_letter_collection",
        query_vector=embeddings[0],
        with_payload=True,
        limit=3,
    )
    return search_result

# 데이터 로드
load_data()

# 검색 예시
text = "저는 성실하게 동아리 활동을 해오면서 책임감과 소속감을 길렀습니다."
similar_sentences = search_similar(text)
print(similar_sentences)
