import json
import random
from pprint import pprint
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import torch
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# 임베딩 모델 정의
model_name = 'BAAI/bge-m3'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).eval()

def embed_text(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
    return embeddings

client = QdrantClient(url="http://localhost:6333")

def load():
    collection_name = "sa_collection_1"
    
    # 컬렉션 생성
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE),
    )

    # 데이터 로드
    with open("./data/final_information.json", "r") as f:
        rows = json.load(f)

    # 데이터 전처리
    for i, row in enumerate(rows):
        items = list(map(lambda x: x.split("\n"), row))[0]
        for j, item in enumerate(items):
            if j == 0:
                items[j] = {
                    "doc_id": i,
                    "category": "title",
                    "content": item,
                }
            else:
                items[j] = {
                    "doc_id": i,
                    "category": "content",
                    "content": item
                }
        rows[i] = items

    random.shuffle(rows)
    rows = rows[:10]  # 샘플로 10개의 데이터만 사용

    data = []
    count = 1
    for row in tqdm(rows, position=0):
        for item in tqdm(row, position=1):
            embeddings = embed_text(item["content"])
            data.append(
                PointStruct(id=count, vector=embeddings[0], payload=item),
            )
            count += 1

    # 컬렉션에 데이터 삽입
    operation_info = client.upsert(
        collection_name=collection_name,
        wait=True,
        points=data,
    )
    print(operation_info)

def search():
    text = input("검색어 입력: ")
    embeddings = embed_text(text)
    
    search_result = client.search(
        collection_name="sa_collection_1",
        query_vector=embeddings[0],
        with_payload=True,
        limit=3,
    )
    pprint(search_result)

if __name__ == "__main__":
    load()
    search()
