import json
import random
from pprint import pprint
from tqdm import tqdm
from FlagEmbedding import BGEM3FlagModel
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from qdrant_client.models import Filter, FieldCondition, MatchValue

# 임베딩 모델 정의
model = BGEM3FlagModel('BAAI/bge-m3',  
                       use_fp16=True) # Setting use_fp16 to True speeds up computation with a slight performance degradation
client = QdrantClient(url="http://localhost:6333")

def load():
    collection_name = "sa_collection_2"
    # 컬렉션 생성
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
    )

    # 데이터 로드
    with open("./data/final_information.json", "r") as f :
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
    rows = rows[:10]
    data = []
    count = 1
    for row in tqdm(rows, position=0):
        for item in tqdm(row, position=1):
            embeddings = model.encode(item["content"], 
                                batch_size=12, 
                                max_length=8192, # If you don't need such a long length, you can set a smaller value to speed up the encoding process.
                                )['dense_vecs']
            data.append(
                PointStruct(id=count, vector=embeddings, payload=item),
            )
            count += 1

    # 컬렉션에 데이터 삽입
    operation_info = client.upsert(
        collection_name=collection_name,
        wait=True,
        points=data,
    )

load()

def search():
    text = input("검색어 입력:")
    embeddings = model.encode(text, 
                        batch_size=12, 
                        max_length=8192, # If you don't need such a long length, you can set a smaller value to speed up the encoding process.
                        )['dense_vecs']
    search_result = client.search(
        collection_name="sa_collection",
        query_vector=embeddings,
        # query_filter=Filter(
        #     must=[FieldCondition(key="city", match=MatchValue(value="London"))]
        # ),
        with_payload=True,
        limit=3,
    )
    pprint(search_result)
