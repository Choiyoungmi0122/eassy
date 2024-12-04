import json
from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel('BAAI/bge-m3',  
                       use_fp16=True) # Setting use_fp16 to True speeds up computation with a slight performance degradation
with open("/Users/choiyoungmi/Desktop/code/essay/test/data/data_test.json", "r") as f:
    rows = json.load(f)
for i, row in enumerate(rows):
    item = list(map(lambda x: x.split("\n"), row))
    rows[i] = item[0]
for sentence in rows:
    embeddings = model.encode(sentence[0], 
                            batch_size=12, 
                            max_length=8192, # If you don't need such a long length, you can set a smaller value to speed up the encoding process.
                            )['dense_vecs']
    print(sentence[0])
    print(embeddings)
    input()