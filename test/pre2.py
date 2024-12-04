import json
import pandas as pd 
from collections import Counter
# path = 'data/final_information.json'
path = '/Users/choiyoungmi/Desktop/code/essay/test/data/information.json'

with open(path, 'r', encoding="utf-8-sig") as f:
    data = json.load(f)



# counter = Counter(data)

# print(counter)
# # print(counter[""])
print(len(data))