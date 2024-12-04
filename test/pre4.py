import json
import pandas as pd 
import re

path = 'data/final_information.json'

with open(path, 'r', encoding="utf-8-sig") as f:
    data = json.load(f)

data_str = '\n'.join(str(item) for item in data)

# print(type(data_str))

# 문자열에서 []를 기준으로 분할하여 출력하는 함수
def print_segments(input_string):
    segments = input_string.split('[')
    for segment in segments:
        print(segment.strip())
        new_segments = segments.copy()


# 함수 호출
print_segments(data_str)
print(new_segments)
first_key = list(result_dict.keys())[0]
first_value = result_dict[first_key]
# print(f"첫 번째 딕셔너리의 키: {first_key}")
# print(f"첫 번째 딕셔너리의 값: {first_value}")

