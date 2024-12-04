import json
import re
with open('data/information.json') as f:
    data = json.load(f)

# JSON 데이터를 문자열로 변환
data_str = json.dumps(data)

# '{}'를 기준으로 문자열을 분할하여 리스트 생성
data_list = []
start_index = 0
while True:
    start_brace = data_str.find('{', start_index)
    if start_brace == -1:
        break
    end_brace = data_str.find('}', start_brace)
    if end_brace == -1:
        break
    json_str = data_str[start_brace:end_brace+1]
    data_list.append(json.loads(json_str))
    start_index = end_brace + 1

# 각 요소에서 'title' 키의 값을 출력
for item in data_list:
    del item['title']
    del item['spec']
    del item['source_url']

# print(data_list[1])
# print(data_list[2])

# 데이터를 문자열로 변환
data_str = json.dumps(data)

# '{}'를 기준으로 문자열을 분할하여 리스트 생성
data_list = re.split(r'\{\s*\}', data_str)

# 각 요소에서 ','를 기준으로 분할하여 최종 데이터 리스트 생성
final_data = []
for item in data_list:
    final_data.extend(item.split(','))

print(final_data)