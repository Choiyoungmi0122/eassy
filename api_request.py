import requests
import json

request_headers = {
    'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36'
}

url = "https://api.linkareer.com/graphql?operationName=CoverLetterList&variables=%7B%22filterBy%22%3A%7B%22organizationName%22%3A%22%22%2C%22role%22%3A%22%EC%9B%B9%EA%B0%9C%EB%B0%9C%22%2C%22keyword%22%3A%22%22%2C%22types%22%3A%5B%22ALL%22%5D%2C%22status%22%3A%22PUBLISHED%22%7D%2C%22orderBy%22%3A%7B%22field%22%3A%22RELEVANCE%22%2C%22direction%22%3A%22DESC%22%7D%2C%22pagination%22%3A%7B%22page%22%3A1%2C%22pageSize%22%3A20%7D%7D&extensions=%7B%22persistedQuery%22%3A%7B%22version%22%3A1%2C%22sha256Hash%22%3A%225c0cd537900df7638b0a29b29264285662fde73d1ac48c06403987051b144fde%22%7D%7D"

response = requests.get(url, headers=request_headers)

data = json.loads(response.text)

data = data.get('data').get('coverLetters').get('edges')

data_list = []

for dataa in data:
    data_list.append(dataa.get('node'))

with open('./1aa.json', 'w', encoding='utf-8') as json_file:
    json.dump(data_list, json_file, ensure_ascii=False, indent = 4, default=str)