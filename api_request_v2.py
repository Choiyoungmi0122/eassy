import requests
import json

# 기본 URL
base_url = "https://api.linkareer.com/graphql?operationName=CoverLetterList&variables=%7B%22filterBy%22%3A%7B%22types%22%3A%5B%22ALL%22%5D%2C%22status%22%3A%22PUBLISHED%22%7D%2C%22orderBy%22%3A%7B%22field%22%3A%22PASSED_AT%22%2C%22direction%22%3A%22DESC%22%7D%2C%22pagination%22%3A%7B%22page%22%3A{}%2C%22pageSize%22%3A20%7D%7D&extensions=%7B%22persistedQuery%22%3A%7B%22version%22%3A1%2C%22sha256Hash%22%3A%225c0cd537900df7638b0a29b29264285662fde73d1ac48c06403987051b144fde%22%7D%7D"


request_headers = {
    'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36'
}

def fetch_data(page):
    url = base_url.format(page)
    response = requests.get(url, headers=request_headers)
    
    if response.status_code == 200:
        return json.loads(response.text)
    else:
        print(f"Failed to fetch page {page}: {response.status_code}")
        return None

page = 1
data_list = []

while True:
    data = fetch_data(page)
    
    print("Current page: ", page)
    if not data or not data.get('data').get('coverLetters').get('edges'):  # 데이터가 없으면 종료
        print(f"No more data on page {page}. Stopping.")
        break
    
    # 데이터 처리
    edges = data.get('data').get('coverLetters').get('edges')
    
    for edge in edges:
        data_list.append(edge.get('node'))

    page += 1  # 다음 페이지로 이동

# JSON 파일로 저장
with open('./datacollection_1001.json', 'w', encoding='utf-8') as json_file:
    json.dump(data_list, json_file, ensure_ascii=False, indent=4, default=str)

print("Data successfully saved to datacollection_1001.json")
