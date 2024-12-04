from transformers import pipeline

# 텍스트 생성 모델 로드
generator = pipeline('text-generation', model='gpt-3.5-turbo')  # 실제 사용할 모델로 변경

def augment_cover_letter(input_text, similar_sentences):
    context = " ".join([res.payload["text"] for res in similar_sentences])
    prompt = f"{input_text}\n\n유사한 내용:\n{context}\n\n추가 보충:"
    augmented_text = generator(prompt, max_length=300, num_return_sequences=1)
    return augmented_text[0]['generated_text']

# 입력된 텍스트와 유사한 문장 검색
input_text = "저는 성실하게 동아리 활동을 해오면서 책임감과 소속감을 길렀으며, 저의 이런 장점은 회사의 발전에 많은 기여를 할 것입니다."
similar_sentences = search_similar(input_text)

# 보충된 자기소개서 생성
augmented_cover_letter = augment_cover_letter(input_text, similar_sentences)
print(augmented_cover_letter)
