from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import warnings
import os
import time
import csv
import google.generativeai as genai
from google.generativeai.types import HarmBlockThreshold, HarmCategory
from tqdm import tqdm
import os
# 경고 메시지 무시
warnings.filterwarnings("ignore")
# 2번 GPU만 사용하도록 설정
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
# 명시적으로 2번 GPU 설정
torch.cuda.set_device(0)
# GPU ID 확인
gpu_ids = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
real_gpu_num = gpu_ids[torch.cuda.current_device()]
# print('Real GPU number being used:', real_gpu_num)
# print('GPU Name:', torch.cuda.get_device_name(0))
# CUDA 장치 사용 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# 파인튜닝된 모델 경로
model_path = "/home/dxlab/team_dxlab_young/model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
causal_lm_model = AutoModelForCausalLM.from_pretrained(model_path).to(device)

#-----------------------------------------------------------------------------------------------
keys = ["APIKEY"]


def extract_keywords(question, answer):
    generation_config = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 8192,
        "response_mime_type": "application/json",
    }
    rows = []
    
    genai.configure(api_key="AIzaSyDYv_EhbPMWTApxQgsjHUqGDKLeTUSu-aM")
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config,
    )
    chat_session = model.start_chat(
            history=[
                {
                    "role": "user",
                    "parts": 
                        f"<자기소개서 카테고리>\n{question}\n</자기소개서 카테고리>\n\n<내용>\n{answer}\n</내용>\n\n<지시사항>1. 위 <내용>을 참고하여 핵심 키워드를 나열해줘.\n2. 나열된 키워드는 <자기소개서 카테고리>와 관련된 항목에 맞춰서 구체적인 키워드와 더불어, 내용 전체에서 유의미한 일반적인 키워드도 포함해줘.\n3. 출력 형식은 keyword: List[str] \n4. 카테고리 기반 키워드는 먼저 나열하고, 그 다음 일반 키워드를 추가해줘.</지시사항>"
                },
            ]
        )
    
    try:
            response = chat_session.send_message(answer, safety_settings={
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            })
            keywords_text = response.text.strip()
            keyword_list = keywords_text[keywords_text.find("[") + 1:keywords_text.find("]")].replace("'", "").split(", ")
        
            return keyword_list
    
    except Exception as e:
        if "Rate limit" in str(e):
            print("Rate limit encountered. Pausing for a while.")
            time.sleep(60)
        else:
            raise e

        
def generate_sentence(original_content, keywords, category):
    # 키워드가 리스트가 아니라면 빈 리스트로 처리
    if not isinstance(keywords, list):
        keywords = []
        
    prompt = (
        f"<start_of_turn>user\n"
        f"원본 내용:\n{original_content}\n\n"  
        f"글쓰기 유형: {category}\n\n" 
        f"키워드:\n- " + "\n- ".join(keywords) + "\n\n"
        f"위 원본 내용을 글쓰기 유형의 목적에 맞게 자연스럽고 문맥에 맞게 작성해 주세요.\n"
        f"**반드시 한국어로만 작성해 주세요. 영어 또는 다른 언어는 절대 포함되지 않도록 주의하세요.**\n"
        f"350자를 넘지 않아도 괜찮습니다. 필요한 경우 문장이 자연스럽게 끝나면 멈추세요.\n"
        f"<end_of_turn>\n<start_of_turn>assistant\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output = causal_lm_model.generate(**inputs, max_new_tokens=450, do_sample=False, temperature=0.7)
    
    # 생성된 텍스트 디코딩
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    # 'assistant' 이후의 내용만 추출
    assistant_index = generated_text.find("assistant")
    if assistant_index != -1:
        generated_text = generated_text[assistant_index + len("assistant"):].strip()
    return generated_text



question = input("자기소개서 작성 카테고리를 입력하세요(지원 동기, 경험, 성격의 장단점, 직무 역량, 협업 경험, 입사 후 포부, 기타, 자유 양식, 성장 과정, 사회 이슈): ")
answer = input("내용을 입력하세요: ")
# 키워드 추출 실행

keywords = extract_keywords(question, answer)
generated_text = generate_sentence(answer, keywords, question)
print("\n\nGenerated Text:\n", generated_text)
