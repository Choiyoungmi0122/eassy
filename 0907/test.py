import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import init_empty_weights, infer_auto_device_map

# 데이터셋 로드 (사용자가 업로드한 데이터셋 사용)
dataset_path = '/Users/choiyoungmi/Desktop/code/essay/0904/test_dataset_0907_v1.csv'
dataset = pd.read_csv(dataset_path)
# Youngmi02/eassy_testset_v1
# 모델과 토크나이저 설정
BASE_MODEL = "google/gemma-2b-it"
with init_empty_weights():
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)

# 디바이스 맵 설정 및 모델 로드
device_map = infer_auto_device_map(model, max_memory={"cpu": "48GiB", "cuda:0": "12GiB"}, no_split_module_classes=["GPT2Block"])
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map=device_map, offload_folder="offload")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, add_special_tokens=True)

# 사용자가 입력할 종류를 dataset의 question에서 선택하게 함
available_kinds = dataset['question'].unique()
print("선택 가능한 종류:")
for kind in available_kinds:
    print(f"- {kind}")

input_kind = input("종류를 입력하세요 (위 목록에서 선택): ")

# 사용자가 입력한 종류가 dataset에 있는지 확인
if input_kind not in available_kinds:
    raise ValueError("잘못된 종류를 입력했습니다. 목록에서 선택해주세요.")

# 사용자가 입력한 글자 받기
input_text = input(f"{input_kind}에 해당하는 글자를 입력하세요: ")

# 선택한 종류에 해당하는 답변을 데이터셋에서 가져옴
matching_answers = dataset[dataset['question'] == input_kind]['answer'].tolist()

# answer에서 랜덤으로 한 개를 선택해서 프롬프트에 추가
import random
selected_answer = random.choice(matching_answers)

# 프롬프트 생성: 사용자의 입력과 선택된 데이터셋 answer를 섞음
prompt = f"다음은 {input_kind}에 대한 내용입니다:\n{input_text}\n\n데이터셋에서 가져온 문장:\n{selected_answer}\n\n이 두 문장을 기반으로 새로운 문장을 생성하세요."

# 토큰화 처리
inputs = tokenizer(prompt, return_tensors="pt")

# 모델에 입력 전달 후 출력 확인
outputs = model.generate(
    **inputs,
    max_new_tokens=100,        # 최대 생성 토큰 수
    repetition_penalty=1.2,    # 반복 억제
    temperature=0.7,           # 다양성 조절
    top_p=0.9,                 # 상위 확률 토큰 선택
    top_k=50                   # 상위 k개 토큰만 고려
)

# 출력된 토큰을 텍스트로 변환
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

# 결과 출력
print("\n[생성된 문장]")
print(response)
