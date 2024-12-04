# Import Libraries
from IPython.display import Markdown
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAI
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
import json
from sentence_transformers import SentenceTransformer, util
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema.runnable import RunnableMap
import os
from transformers import AutoTokenizer
import torch
import warnings

# Suppress specific FutureWarning related to tokenization spaces
# warnings.filterwarnings("ignore", category=FutureWarning, message="`clean_up_tokenization_spaces` was not set")
warnings.filterwarnings("ignore", category=FutureWarning)

# Initialize Gemini API (using ChatGoogleGenerativeAI)
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

gemini = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

def make_gemini_text(chain, question: str) -> str:
    new_question = question
    result = chain.invoke({'question': new_question})

    Markdown(result.content)

    return result.content

def chainMaker():
    with open("./information-allfinal_result.json", "r", encoding="utf-8") as file:
        data = json.load(file)
    
    texts_to_split = [item for sublist in data for item in sublist]

    model = SentenceTransformer('jhgan/ko-sbert-nli')

    corpus_embeddings = model.encode(texts_to_split, convert_to_tensor=True)

    retriver = BM25Retriever.from_texts(texts_to_split)

    prompt_template = """
        Given the following question and relevant context, provide a refined response:

        Question: {question}

        Context: {context}

        Refined Response:
    """

    prompt = PromptTemplate(template=prompt_template, input_variables=["question", "context"])

    chain = RunnableMap({
        "context": lambda x: retriver.invoke(x['question']),
        "question": lambda x: x['question']
    }) | prompt | gemini

    return chain

new_question = "저는 성실합니다. 이런 저의 장점이 회사에 도움이 될거라고 생각합니다."

chain = chainMaker()

gemini_text = make_gemini_text(chain, new_question)

print(gemini_text)






# # Load your JSON data
# with open("./information-allfinal.json", "r", encoding="utf-8") as file:
#     data = json.load(file)

# texts_to_split = [item for sublist in data for item in sublist]

# # Initialize SentenceTransformer model for embeddings
# model = SentenceTransformer('jhgan/ko-sbert-nli')

# # Encode the texts using SentenceTransformer
# corpus_embeddings = model.encode(texts_to_split, convert_to_tensor=True)

# # Initialize your retriever
# retriever = BM25Retriever.from_texts(texts_to_split)  # Updated to BM25Retriever from langchain_community



# # Define the prompt template
# prompt_template = """
# Given the following question and relevant context, provide a refined response:

# Question: {question}

# Context: {context}

# Refined Response:
# """

# prompt = PromptTemplate(template=prompt_template, input_variables=["question", "context"])

# # Define a chain that first retrieves context, then generates the response using Gemini
# chain = RunnableMap({
#     "context": lambda x: retriever.invoke(x['question']),  # Ensure x['question'] is a string
#     "question": lambda x: x['question']
# }) | prompt | gemini

# # New input from the user
# # new_question = "저의 첫 번째 장점은 세심함입니다. 저는 학부연구생으로서 크리스틴이라는 회사와의 산학과제 협업을 통해 이 성격의 장점을 발휘한 경험이 있습니다. 데이터 분석 과정에서 세심한 주의를 기울이며, 데이터의 정확성을 보장하려고 합니다. 작업 중 발견한 데이터 충돌로 인한 중복 오류를 신속하게 수정하고, 시간을 들여 데이터가 중복이 일어날 시에 데이터 중복을 제거하기 위해 새로운 데이터와 비교하여 더 좋은 데이터를 채택하도록 처리하였습니다."
# new_question = "저는 성실합니다. 이런 저의 장점이 회사에 도움이 될거라고 생각합니다."
# # Run the chain
# result = chain.invoke({'question': new_question})

# # Display the result
# Markdown(result.content)  # Directly access the content attribute of the AIMessage object
# print(result.content)