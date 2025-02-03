################################################## LIBRARY #########################################################
# Basic
import os
import openai
from dotenv import load_dotenv

# Chain
from langchain_milvus import Milvus
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Tool
from pydantic import BaseModel, Field

# DB
from typing import TypedDict, Annotated, List

# Error
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Module
from state.question_state import QuestionState
from etc.evaluator import GroundednessChecker
from etc.etcc import format_docs
####################################################################################################################
################################################### STATE ########################################################### 청크 합치기
# 환경설정
load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY')
embeddings = OpenAIEmbeddings()

# Input 노드 정의
def input(state: QuestionState):
    return state
    
# 문서 검색 노드
def retrieve_document(state: QuestionState, collection_name: str, class_id: str):
    # 질문을 상태에서 가져옵니다.
    latest_question = state[f"{collection_name}_query"]
    state_class_id = state[f'{class_id}']
    # Milvus vectorstore 설정
    vectorstore_resume = Milvus(
        collection_name=collection_name,  # 기존 컬렉션 이름
        embedding_function=embeddings,  # 임베딩 함수
        connection_args={
            "uri": os.environ['CLUSTER_ENDPOINT'],
            "token": os.environ['TOKEN'],
        }
    )
    # vectorstore를 retriever로 변환
    retriever = vectorstore_resume.as_retriever(search_kwargs={
            'expr' : f"{class_id} == {state_class_id}",
        }
    )
    
    # 문서에서 검색하여 관련성 있는 문서를 찾습니다.
    retrieved_docs = retriever.invoke(latest_question)
    # 검색된 문서를 형식화합니다.(프롬프트 입력으로 넣어주기 위함)
    retrieved_docs = format_docs(retrieved_docs)
    
    # 검색된 문서를 context 키에 저장합니다.
    return retrieved_docs

# 기술 중심 자소서 관련성 체크 노드
def relevance_check(state: QuestionState, key: str):
    # 관련성 평가기를 생성합니다.
    question_answer_relevant = GroundednessChecker(
        llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0), target="generate-question-retrieval"
    ).create()

    # 관련성 체크를 실행("yes" or "no")
    response = question_answer_relevant.invoke(
        {"question": state[f'{key}_query'], "context1": state[key]}
    )

    print(f"==== [{key} CHECK] ====")
    print(response.score)

    # 참고: 여기서의 관련성 평가기는 각자의 Prompt 를 사용하여 수정할 수 있습니다. 여러분들의 Groundedness Check 를 만들어 사용해 보세요!
    return response.score

# 경험 중심 자소서 관련성 체크 노드
def experience_relevance_check(state: QuestionState, key: str):
    # 관련성 평가기를 생성합니다.
    question_answer_relevant = GroundednessChecker(
        llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0), target="score-question-retrieval"
    ).create()

    # 관련성 체크를 실행("yes" or "no")
    response = question_answer_relevant.invoke(
        {"question": state[f'{key}_query'], "context1": state[key]}
    )

    print(f"==== [{key} CHECK] ====")
    print(response.score)

    # 참고: 여기서의 관련성 평가기는 각자의 Prompt 를 사용하여 수정할 수 있습니다. 여러분들의 Groundedness Check 를 만들어 사용해 보세요!
    return response.score

def rewrite_question(state: QuestionState, prompt: PromptTemplate, collection_name: str):
    # 1. 모델 선언
    model = ChatOpenAI(model='gpt-4o', streaming=True)
    
    # 3. llm + PydanticOutputParser 바인딩 체인 생성
    chain = prompt | model | StrOutputParser()

    response = chain.invoke(
        {"question": state[f'{collection_name}_query']}
    )
    
    return response


# PydanticOutputParser
class question(BaseModel):
    # 질문 형식 지정
    interview:List[str] = Field(
        description="A list containing two interview questions as plain text strings."
    )

# 결과 종합 노드
def combine_prompt(state: QuestionState, prompt: PromptTemplate):
    # State 변수 선언 지정
    job = state['job']
    resume = state['resume']
    evaluation = state['evaluation']
    
    # 1. 모델 선언
    model = ChatOpenAI(model='gpt-4o', streaming=True)
    
    # 2. 구조화된 출력을 위한 LLM 설정
    llm_with_tool = model.with_structured_output(question)
    
    # 3. llm + PydanticOutputParser 바인딩 체인 생성
    chain = prompt | llm_with_tool

    # 4. 질문 생성 LLM 실행
    answer_middle = chain.invoke({'resume' : resume, 'evaluation' : evaluation, 'job' : job})

    # 5. 질문 추출
    question_score = answer_middle.interview
    
    return {'final_question':question_score}

# 관련성 체크 노드
def fact_checking(state: QuestionState):
    # 1. 관련성 평가기를 생성
    question_answer_relevant = GroundednessChecker(
        llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0), target="question-fact-check"
    ).create()

    # 2. 관련성 체크를 실행("yes" or "no")
    response = question_answer_relevant.invoke(
        {"original_document_1": state['resume'], 'original_document_2':state['evaluation'],"question": state['final_question']}
    )

    print("==== [FACT CHECK] ====")
    print(response.score)

    return {'fact':response.score}