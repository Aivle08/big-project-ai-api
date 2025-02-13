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
from etc.validator import format_docs
####################################################################################################################
################################################### STATE ########################################################### 청크 합치기
# 환경설정
load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY')
embeddings = OpenAIEmbeddings()


def input(state: QuestionState):
    """
    langgraph 시작 노드

    Args:
        state (QuestionState): 질문의 상태 정보를 포함하는 객체.

    Returns:
        QuestionState: 질문의 상태 정보 
    """
    return state
    
def retrieve_document(state: QuestionState, collection_name: str, class_id: str):
    """
    문서 검색 노드

    Args:
        state (QuestionState): 질문의 상태 정보를 포함하는 객체.
        collection_name (str): zilliz collection명 (예: "resume" 또는 "evaluation").
        class_id (str): 지원자, 회사 고유 ID

    Returns:
        QuestionState: 질문의 상태 정보 
    """
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
    # 검색한 chunk 저장
    for chunk in retrieved_docs:
        state['resume_chunk'].append(chunk.page_content)
    # 검색된 문서를 형식화합니다.(프롬프트 입력으로 넣어주기 위함)
    retrieved_docs = format_docs(retrieved_docs)
    
    # 검색된 문서를 context 키에 저장합니다.
    return retrieved_docs

def relevance_check(state: QuestionState, key: str):
    """
    기술 중심 자소서의 질문과 문맥 간 관련성을 평가하는 함수.

    Args:
        state (QuestionState): 현재 질문의 상태를 저장하는 객체.
        key (str): 관련성을 평가할 대상의 키값 (예: "resume" 또는 "evaluation").

    Returns:
        str: 관련성이 있는 경우 "yes", 관련성이 없는 경우 "no".
    """
    # 관련성 평가기를 생성합니다.
    question_answer_relevant = GroundednessChecker(
        llm=ChatOpenAI(model="gpt-4o", temperature=0), target="generate-question-retrieval"
    ).create()

    # 관련성 체크를 실행("yes" or "no")
    response = question_answer_relevant.invoke(
        {"question": state[f'{key}_query'], "context1": state[key]}
    )

    print(f"==== [{key} CHECK] ====")
    print(response.score)

    # 참고: 여기서의 관련성 평가기는 각자의 Prompt 를 사용하여 수정할 수 있습니다. 여러분들의 Groundedness Check 를 만들어 사용해 보세요!
    return response.score

def experience_work_fact_checking(state: QuestionState, key: str):
    """
    경험 중심 또는 경력 중심 자소서의 질문과 문맥 간 관련성을 평가하는 함수.

    Args:
        state (QuestionState): 현재 질문의 상태를 저장하는 객체.
        key (str): 관련성을 평가할 대상의 키값 (예: "resume" 또는 "evaluation").

    Returns:
        str: 관련성이 있는 경우 "yes", 관련성이 없는 경우 "no".
    """
    # 관련성 평가기를 생성합니다.
    question_answer_relevant = GroundednessChecker(
        llm=ChatOpenAI(model="gpt-4o", temperature=0), target="score-question-retrieval"
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
    """
    Query 재작성 노드

    Args:
        state (QuestionState): 질문의 상태 정보를 포함하는 객체.
        prompt (PromptTemplate): 질문 생성을 위한 프롬프트 템플릿(기술 중심, 경력 중심, 경험 중심 프롬프트)
        collection_name (str): 지원자, 회사 고유 ID

    Returns:
        str: 재작성한 Query
    """
    print('이전 query: ', state[f'{collection_name}_query'])
    # 1. 모델 선언
    model = ChatOpenAI(model='gpt-4o', streaming=True)
    
    # 3. llm + PydanticOutputParser 바인딩 체인 생성
    chain = prompt | model | StrOutputParser()

    response = chain.invoke(
        {"question": state[f'{collection_name}_query']}
    )
    
    print('rewrite query: ', response)
    
    return response


# PydanticOutputParser
class question(BaseModel):
    # 질문 형식 지정
    interview:List[str] = Field(
        description="A list containing two interview questions as plain text strings."
    )

def combine_prompt(state: QuestionState, prompt: PromptTemplate):
    """
    지원자의 자소서, 평가 기준, 직무 정보를 기반으로 최종 질문을 생성하는 함수.

    Args:
        state (QuestionState): 현재 질문 생성에 필요한 상태 정보 저장 객체.
        prompt (PromptTemplate): 질문 생성을 위한 프롬프트 템플릿(기술 중심, 경력 중심, 경험 중심 프롬프트)

    Returns:
        dict: 생성된 최종 질문을 포함하는 딕셔너리. (예: {'final_question': [질문1, 질문2]})
    """
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

# 관련성 체크 노드 (기술 중심)
def fact_checking(state: QuestionState):
    """
    지원자의 자소서 및 회사 평가 기준을 기반으로 생성된 질문이 사실에 근거하고 있는지 검증하는 함수.

    Args:
        state (QuestionState): 현재 상태 정보를 저장하는 객체로, 자소서, 평가 기준, 생성된 질문을 포함.

    Returns:
        dict: 질문의 사실 여부를 나타내는 결과값을 포함하는 딕셔너리. (예: {'fact': 'yes' 또는 'no'})
    """
    # 1. 관련성 평가기를 생성
    question_answer_relevant = GroundednessChecker(
        llm=ChatOpenAI(model='gpt-4o', temperature=0), target="question-fact-check"
    ).create()

    # 2. 관련성 체크를 실행("yes" or "no")
    response = question_answer_relevant.invoke(
        {"original_document_1": state['resume'], 'original_document_2':state['evaluation'],"question": state['final_question']}
    )

    print("==== [FACT CHECK] ====")
    print(response.score)

    return {'fact':response.score}