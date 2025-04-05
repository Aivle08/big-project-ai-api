################################################## LIBRARY #########################################################
# Basic
import os
import openai
from dotenv import load_dotenv

# Chain
from langchain_milvus import Milvus
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate

# Tool
from pydantic import BaseModel, Field

# DB
from typing import TypedDict, Annotated, List,Dict, Any

# Error
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Module
from state.score_state import ScoreState
from etc.evaluator import GroundednessChecker
from etc.validator import format_docs
from retriever.hybrid import hybrid_retriever
####################################################################################################################
################################################### STATE ########################################################### 청크 합치기
# 환경설정
load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY')
embeddings = OpenAIEmbeddings()
    
# Retriever 노드
def retrieve_document(state: ScoreState, collection_name: str, class_id: str):
    """
    벡터 데이터베이스(Zilliz)를 활용하여 지정된 컬렉션에서 관련 문서를 검색하는 함수.
    
    Args:
        state (ScoreState): 현재 상태 정보를 저장하는 객체로, 검색할 질문(query) 및 식별자(class_id)를 포함.
        collection_name (str): 검색을 수행할 Zilliz 컬렉션의 이름.
        class_id (str): 검색 대상의 특정 ID(예: applicant_id, company_id 등).

    Returns:
        dict: 검색된 문서를 지정된 collection_name 키에 저장하여 반환.
    """

    latest_question = state["query_main"]
    state_class_id = state[f'{class_id}']
    
    # vectorstore를 retriever로 변환
    retriever = hybrid_retriever(collection_name,state_class_id,class_id)
    
    retrieved_docs = retriever.invoke(latest_question)
    #print('chunks: ', retrieved_docs)
    # 검색한 chunk 저장
    for chunk in retrieved_docs:
        state['resume_chunk'].append(chunk.page_content)
    
    retrieved_docs = format_docs(retrieved_docs)
    
    #print(retrieved_docs)
    
    # 검색된 문서를 context 키에 저장합니다.
    return {f'{collection_name}':retrieved_docs}


# 관련성 체크 노드
def relevance_check(state: ScoreState):
    """
    지원자의 자기소개서와 평가 기준이 주어진 질문과 관련성이 있는지 평가하는 함수.

    Args:
        state (ScoreState): 현재 상태 정보를 저장하는 객체로, 검색할 질문(query) 및 지원자의 자기소개서(resume)를 포함.

    Returns:
        dict: 관련성이 있는 경우 'yes', 관련성이 없으면 'no'를 반환하는 딕셔너리.
    """
    # 관련성 평가기를 생성합니다.
    question_answer_relevant = GroundednessChecker(
        llm=ChatOpenAI(model="gpt-4o", temperature=0), target="score-question-retrieval"
    ).create()

    # 관련성 체크를 실행("yes" or "no")
    response = question_answer_relevant.invoke(
        {"question": state['query_main'], "context1": state['resume']}
    )

    print("==== [RELEVANCE CHECK] ====")
    print(response.score)
    
    # 참고: 여기서의 관련성 평가기는 각자의 Prompt 를 사용하여 수정할 수 있습니다. 여러분들의 Groundedness Check 를 만들어 사용해 보세요!
    return {'yes_or_no':response.score}


def no_relevance(state: ScoreState):
    """
    관련성이 없는 경우 지원자에게 최하 점수(0점)를 부여하는 함수.

    Args:
        state (ScoreState): 현재 상태 정보를 저장하는 객체로, 평가 항목(eval_item)과 점수를 저장.

    Returns:
        ScoreState: 평가 점수가 포함된 상태 객체를 반환.
    """
    state['eval_resume'] = {'eval_resume': ['0',f"지원자는 {state['eval_item']}에 관련 정보가 부족하여 최하 점수를 부여하였습니다."]}
    return state

# PydanticOutputParser
class score(BaseModel):    # 점수 산출 형식 지정    
    score_output: List[str] = Field(
        description="""
        A List containing evaluation scores for different resume categories.
        The first element is a score between 0 and 100, and the second element is a description as a string.
        """    
    )

# 자소서 점수 측정 노드
def score_resume(state: ScoreState, prompt: PromptTemplate):
    """
    지원자의 자기소개서를 평가하여 점수를 부여하는 함수.

    Args:
        state (ScoreState): 현재 상태 정보를 저장하는 객체로, 지원자의 이력서(resume), 평가 항목(eval_item),
                            평가 기준 내용(eval_item_content), 지원 직무(job) 정보를 포함함.
        prompt (PromptTemplate): LLM에 전달할 프롬프트 템플릿.

    Returns:
        dict: 평가 점수 및 평가 근거를 포함하는 딕셔너리. 
              {"eval_resume": [점수(str), 평가 근거(str)]} 형식으로 반환됨.
    """
    
    # State 변수 선언
    job = state['job']
    resume = state['resume']
    eval_item = state['eval_item']
    eval_item_content = state['eval_item_content']

    # 1. 모델 선언
    model = ChatOpenAI(model='gpt-4o', streaming=True, temperature=0)
    
    # 2. 구조화된 출력을 위한 LLM 설정
    llm_with_tool = model.with_structured_output(score)

    # PromptTemplate + model + StrOutputParser 체인 생성
    chain = prompt | llm_with_tool

    answer_middle = chain.invoke({
            "resume": resume,
            "eval_item_content": eval_item_content,
            "job": job,
            "eval_item": eval_item
        })
    
    # 질문 추출
    question_score = answer_middle.score_output
    
    return {'eval_resume':question_score}

# 팩트 체크 노드
def fact_checking(state: ScoreState):
    """
    지원자의 자기소개서 평가 결과가 사실에 기반한 내용인지 검증하는 함수.

    Args:
        state (ScoreState): 현재 상태 정보를 저장하는 객체로, 지원자의 자기소개서(resume),
                            평가 기준 내용(eval_item_content), 평가 결과(eval_resume)를 포함함.

    Returns:
        dict: 평가 결과의 사실 여부를 나타내는 딕셔너리. {"yes_or_no": "yes" 또는 "no"} 형태로 반환됨.
    """
    # 1. 관련성 평가기를 생성
    question_answer_relevant = GroundednessChecker(
        llm=ChatOpenAI(model="gpt-4o", temperature=0), target="score-fact-check"
    ).create()

    #print('='*30,state['eval_resume']['eval_resume'][1])
    # 2. 관련성 체크를 실행("yes" or "no")
    response = question_answer_relevant.invoke(
        {"original_document1": state['resume'],"original_document2": state['eval_item_content'], "eval_document": state['eval_resume']['eval_resume'][1]}
    )
    
    print("==== [FACT CHECK] ====")
    print(response.score)

    return {'yes_or_no':response.score} 