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
from pymilvus import Collection, connections

# Error
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Module
from state.summary_state import SummaryState, extractionState
from etc.evaluator import GroundednessChecker
from etc.validator import format_docs
####################################################################################################################
################################################### STATE ##########################################################
# 환경설정
load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY')
embeddings = OpenAIEmbeddings()
    
# 자소서 로드 노드
def resume_load(state: SummaryState, collection_name: str, class_id: str):
    """
    class_id에 해당하는 지원자 자소서 Load하는 노드

    Args:
        state (SummaryState): 질문의 상태 정보를 포함하는 객체.
        collection_name (str): zilliz collection명 (예: "resume" 또는 "evaluation").
        class_id (str): 지원자, 회사 고유 ID

    Returns:
        str: 질문의 상태 정보 
    """
    applicant_id = state['applicant_id']
    # Milvus/Zilliz에 연결
    connections.connect(
        alias="default",                    # 연결 이름 (default 권장)
        uri=os.getenv('CLUSTER_ENDPOINT'),  # Zilliz 클라우드 URI
        secure=True,                        # HTTPS 사용 여부 (Zilliz Cloud는 보통 True)
        token=os.getenv('TOKEN')            # Zilliz Cloud API 토큰
    )

    # collection Load
    collection = Collection(collection_name, using="default")

    # 메타데이터 조건으로 데이터 검색
    results = collection.query(
        expr=f"{class_id} == {state['applicant_id']}",
        output_fields=["text"]
    )
    
    text = ""
    for result in results:
        text += result['text']
    
    # 검색된 문서를 context 키에 저장합니다.
    return text

# PydanticOutputParser
class summary(BaseModel):
    # 요약 형식 지정
    summary: str = Field(
        description="A multi-line summarized string containing key points from the applicant's resume.write more than 500 characters."
    )

# 자소서 요약 노드
def resume_summary(state: SummaryState, prompt: PromptTemplate):
    """
    지원자의 자소서, 직무 정보를 기반으로 자소서를 요약하는 함수.

    Args:
        state (QuestionState): 현재 질문 생성에 필요한 상태 정보 저장 객체.
        prompt (PromptTemplate): 질문 생성을 위한 프롬프트 템플릿(기술 중심, 경력 중심, 경험 중심 프롬프트)

    Returns:
        dict: 생성된 요약을 포함하는 딕셔너리.
    """
    # State 변수 선언 지정
    job = state['job']
    resume = state['resume']
    
    # 1. 모델 선언
    model = ChatOpenAI(model='gpt-3.5-turbo', streaming=True)
    
    # 2. 구조화된 출력을 위한 LLM 설정
    llm_with_tool = model.with_structured_output(summary)
    
    # 3. llm + PydanticOutputParser 바인딩 체인 생성
    chain = prompt | llm_with_tool

    # 4. 질문 생성 LLM 실행
    answer_middle = chain.invoke({'resume' : resume, 'job' : job})

    # 5. 질문 추출
    summary_score = answer_middle.summary

    return summary_score
    
# 팩트 체크 노드
def fact_checking(state: SummaryState):
    """
    지원자의 자소서 및 회사 평가 기준을 기반으로 생성된 질문이 사실에 근거하고 있는지 검증하는 함수.

    Args:
        state (SummaryState): 현재 상태 정보를 저장하는 객체로, 자소서, 생성된 요약을 포함.

    Returns:
        dict: 질문의 사실 여부를 나타내는 결과값을 포함하는 딕셔너리. (예: {'fact': 'yes' 또는 'no'})
    """
    # 1. 관련성 평가기를 생성
    question_answer_relevant = GroundednessChecker(
        llm=ChatOpenAI(model="gpt-4o", temperature=0), target="summary-fact-check"
    ).create()

    # 2. 관련성 체크를 실행("yes" or "no")
    response = question_answer_relevant.invoke(
        {"original_document": state['resume'], "summarized_document": state['summary_result']}
    )
    # print('origin: ', state['resume'])
    print('summarized: ', state['summary_result'])
    print("==== [RELEVANCE CHECK] ====")
    print(response.score)

    return {'yes_or_no':response.score}

def retrieve_document(state: extractionState, collection_name: str, class_id: str):
    """
    벡터 데이터베이스(Zilliz)를 활용하여 지정된 컬렉션에서 관련 문서를 검색하는 함수.

    Args:
        state (extractionState): 추출 정보의 상태 정보를 포함하는 객체.
        collection_name (str): 지원자, 회사 고유 ID
        class_id (str): 검색 대상의 특정 ID(예: applicant_id, company_id 등).

    Returns:
        str: 검색된 문서를 지정된 collection_name 키에 저장하여 반환.
    """
    # 질문을 상태에서 가져옵니다.
    latest_question = state["query_main"]
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
    
    print(retrieved_docs)
    
    # 검색된 문서를 context 키에 저장합니다.
    return {f'{collection_name}':retrieved_docs}

# 결과 종합 노드
def combine_prompt(state: extractionState, prompt: PromptTemplate):
    """
    지원자의 자소서, 직무 정보를 기반으로 자소서를 요약하는 함수.

    Args:
        state (extractionState): 현재 요약에 필요한 상태 정보 저장 객체.
        prompt (PromptTemplate): 질문 생성을 위한 프롬프트 템플릿

    Returns:
        dict: 생성된 요약을 포함하는 딕셔너리. (예: {"summary_result"': "요약"})
    """
    # State 변수 선언
    resume = state['resume']
    output_form = state['output_form']

    # 1. 모델 선언
    model = ChatOpenAI(model='gpt-4o', streaming=True, temperature=0)
    
    # 구조화된 출력을 위한 LLM 설정
    # llm_with_tool = model.with_structured_output(question)

    # PromptTemplate + model + StrOutputParser 체인 생성
    chain = prompt| model | StrOutputParser()

    answer_middle = chain.invoke({
            "resume": resume,
            "output_form": output_form,
        })
    
    # 질문 추출
    question_score = answer_middle
    
    return question_score