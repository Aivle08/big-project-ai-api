################################################## LIBRARY #########################################################
# Basic
import pandas as pd
import numpy as np
import os
import openai
from dotenv import load_dotenv

# Chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader, PyMuPDFLoader
from langchain_milvus import Milvus, Zilliz
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

# Graph
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import GraphRecursionError

# Tool
from pydantic import BaseModel, Field

# DB
from pymilvus import Collection, connections
from typing import TypedDict, Annotated, List

# Error
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Module
from state.summary_state import SummaryState, yonghunState
from etc.evaluator import GroundednessChecker
from etc.graphs import visualize_graph
from etc.etcc import format_docs, is_fact
####################################################################################################################
################################################### STATE ########################################################### 청크 합치기
# 환경설정
load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY')
embeddings = OpenAIEmbeddings()
    
# 자소서 로드 노드
def resume_load(state: SummaryState, collection_name: str, class_id: str):
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
    # 1. 관련성 평가기를 생성
    question_answer_relevant = GroundednessChecker(
        llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0), target="fact-check"
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

def retrieve_document(state: yonghunState, collection_name: str, class_id: str):
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
def combine_prompt(state: yonghunState):
    # State 변수 선언
    resume = state['resume']
    output_form = state['output_form']

    # 1. 모델 선언
    model = ChatOpenAI(model='gpt-3.5-turbo', streaming=True, temperature=0)
    
    # 구조화된 출력을 위한 LLM 설정
    # llm_with_tool = model.with_structured_output(question)

    # Prompt 생성
    prompt = PromptTemplate(
        template="""
        You are a function summarizing resumes.

        ### Instructions:
        1. Organize the content according to the format below. Leave fields blank if the information is not provided.
        2. Avoid repetition by outputting duplicate information only once.
        3. Do not omit important details (e.g., major, GPA, etc.).
       
        Please write the basis in Korean.
        ### Output Format:
        Return your response strictly in this JSON format:
            {output_form}

        ### Resume:
        {resume}
        """,
        input_variables=["resume", "output_form"]
    )

    # PromptTemplate + model + StrOutputParser 체인 생성
    chain = prompt| model | StrOutputParser()

    answer_middle = chain.invoke({
            "resume": resume,
            "output_form": output_form,
        })
    
    # 질문 추출
    question_score = answer_middle
    
    return {'final_result':question_score}