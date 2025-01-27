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
from langchain_core.documents import Document

# Graph
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import GraphRecursionError

# Tool
from pydantic import BaseModel, Field

# DB
from typing import TypedDict, Annotated, List

# Error
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Module
from state.question_state import GraphState
from etc.evaluator import GroundednessChecker
from etc.graphs import visualize_graph
from etc.etcc import format_docs, is_relevant
####################################################################################################################
################################################### STATE ########################################################### 청크 합치기
# 환경설정
load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY')
embeddings = OpenAIEmbeddings()
    
# 문서 검색 노드
def retrieve_document(state: GraphState, collection_name: str, class_id: str):
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
    
    # 검색된 문서를 context 키에 저장합니다.
    return retrieved_docs
    

# 관련성 체크 노드
def relevance_check(state: GraphState):
    # 1. 관련성 평가기를 생성
    question_answer_relevant = GroundednessChecker(
        llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0), target="question-retrieval"
    ).create()

    # 2. 관련성 체크를 실행("yes" or "no")
    response = question_answer_relevant.invoke(
        {"question": state['final_question'], "context1": state['resume'], 'context2':state['evaluation']}
    )

    print("==== [RELEVANCE CHECK] ====")
    print(response.score)

    return {'relevance':response.score}


# PydanticOutputParser
class question(BaseModel):
    # 질문 형식 지정
    interview:List[str] = Field(
        description="A list containing two interview questions as plain text strings."
    )

# 결과 종합 노드
def combine_prompt(state: GraphState, prompt: PromptTemplate):
    # State 변수 선언 지정
    job = state['job']
    resume = state['resume']
    evaluation = state['evaluation']
    
    # 1. 모델 선언
    model = ChatOpenAI(model='gpt-3.5-turbo', streaming=True)
    
    # 2. 구조화된 출력을 위한 LLM 설정
    llm_with_tool = model.with_structured_output(question)
    
    # 3. llm + PydanticOutputParser 바인딩 체인 생성
    chain = prompt | llm_with_tool

    # 4. 질문 생성 LLM 실행
    answer_middle = chain.invoke({'resume' : resume, 'evaluation' : evaluation, 'job' : job})

    # 5. 질문 추출
    question_score = answer_middle.interview
    
    return {'final_question':question_score}
