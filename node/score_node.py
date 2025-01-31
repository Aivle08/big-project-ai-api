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
from langchain_milvus import Milvus
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
from state.score_state import ScoreState
from etc.evaluator import GroundednessChecker
from etc.graphs import visualize_graph
from etc.etcc import format_docs, is_fact
####################################################################################################################
################################################### STATE ########################################################### 청크 합치기
# 환경설정
load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY')
embeddings = OpenAIEmbeddings()
    
# PydanticOutputParser
class score(BaseModel):
    # 점수 산출 형식 지정
    interview: str = Field(
        description="""
            "{eval_item}": <Score>,
            "{eval_item} of Reason": "<Description of evaluation basis>"
        """
    )
    
# Retriever 노드
def retrieve_document(state: ScoreState, collection_name: str, class_id: str):
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
    
    retrieved_docs = retriever.invoke(latest_question)
    retrieved_docs = format_docs(retrieved_docs)
    
    print(retrieved_docs)
    
    # 검색된 문서를 context 키에 저장합니다.
    return {f'{collection_name}':retrieved_docs}

# 관련성 체크 노드
def relevance_check(state: ScoreState):
    # 관련성 평가기를 생성합니다.
    question_answer_relevant = GroundednessChecker(
        llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0), target="question-retrieval"
    ).create()

    # 관련성 체크를 실행("yes" or "no")
    response = question_answer_relevant.invoke(
        {"question": state['final_result'], "context1": state['resume'], 'context2':state['query_main']}
    )

    print("==== [RELEVANCE CHECK] ====")
    print(response.score)

    # 참고: 여기서의 관련성 평가기는 각자의 Prompt 를 사용하여 수정할 수 있습니다. 여러분들의 Groundedness Check 를 만들어 사용해 보세요!
    return {'relevance':response.score}


# 자소서 점수 측정 노드
def score_resume(state: ScoreState, prompt: PromptTemplate):
    # State 변수 선언
    job = state['job']
    resume = state['resume']
    eval_item = state['eval_item']
    eval_item_content = state['eval_item_content']

    # 1. 모델 선언
    model = ChatOpenAI(model='gpt-4o', streaming=True, temperature=0)

    # PromptTemplate + model + StrOutputParser 체인 생성
    chain = prompt | model | StrOutputParser()

    answer_middle = chain.invoke({
            "resume": resume,
            "eval_item_content": eval_item_content,
            "job": job,
            "eval_item": eval_item
        })
    
    # 질문 추출
    question_score = answer_middle
    
    return {'eval_resume':question_score}

# 팩트 체크 노드
def fact_checking(state: ScoreState):
    # 1. 관련성 평가기를 생성
    question_answer_relevant = GroundednessChecker(
        llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0), target="fact-check"
    ).create()

    # 2. 관련성 체크를 실행("yes" or "no")
    response = question_answer_relevant.invoke(
        {"original_document": state['resume'], "eval_document": state['summary_result']}
    )
    
    print('summarized: ', state['summary_result'])
    print("==== [RELEVANCE CHECK] ====")
    print(response.score)

    return {'yes_or_no':response.score}