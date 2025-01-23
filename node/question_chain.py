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
# from evaluator import GroundednessChecker
# from graphs import visualize_graph
####################################################################################################################
################################################### STATE ########################################################### 청크 합치기
# 환경설정
load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY')
embeddings = OpenAIEmbeddings()

def format_docs(docs):
    return "\n".join(
        [
            f"{doc.page_content}"
            for doc in docs
        ]
    )

# 관련성 체크하는 함수(router)
def is_relevant(state: GraphState):
    if state["relevance"] == "yes":
        return "relevant"
    else:
        return "not relevant"
    
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

# #retriever_1.invoke("인재상")

    
    

# 관련성 체크 노드
def relevance_check(state: GraphState):
    # 관련성 평가기를 생성합니다.
    question_answer_relevant = GroundednessChecker(
        llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0), target="question-retrieval"
    ).create()

    # 관련성 체크를 실행("yes" or "no")
    response = question_answer_relevant.invoke(
        {"question": state['final_question'], "context1": state['resume'], 'context2':state['evaluation']}
    )

    print("==== [RELEVANCE CHECK] ====")
    print(response.score)

    # 참고: 여기서의 관련성 평가기는 각자의 Prompt 를 사용하여 수정할 수 있습니다. 여러분들의 Groundedness Check 를 만들어 사용해 보세요!
    return {'relevance':response.score}


# PydanticOutputParser
class question(BaseModel):
    # 질문 형식 지정
    interview:List[str] = Field(
        description="A list containing two interview questions as plain text strings."
    )

# 결과 종합 노드
def combine_prompt(state: GraphState):
    # State 변수 선언 지정
    job = state['job']
    resume = state['resume']
    evaluation = state['evaluation']
    
    # 1. 모델 선언
    model = ChatOpenAI(model='gpt-3.5-turbo', streaming=True)
    
    # 구조화된 출력을 위한 LLM 설정
    llm_with_tool = model.with_structured_output(question)
    
    # Prompt
    prompt = PromptTemplate(
        # 귀하는 지원자의 자소서와 채용하는 회사 기준 db의 관련성을 파악해 채용 질문을 생성하는 기계입니다.
        # 지원자의 자소서는 다음과 같습니다: {resume}
        # 채용하는 회사 기준db는 다음과 같습니다: {recruit}
        # 회사 기준db를 바탕으로 지원자의 자소서를 포함하여 채용 질문을 뽑아냅니다.   
        # 해당 {직무}에서 심화적이고 기술중심적으로 채용 질문을 생성합니다.
        # 절대 회사 기준 db와 지원자의 자소서에 있는 내용 그대로 질문에 포함하면 안됩니다.  
        template = """
        You are a machine designed to generate interview questions by analyzing the relevance between an applicant's resume and the recruiting company's database.
 
        Here is the resume : {resume}
        Here is the recruiting company's database: {evaluation}
       
        Based on the company's database, generate interview questions that incorporate the applicant's resume.
        Create interview questions that are advanced and highly technical, specifically tailored for the {job}.
       
        Do not directly include the exact content from the company's database or the applicant's resume in the questions.
        
        Please write the questions in Korean.
        
        Write the questions as follows:
        (first question)
        (second question)
        """,
        input_variables=['resume', 'evaluation', 'job']
    )
    
    # llm + PydanticOutputParser 바인딩 체인 생성
    chain = prompt | llm_with_tool

    # 질문 생성 LLM 실행
    answer_middle = chain.invoke({'resume' : resume, 'evaluation' : evaluation, 'job' : job})

    # 질문 추출
    question_score = answer_middle.interview
    
    return {'final_question':question_score}
