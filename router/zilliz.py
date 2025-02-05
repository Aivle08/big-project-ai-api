#########################################################################################
# Basic
import os
import pandas as pd
import traceback
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Fastapi
from fastapi import APIRouter, HTTPException, status, File, UploadFile

from sentence_transformers import SentenceTransformer
from pymilvus import connections, Collection, FieldSchema, DataType, CollectionSchema, utility, MilvusClient
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter,MarkdownTextSplitter
from langchain_milvus import Milvus, Zilliz
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# zilliz CLUSTER_ENDPOINT, TOKEN
# from main import cluster_endpoint, token

# TechDTO
from dto.zilliz_dto import ResumeInsertDTO, EvalInsertDTO, ResumeDeleteDTO, EvalDeleteDTO
#########################################################################################
zilliz = APIRouter(prefix='/zilliz')

# 환경 변수 로드
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
cluster_endpoint = os.getenv("CLUSTER_ENDPOINT")
token = os.getenv("TOKEN")

# 1. Set up a Milvus client
client = MilvusClient(
    uri=os.environ['CLUSTER_ENDPOINT'],
    token=os.environ['TOKEN']
)

# LangChain용 OpenAI Embeddings 설정
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

def milvus_connect():
    connections.connect(uri=cluster_endpoint, token=token)

def disconnect_milvus():
    connections.disconnect("default")

##### 데이터 삽입 #####
# resume에 지원서 pdf 로드하기
def insert_data_resume(pdf_name, applicant_id):
    # 컬렉션 연결
    
    collection_name = "resume"
    collection = Collection(name=collection_name)

    loader = PyMuPDFLoader(pdf_name)
    
    docs = loader.load()
    
    for doc in docs :
        # 텍스트를 청크화
        text = doc.page_content
        text_splitter = MarkdownTextSplitter(chunk_size=250, chunk_overlap=20)
        chunks = text_splitter.split_text(text)
        
        for chunk in chunks:
            vector = embeddings.embed_query(chunk)
            
            data = {
                'applicant_id' : applicant_id,
                'vector':vector,
                'text' : chunk,
            }
            
            collection.insert(collection = collection_name, data = data,) 

# evaluation에 평가 기준 로드하기
def insert_data_evaluation(recruitment_id, detail_list):
    """
    evaluation 컬렉션에 PDF 데이터를 삽입하는 함수.
    
    pdf_folder (str): PDF 파일이 위치한 폴더 경로.
    """
    # 컬렉션 이름
    collection_name = "evaluation"
    collection = Collection(name=collection_name)
    
    total_detail = ''
    
    for detail in detail_list :
        total_detail += detail
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_text(total_detail)
    
    for chunk in chunks:
            vector = embeddings.embed_query(chunk)
            
            data = {
                'company_id' : recruitment_id,
                'vector':vector,
                'text' : chunk,
            }
            
            collection.insert(collection = collection_name, data = data,) 

##### 데이터 삭제 #####
# resume에 지원자자 데이터 삭제 
def delete_data_resume(applicant_id):
    # 컬렉션 연결
    collection_name = "resume"
    collection = Collection(name=collection_name)
    
    collection.delete(f"applicant_id in [{applicant_id}]")

# evaluation에 공고 기준 삭제 
def delete_data_evaluation(recruitment_id):
    # 컬렉션 연결
    collection_name = "evaluation"
    collection = Collection(name=collection_name)
    
    collection.delete(f"company_id in [{recruitment_id}]")
    
# zillz에 이력서 데이터 추가
# @zilliz.post("/insertResume", status_code = status.HTTP_200_OK, tags=['zilliz'])
# async def insert_resume(item: ResumeInsertDTO):
#     print('\n\033[36m[AI-API] \033[32m 질문 추출(기술)')
#     try:
#         milvus_connect()
#         insert_data_resume(item.pdf_name, item.applicant_id)
#         disconnect_milvus()
        
#         return {
#             "status": "success",  # 응답 상태
#             "code": 200,  # HTTP 상태 코드
#             "message": "이력서 데이터 추가 완료",  # 응답 메시지
#         }
        
#     except Exception as e:
#             traceback.print_exc()
#             return {
#                 "status": "error",
#                 "message": f"에러 발생: {str(e)}"
#             }

# zillz에 평가 항목 상세 내용 추가
@zilliz.post("/insertDetail", status_code = status.HTTP_200_OK, tags=['zilliz'])
async def insert_detail(item: EvalInsertDTO):
    print('\n\033[36m[AI-API] \033[32m 질문 추출(기술)')
    try:
        milvus_connect()
        insert_data_evaluation(item.recruitment_id, item.detail_list)
        disconnect_milvus()
        
        return {
            "status": "success",  # 응답 상태
            "code": 200,  # HTTP 상태 코드
            "message": "평가 항목 상세 내용 추가 완료",  # 응답 메시지
        }
        
    except Exception as e:
            traceback.print_exc()
            return {
                "status": "error",
                "message": f"에러 발생: {str(e)}"
            }
            
# zillz에서 이력서 데이터 삭제
# 이거 리스트 형태로 수정 필요할 듯 공고를 삭제하면서 이력서 내용을 삭제하는 것것
# @zilliz.post("/deleteResume", status_code = status.HTTP_200_OK, tags=['zilliz'])
# async def delete_Resume(item: ResumeDeleteDTO):
#     print('\n\033[36m[AI-API] \033[32m 질문 추출(기술)')
#     try:
#         milvus_connect()
#         delete_data_resume(item.applicant_id)
#         disconnect_milvus()
        
#         return {
#             "status": "success",  # 응답 상태
#             "code": 200,  # HTTP 상태 코드
#             "message": "이력서 데이터 삭제 완료",  # 응답 메시지
#         }
        
#     except Exception as e:
#             traceback.print_exc()
#             return {
#                 "status": "error",
#                 "message": f"에러 발생: {str(e)}"
#             }

# zillz에서 공고 데이터 삭제
@zilliz.post("/deleteDetial", status_code = status.HTTP_200_OK, tags=['zilliz'])
async def delete_detail(item: EvalDeleteDTO):
    print('\n\033[36m[AI-API] \033[32m 질문 추출(기술)')
    try:
        milvus_connect()
        delete_data_evaluation(item.recruitment_id)
        disconnect_milvus()
        
        return {
            "status": "success",  # 응답 상태
            "code": 200,  # HTTP 상태 코드
            "message": "평가 항목 상세 내용 삭제제 완료",  # 응답 메시지
        }
        
    except Exception as e:
            traceback.print_exc()
            return {
                "status": "error",
                "message": f"에러 발생: {str(e)}"
            }