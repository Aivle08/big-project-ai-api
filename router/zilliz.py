#########################################################################################
# Basic
import os
import pandas as pd
import traceback
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import io
import tempfile

# Fastapi
from fastapi import APIRouter, HTTPException, status, File, UploadFile

from sentence_transformers import SentenceTransformer
from pymilvus import connections, Collection, FieldSchema, DataType, CollectionSchema, utility, MilvusClient
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter,MarkdownTextSplitter
from langchain_milvus import Milvus, Zilliz
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# AWS
import logging
import boto3
from botocore.exceptions import ClientError

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

############################# zilliz Module #############################
def milvus_connect():
    connections.connect(uri=cluster_endpoint, token=token)

def disconnect_milvus():
    connections.disconnect("default")

##### 데이터 삽입 #####
# resume에 지원서 pdf 로드하기
def insert_data_resume(item: ResumeInsertDTO):
    """
    지원자의 이력서를 PDF 형태로 S3에서 다운로드하여 Zilliz에 저장하는 함수.
    - PDF 파일을 PyMuPDFLoader로 로드하여 텍스트 추출
    - MarkdownTextSplitter를 사용하여 청크(Chunk) 단위로 나눔
    - OpenAIEmbeddings을 활용하여 벡터 변환 후 Zilliz에 저장
    """

    # 컬렉션 연결
    collection_name = "resume"
    collection = Collection(name=collection_name)
    
    client_s3 = boto3.client(
        's3',
        aws_access_key_id=os.getenv("CREDENTIALS_ACCESS_KEY"),
        aws_secret_access_key=os.getenv("CREDENTIALS_SECRET_KEY"),
        region_name = os.getenv("AWS_REGION")
    )
    
    bucket = os.getenv("S3_BUCKET")

    # S3에서 파일 가져오기 (다운로드 없이 메모리에서 읽기)
    for pdf_info in item.pdf_info_list :
        response = client_s3.get_object(Bucket=bucket, Key=pdf_info.pdf_name)
        pdf_bytes = response["Body"].read()  # PDF 파일을 바이트 형태로 읽음
        
        # 임시 파일 생성 후 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            temp_pdf.write(pdf_bytes)
            temp_pdf_path = temp_pdf.name  # 임시 파일 경로 저장
        
        print(f"✅ PDF 임시 파일 저장 완료: {temp_pdf_path}")
        
        # 메모리에서 PDF 로드 (파일 저장 없이 사용)
        pdf_loader = PyMuPDFLoader(temp_pdf_path)
        
        docs = pdf_loader.load()
        
        for doc in docs :
            # 텍스트를 청크화
            text = doc.page_content
            print(text)
            text_splitter = MarkdownTextSplitter(chunk_size=250, chunk_overlap=20)
            chunks = text_splitter.split_text(text)
            
            for chunk in chunks:
                vector = embeddings.embed_query(chunk)
                
                data = {
                    'applicant_id' : pdf_info.applicant_id,
                    'vector':vector,
                    'text' : chunk,
                }
                
                collection.insert(collection = collection_name, data = data,) 
                
        # 임시 파일 삭제
        if os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)
            print(f"🗑️ 임시 파일 삭제 완료: {temp_pdf_path}")

# evaluation에 평가 기준 로드하기
def insert_data_evaluation(recruitment_id, detail_list):
    """
    채용 공고의 평가 기준 데이터를 Zilliz에 삽입하는 함수.
    - 텍스트를 RecursiveCharacterTextSplitter로 분할하여 청크화
    - OpenAIEmbeddings을 사용하여 벡터 변환 후 Zilliz에 저장
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
def delete_data_resume(applicant_id_list):
    
    # 컬렉션 연결
    collection_name = "resume"
    collection = Collection(name=collection_name)
    
    collection.delete(f"applicant_id in {applicant_id_list}")

# evaluation에 공고 기준 삭제 
def delete_data_evaluation(recruitment_id):
    # 컬렉션 연결
    collection_name = "evaluation"
    collection = Collection(name=collection_name)
    
    collection.delete(f"company_id in [{recruitment_id}]")

############################# s3 Module #############################

    

############################# FASTAPI #############################
# zillz에 이력서 데이터 추가
@zilliz.post("/insertResume", status_code = status.HTTP_200_OK, tags=['zilliz'])
async def insert_resume(item: ResumeInsertDTO):
    print('\n\033[36m[AI-API] \033[32m 질문 추출(기술)')
    try:
        milvus_connect()
        insert_data_resume(item)
        disconnect_milvus()
        
        return {
            "status": "success",  # 응답 상태
            "code": 200,  # HTTP 상태 코드
            "message": "이력서 데이터 추가 완료",  # 응답 메시지
        }
        
    except Exception as e:
            traceback.print_exc()
            return {
                "status": "error",
                "message": f"에러 발생: {str(e)}"
            }

# zillz에 평가 항목 상세 내용 추가
@zilliz.post("/insertDetail", status_code = status.HTTP_200_OK, tags=['zilliz'])
async def insert_detail(item: EvalInsertDTO):
    """
    채용 공고의 평가 기준 데이터를 Milvus(Zilliz)에 저장하는 API.
    """
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
@zilliz.post("/deleteResume", status_code = status.HTTP_200_OK, tags=['zilliz'])
async def delete_Resume(item: ResumeDeleteDTO):
    """
    지원자의 이력서 데이터를 Zilliz에서 삭제하는 API.
    """
    print('\n\033[36m[AI-API] \033[32m 질문 추출(기술)')
    try:
        milvus_connect()
        delete_data_resume(item.applicant_id_list)
        disconnect_milvus()
        
        return {
            "status": "success",  # 응답 상태
            "code": 200,  # HTTP 상태 코드
            "message": "이력서 데이터 삭제 완료",  # 응답 메시지
        }
        
    except Exception as e:
            traceback.print_exc()
            return {
                "status": "error",
                "message": f"에러 발생: {str(e)}"
            }

# zillz에서 공고 데이터 삭제
@zilliz.post("/deleteDetial", status_code = status.HTTP_200_OK, tags=['zilliz'])
async def delete_detail(item: EvalDeleteDTO):
    """
    지원자의 이력서 데이터를 Zilliz에서 삭제하는 API.
    """

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