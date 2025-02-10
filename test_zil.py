import os
import pandas as pd
import numpy as np
import openai
from dotenv import load_dotenv

from sentence_transformers import SentenceTransformer
from pymilvus import connections, Collection, FieldSchema, DataType, CollectionSchema, utility, MilvusClient
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter,MarkdownTextSplitter
from langchain_milvus import Milvus, Zilliz
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

def format_docs(docs):
    return "\n".join(
        [
            f"<document><content>{doc.page_content}</content><source>{doc.metadata['source']}</source><page>{int(doc.metadata['page'])+1}</page></document>"
            for doc in docs
        ]
    )
    
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

# 1. Set up a Milvus client
# Milvus 클라이언트 연결
connections.connect(uri=cluster_endpoint, token=token)

def check_collection_exists(collection_name):
    """
    컬렉션이 존재하는지 확인하는 함수.
    """
    collection_exists = Collection(name=collection_name).exists()
    if not collection_exists:
        raise ValueError(f"컬렉션 '{collection_name}'이 존재하지 않습니다. Zilliz 클라우드에서 확인하세요.")
    print(f"컬렉션 '{collection_name}' 확인 완료.")

# resume에 지원서 pdf 로드하기
def insert_data_resume(pdf_folder):
    """
    resume 컬렉션에 PDF 데이터를 삽입하는 함수.
    
    pdf_folder (str): PDF 파일이 위치한 폴더 경로.
    """
    # 컬렉션 연결
    collection_name = "resume"
    collection = Collection(name=collection_name)

    # path = './' + pdf_folder + 
    # PDF 파일 로드
    #pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith(".pdf")]

    # 데이터 삽입
    resume_id = 1
    applicant_id = 1
    for pdf_file in pdf_folder:
        loader = PyMuPDFLoader(pdf_folder)
        docs = loader.load()
        
        for doc in docs:
            text = doc.page_content  # PDF에서 읽은 원본 텍스트
            
            print('#'*30)
            print('text')
            #print(text)
            
            # 텍스트를 청크화(페이지가 아니라 우리가 지정해서 청크화)
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            chunks = text_splitter.split_text(text)
            print('#'*30)
            print('chunks')
            print(chunks)
            # # 각 청크를 벡터화하고 삽입
            # for chunk in chunks:
            #     vector = embeddings.embed_query(chunk)  # 벡``````````터화

            #     # 데이터 삽입
            #     collection.insert([
            #         [resume_id, vector, applicant_id, text]  # 원본 텍스트를 포함한 데이터 삽입
            #     ])
            #     print(f"Inserted into resume: resume_id={resume_id}, applicant_id={applicant_id}, text_length={len(text)}")
            #     resume_id += 1
            #     applicant_id += 1


# evaluation에 회사 기준 pdf 로드하기
def insert_data_evaluation(file_name):
    """
    evaluation 컬렉션에 PDF 데이터를 삽입하는 함수.
    
    pdf_folder (str): PDF 파일이 위치한 폴더 경로.
    """
    # 컬렉션 이름
    collection_name = "resume"
    collection = Collection(name=collection_name)

    path = f'./{file_name}/'

    file_list = os.listdir(path)
    for idx, file in enumerate(file_list):
        for pdf_file in file:
            loader = PyMuPDFLoader(path + file)
            docs = loader.load()

            for doc in docs:
                text = doc.page_content  # 원본 텍스트 데이터
                
                # 텍스트를 청크화
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                chunks = text_splitter.split_text(text)

                # 각 청크를 벡터화하고 삽입
                for chunk in chunks:
                    print(chunk)
                    #vector = embeddings.embed_query(chunk)

                # 데이터 삽입
                # collection.insert([
                #     [evaluation_id, vector, text]  # 원본 텍스트를 포함한 데이터 삽입
                # ])
                # print(f"Inserted into evaluation: evaluation_id={evaluation_id}, text_length={len(text)}")
                # evaluation_id += 1
            
 
def load_data():
    results = client.query(
    collection_name="resume",
    filter=None,  # 필터를 완전히 제거
    output_fields=["applicant_id"],  # 모든 필드 가져오기
    limit=100  # 최대 100개 데이터 조회
    )
    
    print(results)
     
file_name = '이력서'
# 컬렉션 이름
collection_name = "resume"
collection = Collection(name=collection_name)

path = f'./{file_name}/'

file_list = os.listdir(path)
print(file_list)

load_data()

start = 200

for idx, file in enumerate(file_list):
    loader = PyMuPDFLoader(path + file)
    # 페이지 별
    docs = loader.load()
    
    print(idx)
    print('-----' * 10)
    index = start + idx 
    
    for doc in docs:
        # 페이지 별 chunk화
        text = doc.page_content  # 원본 텍스트 데이터
        
        # 텍스트를 청크화
        text_splitter = MarkdownTextSplitter(chunk_size=250, chunk_overlap=20)
        chunks = text_splitter.split_text(text)
        print(chunks)
        # 각 청크를 벡터화하고 삽입
        for chunk in chunks:
            vector = embeddings.embed_query(chunk)
            
            data = {
                'applicant_id' : index,
                'vector':vector,
                'text' : chunk,
            }
            collection.insert(
                collection = collection_name,
                data = data,
            )
            
print("완")