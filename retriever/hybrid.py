import os
import pickle
import openai
from dotenv import load_dotenv

# Chain
from langchain_milvus import Milvus
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.text_splitter import MarkdownTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.schema import Document

# 환경설정
load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY')
embeddings = OpenAIEmbeddings()


# 청크화
def chunking(pdf_path: str):
    documents = []
    file_list = os.listdir(pdf_path)
    
    for idx, file in enumerate(file_list):
        print('file_name = ', file)
        print('='*100)
        
        loader = PyMuPDFLoader(pdf_path+'/'+file)
        docs = loader.load()

        for doc in docs:
            text = doc.page_content  # 원본 텍스트 데이터
            
            # 텍스트를 청크화
            text_splitter = MarkdownTextSplitter(chunk_size=150, chunk_overlap=20)
            chunks = text_splitter.split_text(text)

            # 각 청크를 벡터화하고 삽입
            for chunk in chunks:
                print('*'*100)
                print(chunk)
                documents.append(chunk)
                
    return documents

def sparse_retriever_save(pkl_name: str, pdf_path: str):
    # 예시로 문서 객체 생성
    documents = chunking(pdf_path)
    document_objects = [Document(page_content=doc) for doc in documents]
    # BM25Retriever 초기화
    sparse_retriever = BM25Retriever.from_documents(document_objects)
 
    # 객체를 파일로 저장
    with open(f'bm25_db/{pkl_name}.pkl', 'wb') as file:
        pickle.dump(sparse_retriever, file)

    print(f"BM25Retriever 객체가 {pkl_name}.pkl 파일에 저장되었습니다.")

def sparse_retriever_load(pkl_name: str):
    with open(f'bm25_db/{pkl_name}.pkl', 'rb') as file:
        sparse_retriever = pickle.load(file)
    return sparse_retriever 
    
def hybrid_retriever(collection_name: str, state_class_id: str, class_id: str):
    print("hybrid_retriever 시작")
    
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
    dense_retriever = vectorstore_resume.as_retriever(search_kwargs={
            'expr' : f"{class_id} == {state_class_id}",
        }
    )
    
    # 저장된 BM25Retriever 객체 불러오기
    with open(f"bm25_{collection_name}_retriever.pkl", 'rb') as file:
        sparse_retriever = pickle.load(file)

    print("BM25Retriever 객체가 성공적으로 불러와졌습니다.")
    
    # 앙상블 리트리버 생성
    ensemble_retriever = EnsembleRetriever(
        retrievers=[dense_retriever, sparse_retriever],
        weights=[0.5, 0.5]  # 가중치 설정 (가중치의 합은 1.0)
    )
    
    return ensemble_retriever
    
# if __name__ == '__main__':
#     sparse_retriever_save('bm25_evaluation_retriever', '회사기준_pdf')



    
    