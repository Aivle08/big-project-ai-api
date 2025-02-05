# Base
import os
from dotenv import load_dotenv

# FastAPI
import uvicorn
from fastapi import FastAPI, HTTPException, Depends, status
#from fastapi.middleware.cors import CORSMiddleware

# DB
#from sqlalchemy.orm import Session

# Module
from router.summary import summary
from router.question import question
from router.score import score
from router.zilliz import zilliz

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

app = FastAPI()

# 미들웨어
# CORS 정책
# origins = [
#     "*"
#     ]
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins = origins,
#     allow_credentials = True,
#     allow_methods = ["*"],
#     allow_headers = ["*"],
# )


@app.get("/")
def say_hello():
    return {"message": "Hello world from FastAPI1111111"}

# 질문 모델
app.include_router(question)

# 요약 모델
app.include_router(score)

# 요약 모델
app.include_router(summary)

# zilliz 모델
app.include_router(zilliz)


print(f'Documents: http://localhost:8000/docs')

if __name__ == '__main__':
    uvicorn.run("main:app", reload=True)