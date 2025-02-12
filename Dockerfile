# 1. 파이썬 3.11 이미지를 기반으로 설정
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc libffi-dev libssl-dev libbz2-dev liblzma-dev libsqlite3-dev && \
    rm -rf /var/lib/apt/lists/*

# 2. 작업 디렉토리 설정
WORKDIR /app

# 3. 의존성 파일 복사
COPY requirements.txt /app/

# 4. 의존성 설치
RUN pip install --no-cache-dir -r requirements.txt

# 5. 애플리케이션 코드 복사
COPY . /app/

# 6. FastAPI 애플리케이션 실행
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]