FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc libffi-dev libssl-dev libbz2-dev liblzma-dev libsqlite3-dev && \
    rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt
COPY . /app/
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]