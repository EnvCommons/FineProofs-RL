FROM python:3.11-slim

RUN apt update && apt upgrade -y && apt install -y curl

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server.py .

EXPOSE 8000

# Download parquet from HuggingFace during build
RUN curl -L -o fineproofs_train.parquet \
https://huggingface.co/datasets/lm-provers/FineProofs-RL/resolve/main/data/train-00000-of-00001.parquet

CMD ["python", "server.py"]
