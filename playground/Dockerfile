FROM python:3.12-slim
WORKDIR /app
COPY playground/requirements.txt /app/playground/requirements.txt
RUN pip install --no-cache-dir -r /app/playground/requirements.txt
# 벤치와 같은 코드 경로: eval/ train/ configs/ 를 그대로 싣는다
COPY configs /app/configs
COPY train /app/train
COPY eval /app/eval
COPY release /app/release
COPY playground /app/playground
ENV PYTHONUNBUFFERED=1 HF_HOME=/hf
WORKDIR /app/playground
EXPOSE 8080
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080", "--no-access-log"]
