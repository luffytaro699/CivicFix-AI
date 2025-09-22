FROM pytorch/pytorch:2.8.0-cuda12.1-cudnn9-runtime
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["sh", "-c", "uvicorn ai_services.main:app --host 0.0.0.0 --port ${PORT}"]
