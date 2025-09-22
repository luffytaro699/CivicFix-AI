# Use official Python slim image (lightweight, flexible)
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies required by OpenCV, Pydub/FFmpeg, Whisper
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    build-essential \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for Docker caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port (Render uses $PORT)
EXPOSE 10000

# Run FastAPI using Uvicorn, using Render's $PORT environment variable
CMD ["sh", "-c", "uvicorn ai_services.main:app --host 0.0.0.0 --port ${PORT}"]
