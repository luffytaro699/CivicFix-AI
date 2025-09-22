FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your entire project code into the container
COPY . .

# Use the port Render provides
EXPOSE 10000

# Command to run the application using Gunicorn + Uvicorn worker
# Bind to the environment variable $PORT, not hardcoded 8000
CMD ["sh", "-c", "uvicorn ai_services.main:app --host 0.0.0.0 --port ${PORT}"]

