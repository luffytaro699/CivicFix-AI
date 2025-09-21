# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your entire project code into the container
# This includes ai_services, models, routers, etc.
COPY . .

# Expose the port the app will run on
EXPOSE 8000

# Command to run the application using Gunicorn
# This command tells Gunicorn where to find your FastAPI 'app' object
CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "ai_services.main:app", "--bind", "0.0.0.0:8000"]