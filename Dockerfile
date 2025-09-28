# Use a Python image
FROM python:3.12-slim

# Install system build deps (cryptography/pdf libs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libssl-dev \
    libffi-dev \
    cargo \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt /app/
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY . /app

# Expose port Render provides via $PORT
ENV PORT 10000
# Use Gunicorn with Uvicorn workers for concurrency
CMD gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app --bind 0.0.0.0:${PORT} --timeout 40
