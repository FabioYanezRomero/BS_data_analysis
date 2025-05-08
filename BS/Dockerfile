FROM python:3.8-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data
COPY BS_Code/download_nltk_data.py .
RUN python download_nltk_data.py

# Copy source code
COPY . .

# Set working directory to the code directory
WORKDIR /app

CMD ["bash"]
