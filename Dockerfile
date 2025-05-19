FROM python:3.9-slim

WORKDIR /usrvol

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install plotly and other required packages explicitly
RUN pip install --no-cache-dir \
    plotly \
    numpy \
    scikit-learn \
    pandas \
    scipy \
    nltk \
    matplotlib \
    seaborn \
    torch \
    transformers \
    sentence-transformers \
    huggingface-hub

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/usrvol

# Default command
CMD ["/bin/bash"]
