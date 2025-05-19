FROM python:3.9-slim

WORKDIR /usrvol

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file if it exists
COPY requirements.txt .

# Install Python dependencies
RUN if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/usrvol

# Default command
CMD ["/bin/bash"]
