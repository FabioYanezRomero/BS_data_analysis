FROM ubuntu:22.04

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv python3-distutils python3-setuptools python3-wheel python3-dev \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies inside virtualenv
RUN python3 -m venv /opt/venv
RUN ln -sf /opt/venv/bin/python /opt/venv/bin/python3
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Set working directory to the code directory
WORKDIR /app

CMD ["bash"]
