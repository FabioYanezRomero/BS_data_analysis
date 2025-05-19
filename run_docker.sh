#!/bin/bash

# Define the Docker image name
IMAGE_NAME="plotly-high-quality"

# Build the Docker image
echo "Building Docker image: $IMAGE_NAME..."
docker build -t $IMAGE_NAME .

# Run the Docker container with proper volume mounting
echo "Running Docker container..."
docker run -it --rm \
  -v "$(pwd):/home/plotuser/app" \
  -p 8888:8888 \
  $IMAGE_NAME bash -c "cd /home/plotuser/app && jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root"