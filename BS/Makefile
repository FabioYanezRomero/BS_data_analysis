.PHONY: build run run-interactive attach nlp similarity clean

IMAGE_NAME = bs_data_analysis
CONTAINER_NAME = bs_analysis_container

# Build Docker image
build:
	docker build -t $(IMAGE_NAME) .

# Run container with HTTP server for port detection
run:
	docker run -dit --name $(CONTAINER_NAME) \
		-p 8000:8000 \
		-v $(PWD)/..:/app \
		$(IMAGE_NAME) bash -c "python3 -u -m http.server 8000 & exec bash"

run-interactive:
	docker run -it --name $(CONTAINER_NAME) \
		-v $(PWD)/..:/app \
		$(IMAGE_NAME) bash

attach:
	docker exec -it $(CONTAINER_NAME) bash

# Run NLP analysis
nlp:
	docker run -it --rm --name $(CONTAINER_NAME) \
		-v $(PWD)/..:/app \
		$(IMAGE_NAME) python -m src.statistical_nlp_analysis

# Run semantic similarity analysis
similarity:
	docker run -it --rm --name $(CONTAINER_NAME) \
		-v $(PWD)/..:/app \
		$(IMAGE_NAME) python -m src.semantic_intent_similarity

# Clean up Docker resources
clean:
	docker rm -f $(CONTAINER_NAME) 2>/dev/null || true
	docker rmi $(IMAGE_NAME) 2>/dev/null || true
