.PHONY: build run run-interactive attach process-batch intent-metrics entity-metrics lexical distribution semantic clean

IMAGE_NAME = chatbot_data_analysis
CONTAINER_NAME = chatbot_analysis_container

# Build Docker image
build:
	docker build -t $(IMAGE_NAME) .

# Run container with HTTP server for port detection
run:
	docker run -dit --name $(CONTAINER_NAME) \
		-p 8000:8000 \
		-v $(PWD):/app \
		$(IMAGE_NAME) bash -c "python3 -u -m http.server 8000 & exec bash"

run-interactive:
	docker run -it --name $(CONTAINER_NAME) \
		-v $(PWD):/app \
		$(IMAGE_NAME) bash

attach:
	docker exec -it $(CONTAINER_NAME) bash

# Process batch testing files
process-batch:
	docker run -it --rm --name $(CONTAINER_NAME) \
		-v $(PWD):/app \
		$(IMAGE_NAME) python src/process_batch_raw.py

# Calculate intent metrics
intent-metrics:
	docker run -it --rm --name $(CONTAINER_NAME) \
		-v $(PWD):/app \
		$(IMAGE_NAME) python src/calculate_intent_metrics.py

# Calculate entity metrics
entity-metrics:
	docker run -it --rm --name $(CONTAINER_NAME) \
		-v $(PWD):/app \
		$(IMAGE_NAME) python src/calculate_entity_metrics.py

# Run lexical analysis
lexical:
	docker run -it --rm --name $(CONTAINER_NAME) \
		-v $(PWD):/app \
		$(IMAGE_NAME) python src/analyze_lexical.py

# Run distribution analysis
distribution:
	docker run -it --rm --name $(CONTAINER_NAME) \
		-v $(PWD):/app \
		$(IMAGE_NAME) python src/analyze_distribution.py

# Run semantic analysis
semantic:
	docker run -it --rm --name $(CONTAINER_NAME) \
		-v $(PWD):/app \
		$(IMAGE_NAME) python src/analyze_semantic.py

# Clean up Docker resources
clean:
	docker rm -f $(CONTAINER_NAME) || true
	docker rmi -f $(IMAGE_NAME) || true
