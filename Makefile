.PHONY: help setup test run dashboard clean install docker-build docker-run

help:
	@echo "Warp-Claw Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  setup        - Install dependencies and download models"
	@echo "  test        - Run test suite"
	@echo "  run         - Start API server"
	@echo "  dashboard   - Start dashboard"
	@echo "  clean       - Clean cache and temp files"
	@echo "  install    - Install package"
	@echo "  docker-build - Build Docker image"
	@echo "  docker-run  - Run Docker container"

install:
	pip install -e .

setup: install
	@echo "Setting up M1 environment..."
	@bash scripts/setup_m1.sh || echo "Manual setup required"

test:
	pytest tests/ -v

run:
	python -m src.interfaces.openai_api

run-verbose:
	python -m src.interfaces.openai_api --log-level debug

dashboard:
	streamlit run src/dashboard/app.py

clean:
	rm -rf data/models/* data/knowledge/* data/cache/*
	rm -rf __pycache__ src/**/__pycache__ tests/__pycache__
	rm -rf .pytest_cache
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete

docker-build:
	docker build -t warp-agent:latest .

docker-run:
	docker run -p 8000:8000 -p 8501:8501 warp-agent:latest