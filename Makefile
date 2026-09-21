.PHONY: install install-dev run run-prod lint format test test-cov docker-build docker-up docker-down docker-logs ingest evaluate benchmark clean help

# Installation
install:
	pip install -r requirements.txt

install-dev: install
	pip install -e .
	pip install black ruff pytest-cov ipython

# Development Server
run:
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

run-prod:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 4

# Code Quality & Formatting
lint:
	ruff check src/ tests/

format:
	ruff check --fix src/ tests/
	ruff format src/ tests/

# Testing Suite
test:
	pytest tests/ -v

test-cov:
	pytest tests/ --cov=src --cov-report=html --cov-report=term

# Container Orchestration (Docker Compose v2)
docker-build:
	docker build -f docker/Dockerfile -t multimodal-rag-agent .

docker-up:
	docker compose -f docker/docker-compose.yml up -d

docker-down:
	docker compose -f docker/docker-compose.yml down

docker-logs:
	docker compose -f docker/docker-compose.yml logs -f

# Data Pipelines & Evaluation Scripts
ingest:
	python scripts/ingest_documents.py

evaluate:
	python scripts/eval_pipeline.py

benchmark:
	python scripts/benchmark.py

# Cleanup Artifacts
clean:
	python -c "import pathlib, shutil; [shutil.rmtree(p) for p in pathlib.Path('.').rglob('__pycache__')]; [p.unlink() for p in pathlib.Path('.').rglob('*.py[cod]')]; [shutil.rmtree(p) for p in pathlib.Path('.').rglob('*.egg-info')]; [shutil.rmtree(p, ignore_errors=True) for p in ['.pytest_cache', '.coverage', 'htmlcov', 'build', 'dist']]"

# Help Information
help:
	@echo "Available commands:"
	@echo "  make install      - Install runtime dependencies"
	@echo "  make install-dev  - Install dev tools and editable package"
	@echo "  make run          - Run FastAPI dev server with auto-reload"
	@echo "  make test         - Run full pytest test suite"
	@echo "  make test-cov     - Run tests with terminal and HTML coverage report"
	@echo "  make lint         - Lint code with Ruff"
	@echo "  make format       - Format code with Ruff"
	@echo "  make docker-up    - Start Docker services (API, Redis, Prometheus, Grafana)"
	@echo "  make docker-down  - Stop Docker services"
	@echo "  make docker-logs  - View aggregated container logs"
	@echo "  make ingest       - Ingest documents via CLI"
	@echo "  make evaluate     - Run evaluation pipeline (scripts/eval_pipeline.py)"
	@echo "  make benchmark    - Run system latency/throughput benchmarks"
	@echo "  make clean        - Cross-platform removal of cache and build artifacts"