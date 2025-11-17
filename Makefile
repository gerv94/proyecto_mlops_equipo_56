.PHONY: help clean clean-pyc clean-build clean-data list requirements create_environment \
        sync_data_up sync_data_down data train mlflow test lint format

# ###############################################################################
# GLOBALS                                                                       #
# ###############################################################################

PROJECT_NAME = proyecto_mlops_equipo_56
PYTHON_VERSION = 3.12
PYTHON_SYSTEM = python3
VENV_NAME = .venv
PYTHON_INTERPRETER = $(VENV_NAME)/bin/python
PIP = $(VENV_NAME)/bin/pip
DVC = $(VENV_NAME)/bin/dvc
MLFLOW = $(VENV_NAME)/bin/mlflow
PYTEST = $(VENV_NAME)/bin/pytest

# ###############################################################################
# COMMANDS                                                                      #
# ###############################################################################

help: ## Show this help message
	@echo "Available commands:"
	@echo ""
	@awk 'BEGIN {FS = ":.*##"; printf ""} /^[a-zA-Z_-]+:.*##/ { printf "  \033[36mmake %-20s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)
	@echo ""

show_env: ## Show Python environment information
	@echo "Python interpreter: $(PYTHON_INTERPRETER)"
	@echo "Python version: $$($(PYTHON_INTERPRETER) --version)"
	@echo "Pip: $(PIP)"
	@echo "DVC: $(DVC)"
	@echo "MLflow: $(MLFLOW)"
	@echo "Virtual environment: $(VENV_NAME)"
	@if [ -d $(VENV_NAME) ]; then echo "Status: Active"; else echo "Status: Not found"; fi

create_environment: ## Create Python virtual environment
	@if [ -d $(VENV_NAME) ]; then \
		echo "Virtual environment already exists at $(VENV_NAME)/"; \
	else \
		echo "Creating virtual environment..."; \
		$(PYTHON_SYSTEM) -m venv $(VENV_NAME); \
		echo "Virtual environment created at $(VENV_NAME)/"; \
	fi

# Internal target to ensure venv exists before running commands
.ensure_venv:
	@if [ ! -d $(VENV_NAME) ]; then \
		echo "Virtual environment not found. Creating it now..."; \
		$(PYTHON_SYSTEM) -m venv $(VENV_NAME); \
		echo "Virtual environment created at $(VENV_NAME)/"; \
		echo ""; \
	fi

requirements: .ensure_venv ## Install Python dependencies from requirements.txt
	@echo "Installing requirements..."
	@$(PIP) install --upgrade pip
	@$(PIP) install --only-binary=:all: -r requirements.txt || \
		$(PIP) install -r requirements.txt
	@echo "Requirements installed successfully."

clean-pyc: ## Delete all compiled Python files
	@echo "Cleaning Python bytecode..."
	@find . -type f -name "*.py[co]" -delete
	@find . -type d -name "__pycache__" -delete
	@echo "Python bytecode cleaned."

clean-build: ## Remove build artifacts
	@echo "Removing build artifacts..."
	@rm -rf build/
	@rm -rf dist/
	@rm -rf .eggs/
	@find . -name '*.egg-info' -exec rm -rf {} +
	@find . -name '*.egg' -exec rm -f {} +
	@echo "Build artifacts removed."

clean-data: ## Remove processed data (keep raw data)
	@echo "Cleaning processed data..."
	@rm -rf data/interim/*
	@rm -rf data/processed/*
	@echo "Processed data cleaned (raw data preserved)."

clean: clean-pyc clean-build clean-data ## Remove all generated artifacts
	@echo "Cleaning models and reports..."
	@rm -rf models/*.joblib
	@rm -rf reports/*.html
	@rm -rf reports/figures/*
	@echo "Full cleanup complete."

list: ## List all files in the project structure
	@echo "Project structure:"
	@tree -L 3 -I '.venv|__pycache__|*.pyc|.git|.dvc|mlruns' || ls -R

sync_data_down: .ensure_venv ## Pull data from DVC remote storage
	@echo "Pulling data from DVC remote..."
	@$(DVC) pull || echo "Warning: Some files could not be pulled from remote, continuing anyway..."
	@echo "Data sync attempted."

sync_data_up: .ensure_venv ## Push data to DVC remote storage
	@echo "Pushing data to DVC remote..."
	@$(DVC) push
	@echo "Data synced to remote."

data: .ensure_venv sync_data_down ## Run the complete DVC pipeline
	@echo "Running DVC pipeline..."
	@$(DVC) repro
	@echo "Pipeline execution complete."

train: .ensure_venv ## Train model using DVC pipeline
	@echo "Training model..."
	@$(DVC) repro
	@echo "Training complete. Check models/ and reports/ for outputs."

train_multiple: .ensure_venv ## Train multiple models with comparison
	@echo "Training multiple models..."
	@PYTHONPATH=. $(PYTHON_INTERPRETER) train/train_multiple_models.py
	@echo "Multiple models training complete."

train_enhanced: .ensure_venv ## Train enhanced models
	@echo "Training enhanced models..."
	@PYTHONPATH=. $(PYTHON_INTERPRETER) train/train_enhanced_models.py
	@echo "Enhanced models training complete."

mlflow: .ensure_venv ## Start MLflow UI server
	@echo "Starting MLflow UI at http://127.0.0.1:5001"
	@echo "Logs will be written to mlflow.log"
	@$(MLFLOW) ui --host 127.0.0.1 --port 5001 2>&1 | tee mlflow.log

test: .ensure_venv ## Run full test suite (unit + integration)
	@echo "Running full test suite (unit + integration)..."
	@PYTHONPATH=. $(PYTEST) tests/ -v || echo "No tests found. Add tests in tests/ directory."

test-unit: .ensure_venv ## Run unit tests only
	@echo "Running unit tests..."
	@PYTHONPATH=. $(PYTEST) tests/unit -v || echo "No unit tests found in tests/unit."

test-integration: .ensure_venv ## Run integration tests only
	@echo "Running integration tests..."
	@PYTHONPATH=. $(PYTEST) tests/integration -v || echo "No integration tests found in tests/integration."

lint: ## Run code quality checks (placeholder)
	@echo "Running linters..."
	@echo "Linting not configured yet. Add ruff/black/isort to requirements and configure."

format: ## Format code (placeholder)
	@echo "Formatting code..."
	@echo "Formatting not configured yet. Add black/isort to requirements and configure."

all: clean data train ## Run full pipeline: clean, data, train
	@echo "Full pipeline executed successfully."
	@echo "To view results, run: make mlflow"

# ###############################################################################
# PROJECT RULES                                                                 #
# ###############################################################################

status: .ensure_venv ## Show current DVC pipeline status
	@echo "DVC pipeline status:"
	@$(DVC) status

dag: .ensure_venv ## Visualize DVC pipeline DAG
	@echo "DVC pipeline DAG:"
	@$(DVC) dag

summary: .ensure_venv ## Generate project summary report
	@echo "Generating project summary..."
	@$(PYTHON_INTERPRETER) project_summary.py || echo "project_summary.py not found or failed."

# ###############################################################################
# Self Documenting Commands                                                     #
# ###############################################################################

.DEFAULT_GOAL := help
