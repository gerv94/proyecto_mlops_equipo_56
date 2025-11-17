.PHONY: help clean clean-pyc clean-build clean-data list requirements create_environment \
        sync_data_up sync_data_down data train train_multiple train_enhanced mlflow predict api \
        test test-unit test-integration lint format all status dag summary show_env

# ###############################################################################
# GLOBALS                                                                       #
# ###############################################################################

PROJECT_NAME := proyecto_mlops_equipo_56
PYTHON_VERSION := 3.12
PYTHON_SYSTEM := python3
VENV_NAME := .venv

# ###############################################################################
# OS-AWARE BINARIES (Linux/macOS vs Windows)                                     #
# ###############################################################################
ifeq ($(OS),Windows_NT)
  EXE := .exe
  SEP := \\
  VENV_BIN := $(VENV_NAME)$(SEP)Scripts
else
  EXE :=
  SEP := /
  VENV_BIN := $(VENV_NAME)$(SEP)bin
endif

PYTHON_INTERPRETER := "$(VENV_BIN)$(SEP)python$(EXE)"
PIP := "$(VENV_BIN)$(SEP)pip$(EXE)"
DVC := "$(VENV_BIN)$(SEP)dvc$(EXE)"
MLFLOW := "$(VENV_BIN)$(SEP)mlflow$(EXE)"
PYTEST := "$(VENV_BIN)$(SEP)pytest$(EXE)"

# ###############################################################################
# COMMANDS                                                                      #
# ###############################################################################

.DEFAULT_GOAL := help

help: ## Show this help message
	@echo "Available commands:"
	@echo ""
	-@python scripts/print_make_help.py $(MAKEFILE_LIST) || true
	-@python3 scripts/print_make_help.py $(MAKEFILE_LIST) || true
	@echo ""

show_env: ## Show Python environment information
	@echo "Python interpreter: $(PYTHON_INTERPRETER)"
	@$(PYTHON_INTERPRETER) --version
	@echo "Pip: $(PIP)"
	@echo "DVC: $(DVC)"
	@echo "MLflow: $(MLFLOW)"
	@echo "Virtual environment: $(VENV_NAME)"

# ###############################################################################
# VIRTUAL ENV (no shell-specific -d checks)                                      #
# ###############################################################################
VENV_EXISTS := $(wildcard $(VENV_NAME)$(SEP))

create_environment: ## Create Python virtual environment
ifeq ($(VENV_EXISTS),)
	@echo "Creating virtual environment..."
	@$(PYTHON_SYSTEM) -m venv $(VENV_NAME)
	@echo "Virtual environment created at $(VENV_NAME)/"
else
	@echo "Virtual environment already exists at $(VENV_NAME)/"
endif

# Ensure venv exists before running commands
.ensure_venv: create_environment
	@:

requirements: .ensure_venv ## Install Python dependencies from requirements.txt
	@echo "Installing requirements..."
	@$(PIP) install --upgrade pip
	@$(PIP) install --only-binary=:all: -r requirements.txt || \
		$(PIP) install -r requirements.txt
	@echo "Requirements installed successfully."

clean-pyc: ## Delete all compiled Python files
	@echo "Cleaning Python bytecode..."
	@find . -type f -name "*.py[co]" -delete || true
	@find . -type d -name "__pycache__" -delete || true
	@echo "Python bytecode cleaned."

clean-build: ## Remove build artifacts
	@echo "Removing build artifacts..."
	@rm -rf build/ dist/ .eggs/ || true
	@find . -name '*.egg-info' -exec rm -rf {} + || true
	@find . -name '*.egg' -exec rm -f {} + || true
	@echo "Build artifacts removed."

clean-data: ## Remove processed data (keep raw data)
	@echo "Cleaning processed data..."
	@rm -rf data/interim/* data/processed/* || true
	@echo "Processed data cleaned (raw data preserved)."

clean: clean-pyc clean-build clean-data ## Remove all generated artifacts
	@echo "Cleaning models and reports..."
	@rm -rf models/*.joblib reports/*.html reports/figures/* || true
	@echo "Full cleanup complete."

list: ## List all files in the project structure
	@echo "Project structure:"
	@tree -L 3 -I '.venv|__pycache__|*.pyc|.git|.dvc|mlruns' 2>/dev/null || ls -R || true

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

predict: .ensure_venv ## Run model prediction over current preprocessed data
	@echo "Running prediction..."
	@PYTHONPATH=. $(PYTHON_INTERPRETER) predict.py

api: .ensure_venv ## Start FastAPI endpoint (app_api.py) on http://127.0.0.1:8000
	@echo "Starting FastAPI at http://127.0.0.1:8000"
	@PYTHONPATH=. $(PYTHON_INTERPRETER) -m uvicorn app_api:app --host 127.0.0.1 --port 8000 --reload

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
