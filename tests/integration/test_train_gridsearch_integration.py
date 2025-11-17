"""
Pruebas de integración para el módulo train/train_gridsearch.py

Estas pruebas validan el flujo completo de grid search, verificando que
los métodos funcionan correctamente juntos.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
from unittest.mock import patch, MagicMock

from train.train_gridsearch import GridSearchTrainer


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def temp_data_dir():
    """Directorio temporal para pruebas de archivos."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_csv(temp_data_dir):
    """Crea un CSV de prueba con estructura similar al dataset real."""
    interim_dir = temp_data_dir / "interim"
    interim_dir.mkdir(parents=True)
    
    # Crear DataFrame con suficientes datos para split estratificado
    data = {
        "Gender": ["M", "F"] * 50,
        "Caste": ["A", "B"] * 50,
        "coaching": ["Yes", "No"] * 50,
        "time": ["Morning", "Evening"] * 50,
        "Class_ten_education": ["CBSE", "State"] * 50,
        "twelve_education": ["CBSE", "State"] * 50,
        "medium": ["English", "Hindi"] * 50,
        "Class_ X_Percentage": ["A", "B"] * 50,
        "Class_XII_Percentage": ["A", "B"] * 50,
        "Father_occupation": ["Engineer", "Teacher"] * 50,
        "Mother_occupation": ["Doctor", "Nurse"] * 50,
        "Performance": ["good"] * 33 + ["average"] * 33 + ["excellent"] * 34,
        "numeric_col": list(range(100)),
    }
    
    df = pd.DataFrame(data)
    csv_path = interim_dir / "student_interim_clean.csv"
    df.to_csv(csv_path, index=False)
    
    return csv_path


@pytest.fixture
def trainer(temp_data_dir, sample_csv):
    """Instancia de GridSearchTrainer con paths temporales."""
    return GridSearchTrainer(
        seed=42,
        data_path=str(sample_csv),
        models_dir=str(temp_data_dir / "models"),
        reports_dir=str(temp_data_dir / "reports"),
        cv_folds=3,  # Reducido para pruebas más rápidas
        top_n_models=3,  # Reducido para pruebas
    )


# ============================================================
# Tests de integración
# ============================================================

class TestGridSearchPipelineIntegration:
    """Tests de integración para el pipeline completo de grid search"""

    def test_load_encode_split_chain(self, trainer):
        """Test que load_data, encode_target y split_data funcionan juntos."""
        # Load
        X, y = trainer.load_data()
        assert len(X) > 0
        assert len(y) > 0
        
        # Encode
        y_enc = trainer.encode_target(y)
        assert len(y_enc) == len(y)
        assert trainer.label_encoder is not None
        
        # Split
        X_train, X_test, y_train, y_test = trainer.split_data(X, y_enc)
        assert len(X_train) > 0
        assert len(X_test) > 0

    def test_preprocessor_pipeline_gridsearch_chain(self, trainer):
        """Test que create_preprocessor, create_pipeline y create_grid_search funcionan juntos."""
        # Load y preparar datos
        X, y = trainer.load_data()
        y_enc = trainer.encode_target(y)
        cat_cols = trainer.get_categorical_columns(X)
        X_train, X_test, y_train, y_test = trainer.split_data(X, y_enc)
        
        # Crear preprocesador
        preprocessor = trainer.create_preprocessor(cat_cols)
        assert preprocessor is not None
        
        # Crear pipeline
        pipeline = trainer.create_pipeline(preprocessor)
        assert pipeline is not None
        
        # Crear grid search
        param_grid = trainer.get_param_grid()
        grid_search = trainer.create_grid_search(pipeline, param_grid)
        assert grid_search is not None
        assert grid_search.cv == trainer.cv_folds

    def test_get_top_models_from_grid_search(self, trainer):
        """Test que get_top_models funciona con resultados de grid search."""
        from sklearn.model_selection import GridSearchCV
        from sklearn.pipeline import Pipeline
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.ensemble import RandomForestClassifier
        
        # Preparar datos
        X, y = trainer.load_data()
        y_enc = trainer.encode_target(y)
        cat_cols = trainer.get_categorical_columns(X)
        X_train, X_test, y_train, y_test = trainer.split_data(X, y_enc)
        
        # Crear grid search con scoring múltiple
        preprocessor = ColumnTransformer([
            ("ohe", OneHotEncoder(handle_unknown="ignore"), cat_cols)
        ])
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=10, random_state=42))
        ])
        
        param_grid = {"classifier__n_estimators": [10, 20]}
        scoring = {
            'accuracy': 'accuracy',
            'f1_weighted': 'f1_weighted',
            'f1_macro': 'f1_macro',
        }
        grid_search = GridSearchCV(
            pipeline, param_grid, cv=2, scoring=scoring, refit='f1_weighted'
        )
        
        # Entrenar (rápido con pocos datos)
        grid_search.fit(X_train, y_train)
        
        # Obtener top modelos
        top_indices = trainer.get_top_models(grid_search)
        
        assert len(top_indices) <= trainer.top_n_models
        assert len(top_indices) > 0

    def test_log_model_to_mlflow_integration(self, trainer, temp_data_dir):
        """Test que log_model_to_mlflow funciona con datos reales."""
        from sklearn.pipeline import Pipeline
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.ensemble import RandomForestClassifier
        
        # Preparar datos
        X, y = trainer.load_data()
        y_enc = trainer.encode_target(y)
        cat_cols = trainer.get_categorical_columns(X)
        X_train, X_test, y_train, y_test = trainer.split_data(X, y_enc)
        
        # Crear preprocesador
        preprocessor = trainer.create_preprocessor(cat_cols)
        trainer.models_dir = temp_data_dir / "models"
        trainer._ensure_directories()  # Crear directorios primero
        
        # Mock grid_search con resultados
        mock_grid_search = MagicMock()
        mock_grid_search.cv_results_ = {
            "params": [{"classifier__n_estimators": 10}],
            "mean_test_accuracy": [0.8],
            "std_test_accuracy": [0.05],
            "mean_test_f1_weighted": [0.75],
            "std_test_f1_weighted": [0.04],
            "mean_test_f1_macro": [0.73],
        }
        
        with patch("train.train_gridsearch.mlflow") as mock_mlflow:
            trainer.log_model_to_mlflow(
                mock_grid_search, 0, 1, X_train, X_test, y_train, y_test, preprocessor
            )
            
            mock_mlflow.log_params.assert_called_once()
            mock_mlflow.log_metrics.assert_called_once()

    def test_print_grid_search_info_output(self, trainer, capsys):
        """Test que print_grid_search_info imprime información correcta."""
        param_grid = trainer.get_param_grid()
        trainer.print_grid_search_info(param_grid)
        
        captured = capsys.readouterr()
        output = captured.out
        
        assert "GRID SEARCH" in output
        assert "combinaciones" in output.lower()
        assert "f1_weighted" in output.lower()

    def test_full_pipeline_setup(self, trainer):
        """Test que todos los componentes se configuran correctamente."""
        # Setup
        trainer._ensure_directories()
        trainer._setup_mlflow()
        
        # Load and prepare
        X, y = trainer.load_data()
        y_enc = trainer.encode_target(y)
        cat_cols = trainer.get_categorical_columns(X)
        X_train, X_test, y_train, y_test = trainer.split_data(X, y_enc)
        
        # Create components
        preprocessor = trainer.create_preprocessor(cat_cols)
        pipeline = trainer.create_pipeline(preprocessor)
        param_grid = trainer.get_param_grid()
        grid_search = trainer.create_grid_search(pipeline, param_grid)
        
        # Verificar que todo está configurado
        assert X_train.shape[0] > 0
        assert X_test.shape[0] > 0
        assert len(cat_cols) > 0
        assert preprocessor is not None
        assert pipeline is not None
        assert grid_search is not None
        assert len(param_grid) > 0


# ============================================================
# Tests de flujo completo (mockeado)
# ============================================================

class TestFullGridSearchFlow:
    """Tests del flujo completo de grid search (con mocks)"""

    @patch("train.train_gridsearch.mlflow")
    def test_run_method_setup(self, mock_mlflow, trainer):
        """Test que el método run() configura todo correctamente."""
        # Mock MLflow start_run
        mock_run = MagicMock()
        mock_mlflow.start_run.return_value.__enter__.return_value = mock_run
        
        # Mock grid_search.fit para evitar ejecución larga
        with patch.object(trainer, "run_grid_search") as mock_fit:
            mock_fit.return_value = MagicMock()
            mock_fit.return_value.cv_results_ = {
                "mean_test_f1_weighted": [0.7, 0.8, 0.75],
            }
            
            try:
                trainer.run()
            except Exception:
                # Puede fallar en diferentes puntos, pero verificamos setup
                pass
        
        # Verificar que MLflow fue configurado
        mock_mlflow.set_tracking_uri.assert_called()
        mock_mlflow.set_experiment.assert_called()

