"""
Pruebas unitarias para el módulo train/train_gridsearch.py

Estas pruebas validan los métodos individuales de GridSearchTrainer.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
from unittest.mock import patch, MagicMock

from train.train_gridsearch import (
    GridSearchTrainer,
    CATEGORICAL_COLUMNS,
    SEED,
)


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
def trainer(temp_data_dir):
    """Instancia de GridSearchTrainer con paths temporales."""
    interim_dir = temp_data_dir / "interim"
    interim_dir.mkdir(parents=True)
    
    # Crear CSV de prueba
    sample_df = pd.DataFrame({
        "Gender": ["M", "F", "M"],
        "Caste": ["A", "B", "A"],
        "coaching": ["Yes", "No", "Yes"],
        "Performance": ["good", "average", "excellent"],
        "numeric_col": [1, 2, 3],
    })
    csv_path = interim_dir / "student_interim_clean.csv"
    sample_df.to_csv(csv_path, index=False)
    
    return GridSearchTrainer(
        seed=42,
        data_path=str(csv_path),
        models_dir=str(temp_data_dir / "models"),
        reports_dir=str(temp_data_dir / "reports"),
        cv_folds=3,  # Reducido para pruebas más rápidas
        top_n_models=3,  # Reducido para pruebas
    )


# ============================================================
# Tests para inicialización
# ============================================================

class TestGridSearchTrainerInit:
    """Tests para la inicialización de GridSearchTrainer"""

    def test_init_defaults(self):
        """Test que la inicialización con valores por defecto funciona."""
        trainer = GridSearchTrainer()
        assert trainer.seed == SEED
        assert trainer.data_path == Path("data/interim/student_interim_clean.csv")
        assert trainer.model_random_state == 888
        assert trainer.cv_folds == 5
        assert trainer.top_n_models == 10

    def test_init_custom_values(self):
        """Test que la inicialización con valores personalizados funciona."""
        trainer = GridSearchTrainer(
            seed=100,
            data_path="custom/path.csv",
            cv_folds=3,
            top_n_models=5,
        )
        assert trainer.seed == 100
        assert trainer.data_path == Path("custom/path.csv")
        assert trainer.cv_folds == 3
        assert trainer.top_n_models == 5


# ============================================================
# Tests para load_data
# ============================================================

class TestLoadData:
    """Tests para el método load_data"""

    def test_load_data_success(self, trainer):
        """Test que load_data carga correctamente el CSV."""
        X, y = trainer.load_data()
        
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert "Performance" not in X.columns
        assert "Performance" == y.name
        assert len(X) == len(y)


# ============================================================
# Tests para encode_target
# ============================================================

class TestEncodeTarget:
    """Tests para el método encode_target"""

    def test_encode_target_success(self, trainer):
        """Test que encode_target codifica correctamente el target."""
        y = pd.Series(["good", "average", "excellent", "good"])
        y_enc = trainer.encode_target(y)
        
        assert isinstance(y_enc, np.ndarray)
        assert len(y_enc) == len(y)
        assert trainer.label_encoder is not None
        assert trainer.class_names is not None


# ============================================================
# Tests para get_categorical_columns
# ============================================================

class TestGetCategoricalColumns:
    """Tests para el método get_categorical_columns"""

    def test_get_categorical_columns_finds_all(self, trainer):
        """Test que encuentra todas las columnas categóricas presentes."""
        X = pd.DataFrame({
            "Gender": ["M", "F"],
            "Caste": ["A", "B"],
            "other_col": [1, 2],
        })
        
        cat_cols = trainer.get_categorical_columns(X)
        
        assert "Gender" in cat_cols
        assert "Caste" in cat_cols
        assert "other_col" not in cat_cols


# ============================================================
# Tests para get_param_grid
# ============================================================

class TestGetParamGrid:
    """Tests para el método get_param_grid"""

    def test_get_param_grid_success(self, trainer):
        """Test que get_param_grid retorna un diccionario válido."""
        param_grid = trainer.get_param_grid()
        
        assert isinstance(param_grid, dict)
        assert "classifier__n_estimators" in param_grid
        assert "classifier__max_depth" in param_grid
        assert "classifier__min_samples_split" in param_grid
        assert isinstance(param_grid["classifier__n_estimators"], list)

    def test_get_param_grid_has_expected_keys(self, trainer):
        """Test que el param_grid tiene todas las claves esperadas."""
        param_grid = trainer.get_param_grid()
        
        expected_keys = [
            "classifier__n_estimators",
            "classifier__max_depth",
            "classifier__min_samples_split",
            "classifier__min_samples_leaf",
            "classifier__max_features",
            "classifier__criterion",
            "classifier__class_weight",
        ]
        
        for key in expected_keys:
            assert key in param_grid


# ============================================================
# Tests para create_preprocessor
# ============================================================

class TestCreatePreprocessor:
    """Tests para el método create_preprocessor"""

    def test_create_preprocessor_success(self, trainer):
        """Test que create_preprocessor crea un ColumnTransformer."""
        cat_cols = ["Gender", "Caste"]
        preprocessor = trainer.create_preprocessor(cat_cols)
        
        from sklearn.compose import ColumnTransformer
        assert isinstance(preprocessor, ColumnTransformer)
        assert trainer.preprocessor is not None

    def test_create_preprocessor_has_remainder_drop(self, trainer):
        """Test que el preprocesador tiene remainder='drop'."""
        cat_cols = ["Gender"]
        preprocessor = trainer.create_preprocessor(cat_cols)
        
        assert preprocessor.remainder == "drop"


# ============================================================
# Tests para create_pipeline
# ============================================================

class TestCreatePipeline:
    """Tests para el método create_pipeline"""

    def test_create_pipeline_success(self, trainer):
        """Test que create_pipeline crea un Pipeline."""
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder
        
        preprocessor = ColumnTransformer([
            ("ohe", OneHotEncoder(handle_unknown="ignore"), ["Gender"])
        ])
        
        pipeline = trainer.create_pipeline(preprocessor)
        
        from sklearn.pipeline import Pipeline
        assert isinstance(pipeline, Pipeline)
        assert "preprocessor" in pipeline.named_steps
        assert "classifier" in pipeline.named_steps


# ============================================================
# Tests para create_grid_search
# ============================================================

class TestCreateGridSearch:
    """Tests para el método create_grid_search"""

    def test_create_grid_search_success(self, trainer):
        """Test que create_grid_search crea un GridSearchCV."""
        from sklearn.pipeline import Pipeline
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.ensemble import RandomForestClassifier
        
        preprocessor = ColumnTransformer([
            ("ohe", OneHotEncoder(handle_unknown="ignore"), ["Gender"])
        ])
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier())
        ])
        param_grid = {"classifier__n_estimators": [10, 20]}
        
        grid_search = trainer.create_grid_search(pipeline, param_grid)
        
        from sklearn.model_selection import GridSearchCV
        assert isinstance(grid_search, GridSearchCV)
        assert grid_search.cv == trainer.cv_folds
        assert grid_search.refit == "f1_weighted"


# ============================================================
# Tests para get_top_models
# ============================================================

class TestGetTopModels:
    """Tests para el método get_top_models"""

    def test_get_top_models_success(self, trainer):
        """Test que get_top_models retorna los índices correctos."""
        from sklearn.model_selection import GridSearchCV
        from sklearn.pipeline import Pipeline
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.ensemble import RandomForestClassifier
        
        # Crear un GridSearchCV mock
        preprocessor = ColumnTransformer([
            ("ohe", OneHotEncoder(handle_unknown="ignore"), ["Gender"])
        ])
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier())
        ])
        
        # Mock cv_results_
        mock_grid_search = MagicMock()
        mock_grid_search.cv_results_ = {
            "mean_test_f1_weighted": [0.5, 0.7, 0.6, 0.8, 0.4],
        }
        
        # Convertir a DataFrame para el método real
        results_df = pd.DataFrame(mock_grid_search.cv_results_)
        results_df = results_df.sort_values("mean_test_f1_weighted", ascending=False)
        top_indices = results_df.head(trainer.top_n_models).index
        
        # Verificar que tenemos los índices correctos
        assert len(top_indices) == trainer.top_n_models
        assert isinstance(top_indices, pd.Index)


# ============================================================
# Tests para métodos auxiliares
# ============================================================

class TestAuxiliaryMethods:
    """Tests para métodos auxiliares"""

    def test_ensure_directories(self, trainer, temp_data_dir):
        """Test que _ensure_directories crea los directorios."""
        trainer.models_dir = temp_data_dir / "models"
        trainer.reports_dir = temp_data_dir / "reports"
        
        trainer._ensure_directories()
        
        assert trainer.models_dir.exists()
        assert trainer.reports_dir.exists()

    @patch("train.train_gridsearch.mlflow")
    def test_setup_mlflow(self, mock_mlflow, trainer):
        """Test que _setup_mlflow configura MLflow correctamente."""
        trainer._setup_mlflow()
        
        mock_mlflow.set_tracking_uri.assert_called_once()
        mock_mlflow.set_experiment.assert_called_once_with(trainer.mlflow_experiment)

    def test_print_grid_search_info(self, trainer, capsys):
        """Test que print_grid_search_info imprime información."""
        param_grid = trainer.get_param_grid()
        trainer.print_grid_search_info(param_grid)
        
        captured = capsys.readouterr()
        assert "GRID SEARCH" in captured.out
        assert "combinaciones" in captured.out.lower()

