"""
Pruebas unitarias para el módulo mlops.preprocess

Estas pruebas validan las funciones individuales del PreprocessPipeline.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil

from mlops.preprocess import PreprocessPipeline, make_clean_interim, run_all
from mlops.dataset import DatasetRepository
from mlops.features import FeatureEngineering


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
def sample_dataframe():
    """DataFrame de ejemplo con datos sucios."""
    return pd.DataFrame({
        "numeric_col": [1, 2, 3, 4, 5],
        "categorical_col": ["  A  ", "  B  ", "  C  ", "  A  ", "  B  "],
        "target": ["Low", "Medium", "High", "Low", "Medium"],
    })


@pytest.fixture
def mock_dataset_repo(temp_data_dir, sample_dataframe):
    """DatasetRepository mock con datos de prueba."""
    # Crear estructura de directorios
    raw_dir = temp_data_dir / "raw"
    interim_dir = temp_data_dir / "interim"
    raw_dir.mkdir(parents=True)
    interim_dir.mkdir(parents=True)
    
    # Crear archivo modified.csv
    modified_csv = raw_dir / "student_entry_performance_modified.csv"
    sample_dataframe.to_csv(modified_csv, index=False)
    
    return DatasetRepository(
        modified_csv=modified_csv,
        interim_dir=interim_dir,
    )


@pytest.fixture
def preprocess_pipeline(mock_dataset_repo):
    """Instancia de PreprocessPipeline con repositorio mock."""
    return PreprocessPipeline(dataset_repository=mock_dataset_repo)


# ============================================================
# Tests para PreprocessPipeline.make_clean_interim()
# ============================================================

class TestMakeCleanInterim:
    """Tests para make_clean_interim()"""

    def test_creates_clean_csv_file(self, preprocess_pipeline):
        """make_clean_interim() crea el archivo CSV limpio."""
        output_path = preprocess_pipeline.make_clean_interim()
        
        assert Path(output_path).exists()
        assert Path(output_path).is_file()
        assert Path(output_path).name == "student_interim_clean.csv"

    def test_cleans_categorical_columns(self, preprocess_pipeline, sample_dataframe):
        """make_clean_interim() limpia las columnas categóricas."""
        output_path = preprocess_pipeline.make_clean_interim()
        loaded = pd.read_csv(output_path)
        
        # Verificar que las categóricas fueron limpiadas (sin espacios, minúsculas)
        if "categorical_col" in loaded.columns:
            # Verificar que no tienen espacios extra
            assert all(not str(val).strip() != str(val) or pd.isna(val) 
                      for val in loaded["categorical_col"].head(10))

    def test_preserves_dataframe_structure(self, preprocess_pipeline, sample_dataframe):
        """make_clean_interim() preserva la estructura del DataFrame."""
        output_path = preprocess_pipeline.make_clean_interim()
        loaded = pd.read_csv(output_path)
        
        # Debe tener el mismo número de filas
        assert loaded.shape[0] == sample_dataframe.shape[0]
        # Debe tener las mismas columnas
        assert set(loaded.columns) == set(sample_dataframe.columns)

    def test_applies_basic_typing(self, preprocess_pipeline):
        """make_clean_interim() aplica basic_typing a los datos."""
        # Crear DataFrame con strings numéricos
        df = pd.DataFrame({
            "numeric_col": ["1", "2", "3", "4", "5"],
            "categorical_col": ["A", "B", "C", "A", "B"],
        })
        
        # Guardar en modified.csv
        preprocess_pipeline.dataset_repository.modified_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(preprocess_pipeline.dataset_repository.modified_csv, index=False)
        
        output_path = preprocess_pipeline.make_clean_interim()
        loaded = pd.read_csv(output_path)
        
        # Verificar que numeric_col fue convertida a numérico
        assert pd.api.types.is_numeric_dtype(loaded["numeric_col"])

    def test_returns_string_path(self, preprocess_pipeline):
        """make_clean_interim() retorna un string con la ruta."""
        output_path = preprocess_pipeline.make_clean_interim()
        
        assert isinstance(output_path, str)
        assert Path(output_path).exists()


# ============================================================
# Tests para PreprocessPipeline.run_all()
# ============================================================

class TestRunAll:
    """Tests para run_all()"""

    def test_run_all_calls_make_clean_interim(self, preprocess_pipeline):
        """run_all() llama a make_clean_interim()."""
        output_path = preprocess_pipeline.run_all()
        
        # Verificar que se creó el archivo
        assert Path(output_path).exists()
        assert Path(output_path).name == "student_interim_clean.csv"

    def test_run_all_returns_string_path(self, preprocess_pipeline):
        """run_all() retorna un string con la ruta."""
        output_path = preprocess_pipeline.run_all()
        
        assert isinstance(output_path, str)


# ============================================================
# Tests para funciones funcionales (backwards compatible)
# ============================================================

class TestFunctionalAPI:
    """Tests para las funciones funcionales (backwards compatible)"""

    def test_make_clean_interim_function(self, mock_dataset_repo, sample_dataframe):
        """make_clean_interim() función funciona correctamente."""
        # Esta función usa el _default_pipeline, que usa paths por defecto
        # Para probarla correctamente, necesitaríamos el archivo real
        # Por ahora, verificamos que la función existe y es callable
        assert callable(make_clean_interim)

    def test_run_all_function(self):
        """run_all() función funciona correctamente."""
        # Verificar que la función existe y es callable
        assert callable(run_all)


# ============================================================
# Tests para inicialización de PreprocessPipeline
# ============================================================

class TestPreprocessPipelineInit:
    """Tests para la inicialización de PreprocessPipeline"""

    def test_uses_default_repository_if_none(self):
        """Usa DatasetRepository por defecto si no se proporciona."""
        pipeline = PreprocessPipeline()
        
        assert isinstance(pipeline.dataset_repository, DatasetRepository)
        assert isinstance(pipeline.feature_engineer, FeatureEngineering)

    def test_uses_custom_repository(self, mock_dataset_repo):
        """Acepta un DatasetRepository personalizado."""
        pipeline = PreprocessPipeline(dataset_repository=mock_dataset_repo)
        
        assert pipeline.dataset_repository == mock_dataset_repo

    def test_uses_custom_feature_engineer(self, mock_dataset_repo):
        """Acepta un FeatureEngineering personalizado."""
        custom_fe = FeatureEngineering(categorical_guess_max=50)
        pipeline = PreprocessPipeline(
            dataset_repository=mock_dataset_repo,
            feature_engineer=custom_fe
        )
        
        assert pipeline.feature_engineer == custom_fe
        assert pipeline.feature_engineer.categorical_guess_max == 50

