"""
Pruebas de integración para el módulo mlops.preprocess

Estas pruebas validan el flujo completo del pipeline de preprocesamiento,
probando la interacción entre PreprocessPipeline, DatasetRepository y FeatureEngineering.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil

from mlops.preprocess import PreprocessPipeline
from mlops.dataset import DatasetRepository
from mlops.features import FeatureEngineering


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def temp_data_dir():
    """Directorio temporal para pruebas de integración."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_raw_dataframe():
    """DataFrame de ejemplo con datos 'sucios' (como vendrían del CSV raw)."""
    return pd.DataFrame({
        "numeric_col": ["1", "2", "3", "4", "5"],  # Strings que deben convertirse
        "categorical_col": ["  A  ", "  B  ", "  C  ", "  A  ", "  B  "],  # Con espacios
        "mixed_col": ["$100", "$200", "$300", "$400", "$500"],  # Con símbolos
        "target": ["Low", "Medium", "High", "Low", "Medium"],
    })


@pytest.fixture
def dataset_repo_with_data(temp_data_dir, sample_raw_dataframe):
    """DatasetRepository con datos de prueba guardados."""
    # Crear estructura de directorios
    raw_dir = temp_data_dir / "raw"
    interim_dir = temp_data_dir / "interim"
    raw_dir.mkdir(parents=True)
    interim_dir.mkdir(parents=True)
    
    # Guardar datos 'raw' en modified.csv
    modified_csv = raw_dir / "student_entry_performance_modified.csv"
    sample_raw_dataframe.to_csv(modified_csv, index=False)
    
    return DatasetRepository(
        modified_csv=modified_csv,
        interim_dir=interim_dir,
    )


# ============================================================
# Tests de integración: Flujo completo de preprocesamiento
# ============================================================

@pytest.mark.integration
class TestPreprocessFullPipeline:
    """Pruebas de integración para el flujo completo de preprocesamiento."""

    def test_full_pipeline_loads_and_cleans_data(self, dataset_repo_with_data, sample_raw_dataframe):
        """Prueba el flujo completo: load → typing → split → clean → save."""
        pipeline = PreprocessPipeline(dataset_repository=dataset_repo_with_data)
        
        # Ejecutar pipeline completo
        output_path = pipeline.make_clean_interim()
        
        # Verificar que se creó el archivo
        assert Path(output_path).exists()
        
        # Cargar y verificar resultado
        cleaned_df = pd.read_csv(output_path)
        
        # Verificaciones del resultado
        assert cleaned_df.shape[0] == sample_raw_dataframe.shape[0]
        assert set(cleaned_df.columns) == set(sample_raw_dataframe.columns)
        
        # Verificar que numeric_col fue convertida a numérico
        assert pd.api.types.is_numeric_dtype(cleaned_df["numeric_col"])
        
        # Verificar que categorical_col fue limpiada (sin espacios extra)
        for val in cleaned_df["categorical_col"]:
            if pd.notna(val):
                assert val.strip() == val  # Sin espacios al inicio/final
                assert val.islower()  # En minúsculas

    def test_pipeline_integrates_with_feature_engineering(self, dataset_repo_with_data):
        """Prueba que el pipeline integra correctamente con FeatureEngineering."""
        # Crear FeatureEngineering personalizado
        custom_fe = FeatureEngineering(categorical_guess_max=10)
        pipeline = PreprocessPipeline(
            dataset_repository=dataset_repo_with_data,
            feature_engineer=custom_fe
        )
        
        output_path = pipeline.make_clean_interim()
        cleaned_df = pd.read_csv(output_path)
        
        # Verificar que se usó el FeatureEngineering personalizado
        # (esto se verifica indirectamente por el resultado)
        assert cleaned_df.shape[0] > 0
        assert len(cleaned_df.columns) > 0

    def test_pipeline_preserves_all_columns(self, dataset_repo_with_data, sample_raw_dataframe):
        """Prueba que el pipeline preserva todas las columnas del DataFrame original."""
        pipeline = PreprocessPipeline(dataset_repository=dataset_repo_with_data)
        
        output_path = pipeline.make_clean_interim()
        cleaned_df = pd.read_csv(output_path)
        
        # Verificar que todas las columnas están presentes
        original_cols = set(sample_raw_dataframe.columns)
        cleaned_cols = set(cleaned_df.columns)
        assert original_cols == cleaned_cols

    def test_pipeline_handles_mixed_data_types(self, temp_data_dir):
        """Prueba que el pipeline maneja correctamente tipos de datos mixtos."""
        # Crear DataFrame con varios tipos de datos 'sucios'
        df = pd.DataFrame({
            "int_strings": ["1", "2", "3"],
            "float_strings": ["1.5", "2.5", "3.5"],
            "currency": ["$100", "$200", "$300"],
            "percent": ["50%", "75%", "100%"],
            "dirty_cats": ["  HELLO  ", "  WORLD  ", "  TEST  "],
        })
        
        raw_dir = temp_data_dir / "raw"
        interim_dir = temp_data_dir / "interim"
        raw_dir.mkdir(parents=True)
        interim_dir.mkdir(parents=True)
        
        modified_csv = raw_dir / "student_entry_performance_modified.csv"
        df.to_csv(modified_csv, index=False)
        
        repo = DatasetRepository(
            modified_csv=modified_csv,
            interim_dir=interim_dir,
        )
        
        pipeline = PreprocessPipeline(dataset_repository=repo)
        output_path = pipeline.make_clean_interim()
        cleaned_df = pd.read_csv(output_path)
        
        # Verificar que los números fueron convertidos
        assert pd.api.types.is_numeric_dtype(cleaned_df["int_strings"])
        assert pd.api.types.is_numeric_dtype(cleaned_df["float_strings"])
        assert pd.api.types.is_numeric_dtype(cleaned_df["currency"])
        assert pd.api.types.is_numeric_dtype(cleaned_df["percent"])
        
        # Verificar que las categóricas fueron limpiadas
        assert cleaned_df["dirty_cats"].iloc[0] == "hello"


# ============================================================
# Tests de integración: Interacción entre componentes
# ============================================================

@pytest.mark.integration
class TestPreprocessComponentIntegration:
    """Pruebas de integración para la interacción entre componentes."""

    def test_dataset_repository_and_feature_engineering_integration(self, dataset_repo_with_data):
        """Prueba la integración entre DatasetRepository y FeatureEngineering."""
        pipeline = PreprocessPipeline(dataset_repository=dataset_repo_with_data)
        
        # El pipeline debe usar ambos componentes correctamente
        output_path = pipeline.make_clean_interim()
        
        # Verificar que DatasetRepository guardó el archivo
        assert Path(output_path).exists()
        
        # Verificar que FeatureEngineering procesó los datos
        cleaned_df = pd.read_csv(output_path)
        assert cleaned_df.shape[0] > 0

    def test_basic_typing_before_feature_engineering(self, temp_data_dir):
        """Prueba que basic_typing se aplica antes de feature engineering."""
        # Crear DataFrame con strings numéricos
        df = pd.DataFrame({
            "numeric": ["1", "2", "3"],
            "categorical": ["A", "B", "C"],
        })
        
        raw_dir = temp_data_dir / "raw"
        interim_dir = temp_data_dir / "interim"
        raw_dir.mkdir(parents=True)
        interim_dir.mkdir(parents=True)
        
        modified_csv = raw_dir / "student_entry_performance_modified.csv"
        df.to_csv(modified_csv, index=False)
        
        repo = DatasetRepository(
            modified_csv=modified_csv,
            interim_dir=interim_dir,
        )
        
        pipeline = PreprocessPipeline(dataset_repository=repo)
        output_path = pipeline.make_clean_interim()
        cleaned_df = pd.read_csv(output_path)
        
        # Verificar que basic_typing convirtió numeric antes de que
        # FeatureEngineering lo procesara
        assert pd.api.types.is_numeric_dtype(cleaned_df["numeric"])

    def test_run_all_equivalent_to_make_clean_interim(self, dataset_repo_with_data):
        """Prueba que run_all() es equivalente a make_clean_interim()."""
        pipeline = PreprocessPipeline(dataset_repository=dataset_repo_with_data)
        
        # Ejecutar ambos métodos
        path1 = pipeline.make_clean_interim()
        path2 = pipeline.run_all()
        
        # Verificar que ambos retornan el mismo path
        assert path1 == path2
        
        # Verificar que el archivo existe
        assert Path(path1).exists()


# ============================================================
# Tests de integración: Casos edge del flujo completo
# ============================================================

@pytest.mark.integration
class TestPreprocessEdgeCases:
    """Pruebas de integración para casos edge del flujo completo."""

    def test_pipeline_handles_empty_dataframe(self, temp_data_dir):
        """Prueba que el pipeline maneja DataFrames con filas vacías."""
        # Crear DataFrame con columnas pero sin filas (más realista)
        df = pd.DataFrame(columns=["col1", "col2"])
        
        raw_dir = temp_data_dir / "raw"
        interim_dir = temp_data_dir / "interim"
        raw_dir.mkdir(parents=True)
        interim_dir.mkdir(parents=True)
        
        modified_csv = raw_dir / "student_entry_performance_modified.csv"
        df.to_csv(modified_csv, index=False)
        
        repo = DatasetRepository(
            modified_csv=modified_csv,
            interim_dir=interim_dir,
        )
        
        pipeline = PreprocessPipeline(dataset_repository=repo)
        output_path = pipeline.make_clean_interim()
        cleaned_df = pd.read_csv(output_path)
        
        # Debe crear el archivo aunque esté vacío
        assert Path(output_path).exists()
        assert cleaned_df.shape[0] == 0
        assert len(cleaned_df.columns) > 0  # Debe tener columnas

    def test_pipeline_handles_dataframe_with_only_numerics(self, temp_data_dir):
        """Prueba que el pipeline maneja DataFrames solo con columnas numéricas."""
        df = pd.DataFrame({
            "col1": [1, 2, 3],
            "col2": [1.1, 2.2, 3.3],
        })
        
        raw_dir = temp_data_dir / "raw"
        interim_dir = temp_data_dir / "interim"
        raw_dir.mkdir(parents=True)
        interim_dir.mkdir(parents=True)
        
        modified_csv = raw_dir / "student_entry_performance_modified.csv"
        df.to_csv(modified_csv, index=False)
        
        repo = DatasetRepository(
            modified_csv=modified_csv,
            interim_dir=interim_dir,
        )
        
        pipeline = PreprocessPipeline(dataset_repository=repo)
        output_path = pipeline.make_clean_interim()
        cleaned_df = pd.read_csv(output_path)
        
        # Debe procesar correctamente
        assert cleaned_df.shape[0] == df.shape[0]
        assert set(cleaned_df.columns) == set(df.columns)

    def test_pipeline_handles_dataframe_with_only_categoricals(self, temp_data_dir):
        """Prueba que el pipeline maneja DataFrames solo con columnas categóricas."""
        df = pd.DataFrame({
            "cat1": ["A", "B", "C"],
            "cat2": ["X", "Y", "Z"],
        })
        
        raw_dir = temp_data_dir / "raw"
        interim_dir = temp_data_dir / "interim"
        raw_dir.mkdir(parents=True)
        interim_dir.mkdir(parents=True)
        
        modified_csv = raw_dir / "student_entry_performance_modified.csv"
        df.to_csv(modified_csv, index=False)
        
        repo = DatasetRepository(
            modified_csv=modified_csv,
            interim_dir=interim_dir,
        )
        
        pipeline = PreprocessPipeline(dataset_repository=repo)
        output_path = pipeline.make_clean_interim()
        cleaned_df = pd.read_csv(output_path)
        
        # Debe procesar correctamente y limpiar las categóricas
        assert cleaned_df.shape[0] == df.shape[0]
        assert set(cleaned_df.columns) == set(df.columns)
        
        # Verificar que fueron limpiadas
        assert cleaned_df["cat1"].iloc[0] == "a"

    def test_pipeline_removes_mixed_type_col(self, temp_data_dir):
        """Prueba que el pipeline elimina 'mixed_type_col' si existe."""
        # Crear DataFrame con mixed_type_col
        df = pd.DataFrame({
            "numeric_col": [1, 2, 3],
            "categorical_col": ["A", "B", "C"],
            "mixed_type_col": [1, 2, 3],  # Columna que debe eliminarse
            "target": ["Low", "Medium", "High"],
        })
        
        raw_dir = temp_data_dir / "raw"
        interim_dir = temp_data_dir / "interim"
        raw_dir.mkdir(parents=True)
        interim_dir.mkdir(parents=True)
        
        modified_csv = raw_dir / "student_entry_performance_modified.csv"
        df.to_csv(modified_csv, index=False)
        
        repo = DatasetRepository(
            modified_csv=modified_csv,
            interim_dir=interim_dir,
        )
        
        pipeline = PreprocessPipeline(dataset_repository=repo)
        output_path = pipeline.make_clean_interim()
        cleaned_df = pd.read_csv(output_path)
        
        # Verificar que mixed_type_col fue eliminada
        assert "mixed_type_col" not in cleaned_df.columns
        # Verificar que otras columnas se preservaron
        assert "numeric_col" in cleaned_df.columns
        assert "categorical_col" in cleaned_df.columns
        assert "target" in cleaned_df.columns
        # Verificar que el número de filas se preservó
        assert cleaned_df.shape[0] == df.shape[0]

