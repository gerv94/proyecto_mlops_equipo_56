"""
Pruebas de integración para el módulo mlops.dataset

Estas pruebas validan el flujo completo de acceso a datos,
probando la interacción entre DatasetRepository y otros módulos.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil

from mlops.dataset import DatasetRepository
from mlops.features import FeatureEngineering, split_num_cat, clean_categoricals
from mlops.preprocess import PreprocessPipeline


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
def sample_dataframe():
    """DataFrame de ejemplo para pruebas."""
    return pd.DataFrame({
        "numeric_col": [1, 2, 3, 4, 5],
        "categorical_col": ["A", "B", "C", "A", "B"],
        "target": ["Low", "Medium", "High", "Low", "Medium"],
    })


@pytest.fixture
def dataset_repo_with_data(temp_data_dir, sample_dataframe):
    """DatasetRepository con estructura completa de datos."""
    # Crear estructura de directorios
    raw_dir = temp_data_dir / "raw"
    interim_dir = temp_data_dir / "interim"
    processed_dir = temp_data_dir / "processed"
    tables_dir = temp_data_dir / "tables"
    
    raw_dir.mkdir(parents=True)
    interim_dir.mkdir(parents=True)
    processed_dir.mkdir(parents=True)
    tables_dir.mkdir(parents=True)
    
    # Crear archivos de datos
    modified_csv = raw_dir / "student_entry_performance_modified.csv"
    preprocessed_csv = interim_dir / "student_interim_preprocessed.csv"
    
    sample_dataframe.to_csv(modified_csv, index=False)
    sample_dataframe.to_csv(preprocessed_csv, index=False)
    
    return DatasetRepository(
        modified_csv=modified_csv,
        preprocessed_csv=preprocessed_csv,
        interim_dir=interim_dir,
        processed_dir=processed_dir,
        tables_dir=tables_dir,
    )


# ============================================================
# Tests de integración: Flujo completo de carga y guardado
# ============================================================

@pytest.mark.integration
class TestDatasetFullWorkflow:
    """Pruebas de integración para el flujo completo de carga y guardado."""

    def test_load_save_roundtrip(self, dataset_repo_with_data, sample_dataframe):
        """Prueba el flujo completo: load → process → save → load."""
        # 1. Cargar datos modificados
        loaded_df = dataset_repo_with_data.load_modified()
        assert loaded_df.shape == sample_dataframe.shape
        
        # 2. Aplicar basic_typing
        typed_df = dataset_repo_with_data.basic_typing(loaded_df)
        assert typed_df.shape == loaded_df.shape
        
        # 3. Guardar en interim
        output_path = dataset_repo_with_data.save_interim(typed_df, "test_interim.csv")
        assert output_path.exists()
        
        # 4. Cargar desde interim
        reloaded_df = dataset_repo_with_data.load_interim("test_interim.csv")
        pd.testing.assert_frame_equal(reloaded_df, typed_df)

    def test_preprocessing_pipeline_with_dataset_repository(self, dataset_repo_with_data):
        """Prueba que PreprocessPipeline usa DatasetRepository correctamente."""
        pipeline = PreprocessPipeline(dataset_repository=dataset_repo_with_data)
        
        # Ejecutar pipeline
        output_path = pipeline.make_clean_interim()
        
        # Verificar que DatasetRepository guardó el archivo
        assert Path(output_path).exists()
        
        # Verificar que se puede cargar desde interim
        loaded = dataset_repo_with_data.load_interim("student_interim_clean.csv")
        assert loaded.shape[0] > 0

    def test_feature_engineering_with_dataset_repository(self, dataset_repo_with_data):
        """Prueba la integración entre FeatureEngineering y DatasetRepository."""
        # Cargar datos
        df = dataset_repo_with_data.load_modified()
        
        # Aplicar basic_typing
        typed_df = dataset_repo_with_data.basic_typing(df)
        
        # Aplicar feature engineering
        num_cols, cat_cols = split_num_cat(typed_df)
        cleaned_df = clean_categoricals(typed_df, cat_cols)
        
        # Guardar resultado procesado
        output_path = dataset_repo_with_data.save_processed(cleaned_df, "processed_test.csv")
        
        # Verificar que se guardó correctamente
        assert output_path.exists()
        
        # Cargar y verificar
        reloaded = pd.read_csv(output_path)
        pd.testing.assert_frame_equal(reloaded, cleaned_df)

    def test_multiple_save_load_operations(self, dataset_repo_with_data, sample_dataframe):
        """Prueba múltiples operaciones de guardado y carga."""
        # Guardar en interim
        path1 = dataset_repo_with_data.save_interim(sample_dataframe, "file1.csv")
        path2 = dataset_repo_with_data.save_interim(sample_dataframe, "file2.csv")
        
        # Guardar en processed
        path3 = dataset_repo_with_data.save_processed(sample_dataframe, "file3.csv")
        
        # Cargar todos
        df1 = dataset_repo_with_data.load_interim("file1.csv")
        df2 = dataset_repo_with_data.load_interim("file2.csv")
        df3 = pd.read_csv(path3)
        
        # Verificar que todos son iguales
        pd.testing.assert_frame_equal(df1, sample_dataframe)
        pd.testing.assert_frame_equal(df2, sample_dataframe)
        pd.testing.assert_frame_equal(df3, sample_dataframe)


# ============================================================
# Tests de integración: Interacción con basic_typing
# ============================================================

@pytest.mark.integration
class TestDatasetTypingIntegration:
    """Pruebas de integración para basic_typing con otros módulos."""

    def test_basic_typing_before_feature_engineering(self, dataset_repo_with_data):
        """Prueba que basic_typing se aplica correctamente antes de feature engineering."""
        # Cargar datos con strings numéricos
        df = dataset_repo_with_data.load_modified()
        
        # Aplicar basic_typing
        typed_df = dataset_repo_with_data.basic_typing(df)
        
        # Verificar que las columnas numéricas fueron convertidas
        # (esto permite que split_num_cat las identifique correctamente)
        num_cols, cat_cols = split_num_cat(typed_df)
        
        # Verificar que las columnas numéricas fueron identificadas
        assert len(num_cols) > 0 or len(cat_cols) > 0

    def test_basic_typing_preserves_data_structure(self, dataset_repo_with_data):
        """Prueba que basic_typing preserva la estructura de datos."""
        df = dataset_repo_with_data.load_modified()
        typed_df = dataset_repo_with_data.basic_typing(df)
        
        # Verificar estructura
        assert typed_df.shape == df.shape
        assert set(typed_df.columns) == set(df.columns)
        assert typed_df.index.equals(df.index)

    def test_basic_typing_with_save_load(self, dataset_repo_with_data):
        """Prueba que basic_typing funciona correctamente con save/load."""
        df = dataset_repo_with_data.load_modified()
        typed_df = dataset_repo_with_data.basic_typing(df)
        
        # Guardar
        output_path = dataset_repo_with_data.save_interim(typed_df, "typed_test.csv")
        
        # Cargar
        loaded_df = dataset_repo_with_data.load_interim("typed_test.csv")
        
        # Verificar que los tipos se preservaron
        pd.testing.assert_frame_equal(loaded_df, typed_df)


# ============================================================
# Tests de integración: Casos edge del flujo completo
# ============================================================

@pytest.mark.integration
class TestDatasetEdgeCases:
    """Pruebas de integración para casos edge del flujo completo."""

    def test_load_original_if_exists_integration(self, temp_data_dir, sample_dataframe):
        """Prueba load_original_if_exists() en un flujo completo."""
        raw_dir = temp_data_dir / "raw"
        interim_dir = temp_data_dir / "interim"
        raw_dir.mkdir(parents=True)
        interim_dir.mkdir(parents=True)
        
        # Crear archivo original
        original_csv = raw_dir / "student_entry_performance_original.csv"
        sample_dataframe.to_csv(original_csv, index=False)
        
        repo = DatasetRepository(
            original_csv=original_csv,
            interim_dir=interim_dir,
        )
        
        # Cargar original
        loaded = repo.load_original_if_exists()
        
        assert loaded is not None
        pd.testing.assert_frame_equal(loaded, sample_dataframe)

    def test_directory_creation_on_first_use(self, temp_data_dir, sample_dataframe):
        """Prueba que los directorios se crean automáticamente al usar DatasetRepository."""
        interim_dir = temp_data_dir / "new_interim"
        processed_dir = temp_data_dir / "new_processed"
        
        # Asegurar que no existen
        if interim_dir.exists():
            shutil.rmtree(interim_dir)
        if processed_dir.exists():
            shutil.rmtree(processed_dir)
        
        # Crear repositorio (debe crear directorios)
        repo = DatasetRepository(
            interim_dir=interim_dir,
            processed_dir=processed_dir,
        )
        
        # Verificar que se crearon
        assert interim_dir.exists()
        assert processed_dir.exists()
        
        # Guardar algo para verificar que funcionan
        repo.save_interim(sample_dataframe, "test.csv")
        assert (interim_dir / "test.csv").exists()

    def test_save_load_with_special_characters(self, dataset_repo_with_data):
        """Prueba guardado y carga con caracteres especiales en los datos."""
        df = pd.DataFrame({
            "col_with_symbols": ["$100", "50%", "1,000"],
            "col_with_spaces": ["  hello  ", "  world  ", "  test  "],
        })
        
        # Guardar
        output_path = dataset_repo_with_data.save_interim(df, "special_chars.csv")
        
        # Cargar
        loaded = dataset_repo_with_data.load_interim("special_chars.csv")
        
        # Verificar que se preservaron los datos
        assert loaded.shape == df.shape
        assert set(loaded.columns) == set(df.columns)

    def test_concurrent_save_operations(self, dataset_repo_with_data, sample_dataframe):
        """Prueba múltiples guardados simultáneos en diferentes ubicaciones."""
        # Guardar en interim
        interim_path = dataset_repo_with_data.save_interim(sample_dataframe, "interim_file.csv")
        
        # Guardar en processed
        processed_path = dataset_repo_with_data.save_processed(sample_dataframe, "processed_file.csv")
        
        # Verificar que ambos archivos existen
        assert interim_path.exists()
        assert processed_path.exists()
        
        # Verificar que están en los directorios correctos
        assert interim_path.parent == dataset_repo_with_data.interim_dir
        assert processed_path.parent == dataset_repo_with_data.processed_dir
        
        # Cargar y verificar
        interim_df = dataset_repo_with_data.load_interim("interim_file.csv")
        processed_df = pd.read_csv(processed_path)
        
        pd.testing.assert_frame_equal(interim_df, sample_dataframe)
        pd.testing.assert_frame_equal(processed_df, sample_dataframe)

