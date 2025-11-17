"""
Pruebas unitarias para el módulo mlops.dataset

Estas pruebas validan las funciones individuales del DatasetRepository
y los helpers de tipado.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil

from mlops.dataset import (
    DatasetRepository,
    coerce_numeric_col,
    basic_typing,
    save_interim,
    save_processed,
    load_interim,
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
def sample_dataframe():
    """DataFrame de ejemplo para pruebas."""
    return pd.DataFrame({
        "numeric_col": [1, 2, 3, 4, 5],
        "categorical_col": ["A", "B", "C", "A", "B"],
        "target": ["Low", "Medium", "High", "Low", "Medium"],
    })


@pytest.fixture
def dataset_repo(temp_data_dir):
    """Instancia de DatasetRepository con directorios temporales."""
    interim_dir = temp_data_dir / "interim"
    processed_dir = temp_data_dir / "processed"
    tables_dir = temp_data_dir / "tables"
    
    modified_csv = temp_data_dir / "modified.csv"
    preprocessed_csv = temp_data_dir / "preprocessed.csv"
    
    return DatasetRepository(
        modified_csv=modified_csv,
        preprocessed_csv=preprocessed_csv,
        interim_dir=interim_dir,
        processed_dir=processed_dir,
        tables_dir=tables_dir,
    )


# ============================================================
# Tests para coerce_numeric_column()
# ============================================================

class TestCoerceNumericColumn:
    """Tests para coerce_numeric_column() - coerción de tipos"""

    def test_preserves_already_numeric_series(self):
        """No modifica series que ya son numéricas."""
        series = pd.Series([1, 2, 3, 4, 5])
        result = coerce_numeric_col(series)
        pd.testing.assert_series_equal(result, series)

    def test_coerces_string_numbers(self):
        """Convierte strings numéricos a números."""
        series = pd.Series(["1", "2", "3", "4", "5"])
        result = coerce_numeric_col(series)
        assert result.dtype in [np.int64, np.float64]
        assert result.iloc[0] == 1

    def test_removes_currency_symbols(self):
        """Elimina símbolos de moneda ($)."""
        series = pd.Series(["$100", "$200", "$300"])
        result = coerce_numeric_col(series)
        assert result.dtype in [np.int64, np.float64]
        assert result.iloc[0] == 100

    def test_removes_percent_symbols(self):
        """Elimina símbolos de porcentaje (%)."""
        series = pd.Series(["50%", "75%", "100%"])
        result = coerce_numeric_col(series)
        assert result.dtype in [np.int64, np.float64]
        assert result.iloc[0] == 50

    def test_removes_commas(self):
        """Elimina comas de números."""
        series = pd.Series(["1,000", "2,000", "3,000"])
        result = coerce_numeric_col(series)
        assert result.dtype in [np.int64, np.float64]
        assert result.iloc[0] == 1000

    def test_handles_mixed_symbols(self):
        """Maneja múltiples símbolos en el mismo string."""
        series = pd.Series(["$1,000", "$2,000", "$3,000"])
        result = coerce_numeric_col(series)
        assert result.dtype in [np.int64, np.float64]
        assert result.iloc[0] == 1000

    def test_strips_whitespace(self):
        """Elimina espacios en blanco."""
        series = pd.Series(["  100  ", "  200  ", "  300  "])
        result = coerce_numeric_col(series)
        assert result.dtype in [np.int64, np.float64]
        assert result.iloc[0] == 100

    def test_handles_non_numeric_strings(self):
        """Maneja strings que no son numéricos (retorna como string)."""
        series = pd.Series(["hello", "world", "test"])
        result = coerce_numeric_col(series)
        # Debe retornar como string si no puede convertir
        assert result.dtype == object

    def test_handles_nan_strings(self):
        """Convierte strings 'nan' y 'None' a None."""
        series = pd.Series(["100", "nan", "None", "200"])
        result = coerce_numeric_col(series)
        assert pd.isna(result.iloc[1])
        assert pd.isna(result.iloc[2])

    def test_handles_empty_strings(self):
        """Convierte strings vacíos a None."""
        series = pd.Series(["100", "", "200"])
        result = coerce_numeric_col(series)
        assert pd.isna(result.iloc[1])


# ============================================================
# Tests para basic_typing()
# ============================================================

class TestBasicTyping:
    """Tests para basic_typing() - tipado automático"""

    def test_applies_coercion_to_all_columns(self):
        """Aplica coerción numérica a todas las columnas."""
        df = pd.DataFrame({
            "col1": ["1", "2", "3"],
            "col2": ["$100", "$200", "$300"],
            "col3": ["A", "B", "C"],
        })
        result = basic_typing(df)
        
        # Verificar que col1 y col2 fueron convertidas
        assert result["col1"].dtype in [np.int64, np.float64]
        assert result["col2"].dtype in [np.int64, np.float64]
        # col3 debe seguir siendo string
        assert result["col3"].dtype == object

    def test_preserves_numeric_columns(self):
        """No modifica columnas que ya son numéricas."""
        df = pd.DataFrame({
            "numeric": [1, 2, 3],
            "float": [1.1, 2.2, 3.3],
        })
        result = basic_typing(df)
        pd.testing.assert_frame_equal(result, df)

    def test_does_not_modify_original_dataframe(self):
        """No modifica el DataFrame original."""
        df = pd.DataFrame({
            "col": ["1", "2", "3"],
        })
        original = df.copy()
        result = basic_typing(df)
        pd.testing.assert_frame_equal(df, original)

    def test_handles_empty_dataframe(self):
        """Maneja DataFrames vacíos."""
        df = pd.DataFrame()
        result = basic_typing(df)
        assert result.shape == (0, 0)


# ============================================================
# Tests para save_interim() y save_processed()
# ============================================================

class TestSaveMethods:
    """Tests para métodos de guardado"""

    def test_save_interim_creates_file(self, dataset_repo, sample_dataframe):
        """save_interim() crea el archivo correctamente."""
        output_path = dataset_repo.save_interim(sample_dataframe, "test.csv")
        
        assert output_path.exists()
        assert output_path.is_file()
        assert output_path.name == "test.csv"
        assert output_path.parent == dataset_repo.interim_dir

    def test_save_interim_preserves_data(self, dataset_repo, sample_dataframe):
        """save_interim() preserva los datos correctamente."""
        output_path = dataset_repo.save_interim(sample_dataframe, "test.csv")
        
        # Cargar y verificar
        loaded = pd.read_csv(output_path)
        pd.testing.assert_frame_equal(loaded, sample_dataframe)

    def test_save_interim_default_name(self, dataset_repo, sample_dataframe):
        """save_interim() usa el nombre por defecto si no se especifica."""
        output_path = dataset_repo.save_interim(sample_dataframe)
        
        assert output_path.name == "student_interim.csv"

    def test_save_processed_creates_file(self, dataset_repo, sample_dataframe):
        """save_processed() crea el archivo correctamente."""
        output_path = dataset_repo.save_processed(sample_dataframe, "test.csv")
        
        assert output_path.exists()
        assert output_path.is_file()
        assert output_path.name == "test.csv"
        assert output_path.parent == dataset_repo.processed_dir

    def test_save_processed_preserves_data(self, dataset_repo, sample_dataframe):
        """save_processed() preserva los datos correctamente."""
        output_path = dataset_repo.save_processed(sample_dataframe, "test.csv")
        
        # Cargar y verificar
        loaded = pd.read_csv(output_path)
        pd.testing.assert_frame_equal(loaded, sample_dataframe)

    def test_save_processed_default_name(self, dataset_repo, sample_dataframe):
        """save_processed() usa el nombre por defecto si no se especifica."""
        output_path = dataset_repo.save_processed(sample_dataframe)
        
        assert output_path.name == "student_processed.csv"


# ============================================================
# Tests para load_interim()
# ============================================================

class TestLoadInterim:
    """Tests para load_interim()"""

    def test_load_interim_loads_file(self, dataset_repo, sample_dataframe):
        """load_interim() carga el archivo correctamente."""
        # Guardar primero
        dataset_repo.save_interim(sample_dataframe, "test.csv")
        
        # Cargar
        loaded = dataset_repo.load_interim("test.csv")
        
        pd.testing.assert_frame_equal(loaded, sample_dataframe)

    def test_load_interim_raises_on_missing_file(self, dataset_repo):
        """load_interim() lanza error si el archivo no existe."""
        with pytest.raises(FileNotFoundError):
            dataset_repo.load_interim("nonexistent.csv")


# ============================================================
# Tests para load_original_if_exists()
# ============================================================

class TestLoadOriginalIfExists:
    """Tests para load_original_if_exists()"""

    def test_load_original_returns_dataframe_when_exists(self, dataset_repo, sample_dataframe):
        """load_original_if_exists() retorna DataFrame si el archivo existe."""
        # Crear archivo original
        dataset_repo.original_csv.parent.mkdir(parents=True, exist_ok=True)
        sample_dataframe.to_csv(dataset_repo.original_csv, index=False)
        
        # Cargar
        loaded = dataset_repo.load_original_if_exists()
        
        assert loaded is not None
        pd.testing.assert_frame_equal(loaded, sample_dataframe)

    def test_load_original_returns_none_when_not_exists(self, dataset_repo):
        """load_original_if_exists() retorna None si el archivo no existe."""
        # Asegurar que el archivo no existe
        if dataset_repo.original_csv.exists():
            dataset_repo.original_csv.unlink()
        
        result = dataset_repo.load_original_if_exists()
        
        assert result is None


# ============================================================
# Tests para inicialización de DatasetRepository
# ============================================================

class TestDatasetRepositoryInit:
    """Tests para la inicialización de DatasetRepository"""

    def test_creates_directories_on_init(self, temp_data_dir):
        """Crea los directorios necesarios al inicializar."""
        interim_dir = temp_data_dir / "interim"
        processed_dir = temp_data_dir / "processed"
        tables_dir = temp_data_dir / "tables"
        
        repo = DatasetRepository(
            interim_dir=interim_dir,
            processed_dir=processed_dir,
            tables_dir=tables_dir,
        )
        
        assert interim_dir.exists()
        assert processed_dir.exists()
        assert tables_dir.exists()

    def test_uses_custom_paths(self, temp_data_dir):
        """Acepta paths personalizados."""
        custom_modified = temp_data_dir / "custom_modified.csv"
        custom_interim = temp_data_dir / "custom_interim"
        
        repo = DatasetRepository(
            modified_csv=custom_modified,
            interim_dir=custom_interim,
        )
        
        assert repo.modified_csv == custom_modified
        assert repo.interim_dir == custom_interim

