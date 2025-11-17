"""
Pruebas de integración para el módulo mlops.features

Estas pruebas validan el flujo completo de feature engineering,
probando múltiples funciones trabajando juntas en escenarios reales.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil

from mlops.features import (
    FeatureEngineering,
    split_num_cat,
    normalize_categories,
    clean_categoricals,
    minimal_preprocess,
    preprocess_advanced,
    preprocess_for_training,
)


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def sample_dataframe():
    """DataFrame de ejemplo más grande para pruebas de integración."""
    np.random.seed(42)
    n_rows = 100
    return pd.DataFrame({
        "numeric_col_1": np.random.randn(n_rows),
        "numeric_col_2": np.random.randn(n_rows) * 10,
        "categorical_col_1": np.random.choice(["A", "B", "C"], n_rows),
        "categorical_col_2": np.random.choice(["X", "Y"], n_rows),
        "target": np.random.choice(["Low", "Medium", "High"], n_rows),
        "id": range(n_rows),
    })


@pytest.fixture
def dataframe_with_nulls():
    """DataFrame con valores nulos para pruebas de integración."""
    return pd.DataFrame({
        "numeric_col": [1, 2, np.nan, 4, 5, 6, 7, np.nan, 9, 10],
        "categorical_col": ["A", "B", None, "A", "B", "C", None, "A", "B", "C"],
        "another_numeric": [10, 20, 30, np.nan, 50, 60, 70, 80, np.nan, 100],
        "target": ["Low", "Medium", "High", "Low", "Medium", "High", "Low", "Medium", "High", "Low"],
    })


@pytest.fixture
def temp_dir():
    """Directorio temporal para pruebas de integración."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


# ============================================================
# Tests de integración: Flujo completo de feature engineering
# ============================================================

@pytest.mark.integration
class TestFeaturesFullPipeline:
    """Pruebas de integración para el flujo completo de feature engineering."""

    def test_full_preprocessing_pipeline(self, sample_dataframe):
        """Prueba el flujo completo: split → clean → preprocess_for_training."""
        # 1. Clasificar columnas
        num_cols, cat_cols = split_num_cat(sample_dataframe)
        assert len(num_cols) > 0
        assert len(cat_cols) > 0
        
        # Excluir target e id de las listas para preservarlos
        num_cols = [col for col in num_cols if col not in ["target", "id"]]
        cat_cols = [col for col in cat_cols if col not in ["target", "id"]]
        
        # 2. Limpiar categóricas
        cleaned_df = clean_categoricals(sample_dataframe, cat_cols)
        assert cleaned_df.shape == sample_dataframe.shape
        
        # 3. Preprocesar para entrenamiento
        processed_df = preprocess_for_training(cleaned_df, num_cols, cat_cols)
        
        # Verificaciones del resultado completo
        assert processed_df.shape[0] == sample_dataframe.shape[0]  # Mismo número de filas
        assert processed_df.shape[1] > sample_dataframe.shape[1]  # Más columnas (one-hot)
        assert "target" in processed_df.columns  # Columna preservada
        assert "id" in processed_df.columns  # Columna preservada
        
        # Verificar que las columnas numéricas están presentes
        for col in num_cols:
            assert col in processed_df.columns
        
        # Verificar que hay columnas one-hot
        onehot_cols = [col for col in processed_df.columns 
                      if any(col.startswith(f"{cat}_") for cat in cat_cols)]
        assert len(onehot_cols) > 0

    def test_preprocessing_with_imputation_pipeline(self, dataframe_with_nulls):
        """Prueba el flujo completo con imputación: split → minimal_preprocess → preprocess."""
        # 1. Preprocesamiento mínimo (imputa valores nulos)
        processed_df, num_cols, cat_cols = minimal_preprocess(dataframe_with_nulls)
        
        # Verificar que no hay nulos
        assert processed_df.isna().sum().sum() == 0
        
        # Excluir target de las listas para preservarlo
        num_cols = [col for col in num_cols if col != "target"]
        cat_cols = [col for col in cat_cols if col != "target"]
        
        # 2. Preprocesar para entrenamiento
        final_df = preprocess_for_training(processed_df, num_cols, cat_cols)
        
        # Verificaciones
        assert final_df.shape[0] == dataframe_with_nulls.shape[0]
        assert "target" in final_df.columns
        assert final_df.isna().sum().sum() == 0

    def test_advanced_preprocessing_pipeline(self, sample_dataframe):
        """Prueba el flujo completo con preprocesamiento avanzado (escalado + PCA)."""
        # 1. Clasificar columnas
        num_cols, cat_cols = split_num_cat(sample_dataframe)
        
        # 2. Preprocesamiento avanzado (escalado + one-hot + PCA)
        processed_df = preprocess_advanced(
            sample_dataframe, 
            num_cols, 
            cat_cols, 
            n_components=3
        )
        
        # Verificaciones
        assert processed_df.shape[0] == sample_dataframe.shape[0]
        
        # Verificar que hay columnas PCA
        assert "PC1" in processed_df.columns
        assert "PC2" in processed_df.columns
        assert "PC3" in processed_df.columns
        
        # Verificar que las columnas numéricas están escaladas (media ~0)
        for col in num_cols:
            if col in processed_df.columns:
                assert abs(processed_df[col].mean()) < 0.1

    def test_pipeline_preserves_target_column(self, sample_dataframe):
        """Verifica que el pipeline completo preserva la columna target."""
        num_cols, cat_cols = split_num_cat(sample_dataframe)
        
        # Excluir target de las listas
        num_cols = [col for col in num_cols if col != "target"]
        cat_cols = [col for col in cat_cols if col != "target"]
        
        processed_df = preprocess_for_training(sample_dataframe, num_cols, cat_cols)
        
        # Verificar que target está preservado con los mismos valores
        assert "target" in processed_df.columns
        pd.testing.assert_series_equal(
            processed_df["target"], 
            sample_dataframe["target"]
        )


# ============================================================
# Tests de integración: Interacción entre funciones
# ============================================================

@pytest.mark.integration
class TestFeaturesInteraction:
    """Pruebas de integración para la interacción entre funciones de features."""

    def test_split_clean_preprocess_chain(self, sample_dataframe):
        """Prueba la cadena: split_num_cat → clean_categoricals → preprocess_for_training."""
        # Paso 1: Clasificar
        num_cols, cat_cols = split_num_cat(sample_dataframe)
        
        # Paso 2: Limpiar
        cleaned = clean_categoricals(sample_dataframe, cat_cols)
        
        # Verificar que las categóricas fueron limpiadas
        for col in cat_cols:
            if col in cleaned.columns:
                # Verificar que están en minúsculas
                assert all(isinstance(val, str) and val.islower() or pd.isna(val) 
                         for val in cleaned[col].dropna().head(10))
        
        # Paso 3: Preprocesar
        processed = preprocess_for_training(cleaned, num_cols, cat_cols)
        
        # Verificar resultado final
        assert processed.shape[0] == sample_dataframe.shape[0]
        assert len(processed.columns) >= len(sample_dataframe.columns)

    def test_normalize_integration_with_clean(self):
        """Prueba que normalize_categories funciona correctamente dentro de clean_categoricals."""
        df = pd.DataFrame({
            "category": ["  HELLO  ", "  WORLD  ", "  TEST  ", "  FOO  "],
            "numeric": [1, 2, 3, 4],
        })
        
        # clean_categoricals usa normalize_categories internamente
        cleaned = clean_categoricals(df, ["category"])
        
        # Verificar que normalize_categories se aplicó correctamente
        assert cleaned["category"].iloc[0] == "hello"
        assert cleaned["category"].iloc[1] == "world"
        assert cleaned["category"].iloc[2] == "test"
        assert cleaned["category"].iloc[3] == "foo"

    def test_minimal_preprocess_uses_split_num_cat(self):
        """Prueba que minimal_preprocess usa split_num_cat internamente."""
        # Crear DataFrame con más valores únicos para asegurar clasificación correcta
        df = pd.DataFrame({
            "numeric_col": list(range(50)),  # 50 valores únicos
            "categorical_col": (["A", "B", "C"] * 17)[:50],  # 3 valores únicos
            "target": (["Low", "Medium", "High"] * 17)[:50],  # 3 valores únicos
        })
        
        processed, num_cols, cat_cols = minimal_preprocess(df)
        
        # Verificar que las columnas fueron clasificadas correctamente
        assert len(num_cols) > 0
        assert len(cat_cols) > 0
        
        # Verificar que todas las columnas están clasificadas
        all_classified = set(num_cols) | set(cat_cols)
        assert all_classified == set(df.columns)
        
        # Verificar que se preservó la estructura
        assert processed.shape == df.shape


# ============================================================
# Tests de integración: Casos edge del flujo completo
# ============================================================

@pytest.mark.integration
class TestFeaturesEdgeCases:
    """Pruebas de integración para casos edge del flujo completo."""

    def test_pipeline_with_empty_categoricals(self):
        """Prueba el pipeline cuando no hay columnas categóricas."""
        df = pd.DataFrame({
            "numeric_1": [1, 2, 3, 4, 5],
            "numeric_2": [1.1, 2.2, 3.3, 4.4, 5.5],
            "target": ["A", "B", "A", "B", "A"],
        })
        
        num_cols, cat_cols = split_num_cat(df)
        # target puede ser categórica, pero la excluimos
        num_cols = [col for col in num_cols if col != "target"]
        cat_cols = [col for col in cat_cols if col != "target"]
        
        processed = preprocess_for_training(df, num_cols, cat_cols)
        
        # Verificar que funciona sin categóricas
        assert processed.shape[0] == df.shape[0]
        assert "target" in processed.columns

    def test_pipeline_with_empty_numerics(self):
        """Prueba el pipeline cuando no hay columnas numéricas."""
        df = pd.DataFrame({
            "cat_1": ["A", "B", "C", "A", "B"],
            "cat_2": ["X", "Y", "X", "Y", "X"],
            "target": ["Low", "Medium", "High", "Low", "Medium"],
        })
        
        num_cols, cat_cols = split_num_cat(df)
        # Excluir target
        cat_cols = [col for col in cat_cols if col != "target"]
        
        processed = preprocess_for_training(df, num_cols, cat_cols)
        
        # Verificar que funciona sin numéricas
        assert processed.shape[0] == df.shape[0]
        assert "target" in processed.columns
        # Debe tener columnas one-hot
        onehot_cols = [col for col in processed.columns 
                      if any(col.startswith(f"{cat}_") for cat in cat_cols)]
        assert len(onehot_cols) > 0

    def test_pipeline_with_all_nulls_handled(self):
        """Prueba que el pipeline maneja correctamente DataFrames con muchos nulos."""
        df = pd.DataFrame({
            "numeric": [1, np.nan, 3, np.nan, 5, np.nan, 7],
            "categorical": ["A", None, "B", None, "A", None, "B"],
            "target": ["Low", "Medium", "High", "Low", "Medium", "High", "Low"],
        })
        
        # Preprocesamiento mínimo debe imputar todos los nulos
        processed, num_cols, cat_cols = minimal_preprocess(df)
        
        assert processed.isna().sum().sum() == 0
        
        # Luego preprocesar para entrenamiento
        final = preprocess_for_training(processed, num_cols, cat_cols)
        
        assert final.isna().sum().sum() == 0
        assert final.shape[0] == df.shape[0]

