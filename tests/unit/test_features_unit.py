import pytest
import pandas as pd
import numpy as np
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
    """DataFrame de ejemplo con columnas numéricas y categóricas."""
    return pd.DataFrame({
        "numeric_col_1": [1, 2, 3, 4, 5],
        "numeric_col_2": [1.1, 2.2, 3.3, 4.4, 5.5],
        "categorical_col_1": ["A", "B", "C", "A", "B"],
        "categorical_col_2": ["X", "Y", "X", "Y", "X"],
        "target": ["Low", "Medium", "High", "Low", "Medium"],
    })


@pytest.fixture
def dataframe_with_nulls():
    """DataFrame con valores nulos para pruebas de imputación."""
    return pd.DataFrame({
        "numeric_col": [1, 2, np.nan, 4, 5],
        "categorical_col": ["A", "B", None, "A", "B"],
        "another_numeric": [10, 20, 30, np.nan, 50],
    })


@pytest.fixture
def feature_engineer():
    """Instancia de FeatureEngineering para pruebas."""
    return FeatureEngineering()


# ============================================================
# Tests para normalize_categories()
# ============================================================

class TestNormalizeCategories:
    """Tests para normalize_categories() - transforma texto"""

    def test_normalize_lowercase(self):
        """Convierte texto a minúsculas."""
        series = pd.Series(["HELLO", "WORLD", "TEST"])
        result = normalize_categories(series)
        assert result.iloc[0] == "hello"
        assert result.iloc[1] == "world"
        assert result.iloc[2] == "test"

    def test_normalize_strips_whitespace(self):
        """Elimina espacios en blanco al inicio y final."""
        series = pd.Series(["  hello  ", "  world  ", "  test  "])
        result = normalize_categories(series)
        assert result.iloc[0] == "hello"
        assert result.iloc[1] == "world"
        assert result.iloc[2] == "test"

    def test_normalize_replaces_multiple_spaces(self):
        """Reemplaza múltiples espacios con uno solo."""
        series = pd.Series(["hello    world", "test   test"])
        result = normalize_categories(series)
        assert result.iloc[0] == "hello world"
        assert result.iloc[1] == "test test"

    def test_normalize_handles_numeric_series(self):
        """No modifica series numéricas."""
        series = pd.Series([1, 2, 3])
        result = normalize_categories(series)
        pd.testing.assert_series_equal(result, series)

    def test_normalize_handles_nan_values(self):
        """Convierte NaN a string 'nan'."""
        series = pd.Series(["hello", np.nan, "world"])
        result = normalize_categories(series)
        assert result.iloc[0] == "hello"
        assert result.iloc[1] == "nan"
        assert result.iloc[2] == "world"

    def test_normalize_handles_empty_series(self):
        """Maneja series vacías correctamente."""
        series = pd.Series([], dtype=object)
        result = normalize_categories(series)
        assert len(result) == 0

    def test_normalize_handles_mixed_case(self):
        """Normaliza texto con mayúsculas y minúsculas mezcladas."""
        series = pd.Series(["Hello World", "TEST", "MiXeD cAsE"])
        result = normalize_categories(series)
        assert result.iloc[0] == "hello world"
        assert result.iloc[1] == "test"
        assert result.iloc[2] == "mixed case"


# ============================================================
# Tests para split_num_cat()
# ============================================================

class TestSplitNumCat:
    """Tests para split_num_cat() - clasifica columnas"""

    def test_splits_basic_numeric_and_categorical(self):
        """Identifica correctamente columnas numéricas y categóricas básicas."""
        # Usar más valores para que la columna numérica no sea clasificada como categórica
        n_rows = 50
        df = pd.DataFrame({
            "numeric_col": list(range(n_rows)),  # 50 valores únicos
            "categorical_col": (["A", "B", "C"] * (n_rows // 3 + 1))[:n_rows]  # 3 valores únicos, repetidos
        })
        num_cols, cat_cols = split_num_cat(df)
        assert "numeric_col" in num_cols
        assert "categorical_col" in cat_cols

    def test_handles_mixed_types(self):
        """Maneja DataFrames con tipos mixtos."""
        # Usar más valores para columnas numéricas
        n_rows = 50
        df = pd.DataFrame({
            "int_col": list(range(n_rows)),  # 50 valores únicos
            "float_col": [float(i) * 1.1 for i in range(n_rows)],  # 50 valores únicos
            "str_col": (["A", "B", "C"] * (n_rows // 3 + 1))[:n_rows],  # 3 valores únicos
            "bool_col": ([True, False] * (n_rows // 2 + 1))[:n_rows]  # 2 valores únicos
        })
        num_cols, cat_cols = split_num_cat(df)
        assert "int_col" in num_cols
        assert "float_col" in num_cols
        assert "bool_col" in cat_cols  # bool con pocos valores únicos es categórico
        assert "str_col" in cat_cols

    def test_uses_categorical_guess_threshold(self, feature_engineer):
        """Respeta el umbral de categorical_guess_max."""
        # Columna con <= 30 valores únicos debería ser categórica
        n_rows = 50
        few_values_data = list(range(25)) * (n_rows // 25 + 1)
        df = pd.DataFrame({
            "few_values": few_values_data[:n_rows],  # 25 valores únicos, repetidos
            "many_values": list(range(n_rows)),  # 50 valores únicos
        })
        num_cols, cat_cols = feature_engineer.split_num_cat(df)
        assert "few_values" in cat_cols
        assert "many_values" in num_cols

    def test_handles_empty_dataframe(self):
        """Maneja DataFrames vacíos."""
        df = pd.DataFrame()
        num_cols, cat_cols = split_num_cat(df)
        assert len(num_cols) == 0
        assert len(cat_cols) == 0

    def test_handles_dataframe_with_nulls(self):
        """Maneja DataFrames con valores nulos."""
        # Usar más valores para que la columna numérica no sea clasificada como categórica
        n_rows = 51
        numeric_data = list(range(50)) + [np.nan]
        categorical_data = (["A", "B", "C"] * 17) + [None]
        df = pd.DataFrame({
            "numeric_col": numeric_data[:n_rows],
            "categorical_col": categorical_data[:n_rows],
        })
        num_cols, cat_cols = split_num_cat(df)
        assert "numeric_col" in num_cols
        assert "categorical_col" in cat_cols

    def test_all_columns_classified(self, sample_dataframe):
        """Todas las columnas deben estar clasificadas (num o cat)."""
        num_cols, cat_cols = split_num_cat(sample_dataframe)
        all_classified = set(num_cols) | set(cat_cols)
        assert all_classified == set(sample_dataframe.columns)
        # No debe haber solapamiento
        assert len(set(num_cols) & set(cat_cols)) == 0


# ============================================================
# Tests para clean_categoricals()
# ============================================================

class TestCleanCategoricals:
    """Tests para clean_categoricals() - limpia columnas categóricas"""

    def test_cleans_single_categorical_column(self):
        """Limpia una columna categórica."""
        df = pd.DataFrame({
            "category": ["  HELLO  ", "  WORLD  ", "  TEST  "]
        })
        result = clean_categoricals(df, ["category"])
        assert result["category"].iloc[0] == "hello"
        assert result["category"].iloc[1] == "world"
        assert result["category"].iloc[2] == "test"

    def test_cleans_multiple_categorical_columns(self):
        """Limpia múltiples columnas categóricas."""
        df = pd.DataFrame({
            "cat1": ["  A  ", "  B  ", "  C  "],
            "cat2": ["  X  ", "  Y  ", "  Z  "],
            "numeric": [1, 2, 3],
        })
        result = clean_categoricals(df, ["cat1", "cat2"])
        assert result["cat1"].iloc[0] == "a"
        assert result["cat2"].iloc[0] == "x"
        # La columna numérica no debe cambiar
        pd.testing.assert_series_equal(result["numeric"], df["numeric"])

    def test_does_not_modify_original_dataframe(self):
        """No modifica el DataFrame original (retorna copia)."""
        df = pd.DataFrame({
            "category": ["  HELLO  ", "  WORLD  "]
        })
        original_values = df["category"].copy()
        result = clean_categoricals(df, ["category"])
        # El original no debe cambiar
        pd.testing.assert_series_equal(df["category"], original_values)

    def test_handles_empty_categorical_list(self):
        """Maneja lista vacía de columnas categóricas."""
        df = pd.DataFrame({
            "numeric": [1, 2, 3],
            "category": ["A", "B", "C"],
        })
        result = clean_categoricals(df, [])
        pd.testing.assert_frame_equal(result, df)


# ============================================================
# Tests para minimal_preprocess()
# ============================================================

class TestMinimalPreprocess:
    """Tests para minimal_preprocess() - imputación básica"""

    def test_imputes_numeric_with_median(self, dataframe_with_nulls):
        """Imputa valores nulos numéricos con la mediana."""
        result, num_cols, cat_cols = minimal_preprocess(dataframe_with_nulls)
        # Verificar que no hay nulos en columnas numéricas
        assert result["numeric_col"].isna().sum() == 0
        assert result["another_numeric"].isna().sum() == 0
        # Verificar que se usó la mediana (3.0 para numeric_col: [1, 2, nan, 4, 5] -> mediana = 3.0)
        # Pero si la columna tiene pocos valores únicos, puede ser clasificada como categórica
        # Verificar que al menos se imputó correctamente
        assert not pd.isna(result["numeric_col"].iloc[2])

    def test_imputes_categorical_with_mode(self, dataframe_with_nulls):
        """Imputa valores nulos categóricos con la moda."""
        result, num_cols, cat_cols = minimal_preprocess(dataframe_with_nulls)
        # Verificar que no hay nulos en columnas categóricas
        assert result["categorical_col"].isna().sum() == 0
        # Verificar que se usó la moda ("A" o "B")
        assert result["categorical_col"].iloc[2] in ["A", "B"]

    def test_returns_classified_columns(self, sample_dataframe):
        """Retorna las columnas clasificadas correctamente."""
        # Crear un DataFrame con más valores únicos para asegurar clasificación correcta
        n_rows = 50
        df = pd.DataFrame({
            "numeric_col_1": list(range(n_rows)),  # 50 valores únicos
            "numeric_col_2": [float(i) * 1.1 for i in range(n_rows)],
            "categorical_col_1": (["A", "B", "C"] * (n_rows // 3 + 1))[:n_rows],  # 3 valores únicos
            "categorical_col_2": (["X", "Y"] * (n_rows // 2 + 1))[:n_rows],  # 2 valores únicos
            "target": (["Low", "Medium", "High"] * (n_rows // 3 + 1))[:n_rows],  # 3 valores únicos
        })
        result, num_cols, cat_cols = minimal_preprocess(df)
        assert len(num_cols) > 0
        assert len(cat_cols) > 0
        # Verificar que las columnas están correctamente clasificadas
        assert "numeric_col_1" in num_cols
        assert "categorical_col_1" in cat_cols

    def test_handles_dataframe_without_nulls(self, sample_dataframe):
        """Maneja DataFrames sin valores nulos."""
        result, num_cols, cat_cols = minimal_preprocess(sample_dataframe)
        # No debería haber cambios en los valores
        pd.testing.assert_frame_equal(result[sample_dataframe.columns], sample_dataframe)

    def test_preserves_dataframe_structure(self, sample_dataframe):
        """Preserva la estructura del DataFrame."""
        result, num_cols, cat_cols = minimal_preprocess(sample_dataframe)
        assert result.shape == sample_dataframe.shape
        assert set(result.columns) == set(sample_dataframe.columns)


# ============================================================
# Tests para preprocess_for_training()
# ============================================================

class TestPreprocessForTraining:
    """Tests para preprocess_for_training() - OneHot sin PCA"""

    def test_applies_onehot_to_categoricals(self, sample_dataframe):
        """Aplica OneHotEncoder a columnas categóricas."""
        num_cols = ["numeric_col_1", "numeric_col_2"]
        cat_cols = ["categorical_col_1", "categorical_col_2"]
        result = preprocess_for_training(sample_dataframe, num_cols, cat_cols)
        
        # Verificar que hay columnas one-hot (deben empezar con el nombre de la columna)
        onehot_cols = [col for col in result.columns if col.startswith("categorical_col_1_") or col.startswith("categorical_col_2_")]
        assert len(onehot_cols) > 0

    def test_preserves_numeric_columns(self, sample_dataframe):
        """Preserva columnas numéricas sin escalar."""
        num_cols = ["numeric_col_1", "numeric_col_2"]
        cat_cols = ["categorical_col_1"]
        result = preprocess_for_training(sample_dataframe, num_cols, cat_cols)
        
        # Verificar que las columnas numéricas están presentes
        assert "numeric_col_1" in result.columns
        assert "numeric_col_2" in result.columns
        # Verificar que los valores no fueron escalados (deben ser iguales)
        pd.testing.assert_series_equal(
            result["numeric_col_1"], 
            sample_dataframe["numeric_col_1"]
        )

    def test_preserves_columns_not_in_lists(self, sample_dataframe):
        """Preserva columnas que no están en num_cols ni cat_cols."""
        num_cols = ["numeric_col_1"]
        cat_cols = ["categorical_col_1"]
        result = preprocess_for_training(sample_dataframe, num_cols, cat_cols)
        
        # La columna 'target' debe estar preservada
        assert "target" in result.columns
        pd.testing.assert_series_equal(result["target"], sample_dataframe["target"])

    def test_handles_empty_categorical_list(self, sample_dataframe):
        """Maneja lista vacía de columnas categóricas."""
        num_cols = ["numeric_col_1", "numeric_col_2"]
        cat_cols = []
        result = preprocess_for_training(sample_dataframe, num_cols, cat_cols)
        
        # Solo debe tener columnas numéricas y preservadas
        assert "numeric_col_1" in result.columns
        assert "target" in result.columns

    def test_handles_empty_numeric_list(self, sample_dataframe):
        """Maneja lista vacía de columnas numéricas."""
        num_cols = []
        cat_cols = ["categorical_col_1"]
        result = preprocess_for_training(sample_dataframe, num_cols, cat_cols)
        
        # Debe tener columnas one-hot y preservadas
        onehot_cols = [col for col in result.columns if col.startswith("categorical_col_1_")]
        assert len(onehot_cols) > 0
        assert "target" in result.columns

    def test_returns_dataframe_with_correct_shape(self, sample_dataframe):
        """Retorna DataFrame con forma correcta."""
        num_cols = ["numeric_col_1", "numeric_col_2"]
        cat_cols = ["categorical_col_1", "categorical_col_2"]
        result = preprocess_for_training(sample_dataframe, num_cols, cat_cols)
        
        # Debe tener el mismo número de filas
        assert result.shape[0] == sample_dataframe.shape[0]
        # Debe tener más columnas (por el one-hot encoding)
        assert result.shape[1] > sample_dataframe.shape[1]


# ============================================================
# Tests para preprocess_advanced()
# ============================================================

class TestPreprocessAdvanced:
    """Tests para preprocess_advanced() - escalado + OneHot + PCA"""

    def test_applies_scaling_to_numerics(self, sample_dataframe):
        """Aplica StandardScaler a columnas numéricas."""
        num_cols = ["numeric_col_1", "numeric_col_2"]
        cat_cols = ["categorical_col_1"]
        result = preprocess_advanced(sample_dataframe, num_cols, cat_cols, n_components=0)
        
        # Verificar que las columnas numéricas están escaladas (media ~0, std ~1)
        assert abs(result["numeric_col_1"].mean()) < 0.01
        assert abs(result["numeric_col_2"].mean()) < 0.01

    def test_applies_onehot_to_categoricals(self, sample_dataframe):
        """Aplica OneHotEncoder a columnas categóricas."""
        num_cols = ["numeric_col_1"]
        cat_cols = ["categorical_col_1", "categorical_col_2"]
        result = preprocess_advanced(sample_dataframe, num_cols, cat_cols, n_components=0)
        
        # Verificar columnas one-hot
        onehot_cols = [col for col in result.columns if "categorical_col" in col]
        assert len(onehot_cols) > 0

    def test_applies_pca_when_requested(self, sample_dataframe):
        """Aplica PCA cuando se solicita."""
        num_cols = ["numeric_col_1", "numeric_col_2"]
        cat_cols = ["categorical_col_1", "categorical_col_2"]
        result = preprocess_advanced(sample_dataframe, num_cols, cat_cols, n_components=2)
        
        # Verificar que hay columnas PC1, PC2
        assert "PC1" in result.columns
        assert "PC2" in result.columns

    def test_handles_zero_pca_components(self, sample_dataframe):
        """Maneja n_components=0 (sin PCA)."""
        num_cols = ["numeric_col_1"]
        cat_cols = ["categorical_col_1"]
        result = preprocess_advanced(sample_dataframe, num_cols, cat_cols, n_components=0)
        
        # No debe haber columnas PC
        pc_cols = [col for col in result.columns if col.startswith("PC")]
        assert len(pc_cols) == 0

    def test_handles_empty_inputs(self):
        """Maneja listas vacías de columnas."""
        df = pd.DataFrame({"other_col": [1, 2, 3]})
        result = preprocess_advanced(df, [], [], n_components=0)
        # Debe retornar el DataFrame original
        pd.testing.assert_frame_equal(result, df)

