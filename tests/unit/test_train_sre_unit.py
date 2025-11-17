"""
Pruebas unitarias para el módulo train/train_model_sre.py

Estas pruebas validan los métodos individuales de StudentPerformanceTrainer.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
from unittest.mock import patch, MagicMock

from train.train_model_sre import (
    StudentPerformanceTrainer,
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
def sample_dataframe():
    """DataFrame de ejemplo con estructura similar al dataset real."""
    return pd.DataFrame({
        "Gender": ["M", "F", "M", "F", "M"],
        "Caste": ["A", "B", "A", "B", "A"],
        "coaching": ["Yes", "No", "Yes", "No", "Yes"],
        "time": ["Morning", "Evening", "Morning", "Evening", "Morning"],
        "Class_ten_education": ["CBSE", "State", "CBSE", "State", "CBSE"],
        "twelve_education": ["CBSE", "State", "CBSE", "State", "CBSE"],
        "medium": ["English", "Hindi", "English", "Hindi", "English"],
        "Class_ X_Percentage": ["A", "B", "A", "B", "A"],
        "Class_XII_Percentage": ["A", "B", "A", "B", "A"],
        "Father_occupation": ["Engineer", "Teacher", "Engineer", "Teacher", "Engineer"],
        "Mother_occupation": ["Doctor", "Nurse", "Doctor", "Nurse", "Doctor"],
        "Performance": ["good", "average", "good", "average", "excellent"],
        "numeric_col": [1, 2, 3, 4, 5],
    })


@pytest.fixture
def trainer(temp_data_dir):
    """Instancia de StudentPerformanceTrainer con paths temporales."""
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
    
    return StudentPerformanceTrainer(
        seed=42,
        data_path=str(csv_path),
        models_dir=str(temp_data_dir / "models"),
        reports_dir=str(temp_data_dir / "reports"),
    )


# ============================================================
# Tests para inicialización
# ============================================================

class TestStudentPerformanceTrainerInit:
    """Tests para la inicialización de StudentPerformanceTrainer"""

    def test_init_defaults(self):
        """Test que la inicialización con valores por defecto funciona."""
        trainer = StudentPerformanceTrainer()
        assert trainer.seed == SEED
        assert trainer.data_path == Path("data/interim/student_interim_clean.csv")
        assert trainer.model_random_state == 888

    def test_init_custom_values(self):
        """Test que la inicialización con valores personalizados funciona."""
        trainer = StudentPerformanceTrainer(
            seed=100,
            data_path="custom/path.csv",
            model_random_state=999,
        )
        assert trainer.seed == 100
        assert trainer.data_path == Path("custom/path.csv")
        assert trainer.model_random_state == 999


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

    def test_load_data_removes_mixed_type_col(self, trainer, temp_data_dir):
        """Test que load_data elimina mixed_type_col si existe."""
        # Crear CSV con mixed_type_col
        df = pd.DataFrame({
            "Gender": ["M", "F"],
            "Performance": ["good", "average"],
            "mixed_type_col": [1, 2],
        })
        csv_path = temp_data_dir / "interim" / "student_interim_clean.csv"
        df.to_csv(csv_path, index=False)
        
        trainer.data_path = csv_path
        X, y = trainer.load_data()
        
        assert "mixed_type_col" not in X.columns

    def test_load_data_preserves_other_columns(self, trainer):
        """Test que load_data preserva otras columnas."""
        X, y = trainer.load_data()
        
        # Debe tener las columnas categóricas y numéricas
        assert len(X.columns) > 0


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
        assert len(trainer.class_names) == 3  # good, average, excellent

    def test_encode_target_preserves_classes(self, trainer):
        """Test que encode_target preserva las clases originales."""
        y = pd.Series(["good", "average", "excellent"])
        y_enc = trainer.encode_target(y)
        
        # Verificar que podemos decodificar
        decoded = trainer.label_encoder.inverse_transform(y_enc)
        assert list(decoded) == list(y)


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
            "coaching": ["Yes", "No"],
            "other_col": [1, 2],  # No categórica
        })
        
        cat_cols = trainer.get_categorical_columns(X)
        
        assert "Gender" in cat_cols
        assert "Caste" in cat_cols
        assert "coaching" in cat_cols
        assert "other_col" not in cat_cols

    def test_get_categorical_columns_filters_missing(self, trainer):
        """Test que solo incluye columnas que existen en el DataFrame."""
        X = pd.DataFrame({
            "Gender": ["M", "F"],
            "other_col": [1, 2],
        })
        
        cat_cols = trainer.get_categorical_columns(X)
        
        assert "Gender" in cat_cols
        assert "Caste" not in cat_cols  # No está en X


# ============================================================
# Tests para split_data
# ============================================================

class TestSplitData:
    """Tests para el método split_data"""

    def test_split_data_success(self, trainer):
        """Test que split_data divide correctamente los datos."""
        X = pd.DataFrame({"col1": range(100), "col2": range(100, 200)})
        y = np.array([0, 1] * 50)
        
        X_train, X_test, y_train, y_test = trainer.split_data(X, y)
        
        assert len(X_train) + len(X_test) == len(X)
        assert len(y_train) + len(y_test) == len(y)
        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)
        # Debe ser aproximadamente 80/20
        assert abs(len(X_train) / len(X) - 0.8) < 0.1

    def test_split_data_stratified(self, trainer):
        """Test que split_data mantiene la estratificación."""
        X = pd.DataFrame({"col1": range(100)})
        y = np.array([0] * 50 + [1] * 50)  # 50% de cada clase
        
        X_train, X_test, y_train, y_test = trainer.split_data(X, y)
        
        # Verificar que ambas clases están en train y test
        assert 0 in y_train and 1 in y_train
        assert 0 in y_test and 1 in y_test


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
        assert len(preprocessor.transformers) == 1
        assert preprocessor.transformers[0][0] == "ohe"

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

    def test_create_pipeline_has_correct_classifier(self, trainer):
        """Test que el pipeline tiene RandomForestClassifier."""
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.ensemble import RandomForestClassifier
        
        preprocessor = ColumnTransformer([
            ("ohe", OneHotEncoder(handle_unknown="ignore"), ["Gender"])
        ])
        
        pipeline = trainer.create_pipeline(preprocessor)
        
        classifier = pipeline.named_steps["classifier"]
        assert isinstance(classifier, RandomForestClassifier)
        assert classifier.random_state == trainer.model_random_state


# ============================================================
# Tests para evaluate
# ============================================================

class TestEvaluate:
    """Tests para el método evaluate"""

    def test_evaluate_success(self, trainer):
        """Test que evaluate calcula métricas correctamente."""
        from sklearn.pipeline import Pipeline
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.dummy import DummyClassifier
        
        # Crear un pipeline simple para testing
        preprocessor = ColumnTransformer([
            ("ohe", OneHotEncoder(handle_unknown="ignore"), ["Gender"])
        ])
        
        # Usar DummyClassifier para pruebas rápidas
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", DummyClassifier(strategy="most_frequent"))
        ])
        
        # Datos de prueba
        X_test = pd.DataFrame({"Gender": ["M", "F", "M"]})
        y_test = np.array([0, 1, 0])
        
        # Entrenar pipeline
        X_train = pd.DataFrame({"Gender": ["M", "F"]})
        y_train = np.array([0, 1])
        pipeline.fit(X_train, y_train)
        
        # Configurar class_names
        trainer.class_names = np.array(["class0", "class1"])
        
        # Evaluar
        y_pred, metrics, classif_rep = trainer.evaluate(pipeline, X_test, y_test)
        
        assert isinstance(y_pred, np.ndarray)
        assert isinstance(metrics, dict)
        assert "accuracy" in metrics
        assert isinstance(classif_rep, str)
        assert len(y_pred) == len(y_test)


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

    @patch("train.train_model_sre.mlflow")
    def test_setup_mlflow(self, mock_mlflow, trainer):
        """Test que _setup_mlflow configura MLflow correctamente."""
        trainer._setup_mlflow()
        
        mock_mlflow.set_tracking_uri.assert_called_once()
        mock_mlflow.set_experiment.assert_called_once_with(trainer.mlflow_experiment)

