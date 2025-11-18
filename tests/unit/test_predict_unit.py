"""
Pruebas unitarias para el módulo predict.py

Estas pruebas validan los métodos individuales de ModelPredictor.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
import joblib
from unittest.mock import patch, MagicMock, Mock
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

from predict import ModelPredictor, ModelPredictorConfig


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
def sample_clean_dataframe():
    """DataFrame de ejemplo con datos limpios."""
    return pd.DataFrame({
        "Gender": ["Male", "Female", "Male"],
        "Caste": ["General", "OBC", "SC"],
        "coaching": ["yes", "no", "yes"],
        "time": ["1-2 hours", "2-3 hours", "3-4 hours"],
        "Class_ten_education": ["CBSE", "State Board", "ICSE"],
        "twelve_education": ["CBSE", "State Board", "ICSE"],
        "medium": ["English", "Hindi", "English"],
        "Class_ X_Percentage": ["good", "vg", "excellent"],
        "Class_XII_Percentage": ["good", "vg", "excellent"],
        "Father_occupation": ["Engineer", "Teacher", "Doctor"],
        "Mother_occupation": ["Doctor", "Nurse", "Engineer"],
        "Performance": ["good", "average", "excellent"],
    })


@pytest.fixture
def mock_pipeline():
    """Pipeline mock para pruebas."""
    preprocessor = ColumnTransformer(
        transformers=[("ohe", OneHotEncoder(handle_unknown="ignore"), ["Gender", "Caste"])],
        remainder="drop"
    )
    classifier = RandomForestClassifier(n_estimators=10, random_state=42)
    classifier.classes_ = np.array([0, 1, 2, 3])  # Índices numéricos
    
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])
    return pipeline


@pytest.fixture
def mock_model_file(temp_data_dir, mock_pipeline):
    """Crea un archivo de modelo mock."""
    model_path = temp_data_dir / "test_model.joblib"
    joblib.dump(mock_pipeline, model_path)
    return model_path


# ============================================================
# Tests de ModelPredictorConfig
# ============================================================

class TestModelPredictorConfig:
    """Tests para ModelPredictorConfig."""
    
    def test_default_config(self):
        """Test que la configuración por defecto es correcta."""
        config = ModelPredictorConfig()
        assert config.model_path == "models/best_gridsearch_amplio.joblib"
        assert config.tracking_uri == "file:./mlruns"
        assert config.clean_data_path == "data/interim/student_interim_clean.csv"
        assert config.target_column == "Performance"
    
    def test_custom_config(self):
        """Test que se puede crear una configuración personalizada."""
        config = ModelPredictorConfig(
            model_path="custom/path.joblib",
            clean_data_path="custom/data.csv"
        )
        assert config.model_path == "custom/path.joblib"
        assert config.clean_data_path == "custom/data.csv"


# ============================================================
# Tests de ModelPredictor - Inicialización
# ============================================================

class TestModelPredictorInit:
    """Tests para la inicialización de ModelPredictor."""
    
    def test_init_with_default_config(self):
        """Test inicialización con configuración por defecto."""
        predictor = ModelPredictor()
        assert predictor.config.model_path == "models/best_gridsearch_amplio.joblib"
        assert predictor.client is None  # Lazy initialization
    
    def test_init_with_custom_config(self):
        """Test inicialización con configuración personalizada."""
        config = ModelPredictorConfig(model_path="custom/path.joblib")
        predictor = ModelPredictor(config)
        assert predictor.config.model_path == "custom/path.joblib"


# ============================================================
# Tests de ModelPredictor - _load_model_from_dvc
# ============================================================

class TestLoadModelFromDVC:
    """Tests para _load_model_from_dvc."""
    
    def test_load_model_from_dvc_success(self, temp_data_dir, mock_model_file):
        """Test carga exitosa de modelo desde DVC."""
        config = ModelPredictorConfig(model_path=str(mock_model_file))
        predictor = ModelPredictor(config)
        
        model = predictor._load_model_from_dvc()
        
        assert model is not None
        assert isinstance(model, Pipeline)
    
    def test_load_model_from_dvc_not_found(self, temp_data_dir):
        """Test cuando el modelo no existe en DVC."""
        model_path = temp_data_dir / "nonexistent.joblib"
        config = ModelPredictorConfig(model_path=str(model_path))
        predictor = ModelPredictor(config)
        
        model = predictor._load_model_from_dvc()
        
        assert model is None
    
    def test_load_model_from_dvc_invalid_file(self, temp_data_dir):
        """Test cuando el archivo existe pero no es un modelo válido."""
        invalid_file = temp_data_dir / "invalid.joblib"
        invalid_file.write_text("not a valid model")
        
        config = ModelPredictorConfig(model_path=str(invalid_file))
        predictor = ModelPredictor(config)
        
        model = predictor._load_model_from_dvc()
        
        assert model is None


# ============================================================
# Tests de ModelPredictor - _load_data
# ============================================================

class TestLoadData:
    """Tests para _load_data."""
    
    def test_load_data_removes_none_performance(self, temp_data_dir, sample_clean_dataframe):
        """Test que elimina registros con Performance='none'."""
        # Agregar registros con 'none'
        df_with_none = sample_clean_dataframe.copy()
        df_with_none.loc[len(df_with_none)] = {
            "Gender": "Male", "Caste": "General", "coaching": "yes",
            "time": "1-2 hours", "Class_ten_education": "CBSE",
            "twelve_education": "CBSE", "medium": "English",
            "Class_ X_Percentage": "good", "Class_XII_Percentage": "good",
            "Father_occupation": "Engineer", "Mother_occupation": "Doctor",
            "Performance": "none"
        }
        
        csv_path = temp_data_dir / "test_data.csv"
        df_with_none.to_csv(csv_path, index=False)
        
        config = ModelPredictorConfig(clean_data_path=str(csv_path))
        predictor = ModelPredictor(config)
        
        X, y = predictor._load_data()
        
        assert "none" not in y.values
        assert len(X) == len(sample_clean_dataframe)  # El registro 'none' fue eliminado
    
    def test_load_data_removes_invalid_features(self, temp_data_dir):
        """Test que elimina registros con valores inválidos en features categóricas."""
        df = pd.DataFrame({
            "Gender": ["Male", "none", "Female"],
            "Caste": ["General", "OBC", "SC"],
            "coaching": ["yes", "no", "yes"],
            "Performance": ["good", "average", "excellent"],
        })
        
        csv_path = temp_data_dir / "test_data.csv"
        df.to_csv(csv_path, index=False)
        
        config = ModelPredictorConfig(clean_data_path=str(csv_path))
        predictor = ModelPredictor(config)
        
        X, y = predictor._load_data()
        
        # El registro con 'none' en Gender debe ser eliminado
        assert len(X) == 2
        assert "none" not in X["Gender"].values
    
    def test_load_data_returns_correct_shape(self, temp_data_dir, sample_clean_dataframe):
        """Test que _load_data retorna X e y con la forma correcta."""
        csv_path = temp_data_dir / "test_data.csv"
        sample_clean_dataframe.to_csv(csv_path, index=False)
        
        config = ModelPredictorConfig(clean_data_path=str(csv_path))
        predictor = ModelPredictor(config)
        
        X, y = predictor._load_data()
        
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert len(X) == len(y)
        assert "Performance" not in X.columns
        assert "Performance" in y.name or y.name is None


# ============================================================
# Tests de ModelPredictor - _load_model
# ============================================================

class TestLoadModel:
    """Tests para _load_model."""
    
    def test_load_model_from_dvc_first(self, temp_data_dir, mock_model_file):
        """Test que intenta cargar desde DVC primero."""
        config = ModelPredictorConfig(model_path=str(mock_model_file))
        predictor = ModelPredictor(config)
        
        model = predictor._load_model()
        
        assert model is not None
        assert isinstance(model, Pipeline)
    
    @patch("predict.ModelPredictor._load_model_from_mlflow")
    def test_load_model_fallback_to_mlflow(self, mock_mlflow, temp_data_dir):
        """Test que hace fallback a MLflow si DVC falla."""
        model_path = temp_data_dir / "nonexistent.joblib"
        config = ModelPredictorConfig(model_path=str(model_path))
        predictor = ModelPredictor(config)
        
        mock_pipeline = MagicMock()
        mock_mlflow.return_value = mock_pipeline
        
        model = predictor._load_model()
        
        assert model is not None
        mock_mlflow.assert_called_once()
    
    def test_load_model_raises_when_both_fail(self, temp_data_dir):
        """Test que lanza excepción cuando ambos métodos fallan."""
        model_path = temp_data_dir / "nonexistent.joblib"
        config = ModelPredictorConfig(model_path=str(model_path))
        predictor = ModelPredictor(config)
        
        with patch("predict.ModelPredictor._load_model_from_mlflow", return_value=None):
            with pytest.raises(FileNotFoundError):
                predictor._load_model()


# ============================================================
# Tests de ModelPredictor - evaluate
# ============================================================

class TestEvaluate:
    """Tests para evaluate."""
    
    def test_evaluate_with_valid_model(self, temp_data_dir, mock_model_file, sample_clean_dataframe):
        """Test evaluación con modelo válido."""
        csv_path = temp_data_dir / "test_data.csv"
        sample_clean_dataframe.to_csv(csv_path, index=False)
        
        # Crear y entrenar un modelo con las mismas clases que los datos
        from sklearn.preprocessing import LabelEncoder
        
        # Obtener clases únicas de los datos
        unique_classes = sorted(sample_clean_dataframe["Performance"].unique())
        le = LabelEncoder()
        le.fit(unique_classes)
        
        # Crear pipeline con el número correcto de clases
        preprocessor = ColumnTransformer(
            transformers=[("ohe", OneHotEncoder(handle_unknown="ignore"), ["Gender", "Caste"])],
            remainder="drop"
        )
        classifier = RandomForestClassifier(n_estimators=10, random_state=42)
        classifier.classes_ = np.array(range(len(unique_classes)))  # Índices numéricos
        
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", classifier)
        ])
        
        # Entrenar con datos dummy
        X_train = pd.DataFrame({
            "Gender": ["Male", "Female", "Male"],
            "Caste": ["General", "OBC", "SC"]
        })
        y_train_encoded = le.transform(unique_classes[:3])  # Usar las primeras 3 clases
        pipeline.fit(X_train, y_train_encoded)
        
        # Guardar el modelo entrenado
        joblib.dump(pipeline, mock_model_file)
        
        config = ModelPredictorConfig(
            model_path=str(mock_model_file),
            clean_data_path=str(csv_path)
        )
        predictor = ModelPredictor(config)
        
        accuracy = predictor.evaluate()
        
        assert isinstance(accuracy, float)
        assert 0.0 <= accuracy <= 1.0
    
    def test_evaluate_raises_when_no_classifier(self, temp_data_dir, mock_model_file):
        """Test que lanza excepción cuando no se encuentra el clasificador."""
        csv_path = temp_data_dir / "test_data.csv"
        df = pd.DataFrame({
            "Gender": ["Male"],
            "Performance": ["good"]
        })
        df.to_csv(csv_path, index=False)
        
        # Crear un pipeline sin clasificador (solo preprocessor)
        preprocessor = ColumnTransformer(
            transformers=[("ohe", OneHotEncoder(handle_unknown="ignore"), ["Gender"])],
            remainder="drop"
        )
        invalid_pipeline = Pipeline([("preprocessor", preprocessor)])
        joblib.dump(invalid_pipeline, mock_model_file)
        
        config = ModelPredictorConfig(
            model_path=str(mock_model_file),
            clean_data_path=str(csv_path)
        )
        predictor = ModelPredictor(config)
        
        with pytest.raises(ValueError, match="No se pudo encontrar el clasificador"):
            predictor.evaluate()
    
    def test_evaluate_handles_class_mismatch(self, temp_data_dir, mock_model_file):
        """Test que maneja correctamente cuando el número de clases no coincide."""
        csv_path = temp_data_dir / "test_data.csv"
        df = pd.DataFrame({
            "Gender": ["Male"],
            "Performance": ["good"]
        })
        df.to_csv(csv_path, index=False)
        
        # Crear modelo con diferente número de clases
        classifier = RandomForestClassifier(n_estimators=10, random_state=42)
        classifier.classes_ = np.array([0, 1, 2, 3, 4])  # 5 clases
        
        pipeline = Pipeline([
            ("preprocessor", ColumnTransformer(
                transformers=[("ohe", OneHotEncoder(handle_unknown="ignore"), ["Gender"])],
                remainder="drop"
            )),
            ("classifier", classifier)
        ])
        
        joblib.dump(pipeline, mock_model_file)
        
        config = ModelPredictorConfig(
            model_path=str(mock_model_file),
            clean_data_path=str(csv_path)
        )
        predictor = ModelPredictor(config)
        
        with pytest.raises(ValueError, match="Número de clases no coincide"):
            predictor.evaluate()

