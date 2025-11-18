"""
Pruebas unitarias para el módulo app_api.py

Estas pruebas validan los componentes individuales de la API FastAPI.
"""

import pytest
from pathlib import Path
import tempfile
import shutil
import joblib
from unittest.mock import patch, MagicMock, Mock
from fastapi.testclient import TestClient
from fastapi import status
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd

# Importar después de configurar el path
import sys
from pathlib import Path as PathLib
project_root = PathLib(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from app_api import (
    app,
    StudentFeatures,
    PredictionRequest,
    PredictionResponse,
    BatchPredictionResponse,
    _try_load_local_joblib,
    _resolve_latest_mlflow_model_uri,
    _try_load_mlflow_model,
    MODEL_PATH,
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
def mock_pipeline():
    """Pipeline mock para pruebas."""
    preprocessor = ColumnTransformer(
        transformers=[("ohe", OneHotEncoder(handle_unknown="ignore"), ["Gender", "Caste"])],
        remainder="drop"
    )
    classifier = RandomForestClassifier(n_estimators=10, random_state=42)
    classifier.classes_ = np.array([0, 1, 2, 3])
    
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])
    
    # Entrenar con datos dummy
    X_train = pd.DataFrame({
        "Gender": ["Male", "Female"],
        "Caste": ["General", "OBC"]
    })
    y_train = np.array([0, 1])
    pipeline.fit(X_train, y_train)
    
    return pipeline


@pytest.fixture
def client():
    """Cliente de prueba para FastAPI."""
    return TestClient(app)


@pytest.fixture
def sample_student_features():
    """Datos de ejemplo para StudentFeatures."""
    return {
        "Gender": "Male",
        "Caste": "General",
        "coaching": "yes",
        "time": "3-4 hours",
        "Class_ten_education": "CBSE",
        "twelve_education": "CBSE",
        "medium": "English",
        "Class_X_Percentage": "vg",
        "Class_XII_Percentage": "vg",
        "Father_occupation": "Business",
        "Mother_occupation": "Housewife"
    }


# ============================================================
# Tests de Modelos Pydantic
# ============================================================

class TestStudentFeatures:
    """Tests para el modelo StudentFeatures."""
    
    def test_valid_student_features(self, sample_student_features):
        """Test que acepta datos válidos."""
        student = StudentFeatures(**sample_student_features)
        assert student.Gender == "Male"
        assert student.Caste == "General"
    
    def test_invalid_gender(self, sample_student_features):
        """Test que rechaza Gender inválido."""
        sample_student_features["Gender"] = "Invalid"
        with pytest.raises(Exception):  # ValidationError de Pydantic
            StudentFeatures(**sample_student_features)
    
    def test_invalid_caste(self, sample_student_features):
        """Test que rechaza Caste inválido."""
        sample_student_features["Caste"] = "Invalid"
        with pytest.raises(Exception):
            StudentFeatures(**sample_student_features)
    
    def test_empty_occupation(self, sample_student_features):
        """Test que rechaza ocupación vacía."""
        sample_student_features["Father_occupation"] = ""
        with pytest.raises(Exception):
            StudentFeatures(**sample_student_features)
    
    def test_class_x_percentage_alias(self, sample_student_features):
        """Test que acepta el alias 'Class_ X_Percentage'."""
        # Usar el alias con espacio
        data = sample_student_features.copy()
        data["Class_ X_Percentage"] = "vg"
        del data["Class_X_Percentage"]
        
        student = StudentFeatures(**data)
        assert student.Class_X_Percentage == "vg"


class TestPredictionRequest:
    """Tests para el modelo PredictionRequest."""
    
    def test_valid_prediction_request(self, sample_student_features):
        """Test que acepta una solicitud válida."""
        request = PredictionRequest(students=[StudentFeatures(**sample_student_features)])
        assert len(request.students) == 1
    
    def test_empty_students_list(self):
        """Test que rechaza lista vacía."""
        with pytest.raises(Exception):
            PredictionRequest(students=[])
    
    def test_multiple_students(self, sample_student_features):
        """Test que acepta múltiples estudiantes."""
        students = [StudentFeatures(**sample_student_features) for _ in range(3)]
        request = PredictionRequest(students=students)
        assert len(request.students) == 3
    
    def test_max_students_limit(self, sample_student_features):
        """Test que respeta el límite máximo de estudiantes."""
        # Crear 101 estudiantes (más del límite de 100)
        students = [StudentFeatures(**sample_student_features) for _ in range(101)]
        with pytest.raises(Exception):
            PredictionRequest(students=students)


# ============================================================
# Tests de Funciones Helper
# ============================================================

class TestTryLoadLocalJoblib:
    """Tests para _try_load_local_joblib."""
    
    def test_load_success(self, temp_data_dir, mock_pipeline):
        """Test carga exitosa de modelo local."""
        model_path = temp_data_dir / "test_model.joblib"
        joblib.dump(mock_pipeline, model_path)
        
        model = _try_load_local_joblib(model_path)
        
        assert model is not None
        assert isinstance(model, Pipeline)
    
    def test_load_not_found(self, temp_data_dir):
        """Test cuando el archivo no existe."""
        model_path = temp_data_dir / "nonexistent.joblib"
        
        model = _try_load_local_joblib(model_path)
        
        assert model is None
    
    def test_load_invalid_file(self, temp_data_dir):
        """Test cuando el archivo no es un modelo válido."""
        invalid_file = temp_data_dir / "invalid.joblib"
        invalid_file.write_text("not a valid model")
        
        model = _try_load_local_joblib(invalid_file)
        
        assert model is None


class TestResolveLatestMLflowModelURI:
    """Tests para _resolve_latest_mlflow_model_uri."""
    
    @patch("app_api.get_mlflow_client")
    @patch("app_api.mlflow.set_tracking_uri")
    def test_resolve_success(self, mock_set_tracking, mock_get_client):
        """Test resolución exitosa de URI de modelo."""
        # Mock del cliente MLflow
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        
        # Mock del experimento
        mock_experiment = MagicMock()
        mock_experiment.experiment_id = "exp_123"
        mock_client.get_experiment_by_name.return_value = mock_experiment
        
        # Mock del run
        mock_run = MagicMock()
        mock_run.info.run_id = "run_456"
        mock_client.search_runs.return_value = [mock_run]
        
        # Mock de artifacts - necesitamos que list_artifacts retorne objetos con .path y .name
        from pathlib import Path as PathLib
        mock_artifact = MagicMock()
        mock_artifact.path = "best_model"
        # Crear un objeto Path mock para el name
        mock_path = MagicMock()
        mock_path.name = "MLmodel"
        # Hacer que Path(artifact_item.path).name funcione
        with patch("app_api.Path", return_value=mock_path):
            mock_client.list_artifacts.return_value = [mock_artifact]
            
            uri = _resolve_latest_mlflow_model_uri()
            
            assert uri is not None
            assert "runs:/run_456/best_model" in uri
    
    @patch("app_api.get_mlflow_client")
    @patch("app_api.mlflow.set_tracking_uri")
    def test_resolve_no_experiment(self, mock_set_tracking, mock_get_client):
        """Test cuando no existe el experimento."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.get_experiment_by_name.return_value = None
        
        uri = _resolve_latest_mlflow_model_uri()
        
        assert uri is None
    
    @patch("app_api.get_mlflow_client")
    @patch("app_api.mlflow.set_tracking_uri")
    def test_resolve_no_runs(self, mock_set_tracking, mock_get_client):
        """Test cuando no hay runs en el experimento."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        
        mock_experiment = MagicMock()
        mock_experiment.experiment_id = "exp_123"
        mock_client.get_experiment_by_name.return_value = mock_experiment
        mock_client.search_runs.return_value = []
        
        uri = _resolve_latest_mlflow_model_uri()
        
        assert uri is None


# ============================================================
# Tests de Endpoints
# ============================================================

class TestRootEndpoint:
    """Tests para el endpoint raíz."""
    
    def test_root_endpoint(self, client):
        """Test que el endpoint raíz responde correctamente."""
        response = client.get("/")
        assert response.status_code == status.HTTP_200_OK
        assert "message" in response.json()


class TestHealthCheckEndpoint:
    """Tests para el endpoint de health check."""
    
    @patch("app_api.model", None)
    def test_health_check_no_model(self, client):
        """Test health check cuando el modelo no está cargado."""
        response = client.get("/health")
        assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    
    @patch("app_api.model")
    def test_health_check_with_model(self, mock_model, client):
        """Test health check cuando el modelo está cargado."""
        mock_model.__bool__ = lambda x: True  # Para que model is not None
        
        response = client.get("/health")
        assert response.status_code == status.HTTP_200_OK
        assert response.json()["status"] == "healthy"
        assert response.json()["model_loaded"] is True


class TestModelInfoEndpoint:
    """Tests para el endpoint de información del modelo."""
    
    @patch("app_api.model", None)
    def test_model_info_no_model(self, client):
        """Test model info cuando el modelo no está cargado."""
        response = client.get("/model/info")
        assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    
    @patch("app_api.model")
    def test_model_info_with_model(self, mock_model, client):
        """Test model info cuando el modelo está cargado."""
        mock_model.__bool__ = lambda x: True
        
        response = client.get("/model/info")
        assert response.status_code == status.HTTP_200_OK
        assert "model_type" in response.json()
        assert "parameters" in response.json()
        assert "classes" in response.json()


class TestPredictEndpoint:
    """Tests para el endpoint de predicción."""
    
    @patch("app_api.model", None)
    def test_predict_no_model(self, client, sample_student_features):
        """Test predict cuando el modelo no está cargado."""
        request_data = {
            "students": [sample_student_features]
        }
        response = client.post("/predict", json=request_data)
        assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    
    @patch("app_api.model")
    def test_predict_empty_students(self, mock_model, client):
        """Test predict con lista vacía de estudiantes."""
        mock_model.__bool__ = lambda x: True
        
        request_data = {"students": []}
        response = client.post("/predict", json=request_data)
        # Pydantic valida antes, así que retorna 422 (Unprocessable Entity)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    @patch("app_api.model")
    def test_predict_invalid_data(self, mock_model, client):
        """Test predict con datos inválidos."""
        mock_model.__bool__ = lambda x: True
        
        request_data = {
            "students": [{"Gender": "Invalid"}]
        }
        response = client.post("/predict", json=request_data)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


class TestPredictSingleEndpoint:
    """Tests para el endpoint de predicción individual."""
    
    @patch("app_api.model", None)
    def test_predict_single_no_model(self, client, sample_student_features):
        """Test predict single cuando el modelo no está cargado."""
        response = client.post("/predict/single", json=sample_student_features)
        assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    
    @patch("app_api.model")
    def test_predict_single_invalid_data(self, mock_model, client):
        """Test predict single con datos inválidos."""
        mock_model.__bool__ = lambda x: True
        
        invalid_data = {"Gender": "Invalid"}
        response = client.post("/predict/single", json=invalid_data)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

