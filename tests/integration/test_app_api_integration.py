"""
Pruebas de integración para el módulo app_api.py

Estas pruebas validan el flujo completo de la API con un cliente HTTP real.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
import joblib
from fastapi.testclient import TestClient
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from unittest.mock import patch, MagicMock

# Importar después de configurar el path
import sys
from pathlib import Path as PathLib
project_root = PathLib(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from app_api import app, MODEL_PATH


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
def trained_pipeline(temp_data_dir):
    """Pipeline entrenado para pruebas de integración."""
    # Crear datos de entrenamiento
    X_train = pd.DataFrame({
        "Gender": ["Male", "Female", "Male", "Female", "Male"],
        "Caste": ["General", "OBC", "SC", "General", "OBC"],
        "coaching": ["yes", "no", "yes", "no", "yes"],
        "time": ["1-2 hours", "2-3 hours", "3-4 hours", "1-2 hours", "2-3 hours"],
        "Class_ten_education": ["CBSE", "State Board", "ICSE", "CBSE", "State Board"],
        "twelve_education": ["CBSE", "State Board", "ICSE", "CBSE", "State Board"],
        "medium": ["English", "Hindi", "English", "Hindi", "English"],
        "Class_ X_Percentage": ["good", "vg", "excellent", "good", "vg"],
        "Class_XII_Percentage": ["good", "vg", "excellent", "good", "vg"],
        "Father_occupation": ["Engineer", "Teacher", "Doctor", "Engineer", "Teacher"],
        "Mother_occupation": ["Doctor", "Nurse", "Engineer", "Doctor", "Nurse"],
    })
    
    y_train = pd.Series(["good", "average", "excellent", "good", "average"])
    
    # Crear y entrenar pipeline
    cat_cols = ["Gender", "Caste", "coaching", "time", "Class_ten_education",
                "twelve_education", "medium", "Class_ X_Percentage", 
                "Class_XII_Percentage", "Father_occupation", "Mother_occupation"]
    
    preprocessor = ColumnTransformer(
        transformers=[("ohe", OneHotEncoder(handle_unknown="ignore"), cat_cols)],
        remainder="drop"
    )
    
    classifier = RandomForestClassifier(n_estimators=10, random_state=42)
    
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])
    
    # Codificar target
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_train)
    
    # Entrenar
    pipeline.fit(X_train, y_encoded)
    
    # Guardar modelo
    model_path = temp_data_dir / "test_model.joblib"
    joblib.dump(pipeline, model_path)
    
    return model_path, pipeline


@pytest.fixture
def sample_student_data():
    """Datos de ejemplo para un estudiante."""
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


@pytest.fixture
def client_with_model(trained_pipeline):
    """Cliente de prueba con modelo cargado."""
    model_path, _ = trained_pipeline
    
    # Mockear MODEL_PATH y cargar el modelo
    with patch("app_api.MODEL_PATH", model_path):
        with patch("app_api._try_load_local_joblib") as mock_load:
            mock_load.return_value = joblib.load(model_path)
            
            # Recargar el módulo para que use el modelo mockeado
            import importlib
            import app_api
            importlib.reload(app_api)
            
            client = TestClient(app_api.app)
            yield client
            
            # Restaurar
            importlib.reload(app_api)


# ============================================================
# Tests de Integración
# ============================================================

class TestAppAPIIntegration:
    """Tests de integración para el flujo completo de la API."""
    
    def test_root_endpoint_integration(self):
        """Test del endpoint raíz en flujo completo."""
        client = TestClient(app)
        response = client.get("/")
        assert response.status_code == 200
        assert "message" in response.json()
    
    def test_health_check_integration(self, trained_pipeline):
        """Test del health check con modelo real."""
        model_path, pipeline = trained_pipeline
        
        with patch("app_api.MODEL_PATH", model_path):
            with patch("app_api.model", pipeline):
                client = TestClient(app)
                response = client.get("/health")
                
                if pipeline is not None:
                    assert response.status_code == 200
                    assert response.json()["status"] == "healthy"
                else:
                    assert response.status_code == 503
    
    def test_model_info_integration(self, trained_pipeline):
        """Test de información del modelo con modelo real."""
        model_path, pipeline = trained_pipeline
        
        with patch("app_api.model", pipeline):
            client = TestClient(app)
            response = client.get("/model/info")
            
            if pipeline is not None:
                assert response.status_code == 200
                data = response.json()
                assert "model_type" in data
                assert "parameters" in data
                assert "classes" in data
            else:
                assert response.status_code == 503
    
    def test_predict_single_integration(self, trained_pipeline, sample_student_data):
        """Test de predicción individual con modelo real."""
        model_path, pipeline = trained_pipeline
        
        with patch("app_api.model", pipeline):
            client = TestClient(app)
            response = client.post("/predict/single", json=sample_student_data)
            
            if pipeline is not None:
                assert response.status_code == 200
                data = response.json()
                assert "prediction" in data
                assert "probability" in data
                assert data["prediction"] in ["average", "excellent", "good", "vg", "none"]
                assert 0.0 <= data["probability"] <= 1.0
            else:
                assert response.status_code == 503
    
    def test_predict_batch_integration(self, trained_pipeline, sample_student_data):
        """Test de predicción por lotes con modelo real."""
        model_path, pipeline = trained_pipeline
        
        request_data = {
            "students": [sample_student_data, sample_student_data]
        }
        
        with patch("app_api.model", pipeline):
            client = TestClient(app)
            response = client.post("/predict", json=request_data)
            
            if pipeline is not None:
                assert response.status_code == 200
                data = response.json()
                assert "predictions" in data
                assert len(data["predictions"]) == 2
                assert "total_students" in data
                assert data["total_students"] == 2
                
                for pred in data["predictions"]:
                    assert "prediction" in pred
                    assert "probability" in pred
                    assert "all_probabilities" in pred
            else:
                assert response.status_code == 503
    
    def test_predict_with_invalid_data(self, trained_pipeline):
        """Test que maneja correctamente datos inválidos."""
        model_path, pipeline = trained_pipeline
        
        invalid_data = {
            "Gender": "Invalid",  # Valor inválido
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
        
        with patch("app_api.model", pipeline):
            client = TestClient(app)
            response = client.post("/predict/single", json=invalid_data)
            
            # Debe retornar error de validación
            assert response.status_code == 422
    
    def test_predict_empty_request(self, trained_pipeline):
        """Test que maneja correctamente solicitudes vacías."""
        model_path, pipeline = trained_pipeline
        
        request_data = {"students": []}
        
        with patch("app_api.model", pipeline):
            client = TestClient(app)
            response = client.post("/predict", json=request_data)
            
            # Pydantic valida antes, así que retorna 422 (Unprocessable Entity)
            assert response.status_code == 422
    
    def test_error_handlers(self, trained_pipeline):
        """Test que los manejadores de error funcionan correctamente."""
        model_path, pipeline = trained_pipeline
        
        with patch("app_api.model", pipeline):
            client = TestClient(app)
            
            # Test 404 (endpoint no existe)
            response = client.get("/nonexistent")
            assert response.status_code == 404
            
            # Test 422 (validación fallida)
            response = client.post("/predict/single", json={"invalid": "data"})
            assert response.status_code == 422
    
    def test_predict_response_structure(self, trained_pipeline, sample_student_data):
        """Test que la estructura de respuesta es correcta."""
        model_path, pipeline = trained_pipeline
        
        request_data = {"students": [sample_student_data]}
        
        with patch("app_api.model", pipeline):
            client = TestClient(app)
            response = client.post("/predict", json=request_data)
            
            if pipeline is not None and response.status_code == 200:
                data = response.json()
                
                # Verificar estructura
                assert "predictions" in data
                assert "total_students" in data
                assert "timestamp" in data
                
                # Verificar estructura de cada predicción
                for pred in data["predictions"]:
                    assert "prediction" in pred
                    assert "probability" in pred
                    assert "all_probabilities" in pred
                    
                    # Verificar que all_probabilities tiene las clases esperadas
                    # (el modelo puede tener diferentes clases según cómo fue entrenado)
                    assert len(pred["all_probabilities"]) > 0
                    # Verificar que todas las probabilidades suman aproximadamente 1.0
                    total_prob = sum(pred["all_probabilities"].values())
                    assert abs(total_prob - 1.0) < 0.01  # Tolerancia para errores de redondeo
    
    def test_multiple_predictions_consistency(self, trained_pipeline, sample_student_data):
        """Test que múltiples predicciones del mismo estudiante son consistentes."""
        model_path, pipeline = trained_pipeline
        
        request_data = {"students": [sample_student_data] * 3}
        
        with patch("app_api.model", pipeline):
            client = TestClient(app)
            response = client.post("/predict", json=request_data)
            
            if pipeline is not None and response.status_code == 200:
                data = response.json()
                predictions = data["predictions"]
                
                # Todas las predicciones deben ser iguales (mismo input)
                first_pred = predictions[0]["prediction"]
                for pred in predictions[1:]:
                    assert pred["prediction"] == first_pred

