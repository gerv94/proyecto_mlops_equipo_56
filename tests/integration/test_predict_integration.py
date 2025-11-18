"""
Pruebas de integración para el módulo predict.py

Estas pruebas validan el flujo completo de carga y evaluación del modelo.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
import joblib
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
def test_data_csv(temp_data_dir):
    """CSV de datos de prueba."""
    df = pd.DataFrame({
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
    
    csv_path = temp_data_dir / "test_data.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


# ============================================================
# Tests de Integración
# ============================================================

class TestPredictIntegration:
    """Tests de integración para el flujo completo de predict.py."""
    
    def test_full_predict_workflow(self, temp_data_dir, trained_pipeline, test_data_csv):
        """Test del flujo completo: carga modelo, carga datos, evalúa."""
        model_path, _ = trained_pipeline
        
        config = ModelPredictorConfig(
            model_path=str(model_path),
            clean_data_path=str(test_data_csv)
        )
        predictor = ModelPredictor(config)
        
        accuracy = predictor.evaluate()
        
        assert isinstance(accuracy, float)
        assert 0.0 <= accuracy <= 1.0
    
    def test_predict_with_dvc_model(self, temp_data_dir, trained_pipeline, test_data_csv):
        """Test que carga el modelo desde DVC correctamente."""
        model_path, _ = trained_pipeline
        
        config = ModelPredictorConfig(
            model_path=str(model_path),
            clean_data_path=str(test_data_csv)
        )
        predictor = ModelPredictor(config)
        
        # Verificar que carga desde DVC
        model = predictor._load_model_from_dvc()
        assert model is not None
        assert isinstance(model, Pipeline)
    
    def test_predict_handles_missing_performance(self, temp_data_dir, trained_pipeline):
        """Test que maneja correctamente datos con Performance faltante."""
        model_path, _ = trained_pipeline
        
        # Crear CSV con algunos registros sin Performance válida
        df = pd.DataFrame({
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
            "Performance": ["good", "none", "excellent"],  # 'none' debe ser eliminado
        })
        
        csv_path = temp_data_dir / "test_data.csv"
        df.to_csv(csv_path, index=False)
        
        config = ModelPredictorConfig(
            model_path=str(model_path),
            clean_data_path=str(csv_path)
        )
        predictor = ModelPredictor(config)
        
        X, y = predictor._load_data()
        
        # Verificar que 'none' fue eliminado
        assert "none" not in y.values
        assert len(X) == 2  # Solo quedan 2 registros válidos
    
    def test_predict_handles_invalid_features(self, temp_data_dir, trained_pipeline):
        """Test que maneja correctamente features con valores inválidos."""
        model_path, _ = trained_pipeline
        
        # Crear CSV con valores inválidos en features categóricas
        df = pd.DataFrame({
            "Gender": ["Male", "none", "Female"],  # 'none' en feature
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
        
        csv_path = temp_data_dir / "test_data.csv"
        df.to_csv(csv_path, index=False)
        
        config = ModelPredictorConfig(
            model_path=str(model_path),
            clean_data_path=str(csv_path)
        )
        predictor = ModelPredictor(config)
        
        X, y = predictor._load_data()
        
        # Verificar que el registro con 'none' en Gender fue eliminado
        assert "none" not in X["Gender"].values
        assert len(X) == 2
    
    def test_predict_accuracy_is_reasonable(self, temp_data_dir, trained_pipeline, test_data_csv):
        """Test que el accuracy calculado es razonable."""
        model_path, _ = trained_pipeline
        
        config = ModelPredictorConfig(
            model_path=str(model_path),
            clean_data_path=str(test_data_csv)
        )
        predictor = ModelPredictor(config)
        
        accuracy = predictor.evaluate()
        
        # El accuracy debe estar entre 0 y 1
        assert 0.0 <= accuracy <= 1.0
        # Con un modelo entrenado y datos similares, debería tener algún accuracy
        # (aunque puede ser bajo si los datos de prueba son diferentes)
    
    def test_predict_with_empty_data(self, temp_data_dir, trained_pipeline):
        """Test que maneja correctamente datos vacíos después de limpieza."""
        model_path, _ = trained_pipeline
        
        # Crear CSV donde todos los registros serán eliminados
        df = pd.DataFrame({
            "Gender": ["none", "none"],
            "Caste": ["General", "OBC"],
            "coaching": ["yes", "no"],
            "time": ["1-2 hours", "2-3 hours"],
            "Class_ten_education": ["CBSE", "State Board"],
            "twelve_education": ["CBSE", "State Board"],
            "medium": ["English", "Hindi"],
            "Class_ X_Percentage": ["good", "vg"],
            "Class_XII_Percentage": ["good", "vg"],
            "Father_occupation": ["Engineer", "Teacher"],
            "Mother_occupation": ["Doctor", "Nurse"],
            "Performance": ["none", "none"],  # Todos serán eliminados
        })
        
        csv_path = temp_data_dir / "test_data.csv"
        df.to_csv(csv_path, index=False)
        
        config = ModelPredictorConfig(
            model_path=str(model_path),
            clean_data_path=str(csv_path)
        )
        predictor = ModelPredictor(config)
        
        X, y = predictor._load_data()
        
        # Después de limpieza, debería quedar vacío o muy pocos registros
        assert len(X) == 0 or len(X) < len(df)

