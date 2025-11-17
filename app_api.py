# ============================================================
# FastAPI Service - Student Performance Prediction
# Modelo: RandomForest (n_estimators=20, max_depth=20, min_samples_split=15)
# Test Accuracy: 53.68%
# ============================================================

from fastapi import FastAPI, HTTPException, status, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field, validator, ValidationError
from typing import List, Literal
import joblib
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime
import traceback
import os
import mlflow
import mlflow.sklearn

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ============================================================
# INICIALIZACIÓN DE LA APLICACIÓN
# ============================================================

app = FastAPI(
    title="Student Performance Prediction API",
    description="API para predecir el rendimiento académico de estudiantes usando RandomForest",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Nota: La documentación completa de OpenAPI está en el archivo openapi.yaml

# ============================================================
# CARGA DEL MODELO
# ============================================================

MODEL_PATH = Path("models/best_gridsearch_amplio.joblib")

try:
    model = joblib.load(MODEL_PATH)
    logger.info(f"Modelo cargado exitosamente desde {MODEL_PATH}")
except FileNotFoundError:
    logger.error(f"Modelo no encontrado en {MODEL_PATH}")
    model = None
except Exception as e:
    logger.error(f"Error al cargar el modelo: {str(e)}")
    model = None

# Clases de predicción (en orden alfabético según LabelEncoder)
CLASS_NAMES = ['average', 'excellent', 'good', 'none', 'vg']

# ============================================================
# MODELOS DE DATOS (PYDANTIC)
# ============================================================

class StudentFeatures(BaseModel):
    """Modelo de entrada para las características del estudiante con validación estricta"""
    
    Gender: Literal["Male", "Female"] = Field(
        ..., 
        description="Género del estudiante", 
        example="Male"
    )
    
    Caste: Literal["General", "OBC", "SC", "ST"] = Field(
        ..., 
        description="Casta del estudiante", 
        example="General"
    )
    
    coaching: Literal["yes", "no"] = Field(
        ..., 
        description="¿Recibe coaching?", 
        example="yes"
    )
    
    time: Literal[
        "less than 1 hour", 
        "1-2 hours", 
        "2-3 hours", 
        "3-4 hours", 
        "more than 4 hours"
    ] = Field(
        ..., 
        description="Tiempo dedicado al estudio", 
        example="3-4 hours"
    )
    
    Class_ten_education: Literal["CBSE", "State Board", "ICSE"] = Field(
        ..., 
        description="Tipo de educación en clase 10", 
        example="CBSE"
    )
    
    twelve_education: Literal["CBSE", "State Board", "ICSE"] = Field(
        ..., 
        description="Tipo de educación en clase 12", 
        example="CBSE"
    )
    
    medium: Literal["English", "Hindi"] = Field(
        ..., 
        description="Medio de instrucción", 
        example="English"
    )
    
    # IMPORTANTE: El dataset tiene un espacio en el nombre: 'Class_ X_Percentage'
    Class_X_Percentage: Literal["average", "good", "vg", "excellent"] = Field(
        ..., 
        alias="Class_ X_Percentage",
        description="Porcentaje en clase X", 
        example="vg"
    )
    
    Class_XII_Percentage: Literal["average", "good", "vg", "excellent"] = Field(
        ..., 
        description="Porcentaje en clase XII", 
        example="vg"
    )
    
    Father_occupation: str = Field(
        ..., 
        min_length=2,
        max_length=50,
        description="Ocupación del padre", 
        example="Business"
    )
    
    Mother_occupation: str = Field(
        ..., 
        min_length=2,
        max_length=50,
        description="Ocupación de la madre", 
        example="Housewife"
    )
    
    @validator('Father_occupation', 'Mother_occupation')
    def validate_occupation(cls, v):
        """Validar que las ocupaciones no estén vacías y sean alfanuméricas"""
        if not v or v.strip() == "":
            raise ValueError("La ocupación no puede estar vacía")
        return v.strip()
    
    class Config:
        populate_by_name = True  # Permite usar tanto el nombre del campo como el alias
        json_schema_extra = {
            "example": {
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
        }

class PredictionRequest(BaseModel):
    """Modelo de solicitud para predicción con validación de tamaño"""
    students: List[StudentFeatures] = Field(
        ..., 
        min_items=1,
        max_items=100,
        description="Lista de estudiantes para predecir (máximo 100)"
    )
    
    @validator('students')
    def validate_students_list(cls, v):
        """Validar que la lista de estudiantes no esté vacía"""
        if not v:
            raise ValueError("Debe proporcionar al menos un estudiante")
        return v

class PredictionResponse(BaseModel):
    """Modelo de respuesta para predicción"""
    prediction: str = Field(
        ..., 
        description="Clase predicha (average, good, vg, excellent, none)",
        example="vg"
    )
    probability: float = Field(
        ..., 
        description="Probabilidad de la clase predicha (0.0 - 1.0)",
        ge=0.0,
        le=1.0,
        example=0.65
    )
    all_probabilities: dict = Field(
        ..., 
        description="Probabilidades de todas las clases posibles",
        example={
            "average": 0.10,
            "excellent": 0.15,
            "good": 0.05,
            "none": 0.05,
            "vg": 0.65
        }
    )
    
class BatchPredictionResponse(BaseModel):
    """Modelo de respuesta para predicción por lotes"""
    predictions: List[PredictionResponse] = Field(
        ..., 
        description="Lista de predicciones para cada estudiante"
    )
    total_students: int = Field(
        ..., 
        description="Número total de estudiantes procesados",
        example=1
    )
    timestamp: str = Field(
        ..., 
        description="Marca de tiempo de la predicción (ISO 8601)",
        example="2025-11-16T19:28:00.000000"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "predictions": [
                    {
                        "prediction": "vg",
                        "probability": 0.65,
                        "all_probabilities": {
                            "average": 0.10,
                            "excellent": 0.15,
                            "good": 0.05,
                            "none": 0.05,
                            "vg": 0.65
                        }
                    }
                ],
                "total_students": 1,
                "timestamp": "2025-11-16T19:28:00.000000"
            }
        }

# ============================================================
# ENDPOINTS
# ============================================================

@app.get(
    "/", 
    tags=["Health"],
    summary="Health check básico",
    description="Verificación rápida de que el servicio está corriendo",
    responses={
        200: {
            "description": "Servicio funcionando correctamente",
            "content": {
                "application/json": {
                    "example": {
                        "message": "Student Performance Prediction API",
                        "status": "running",
                        "model_loaded": True,
                        "version": "1.0.0"
                    }
                }
            }
        }
    }
)
async def root():
    """
    Endpoint raíz para verificación básica de salud.
    
    Returns:
        dict: Estado básico del servicio
    """
    return {
        "message": "Student Performance Prediction API",
        "status": "running",
        "model_loaded": model is not None,
        "version": "1.0.0"
    }

@app.get(
    "/health", 
    tags=["Health"],
    summary="Health check detallado",
    description="Verificación detallada del estado del servicio y del modelo",
    responses={
        200: {
            "description": "Servicio saludable",
            "content": {
                "application/json": {
                    "example": {
                        "status": "healthy",
                        "model_loaded": True,
                        "model_path": "models/best_gridsearch_amplio.joblib",
                        "timestamp": "2025-11-16T19:28:00.000000"
                    }
                }
            }
        },
        503: {
            "description": "Servicio no disponible",
            "content": {
                "application/json": {
                    "example": {
                        "error": "Modelo no cargado. El servicio no está disponible.",
                        "status_code": 503,
                        "timestamp": "2025-11-16T19:28:00.000000"
                    }
                }
            }
        }
    }
)
async def health_check():
    """
    Endpoint de verificación de salud detallada.
    
    Returns:
        dict: Estado detallado del servicio incluyendo modelo cargado
    
    Raises:
        HTTPException 503: Modelo no cargado
    """
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modelo no cargado. El servicio no está disponible."
        )
    
    return {
        "status": "healthy",
        "model_loaded": True,
        "model_path": str(MODEL_PATH),
        "timestamp": datetime.now().isoformat()
    }

@app.get(
    "/model/info", 
    tags=["Model"],
    summary="Información del modelo",
    description="Obtiene información detallada sobre el modelo de ML en producción",
    responses={
        200: {
            "description": "Información del modelo",
            "content": {
                "application/json": {
                    "example": {
                        "model_type": "RandomForestClassifier",
                        "parameters": {
                            "n_estimators": 20,
                            "max_depth": 20,
                            "min_samples_split": 15,
                            "min_samples_leaf": 1,
                            "max_features": "sqrt",
                            "criterion": "gini",
                            "random_state": 888
                        },
                        "performance": {
                            "test_accuracy": 0.5368,
                            "test_f1_weighted": 0.5182
                        },
                        "classes": ["average", "excellent", "good", "none", "vg"],
                        "preprocessing": "OneHotEncoder (remainder='drop')"
                    }
                }
            }
        },
        503: {
            "description": "Modelo no disponible",
            "content": {
                "application/json": {
                    "example": {
                        "error": "Modelo no cargado",
                        "status_code": 503,
                        "timestamp": "2025-11-16T19:28:00.000000"
                    }
                }
            }
        }
    }
)
async def model_info():
    """
    Obtiene información detallada sobre el modelo cargado.
    
    Returns:
        dict: Información del modelo incluyendo tipo, parámetros, performance y clases
    
    Raises:
        HTTPException 503: Modelo no cargado
    """
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modelo no cargado"
        )
    
    return {
        "model_type": "RandomForestClassifier",
        "parameters": {
            "n_estimators": 20,
            "max_depth": 20,
            "min_samples_split": 15,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
            "criterion": "gini",
            "random_state": 888
        },
        "performance": {
            "test_accuracy": 0.5368,
            "test_f1_weighted": 0.5182
        },
        "classes": CLASS_NAMES,
        "preprocessing": "OneHotEncoder (remainder='drop')"
    }

@app.post(
    "/predict", 
    response_model=BatchPredictionResponse, 
    tags=["Prediction"],
    summary="Predicción por lotes",
    description="""
    Realiza predicciones de rendimiento para uno o múltiples estudiantes.
    
    Este endpoint acepta hasta 100 estudiantes por solicitud y retorna predicciones
    con probabilidades para cada clase posible.
    """,
    responses={
        200: {
            "description": "Predicciones realizadas exitosamente",
            "content": {
                "application/json": {
                    "example": {
                        "predictions": [
                            {
                                "prediction": "vg",
                                "probability": 0.65,
                                "all_probabilities": {
                                    "average": 0.10,
                                    "excellent": 0.15,
                                    "good": 0.05,
                                    "none": 0.05,
                                    "vg": 0.65
                                }
                            }
                        ],
                        "total_students": 1,
                        "timestamp": "2025-11-16T19:28:00.000000"
                    }
                }
            }
        },
        400: {
            "description": "Error en los datos de entrada",
            "content": {
                "application/json": {
                    "example": {
                        "error": "Valor inválido",
                        "detail": "La lista de estudiantes está vacía",
                        "timestamp": "2025-11-16T19:28:00.000000"
                    }
                }
            }
        },
        422: {
            "description": "Error de validación de datos",
            "content": {
                "application/json": {
                    "example": {
                        "error": "Error de validación de datos",
                        "detail": "Los datos proporcionados no cumplen con el formato requerido",
                        "errors": [
                            {
                                "field": "students -> 0 -> Gender",
                                "message": "Valor inválido para 'Gender'. Valores permitidos: 'Male', 'Female'",
                                "type": "literal_error",
                                "input": "Masculino"
                            }
                        ],
                        "timestamp": "2025-11-16T19:28:00.000000"
                    }
                }
            }
        },
        500: {
            "description": "Error interno del servidor",
            "content": {
                "application/json": {
                    "example": {
                        "error": "Error interno del servidor",
                        "detail": "Error al procesar la predicción. Verifique los datos de entrada.",
                        "timestamp": "2025-11-16T19:28:00.000000"
                    }
                }
            }
        },
        503: {
            "description": "Servicio no disponible",
            "content": {
                "application/json": {
                    "example": {
                        "error": "Modelo no cargado. No se pueden realizar predicciones.",
                        "status_code": 503,
                        "timestamp": "2025-11-16T19:28:00.000000"
                    }
                }
            }
        }
    }
)
async def predict(request: PredictionRequest):
    """
    Endpoint principal de predicción por lotes.
    
    Args:
        request: Objeto PredictionRequest con lista de estudiantes
    
    Returns:
        BatchPredictionResponse: Predicciones para todos los estudiantes
    
    Raises:
        HTTPException 400: Datos de entrada inválidos
        HTTPException 422: Error de validación
        HTTPException 500: Error interno del servidor
        HTTPException 503: Modelo no disponible
    """
    
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modelo no cargado. No se pueden realizar predicciones."
        )
    
    try:
        # Validar que hay estudiantes
        if not request.students:
            raise ValueError("La lista de estudiantes está vacía")
        
        # Convertir datos de entrada a DataFrame usando by_alias para mantener nombres originales
        students_data = [student.dict(by_alias=True) for student in request.students]
        df = pd.DataFrame(students_data)
        
        logger.info(f"Recibidas {len(students_data)} solicitudes de predicción")
        logger.debug(f"Columnas del DataFrame: {df.columns.tolist()}")
        
        # Validar que el DataFrame tiene las columnas esperadas
        expected_columns = [
            'Gender', 'Caste', 'coaching', 'time', 'Class_ten_education',
            'twelve_education', 'medium', 'Class_ X_Percentage', 'Class_XII_Percentage',
            'Father_occupation', 'Mother_occupation'
        ]
        missing_columns = set(expected_columns) - set(df.columns)
        if missing_columns:
            raise ValueError(f"Faltan columnas requeridas: {missing_columns}")
        
        # Realizar predicciones
        predictions = model.predict(df)
        probabilities = model.predict_proba(df)
        
        # Construir respuesta
        results = []
        for i, (pred, probs) in enumerate(zip(predictions, probabilities)):
            predicted_class = CLASS_NAMES[pred]
            predicted_prob = float(probs[pred])
            
            all_probs = {
                class_name: float(prob) 
                for class_name, prob in zip(CLASS_NAMES, probs)
            }
            
            results.append(PredictionResponse(
                prediction=predicted_class,
                probability=predicted_prob,
                all_probabilities=all_probs
            ))
        
        logger.info(f"Predicciones completadas exitosamente para {len(results)} estudiantes")
        
        return BatchPredictionResponse(
            predictions=results,
            total_students=len(results),
            timestamp=datetime.now().isoformat()
        )
    
    except ValueError as e:
        logger.error(f"Error de valor durante la predicción: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except KeyError as e:
        logger.error(f"Error de columna faltante: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Columna requerida no encontrada: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error inesperado durante la predicción: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error al procesar la predicción. Verifique los datos de entrada."
        )

@app.post(
    "/predict/single", 
    response_model=PredictionResponse, 
    tags=["Prediction"],
    summary="Predicción individual",
    description="""
    Realiza una predicción de rendimiento para un solo estudiante.
    
    Este endpoint es más simple y directo que /predict, ideal para consultas individuales.
    """,
    responses={
        200: {
            "description": "Predicción realizada exitosamente",
            "content": {
                "application/json": {
                    "example": {
                        "prediction": "excellent",
                        "probability": 0.75,
                        "all_probabilities": {
                            "average": 0.05,
                            "excellent": 0.75,
                            "good": 0.10,
                            "none": 0.02,
                            "vg": 0.08
                        }
                    }
                }
            }
        },
        400: {"description": "Error en los datos de entrada"},
        422: {"description": "Error de validación de datos"},
        500: {"description": "Error interno del servidor"},
        503: {"description": "Servicio no disponible"}
    }
)
async def predict_single(student: StudentFeatures):
    """
    Endpoint de predicción para un solo estudiante.
    
    Args:
        student: Objeto StudentFeatures con las características del estudiante
    
    Returns:
        PredictionResponse: Predicción con probabilidades
    
    Raises:
        HTTPException 400: Datos de entrada inválidos
        HTTPException 422: Error de validación
        HTTPException 500: Error interno del servidor
        HTTPException 503: Modelo no disponible
    """
    
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modelo no cargado. No se pueden realizar predicciones."
        )
    
    try:
        # Usar by_alias=True para mantener el nombre con espacio del dataset
        df = pd.DataFrame([student.dict(by_alias=True)])
        prediction = model.predict(df)[0]
        probabilities = model.predict_proba(df)[0]
        
        predicted_class = CLASS_NAMES[prediction]
        predicted_prob = float(probabilities[prediction])
        
        all_probs = {
            class_name: float(prob) 
            for class_name, prob in zip(CLASS_NAMES, probabilities)
        }
        
        logger.info(f"Predicción individual: {predicted_class} (prob: {predicted_prob:.4f})")
        
        return PredictionResponse(
            prediction=predicted_class,
            probability=predicted_prob,
            all_probabilities=all_probs
        )
    
    except ValueError as e:
        logger.error(f"Error de valor durante la predicción individual: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except KeyError as e:
        logger.error(f"Error de columna faltante: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Columna requerida no encontrada: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error inesperado durante la predicción individual: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error al procesar la predicción. Verifique los datos de entrada."
        )

# ============================================================
# MANEJO DE ERRORES MEJORADO
# ============================================================

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Manejo personalizado de errores de validación de Pydantic
    Proporciona mensajes de error detallados y amigables
    """
    errors = []
    for error in exc.errors():
        field = " -> ".join(str(loc) for loc in error["loc"])
        message = error["msg"]
        error_type = error["type"]
        
        # Crear mensaje amigable
        if "literal_error" in error_type:
            allowed_values = error.get("ctx", {}).get("expected", "valores permitidos")
            friendly_message = f"Valor inválido para '{field}'. Valores permitidos: {allowed_values}"
        elif "missing" in error_type:
            friendly_message = f"Campo requerido '{field}' no proporcionado"
        elif "string_too_short" in error_type:
            min_length = error.get("ctx", {}).get("limit_value", "mínimo")
            friendly_message = f"'{field}' debe tener al menos {min_length} caracteres"
        elif "string_too_long" in error_type:
            max_length = error.get("ctx", {}).get("limit_value", "máximo")
            friendly_message = f"'{field}' no debe exceder {max_length} caracteres"
        else:
            friendly_message = f"Error en '{field}': {message}"
        
        errors.append({
            "field": field,
            "message": friendly_message,
            "type": error_type,
            "input": error.get("input")
        })
    
    logger.warning(f"Error de validación: {errors}")
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Error de validación de datos",
            "detail": "Los datos proporcionados no cumplen con el formato requerido",
            "errors": errors,
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """
    Manejo de excepciones HTTP estándar
    """
    logger.warning(f"HTTP Exception: {exc.status_code} - {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """
    Manejo de errores de valor (ValueError)
    """
    logger.error(f"ValueError: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "error": "Valor inválido",
            "detail": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """
    Manejo de excepciones generales no capturadas
    """
    error_trace = traceback.format_exc()
    logger.error(f"Error no manejado: {str(exc)}\n{error_trace}")
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Error interno del servidor",
            "detail": "Ocurrió un error inesperado. Por favor, contacte al administrador.",
            "error_type": type(exc).__name__,
            "timestamp": datetime.now().isoformat()
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)