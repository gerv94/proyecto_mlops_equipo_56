import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib
import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.preprocessing import LabelEncoder


@dataclass
class ModelPredictorConfig:
    """Configuration required to resolve and evaluate the trained model."""

    model_path: str = "models/best_gridsearch_amplio.joblib"
    tracking_uri: str = "file:./mlruns"
    clean_data_path: str = "data/interim/student_interim_clean.csv"
    target_column: str = "Performance"


class ModelPredictor:
    """Encapsulates model resolution from MLflow and evaluation on local data."""

    def __init__(self, config: Optional[ModelPredictorConfig] = None) -> None:
        self.config = config or ModelPredictorConfig()
        # MLflow client solo se inicializa si se necesita (fallback)
        self.client: Optional[MlflowClient] = None

    def _resolve_latest_run_id(self) -> str:
        """Resolve latest run ID from known experiments (same logic as app_api.py)."""
        experiments = [
            "student_performance_gridsearch_amplio",
            "student_performance_experiment_fase2",
        ]
        
        for exp_name in experiments:
            exp = self.client.get_experiment_by_name(exp_name)
            if not exp:
                continue
            runs = self.client.search_runs(
                experiment_ids=[exp.experiment_id],
                order_by=["attributes.end_time DESC"],
                max_results=1,
            )
            if runs:
                run_id = runs[0].info.run_id
                print(f"Usando run_id del experimento '{exp_name}': {run_id}")
                return run_id
        
        raise ValueError("No se encontraron runs en ningún experimento conocido")

    def _find_model_artifact_subpath(self, run_id: str) -> str:
        candidate_names = ["best_model", "student_model", "model", "sklearn-model"]
        artifact_subpath: Optional[str] = None

        for name in candidate_names:
            try:
                self.client.list_artifacts(run_id, name)
                artifact_subpath = name
                break
            except Exception:
                continue

        if artifact_subpath is None:
            root_items = self.client.list_artifacts(run_id)
            for item in root_items:
                if item.is_dir:
                    try:
                        inner_items = self.client.list_artifacts(run_id, item.path)
                        if any(os.path.basename(inner.path) == "MLmodel" for inner in inner_items):
                            artifact_subpath = item.path
                            break
                    except Exception:
                        continue

        assert artifact_subpath is not None, "No encontré un artifact de modelo con archivo MLmodel en el run."
        return artifact_subpath

    def _load_model_from_dvc(self) -> Optional:
        """Try to load model from DVC (local .joblib file).
        
        Returns:
            Loaded model or None if file doesn't exist
        """
        model_path = Path(self.config.model_path)
        if not model_path.exists():
            print(f"Modelo no encontrado en DVC: {model_path}")
            return None
        
        try:
            model = joblib.load(model_path)
            print(f"Modelo cargado exitosamente desde DVC: {model_path}")
            return model
        except Exception as e:
            print(f"Error al cargar modelo desde DVC: {e}")
            return None
    
    def _load_model_from_mlflow(self):
        """Load model from MLflow as fallback."""
        # Inicializar MLflow client solo si se necesita
        if self.client is None:
            mlflow.set_tracking_uri(self.config.tracking_uri)
            self.client = MlflowClient()
        
        run_id = self._resolve_latest_run_id()
        artifact_subpath = self._find_model_artifact_subpath(run_id)

        model_uri = f"runs:/{run_id}/{artifact_subpath}"
        print(f"Model URI (MLflow fallback): {model_uri}")
        loaded_model = mlflow.sklearn.load_model(model_uri)
        print("Modelo cargado correctamente desde MLflow")
        return loaded_model
    
    def _load_model(self):
        """Load model: first try DVC, then fallback to MLflow."""
        # Intento 1: Cargar desde DVC (mejor modelo)
        model = self._load_model_from_dvc()
        
        # Intento 2: Fallback a MLflow si no existe en DVC
        if model is None:
            print("Intentando cargar desde MLflow como fallback...")
            model = self._load_model_from_mlflow()
        
        if model is None:
            raise FileNotFoundError(
                f"No se pudo cargar el modelo. Verifica que existe en DVC ({self.config.model_path}) "
                "o que hay runs en MLflow."
            )
        
        return model

    def _load_data(self) -> tuple[pd.DataFrame, pd.Series]:
        """Load data from clean CSV and prepare for prediction.
        
        The model is a Pipeline that includes preprocessing, so we only need
        the raw clean data. The pipeline will apply OneHotEncoder internally.
        
        This method replicates the same data cleaning logic used in training:
        1. Remove records where Performance='none' or empty
        2. Remove records with 'none', 'None', or empty values in categorical features
        """
        df = pd.read_csv(self.config.clean_data_path)
        
        # PASO 1: ELIMINAR REGISTROS DONDE Performance = 'none', 'None' o vacío
        df = df[df['Performance'].notna()]  # Eliminar NaN
        df = df[~df['Performance'].astype(str).str.lower().isin(['none', 'nan', ''])]  # Eliminar 'none'
        
        # PASO 2: ELIMINAR REGISTROS CON 'none', 'None' o vacíos EN LAS FEATURES
        X = df.drop(columns=[self.config.target_column])
        y = df[self.config.target_column]
        
        # Crear máscara para registros válidos
        valid_mask = pd.Series([True] * len(X), index=X.index)
        
        for col in X.columns:
            if X[col].dtype == 'object':
                # Detectar 'none', 'None', vacíos, NaN
                mask_invalid = (
                    X[col].isna() | 
                    X[col].astype(str).str.strip().str.lower().isin(['none', 'nan', ''])
                )
                
                if mask_invalid.any():
                    valid_mask &= ~mask_invalid
        
        # Aplicar filtro
        X = X[valid_mask]
        y = y[valid_mask]
        
        return X, y

    def evaluate(self) -> float:
        """Load the best model and compute accuracy over current data.
        
        The model is a Pipeline that expects:
        - X: Raw features (will be preprocessed by the pipeline)
        - y: Encoded target (using LabelEncoder with the same classes as training)
        """
        model = self._load_model()
        X, y = self._load_data()
        
        # Obtener las clases del modelo entrenado
        # El Pipeline tiene un clasificador en el paso 'classifier'
        classifier = model.named_steps.get('classifier')
        if classifier is None:
            # Si no tiene 'classifier', buscar cualquier paso que tenga 'classes_'
            for step_name, step in model.named_steps.items():
                if hasattr(step, 'classes_'):
                    classifier = step
                    break
        
        if classifier is None or not hasattr(classifier, 'classes_'):
            raise ValueError("No se pudo encontrar el clasificador con clases en el modelo")
        
        # classifier.classes_ contiene los índices numéricos [0, 1, 2, 3]
        # Necesitamos codificar los strings de y a números usando LabelEncoder
        # LabelEncoder ordena alfabéticamente por defecto
        label_encoder = LabelEncoder()
        
        # Obtener todas las clases únicas de y y del modelo para asegurar consistencia
        unique_y = sorted(y.unique())
        print(f"Clases encontradas en datos: {unique_y}")
        print(f"Clases esperadas por modelo (índices): {classifier.classes_}")
        
        # Fit del LabelEncoder con las clases únicas (orden alfabético)
        # Esto asegura que el orden sea consistente con el entrenamiento
        label_encoder.fit(unique_y)
        
        # Verificar que el número de clases coincida
        if len(label_encoder.classes_) != len(classifier.classes_):
            raise ValueError(
                f"Número de clases no coincide: datos tienen {len(label_encoder.classes_)} "
                f"pero el modelo espera {len(classifier.classes_)}"
            )
        
        # Transformar y a números
        y_encoded = label_encoder.transform(y)
        
        # Evaluar el modelo
        accuracy = model.score(X, y_encoded)
        print(f"Accuracy del modelo cargado: {accuracy:.4f}")
        return accuracy


def main() -> None:
    predictor = ModelPredictor()
    predictor.evaluate()


if __name__ == "__main__":
    main()
