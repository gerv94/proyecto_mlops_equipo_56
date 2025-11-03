from __future__ import annotations
from typing import Any, Dict
from pathlib import Path
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature


class MLflowHelper:
    """Helper class for MLflow operations."""
    
    def __init__(self, tracking_uri: str = "file:./mlruns", experiment_name: str | None = None):
        """
        Initialize MLflow helper.
        
        Args:
            tracking_uri: MLflow tracking URI
            experiment_name: Experiment name to set
        """
        mlflow.set_tracking_uri(tracking_uri)
        if experiment_name:
            mlflow.set_experiment(experiment_name)
    
    @staticmethod
    def log_params_safe(params: Dict[str, Any]) -> None:
        """Log parameters to MLflow, converting values safely."""
        clean_params = {
            key: str(value) if not isinstance(value, (int, float, str, bool)) else value
            for key, value in params.items()
        }
        mlflow.log_params(clean_params)
    
    @staticmethod
    def log_metrics_safe(metrics: Dict[str, float]) -> None:
        """Log metrics to MLflow, filtering non-numeric values."""
        clean_metrics = {
            key: float(value) 
            for key, value in metrics.items() 
            if isinstance(value, (int, float))
        }
        mlflow.log_metrics(clean_metrics)
    
    @staticmethod
    def log_artifacts_safe(artifact_paths: list[str | Path]) -> None:
        """Log multiple artifacts to MLflow."""
        for artifact_path in artifact_paths:
            if Path(artifact_path).exists():
                mlflow.log_artifact(str(artifact_path))
    
    @staticmethod
    def log_model_sklearn(model, artifact_path: str, signature=None, input_example=None):
        """
        Log sklearn model to MLflow with signature.
        
        Args:
            model: Trained sklearn model
            artifact_path: Path within MLflow run
            signature: Model signature (auto-inferred if None)
            input_example: Input example for model
        """
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=artifact_path,
            signature=signature,
            input_example=input_example
        )
    
    @staticmethod
    def create_signature(features, predictions):
        """Create MLflow model signature from features and predictions."""
        return infer_signature(features, predictions)
