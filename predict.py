import os
from dataclasses import dataclass
from typing import Optional

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient


@dataclass
class ModelPredictorConfig:
    """Configuration required to resolve and evaluate the trained model."""

    tracking_uri: str = "file:./mlruns"
    experiment_name: str = "student_performance_experiment_fase2"
    preprocessed_features_path: str = "data/interim/student_interim_preprocessed.csv"
    clean_data_path: str = "data/interim/student_interim_clean.csv"
    target_column: str = "Performance"


class ModelPredictor:
    """Encapsulates model resolution from MLflow and evaluation on local data."""

    def __init__(self, config: Optional[ModelPredictorConfig] = None) -> None:
        self.config = config or ModelPredictorConfig()
        mlflow.set_tracking_uri(self.config.tracking_uri)
        self.client = MlflowClient()

    def _resolve_latest_run_id(self) -> str:
        experiment = self.client.get_experiment_by_name(self.config.experiment_name)
        assert experiment is not None, f"Experimento no existe: {self.config.experiment_name}"

        runs = self.client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["attributes.end_time DESC"],
            max_results=1,
        )
        assert runs, f"No hay runs en el experimento {self.config.experiment_name}"
        run_id = runs[0].info.run_id
        print("Usando run_id:", run_id)
        return run_id

    def _find_model_artifact_subpath(self, run_id: str) -> str:
        candidate_names = ["student_model", "model", "sklearn-model"]
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

        assert artifact_subpath is not None,
        "No encontré un artifact de modelo con archivo MLmodel en el run."
        return artifact_subpath

    def _load_model(self):
        run_id = self._resolve_latest_run_id()
        artifact_subpath = self._find_model_artifact_subpath(run_id)

        model_uri = f"runs:/{run_id}/{artifact_subpath}"
        print("Model URI:", model_uri)
        loaded_model = mlflow.sklearn.load_model(model_uri)
        print("Modelo cargado correctamente")
        return loaded_model

    def _load_data(self) -> tuple[pd.DataFrame, pd.Series]:
        features = pd.read_csv(self.config.preprocessed_features_path)
        labels = pd.read_csv(self.config.clean_data_path)[self.config.target_column]
        return features, labels

    def evaluate(self) -> float:
        """Load the best model and compute accuracy over current data."""

        model = self._load_model()
        X, y = self._load_data()
        accuracy = model.score(X, y)
        print(f"Accuracy del modelo cargado: {accuracy:.4f}")
        return accuracy


def main() -> None:
    predictor = ModelPredictor()
    predictor.evaluate()


if __name__ == "__main__":
    main()
