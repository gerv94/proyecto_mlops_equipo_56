# ============================================================
# Entrenamiento con encoding ORDINAL de la variable objetivo - MLflow
# Basado en train_model_sre.py, pero usando un mapeo fijo:
#   average < good < vg < excellent
# ============================================================

import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, mean_absolute_error,
)
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

logging.basicConfig(
    filename="training_ordinal.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

SEED = 42

# Columnas categóricas conocidas del dataset
CATEGORICAL_COLUMNS = [
    "Gender",
    "Caste",
    "coaching",
    "time",
    "Class_ten_education",
    "twelve_education",
    "medium",
    "Class_ X_Percentage",
    "Class_XII_Percentage",
    "Father_occupation",
    "Mother_occupation",
]


class StudentPerformanceOrdinalTrainer:
    """Entrenamiento con encoding ORDINAL de la variable Performance."""

    def __init__(
        self,
        seed: int = SEED,
        data_path: str = "data/interim/student_interim_clean.csv",
        models_dir: str = "models",
        reports_dir: str = "reports",
        mlflow_tracking_uri: str = "file:./mlruns",
        mlflow_experiment: str = "student_performance_experiment_ordinal",
        model_random_state: int = 888,
    ) -> None:
        """Initialize the trainer with configuration."""
        self.seed = seed
        self.data_path = Path(data_path)
        self.models_dir = Path(models_dir)
        self.reports_dir = Path(reports_dir)
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.mlflow_experiment = mlflow_experiment
        self.model_random_state = model_random_state

        # Attributes to be set during training
        self.pipeline: Optional[Pipeline] = None
        self.class_names: Optional[np.ndarray] = None
        self.target_mapping: Optional[Dict[str, int]] = None

    def _ensure_directories(self) -> None:
        """Ensure required directories exist."""
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def _setup_mlflow(self) -> None:
        """Configure MLflow tracking."""
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        mlflow.set_experiment(self.mlflow_experiment)
        logging.info("Tracking MLflow configurado correctamente (ORDINAL).")

    def load_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Load and prepare data from CSV file.

        Returns:
            Tuple of (features DataFrame, target Series)
        """
        df = pd.read_csv(self.data_path)

        logging.info(f"Datos originales cargados. Dataset completo: {df.shape}")
        logging.info(
            "Distribución original de clases: "
            f"{df['Performance'].value_counts().to_dict()}"
        )

        # PASO 1: ELIMINAR REGISTROS DONDE Performance = 'none', 'None' o vacío
        initial_size = len(df)
        df = df[df["Performance"].notna()]
        df = df[~df["Performance"].astype(str).str.lower().isin(["none", "nan", ""])]

        removed_count = initial_size - len(df)
        if removed_count > 0:
            logging.info(
                f"✓ Eliminados {removed_count} registros con Performance='none' o vacío"
            )
            logging.info(f"Dataset después de limpieza: {df.shape}")

        # PASO 2: ELIMINAR REGISTROS CON 'none', 'None' o vacíos EN LAS FEATURES
        X = df.drop(columns=["Performance"])
        y = df["Performance"]

        valid_mask = pd.Series([True] * len(X), index=X.index)
        for col in X.columns:
            if X[col].dtype == "object":
                mask_invalid = (
                    X[col].isna()
                    | X[col].astype(str).str.strip().str.lower().isin(["none", "nan", ""])
                )
                if mask_invalid.any():
                    invalid_count = mask_invalid.sum()
                    logging.info(
                        f"✓ Detectados {invalid_count} valores inválidos en '{col}'"
                    )
                    valid_mask &= ~mask_invalid

        initial_feature_size = len(X)
        X = X[valid_mask]
        y = y[valid_mask]
        removed_feature_count = initial_feature_size - len(X)

        if removed_feature_count > 0:
            logging.info(
                f"✓ Eliminados {removed_feature_count} registros con valores inválidos en features"
            )
            logging.info(f"Dataset final: {X.shape}")

        logging.info(f"Distribución final de clases: {y.value_counts().to_dict()}")

        return X, y

    def encode_target(self, y: pd.Series) -> np.ndarray:
        """Encode target variable with FIXED ORDINAL mapping.

        Orden impuesto: average < good < vg < excellent
        """
        # Definir el orden ordinal fijo
        ordered_classes = ["average", "good", "vg", "excellent"]
        self.class_names = np.array(ordered_classes)

        # Mapeo texto -> entero
        mapping = {cls: idx for idx, cls in enumerate(ordered_classes)}
        self.target_mapping = mapping

        # Normalizar a minúsculas para el mapeo
        y_norm = y.astype(str).str.strip().str.lower()

        unknown_values = sorted(set(y_norm.unique()) - set(ordered_classes))
        if unknown_values:
            raise ValueError(
                f"Valores de Performance no esperados para encoding ordinal: {unknown_values}"
            )

        y_enc = y_norm.map(mapping).to_numpy()

        logging.info(f"Clases (ordinal): {ordered_classes}")
        logging.info("Target encoding ORDINAL (average < good < vg < excellent)")
        logging.info(f"Target: {len(y_enc)}")

        return y_enc

    def get_categorical_columns(self, X: pd.DataFrame) -> list[str]:
        """Get list of categorical columns present in the DataFrame."""
        cat_cols = [col for col in CATEGORICAL_COLUMNS if col in X.columns]
        logging.info(f"Columnas categóricas a codificar: {cat_cols}")
        return cat_cols

    def split_data(
        self, X: pd.DataFrame, y: np.ndarray
    ) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
        """Split data into train and test sets."""
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=self.seed,
            stratify=y,
        )
        logging.info(f"Split hecho. Train: {X_train.shape}, Test: {X_test.shape}")
        return X_train, X_test, y_train, y_test

    def create_preprocessor(self, cat_cols: list[str]) -> ColumnTransformer:
        """Create preprocessing pipeline."""
        return ColumnTransformer(
            transformers=[
                ("ohe", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ],
            remainder="drop",
        )

    def create_pipeline(self, preprocessor: ColumnTransformer) -> Pipeline:
        """Create the full training pipeline using RandomForestRegressor."""
        rf_params = {
            "random_state": self.model_random_state,
            "n_estimators": 120,
            "max_depth": 12,
            "min_samples_split": 6,
            "min_samples_leaf": 1,
            "criterion": "squared_error",
            "max_features": "sqrt",
            "bootstrap": True,
            "oob_score": False,
        }

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("regressor", RandomForestRegressor(**rf_params)),
            ]
        )

        return pipeline

    def train(self, pipeline: Pipeline, X_train: pd.DataFrame, y_train: np.ndarray) -> Pipeline:
        """Train the pipeline."""
        pipeline.fit(X_train, y_train)
        logging.info("Pipeline (Preprocessing + RandomForestRegressor, ORDINAL) entrenado.")

        if hasattr(pipeline.named_steps["regressor"], "oob_score_"):
            oob_score = pipeline.named_steps["regressor"].oob_score_
            logging.info(f"OOB Score: {oob_score:.4f}")

        return pipeline

    def evaluate(
        self, pipeline: Pipeline, X_test: pd.DataFrame, y_test: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, float], str]:
        """Evaluate the trained pipeline.

        - Usa predicciones continuas del RandomForestRegressor.
        - Calcula MAE sobre predicciones continuas.
        - Reconstruye métricas de clasificación usando predicciones redondeadas 0–3.
        """
        # Predicción continua (regresión)
        y_pred_cont = pipeline.predict(X_test)

        # MAE sobre predicción continua
        mae = mean_absolute_error(y_test, y_pred_cont)

        # Reconstruir clases: redondear y clip 0–3
        y_pred_rounded = np.rint(y_pred_cont).astype(int)
        y_pred_rounded = np.clip(y_pred_rounded, 0, len(self.class_names) - 1)

        metrics = {
            "mae": mae,
            "accuracy": accuracy_score(y_test, y_pred_rounded),
            "precision_weighted": precision_score(
                y_test, y_pred_rounded, average="weighted", zero_division=0
            ),
            "recall_weighted": recall_score(
                y_test, y_pred_rounded, average="weighted", zero_division=0
            ),
            "f1_weighted": f1_score(
                y_test, y_pred_rounded, average="weighted", zero_division=0
            ),
        }

        # Reporte de clasificación usando las clases redondeadas
        classif_rep = classification_report(
            y_test,
            y_pred_rounded,
            target_names=list(self.class_names),
            zero_division=0,
        )

        return y_pred_rounded, metrics, classif_rep

    def save_reports(
        self, classif_rep: str, run_name: str, metrics: Dict[str, float]
    ) -> Path:
        """Save classification reports to disk."""
        report_path = self.reports_dir / f"classification_report_ordinal_{run_name}.txt"
        with open(report_path, "w", encoding="utf-8") as report_file:
            report_file.write(classif_rep)

        report_latest_path = self.reports_dir / "classification_report_ordinal_latest.txt"
        with open(report_latest_path, "w", encoding="utf-8") as latest_file:
            latest_file.write(classif_rep)
            latest_file.write(f"\n\nRun: {run_name}\n")
            latest_file.write(f"Metrics: {metrics}\n")
            latest_file.write(f"Clases (ordial): {list(self.class_names)}\n")

        return report_path

    def create_confusion_matrix(
        self, y_test: np.ndarray, y_pred: np.ndarray, run_name: str
    ) -> Path:
        """Create and save confusion matrix.

        Usa el orden ordinal fijo average < good < vg < excellent.
        """
        classes_ordered = list(self.class_names)
        mapping = self.target_mapping or {
            cls: idx for idx, cls in enumerate(classes_ordered)
        }
        labels_ordered = [mapping[cls] for cls in classes_ordered]

        cm = confusion_matrix(
            y_test,
            y_pred,
            labels=labels_ordered,
        )

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=classes_ordered,
            yticklabels=classes_ordered,
            cbar_kws={"label": "Cantidad"},
        )
        plt.xlabel("Predicción", fontsize=13, fontweight="bold")
        plt.ylabel("Real", fontsize=13, fontweight="bold")
        plt.title("Matriz de confusión (ORDINAL)", fontsize=14, fontweight="bold")
        plt.xticks(rotation=0)
        plt.yticks(rotation=0)

        cm_path = self.models_dir / f"conf_matrix_ordinal_{run_name}.png"
        plt.savefig(cm_path, bbox_inches="tight", dpi=100)
        plt.close()

        logging.info(f"Matriz de confusión (ordinal) guardada en {cm_path}")
        return cm_path

    def save_model(self, pipeline: Pipeline, run_name: str) -> Path:
        """Save the trained model to disk."""
        model_path = self.models_dir / f"model_ordinal_{run_name}.joblib"
        joblib.dump(pipeline, model_path)
        logging.info(f"Pipeline ORDINAL guardado en {model_path}")

        best_model_path = self.models_dir / "best_ordinal_model.joblib"
        joblib.dump(pipeline, best_model_path)
        logging.info(f"✓ Modelo ordinal guardado como {best_model_path}")

        return model_path

    def log_to_mlflow(
        self,
        pipeline: Pipeline,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        metrics: Dict[str, float],
        cat_cols: list[str],
        report_path: Path,
        cm_path: Path,
        model_path: Path,
        run_name: str,
    ) -> None:
        """Log training results to MLflow."""
        params = {
            "random_state": self.model_random_state,
            "n_estimators": 120,
            "max_depth": 12,
            "min_samples_split": 6,
            "min_samples_leaf": 1,
            "criterion": "squared_error",
            "max_features": "sqrt",
            "preprocessing": "OneHotEncoder_only_like_notebook",
            "categorical_features": str(cat_cols),
            "remainder": "drop",
            "target_encoding": "ordinal_average<good<vg<excellent",
            "model_type": "RandomForestRegressor",
        }

        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(report_path))
        mlflow.log_artifact(str(cm_path))
        mlflow.log_artifact(str(model_path))

        if hasattr(pipeline.named_steps["regressor"], "oob_score_"):
            oob_score = pipeline.named_steps["regressor"].oob_score_
            mlflow.log_metric("oob_score", oob_score)

        signature = infer_signature(X_train, pipeline.predict(X_train))
        input_example = X_test.head(5)

        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="student_model_ordinal_regressor",
            signature=signature,
            input_example=input_example,
        )

        logging.info(f"Métricas (ordinal): {metrics}")

    def run(self) -> None:
        """Execute the full ordinal training pipeline."""
        self._ensure_directories()
        np.random.seed(self.seed)
        logging.info("==== Entrenamiento ORDINAL iniciado ====")
        logging.info(f"Semilla global fijada: {self.seed}")

        self._setup_mlflow()

        # Load and prepare data
        X, y = self.load_data()
        y_enc = self.encode_target(y)
        cat_cols = self.get_categorical_columns(X)
        X_train, X_test, y_train, y_test = self.split_data(X, y_enc)

        # Create and train pipeline
        preprocessor = self.create_preprocessor(cat_cols)
        pipeline = self.create_pipeline(preprocessor)
        pipeline = self.train(pipeline, X_train, y_train)
        self.pipeline = pipeline

        # Evaluate
        y_pred, metrics, classif_rep = self.evaluate(pipeline, X_test, y_test)

        # Generate run name
        run_name = f"rf_train_ordinal_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Save artifacts
        report_path = self.save_reports(classif_rep, run_name, metrics)
        cm_path = self.create_confusion_matrix(y_test, y_pred, run_name)
        model_path = self.save_model(pipeline, run_name)

        # Log to MLflow
        with mlflow.start_run(run_name=run_name):
            try:
                self.log_to_mlflow(
                    pipeline,
                    X_train,
                    X_test,
                    metrics,
                    cat_cols,
                    report_path,
                    cm_path,
                    model_path,
                    run_name,
                )
                print(
                    "Entrenamiento ORDINAL (Regressor) completado. "
                    f"Accuracy (clases redondeadas): {metrics['accuracy']:.4f} | "
                    f"MAE (continua): {metrics['mae']:.4f}"
                )
            except Exception as exc:  # noqa: BLE001
                logging.error(f"Error durante el registro en MLflow (ordinal): {str(exc)}")
                raise

        logging.info("==== Entrenamiento ORDINAL completado exitosamente ====")


def main() -> None:
    """Main entry point for ordinal training script."""
    trainer = StudentPerformanceOrdinalTrainer()
    trainer.run()


if __name__ == "__main__":
    main()
