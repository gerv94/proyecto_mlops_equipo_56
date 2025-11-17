# ============================================================
# Entrenamiento reproducible y trazable - Fase 2 (SRE) - MLflow
# Compatible con MLflow 2.15.1 y 3.x
# ============================================================

import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
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
    filename="training.log",
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


class StudentPerformanceTrainer:
    """Encapsulates the full SRE training pipeline for student performance."""

    def __init__(
        self,
        seed: int = SEED,
        data_path: str = "data/interim/student_interim_clean.csv",
        models_dir: str = "models",
        reports_dir: str = "reports",
        mlflow_tracking_uri: str = "file:./mlruns",
        mlflow_experiment: str = "student_performance_experiment_fase2",
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
        self.label_encoder: Optional[LabelEncoder] = None
        self.pipeline: Optional[Pipeline] = None
        self.class_names: Optional[np.ndarray] = None

    def _ensure_directories(self) -> None:
        """Ensure required directories exist."""
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def _setup_mlflow(self) -> None:
        """Configure MLflow tracking."""
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        mlflow.set_experiment(self.mlflow_experiment)
        logging.info("Tracking MLflow configurado correctamente (Fase 2).")

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
        
        # Separar features y target (IGUAL QUE EL NOTEBOOK)
        X = df.drop(columns=["Performance"])
        y = df["Performance"]
        
        # Eliminar columna 'mixed_type_col' si existe (generada durante limpieza)
        if "mixed_type_col" in X.columns:
            X = X.drop(columns=["mixed_type_col"])
            logging.info("Columna 'mixed_type_col' eliminada del dataset")
        
        return X, y

    def encode_target(self, y: pd.Series) -> np.ndarray:
        """Encode target variable with LabelEncoder.
        
        Args:
            y: Target series
            
        Returns:
            Encoded target array
        """
        self.label_encoder = LabelEncoder()
        y_enc = self.label_encoder.fit_transform(y)
        self.class_names = self.label_encoder.classes_
        
        logging.info(f"Clases finales: {list(self.class_names)}")
        logging.info("Target encoding con LabelEncoder (ANTES del split)")
        logging.info(f"Target: {len(y_enc)}")
        
        return y_enc

    def get_categorical_columns(self, X: pd.DataFrame) -> list[str]:
        """Get list of categorical columns present in the DataFrame.
        
        Args:
            X: Features DataFrame
            
        Returns:
            List of categorical column names
        """
        cat_cols = [col for col in CATEGORICAL_COLUMNS if col in X.columns]
        logging.info(f"Columnas categóricas a codificar: {cat_cols}")
        return cat_cols

    def split_data(
        self, X: pd.DataFrame, y: np.ndarray
    ) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
        """Split data into train and test sets.
        
        Args:
            X: Features DataFrame
            y: Target array
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
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
        """Create preprocessing pipeline.
        
        Args:
            cat_cols: List of categorical column names
            
        Returns:
            ColumnTransformer for preprocessing
        """
        return ColumnTransformer(
            transformers=[
                ("ohe", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ],
            remainder="drop",  # IMPORTANTE: elimina columnas numéricas como en el notebook
        )

    def create_pipeline(self, preprocessor: ColumnTransformer) -> Pipeline:
        """Create the full training pipeline.
        
        Args:
            preprocessor: ColumnTransformer for preprocessing
            
        Returns:
            Complete Pipeline with preprocessing and classifier
        """
        rf_params = {
            "random_state": self.model_random_state,
            "n_estimators": 20,
            "max_depth": 20,
            "min_samples_split": 15,
            "criterion": "gini",
            "class_weight": None,
            "max_features": "sqrt",
            "bootstrap": True,
            "oob_score": False,
        }
        
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", RandomForestClassifier(**rf_params)),
            ]
        )
        
        return pipeline

    def train(self, pipeline: Pipeline, X_train: pd.DataFrame, y_train: np.ndarray) -> Pipeline:
        """Train the pipeline.
        
        Args:
            pipeline: Pipeline to train
            X_train: Training features
            y_train: Training target
            
        Returns:
            Trained pipeline
        """
        pipeline.fit(X_train, y_train)
        logging.info("Pipeline (Preprocessing + RandomForest) entrenado.")
        
        if hasattr(pipeline.named_steps["classifier"], "oob_score_"):
            oob_score = pipeline.named_steps["classifier"].oob_score_
            logging.info(f"OOB Score: {oob_score:.4f}")
        
        return pipeline

    def evaluate(
        self, pipeline: Pipeline, X_test: pd.DataFrame, y_test: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, float], str]:
        """Evaluate the trained pipeline.
        
        Args:
            pipeline: Trained pipeline
            X_test: Test features
            y_test: Test target
            
        Returns:
            Tuple of (predictions, metrics dict, classification report)
        """
        y_pred = pipeline.predict(X_test)
        
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision_weighted": precision_score(
                y_test, y_pred, average="weighted", zero_division=0
            ),
            "recall_weighted": recall_score(
                y_test, y_pred, average="weighted", zero_division=0
            ),
            "f1_weighted": f1_score(
                y_test, y_pred, average="weighted", zero_division=0
            ),
        }
        
        classif_rep = classification_report(
            y_test,
            y_pred,
            target_names=self.class_names,
            zero_division=0,
        )
        
        return y_pred, metrics, classif_rep

    def save_reports(
        self, classif_rep: str, run_name: str, metrics: Dict[str, float]
    ) -> Path:
        """Save classification reports to disk.
        
        Args:
            classif_rep: Classification report string
            run_name: Name of the MLflow run
            metrics: Metrics dictionary
            
        Returns:
            Path to the saved report file
        """
        report_path = self.reports_dir / f"classification_report_{run_name}.txt"
        with open(report_path, "w", encoding="utf-8") as report_file:
            report_file.write(classif_rep)
        
        report_latest_path = self.reports_dir / "classification_report_latest.txt"
        with open(report_latest_path, "w", encoding="utf-8") as latest_file:
            latest_file.write(classif_rep)
            latest_file.write(f"\n\nRun: {run_name}\n")
            latest_file.write(f"Metrics: {metrics}\n")
            latest_file.write(f"Clases: {list(self.class_names)}\n")
        
        return report_path

    def create_confusion_matrix(
        self, y_test: np.ndarray, y_pred: np.ndarray, run_name: str
    ) -> Path:
        """Create and save confusion matrix.
        
        Args:
            y_test: True labels
            y_pred: Predicted labels
            run_name: Name of the MLflow run
            
        Returns:
            Path to the saved confusion matrix image
        """
        # Filtrar 'none' class
        if "none" in self.class_names:
            none_label = self.label_encoder.transform(["none"])[0]
            mask_no_none = y_test != none_label
            y_test_filtered = y_test[mask_no_none]
            y_pred_filtered = y_pred[mask_no_none]
        else:
            y_test_filtered = y_test
            y_pred_filtered = y_pred
        
        # Clases ordenadas (sin 'none')
        classes_ordered = ["average", "good", "vg", "excellent"]
        labels_ordered = [
            self.label_encoder.transform([class_name])[0]
            for class_name in classes_ordered
            if class_name in self.class_names
        ]
        
        cm = confusion_matrix(
            y_test_filtered,
            y_pred_filtered,
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
        plt.title("Matriz de confusión (sin 'none')", fontsize=14, fontweight="bold")
        plt.xticks(rotation=0)
        plt.yticks(rotation=0)
        
        cm_path = self.models_dir / f"conf_matrix_{run_name}.png"
        plt.savefig(cm_path, bbox_inches="tight", dpi=100)
        plt.close()
        
        logging.info(
            f"Matriz de confusión guardada en {cm_path} (sin 'none', ordenada)"
        )
        return cm_path

    def save_model(self, pipeline: Pipeline, run_name: str) -> Path:
        """Save the trained model to disk.
        
        Args:
            pipeline: Trained pipeline
            run_name: Name of the MLflow run
            
        Returns:
            Path to the saved model file
        """
        model_path = self.models_dir / f"model_{run_name}.joblib"
        joblib.dump(pipeline, model_path)
        logging.info(f"Pipeline completo guardado en {model_path}")
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
        """Log training results to MLflow.
        
        Args:
            pipeline: Trained pipeline
            X_train: Training features
            X_test: Test features
            metrics: Metrics dictionary
            cat_cols: List of categorical columns
            report_path: Path to classification report
            cm_path: Path to confusion matrix
            model_path: Path to saved model
            run_name: Name of the MLflow run
        """
        params = {
            "random_state": self.model_random_state,
            "n_estimators": 20,
            "max_depth": 20,
            "min_samples_split": 15,
            "criterion": "gini",
            "class_weight": "None",
            "max_features": "sqrt",
            "preprocessing": "OneHotEncoder_only_like_notebook",
            "categorical_features": str(cat_cols),
            "remainder": "drop",
            "based_on": "run_23_RF_GridSearch_notebook_exact_replica",
        }
        
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(report_path))
        mlflow.log_artifact(str(cm_path))
        mlflow.log_artifact(str(model_path))
        
        # Log OOB score if available
        if hasattr(pipeline.named_steps["classifier"], "oob_score_"):
            oob_score = pipeline.named_steps["classifier"].oob_score_
            mlflow.log_metric("oob_score", oob_score)
        
        # Register model
        signature = infer_signature(X_train, pipeline.predict(X_train))
        input_example = X_test.head(5)
        
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="student_model",
            signature=signature,
            input_example=input_example,
        )
        
        logging.info(f"Métricas: {metrics}")

    def run(self) -> None:
        """Execute the full training pipeline (data load, train, log)."""
        # Setup
        self._ensure_directories()
        np.random.seed(self.seed)
        logging.info("==== Entrenamiento iniciado ====")
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
        run_name = f"rf_train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
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
                print(f"Entrenamiento completado. Accuracy: {metrics['accuracy']:.4f}")
            except Exception as exc:  # noqa: BLE001
                logging.error(f"Error durante el registro en MLflow: {str(exc)}")
                raise
        
        logging.info("==== Entrenamiento completado exitosamente ====")


def main() -> None:
    """Main entry point for training script."""
    trainer = StudentPerformanceTrainer()
    trainer.run()


if __name__ == "__main__":
    main()
