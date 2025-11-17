# ============================================================
# Entrenamiento reproducible y trazable - Fase 2 (SRE) - MLflow
# Compatible con MLflow 2.15.1 y 3.x
# ============================================================

import os
import logging
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, LabelEncoder
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
# 1. CONFIGURACIÓN GENERAL Y CLASE PRINCIPAL
# ============================================================

logging.basicConfig(
    filename="training.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

SEED = 42


class StudentPerformanceTrainer:
    """Encapsulates the full SRE training pipeline for student performance."""

    def __init__(self, seed: int = SEED) -> None:
        self.seed = seed

    def run(self) -> None:
        """Execute the full training pipeline (data load, train, log)."""

        # Ensure required directories exist
        os.makedirs("models", exist_ok=True)
        os.makedirs("reports", exist_ok=True)

        logging.info("==== Entrenamiento iniciado ====")

        # Reproducibility
        np.random.seed(self.seed)
        logging.info(f"Semilla global fijada: {self.seed}")

        # ============================================================
        # 2. CONFIGURACIÓN DE MLFLOW
        # ============================================================
        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment("student_performance_experiment_fase2")
        logging.info("Tracking MLflow configurado correctamente (Fase 2).")

        # ============================================================
        # 3. CARGA DE DATOS (SIN ENCODING - EVITA DATA LEAKAGE)
        # ============================================================
        try:
            # Cargar datos limpios (features + target en un solo archivo)
            df = pd.read_csv("data/interim/student_interim_clean.csv")

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

            # Codificar target con LabelEncoder ANTES del split (igual que el notebook)
            label_encoder = LabelEncoder()
            y_enc = label_encoder.fit_transform(y)
            class_names = label_encoder.classes_

            logging.info(f"Clases finales: {list(class_names)}")
            logging.info("Target encoding con LabelEncoder (ANTES del split)")
            logging.info(f"Features: {X.shape}, Target: {len(y_enc)}")

            # Identificar columnas categóricas y numéricas
            cat_cols_explicit = [
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
            cat_cols_to_encode = [column for column in cat_cols_explicit if column in X.columns]
            logging.info(f"Columnas categóricas a codificar: {cat_cols_to_encode}")
            num_cols = [column for column in X.columns if column not in cat_cols_to_encode]
            logging.info(f"Columnas numéricas: {len(num_cols)}")

        except Exception as exc:  # noqa: BLE001
            logging.error(f"Error al cargar datasets: {str(exc)}")
            raise

        # ============================================================
        # 4. DIVISIÓN TRAIN / TEST
        # ============================================================
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y_enc,
            test_size=0.2,
            random_state=self.seed,
            stratify=y_enc,
        )
        logging.info(f"Split hecho. Train: {X_train.shape}, Test: {X_test.shape}")

        # ============================================================
        # 5. PIPELINE DE PREPROCESAMIENTO (EVITA DATA LEAKAGE)
        # ============================================================
        preprocessor = ColumnTransformer(
            transformers=[
                ("ohe", OneHotEncoder(handle_unknown="ignore"), cat_cols_to_encode),
            ],
            remainder="drop",  # IMPORTANTE: elimina columnas numéricas como en el notebook
        )

        # Parámetros del modelo basados en run_23_RF_GridSearch (test_acc: 0.5368)
        rf_params = {
            "random_state": 888,
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

        params = {
            "random_state": 888,
            "n_estimators": 20,
            "max_depth": 20,
            "min_samples_split": 15,
            "criterion": "gini",
            "class_weight": "None",
            "max_features": "sqrt",
            "preprocessing": "OneHotEncoder_only_like_notebook",
            "categorical_features": str(cat_cols_to_encode),
            "remainder": "drop",
            "based_on": "run_23_RF_GridSearch_notebook_exact_replica",
        }

        run_name = f"rf_train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        with mlflow.start_run(run_name=run_name):
            try:
                # ----------------------------------------------------
                # ENTRENAMIENTO (PIPELINE COMPLETO)
                # ----------------------------------------------------
                pipeline.fit(X_train, y_train)
                logging.info("Pipeline (Preprocessing + RandomForest) entrenado.")

                if hasattr(pipeline.named_steps["classifier"], "oob_score_"):
                    oob_score = pipeline.named_steps["classifier"].oob_score_
                    logging.info(f"OOB Score: {oob_score:.4f}")
                    mlflow.log_metric("oob_score", oob_score)

                # ----------------------------------------------------
                # PREDICCIÓN Y MÉTRICAS
                # ----------------------------------------------------
                y_pred = pipeline.predict(X_test)
                metrics = {
                    "accuracy": accuracy_score(y_test, y_pred),
                    "precision_weighted": precision_score(
                        y_test,
                        y_pred,
                        average="weighted",
                        zero_division=0,
                    ),
                    "recall_weighted": recall_score(
                        y_test,
                        y_pred,
                        average="weighted",
                        zero_division=0,
                    ),
                    "f1_weighted": f1_score(
                        y_test,
                        y_pred,
                        average="weighted",
                        zero_division=0,
                    ),
                }

                classif_rep = classification_report(
                    y_test,
                    y_pred,
                    target_names=class_names,
                    zero_division=0,
                )
                report_path = f"reports/classification_report_{run_name}.txt"
                with open(report_path, "w", encoding="utf-8") as report_file:
                    report_file.write(classif_rep)

                report_latest_path = "reports/classification_report_latest.txt"
                with open(report_latest_path, "w", encoding="utf-8") as latest_file:
                    latest_file.write(classif_rep)
                    latest_file.write(f"\n\nRun: {run_name}\n")
                    latest_file.write(f"Metrics: {metrics}\n")
                    latest_file.write(f"Clases: {list(class_names)}\n")

                # ----------------------------------------------------
                # LOG DE PARÁMETROS, MÉTRICAS Y ARTEFACTOS
                # ----------------------------------------------------
                mlflow.log_params(params)
                mlflow.log_metrics(metrics)
                mlflow.log_artifact(report_path)
                logging.info(f"Métricas: {metrics}")

                # ----------------------------------------------------
                # MATRIZ DE CONFUSIÓN (SIN 'NONE', ORDENADA)
                # ----------------------------------------------------
                mask_no_none = y_test != label_encoder.transform(["none"])[0]
                y_test_filtered = y_test[mask_no_none]
                y_pred_filtered = y_pred[mask_no_none]

                classes_ordered = ["average", "good", "vg", "excellent"]
                labels_ordered = [
                    label_encoder.transform([class_name])[0]
                    for class_name in classes_ordered
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
                cm_path = f"models/conf_matrix_{run_name}.png"
                plt.savefig(cm_path, bbox_inches="tight", dpi=100)
                plt.close()
                mlflow.log_artifact(cm_path)
                logging.info(
                    f"Matriz de confusión guardada en {cm_path} (sin 'none', ordenada)"
                )

                # ----------------------------------------------------
                # GUARDAR MODELO LOCAL (PIPELINE COMPLETO)
                # ----------------------------------------------------
                model_path = f"models/model_{run_name}.joblib"
                joblib.dump(pipeline, model_path)
                mlflow.log_artifact(model_path)
                logging.info(f"Pipeline completo guardado en {model_path}")

                # ----------------------------------------------------
                # REGISTRO DE MODELO EN MLFLOW
                # ----------------------------------------------------
                signature = infer_signature(X_train, pipeline.predict(X_train))
                input_example = X_test.head(5)

                mlflow.sklearn.log_model(
                    sk_model=pipeline,
                    name="student_model",
                    signature=signature,
                    input_example=input_example,
                )

                print(f"Entrenamiento completado. Accuracy: {metrics['accuracy']:.4f}")

            except Exception as exc:  # noqa: BLE001
                logging.error(
                    f"Error durante el entrenamiento o registro: {str(exc)}"
                )
                raise

        logging.info("==== Entrenamiento completado exitosamente ====")


def main() -> None:
    trainer = StudentPerformanceTrainer()
    trainer.run()


if __name__ == "__main__":
    main()
