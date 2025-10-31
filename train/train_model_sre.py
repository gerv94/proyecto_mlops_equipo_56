# ============================================================
# Entrenamiento reproducible y trazable - Fase 2 (SRE) - MLflow 3.x
# ============================================================

import os
import logging
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
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
# 1. CONFIGURACIÓN GENERAL
# ============================================================

os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)

logging.basicConfig(
    filename="training.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.info("==== Entrenamiento iniciado ====")

SEED = 42
np.random.seed(SEED)
logging.info(f"Semilla global fijada: {SEED}")

# ============================================================
# 2. CONFIGURACIÓN DE MLFLOW
# ============================================================

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("student_performance_experiment_fase2")
logging.info("Tracking MLflow configurado correctamente (Fase 2).")

# ============================================================
# 3. CARGA DE DATOS
# ============================================================

try:
    X = pd.read_csv("data/interim/student_interim_preprocessed.csv")
    y = pd.read_csv("data/interim/student_interim_clean.csv")["Performance"]
    logging.info(f"Datos cargados. X: {X.shape}, y: {y.shape}")
except Exception as e:
    logging.error(f"Error al cargar datasets: {str(e)}")
    raise e

# ============================================================
# 4. DIVISIÓN TRAIN / TEST
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)
logging.info(f"Split hecho. Train: {X_train.shape}, Test: {X_test.shape}")

# ============================================================
# 5. ENTRENAMIENTO Y TRACKING MLFLOW
# ============================================================

params = {
    "random_state": SEED,
    "n_estimators": 150,
    "max_depth": 8,
    "criterion": "gini"
}

run_name = f"rf_train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
with mlflow.start_run(run_name=run_name):
    try:
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        logging.info("Modelo RandomForest entrenado.")

        y_pred = model.predict(X_test)

        # Métricas con zero_division=0 para evitar warnings cuando hay clases sin predicciones
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision_weighted": precision_score(y_test, y_pred, average="weighted", zero_division=0),
            "recall_weighted": recall_score(y_test, y_pred, average="weighted", zero_division=0),
            "f1_weighted": f1_score(y_test, y_pred, average="weighted", zero_division=0)
        }

        # Reporte por clase (útil para ver qué clase no se predijo)
        classif_rep = classification_report(y_test, y_pred, zero_division=0)
        report_path = f"reports/classification_report_{run_name}.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(classif_rep)

        # Log de parámetros y métricas
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(report_path)
        logging.info(f"Métricas: {metrics}")

        # Matriz de confusión
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicción")
        plt.ylabel("Real")
        plt.title("Matriz de confusión")
        cm_path = f"models/conf_matrix_{run_name}.png"
        plt.savefig(cm_path, bbox_inches="tight")
        plt.close()
        mlflow.log_artifact(cm_path)
        logging.info(f"Matriz de confusión guardada en {cm_path}")

        # Guardar modelo como artefacto (joblib)
        model_path = f"models/model_{run_name}.joblib"
        joblib.dump(model, model_path)
        mlflow.log_artifact(model_path)
        logging.info(f"Modelo guardado en {model_path}")

        # Firma del modelo e input_example (MLflow 3.x)
        # name= sustituye a artifact_path=
        signature = infer_signature(X_train, model.predict(X_train))
        input_example = X_test.head(5)
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            signature=signature,
            input_example=input_example
        )

        print(f"Entrenamiento completado. Accuracy: {metrics['accuracy']:.4f}")

    except Exception as e:
        logging.error(f"Error durante el entrenamiento o registro: {str(e)}")
        raise e

logging.info("==== Entrenamiento completado exitosamente ====")
