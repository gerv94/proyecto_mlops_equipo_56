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
# 3. CARGA DE DATOS (SIN ENCODING - EVITA DATA LEAKAGE)
# ============================================================
try:
    # Cargar datos limpios (features + target en un solo archivo)
    df = pd.read_csv("data/interim/student_interim_clean.csv")
    
    logging.info(f"Datos originales cargados. Dataset completo: {df.shape}")
    logging.info(f"Distribución original de clases: {df['Performance'].value_counts().to_dict()}")
    
    # Separar features y target (IGUAL QUE EL NOTEBOOK)
    X = df.drop(columns=['Performance'])
    y = df['Performance']
    
    # Eliminar columna 'mixed_type_col' si existe (generada durante limpieza)
    if 'mixed_type_col' in X.columns:
        X = X.drop(columns=['mixed_type_col'])
        logging.info("Columna 'mixed_type_col' eliminada del dataset")
    
    # Codificar target con LabelEncoder ANTES del split (igual que el notebook)
    label_encoder = LabelEncoder()
    y_enc = label_encoder.fit_transform(y)
    class_names = label_encoder.classes_
    
    logging.info(f"Clases finales: {list(class_names)}")
    logging.info(f"Target encoding con LabelEncoder (ANTES del split)")
    logging.info(f"Features: {X.shape}, Target: {len(y_enc)}")

    # Identificar columnas categóricas y numéricas
    cat_cols_explicit = ['Gender', 'Caste', 'coaching', 'time', 'Class_ten_education', 
                         'twelve_education', 'medium', 'Class_ X_Percentage', 
                         'Class_XII_Percentage', 'Father_occupation', 'Mother_occupation']
    cat_cols_to_encode = [c for c in cat_cols_explicit if c in X.columns]
    logging.info(f"Columnas categóricas a codificar: {cat_cols_to_encode}")
    num_cols = [c for c in X.columns if c not in cat_cols_to_encode]
    logging.info(f"Columnas numéricas: {len(num_cols)}")

except Exception as e:
    logging.error(f"Error al cargar datasets: {str(e)}")
    raise e

# ============================================================
# 4. DIVISIÓN TRAIN / TEST
# ============================================================
# Split con random_state=42 y stratify sobre el target codificado
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=SEED, stratify=y_enc
)
logging.info(f"Split hecho. Train: {X_train.shape}, Test: {X_test.shape}")

# ============================================================
# 5. PIPELINE DE PREPROCESAMIENTO (EVITA DATA LEAKAGE)
# ============================================================
# Preprocessor simple: OneHotEncoder para TODAS las categóricas, drop el resto
# (replica exactamente el comportamiento del notebook)
preprocessor = ColumnTransformer(
    transformers=[('ohe', OneHotEncoder(handle_unknown='ignore'), cat_cols_to_encode)],
    remainder='drop'  # IMPORTANTE: elimina columnas numéricas como en el notebook
)

# Parámetros del modelo basados en run_23_RF_GridSearch (test_acc: 0.5368)
# IMPORTANTE: El notebook usa random_state=888, NO 42
rf_params = {
    "random_state": 888,           # CLAVE: Notebook usa 888
    "n_estimators": 20,            # Basado en run_23_RF_GridSearch
    "max_depth": 20,               # Basado en run_23_RF_GridSearch
    "min_samples_split": 15,       # Basado en run_23_RF_GridSearch
    "criterion": "gini",
    "class_weight": None,          # Sin balanceo forzado
    "max_features": "sqrt",       # Reduce correlación entre árboles
    "bootstrap": True,
    "oob_score": False             # Notebook no usa OOB
}

# Crear pipeline completo (preprocessing + modelo)
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(**rf_params))
])

# Parámetros para logging (sin prefijo 'classifier__')
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
    "based_on": "run_23_RF_GridSearch_notebook_exact_replica"
}
run_name = f"rf_train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

with mlflow.start_run(run_name=run_name):
    try:
        # ------------------------------------------------------------
        # ENTRENAMIENTO (PIPELINE COMPLETO)
        # ------------------------------------------------------------
        # El pipeline aplica OneHotEncoding SOLO en train durante fit
        pipeline.fit(X_train, y_train)
        logging.info("Pipeline (Preprocessing + RandomForest) entrenado.")
        
        # Log OOB score si está disponible
        if hasattr(pipeline.named_steps['classifier'], 'oob_score_'):
            oob_score = pipeline.named_steps['classifier'].oob_score_
            logging.info(f"OOB Score: {oob_score:.4f}")
            mlflow.log_metric("oob_score", oob_score)

        # ------------------------------------------------------------
        # PREDICCIÓN Y MÉTRICAS
        # ------------------------------------------------------------
        y_pred = pipeline.predict(X_test)
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision_weighted": precision_score(y_test, y_pred, average="weighted", zero_division=0),
            "recall_weighted": recall_score(y_test, y_pred, average="weighted", zero_division=0),
            "f1_weighted": f1_score(y_test, y_pred, average="weighted", zero_division=0)
        }

        classif_rep = classification_report(y_test, y_pred, target_names=class_names, zero_division=0)
        report_path = f"reports/classification_report_{run_name}.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(classif_rep)
        
        # Guardar también con nombre fijo para DVC tracking
        report_latest_path = "reports/classification_report_latest.txt"
        with open(report_latest_path, "w", encoding="utf-8") as f:
            f.write(classif_rep)
            f.write(f"\n\nRun: {run_name}\n")
            f.write(f"Metrics: {metrics}\n")
            f.write(f"Clases: {list(class_names)}\n")

        # ------------------------------------------------------------
        # LOG DE PARÁMETROS, MÉTRICAS Y ARTEFACTOS
        # ------------------------------------------------------------
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(report_path)
        logging.info(f"Métricas: {metrics}")

        # ------------------------------------------------------------
        # MATRIZ DE CONFUSIÓN CON ETIQUETAS (SIN 'NONE', ORDENADA)
        # ------------------------------------------------------------
        # Filtrar 'none' de las predicciones y valores reales para la matriz
        mask_no_none = y_test != label_encoder.transform(['none'])[0]
        y_test_filtered = y_test[mask_no_none]
        y_pred_filtered = y_pred[mask_no_none]
        
        # Clases sin 'none' en el orden deseado: average, good, vg, excellent
        classes_ordered = ['average', 'good', 'vg', 'excellent']
        labels_ordered = [label_encoder.transform([c])[0] for c in classes_ordered]
        
        # Matriz de confusión con orden específico
        cm = confusion_matrix(y_test_filtered, y_pred_filtered, labels=labels_ordered)
        
        # Crear figura más pequeña para mejor visualización
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                    xticklabels=classes_ordered, yticklabels=classes_ordered,
                    cbar_kws={'label': 'Cantidad'})
        plt.xlabel("Predicción", fontsize=13, fontweight='bold')
        plt.ylabel("Real", fontsize=13, fontweight='bold')
        plt.title("Matriz de confusión (sin 'none')", fontsize=14, fontweight='bold')
        plt.xticks(rotation=0)  # Horizontal para mejor lectura
        plt.yticks(rotation=0)
        cm_path = f"models/conf_matrix_{run_name}.png"
        plt.savefig(cm_path, bbox_inches="tight", dpi=100)
        plt.close()
        mlflow.log_artifact(cm_path)
        logging.info(f"Matriz de confusión guardada en {cm_path} (sin 'none', ordenada)")

        # ------------------------------------------------------------
        # GUARDAR MODELO LOCAL (joblib) - AHORA ES EL PIPELINE COMPLETO
        # ------------------------------------------------------------
        model_path = f"models/model_{run_name}.joblib"
        joblib.dump(pipeline, model_path)
        mlflow.log_artifact(model_path)
        logging.info(f"Pipeline completo guardado en {model_path}")

        # ------------------------------------------------------------
        # REGISTRO DE MODELO EN MLFLOW (COMPATIBLE 2.x Y 3.x)
        # ------------------------------------------------------------
        signature = infer_signature(X_train, pipeline.predict(X_train))
        input_example = X_test.head(5)

        mlflow_version = tuple(map(int, mlflow.__version__.split(".")[:2]))

        if mlflow_version >= (3, 0):
            # MLflow 3.x
            mlflow.sklearn.log_model(
                sk_model=pipeline,
                name="student_model",
                signature=signature,
                input_example=input_example
            )
        else:
            # MLflow 2.x
            mlflow.sklearn.log_model(
                sk_model=pipeline,
                artifact_path="student_model",
                signature=signature,
                input_example=input_example
            )

        print(f"Entrenamiento completado. Accuracy: {metrics['accuracy']:.4f}")

    except Exception as e:
        logging.error(f"Error durante el entrenamiento o registro: {str(e)}")
        raise e

logging.info("==== Entrenamiento completado exitosamente ====")