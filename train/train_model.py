import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
import os
from mlops.config import PREPROCESSED_CSV, CLEAN_CSV

# ==============================
# 1. Configuración de MLflow
# ==============================
# Carpeta local donde se guardarán los registros (runs y modelos)
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("student_performance_experiment")

# ==============================
# 2. Cargar datasets
# ==============================
X = pd.read_csv(PREPROCESSED_CSV)
y = pd.read_csv(CLEAN_CSV)["Performance"]

# ==============================
# 3. Dividir datos
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==============================
# 4. Entrenar modelo
# ==============================
model = RandomForestClassifier(random_state=42)

# Inicia registro de experimento en MLflow
with mlflow.start_run():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # ==============================
    # 5. Registrar métricas y modelo
    # ==============================
    mlflow.log_param("random_state", 42)
    mlflow.log_param("n_estimators", model.n_estimators)
    mlflow.log_metric("accuracy", acc)

    # Guarda el modelo serializado (con versión)
    mlflow.sklearn.log_model(model, "model")

    print(f"Modelo entrenado y guardado con accuracy: {acc:.4f}")

