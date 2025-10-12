import mlflow
import mlflow.sklearn
import os
import pandas as pd

# Ruta completa al modelo entrenado
model_path = os.path.join("mlruns", "419557527186161746", "models", "m-2d309fe2933d4f2e9d83e67e4ab855f9", "artifacts")

# Cargar el modelo
loaded_model = mlflow.sklearn.load_model(model_path)
print("✅ Modelo cargado correctamente desde la ruta especificada")

# Validar el modelo con tus datos
X_test = pd.read_csv("data/interim/student_interim_preprocessed.csv")

y_test = pd.read_csv("data/interim/student_interim_clean.csv")["Performance"]

accuracy = loaded_model.score(X_test, y_test)
print(f"Accuracy del modelo cargado: {accuracy:.4f}")