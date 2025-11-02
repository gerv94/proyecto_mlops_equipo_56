import joblib
import pandas as pd

# Carga el modelo estable escrito por training
model = joblib.load("models/model_latest.joblib")

# Carga datasets ya generados por DVC
X = pd.read_csv("data/interim/student_interim_preprocessed.csv")
y = pd.read_csv("data/interim/student_interim_clean.csv")["Performance"]

print("Accuracy:", model.score(X, y))
