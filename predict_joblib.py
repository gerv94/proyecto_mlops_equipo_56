import joblib
import pandas as pd
from mlops.models.evaluator import ModelEvaluator

model = joblib.load("models/model_latest.joblib")

X = pd.read_csv("data/interim/student_interim_preprocessed.csv")
y = pd.read_csv("data/interim/student_interim_clean.csv")["Performance"]

evaluator = ModelEvaluator()
y_pred = model.predict(X)
metrics = evaluator.evaluate_classification(y, y_pred)

print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"F1 (weighted): {metrics['f1_weighted']:.4f}")
