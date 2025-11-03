import joblib
import pandas as pd
from mlops.models.evaluator import ModelEvaluator
from mlops.config import PREPROCESSED_CSV, CLEAN_CSV

model = joblib.load("models/model_latest.joblib")

X = pd.read_csv(PREPROCESSED_CSV)
y = pd.read_csv(CLEAN_CSV)["Performance"]

evaluator = ModelEvaluator()
y_pred = model.predict(X)
metrics = evaluator.evaluate_classification(y, y_pred)

print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"F1 (weighted): {metrics['f1_weighted']:.4f}")
