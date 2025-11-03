import pandas as pd
from sklearn.model_selection import train_test_split
from mlops.models.model_factory import make_estimator
from mlops.models.evaluator import ModelEvaluator

X = pd.read_csv("data/interim/student_interim_preprocessed.csv")
y = pd.read_csv("data/interim/student_interim_clean.csv")["Performance"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = make_estimator("random_forest", {"random_state": 42})
model.fit(X_train, y_train)

evaluator = ModelEvaluator()
y_pred = model.predict(X_test)
metrics = evaluator.evaluate_classification(y_test, y_pred)

print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"F1 (weighted): {metrics['f1_weighted']:.4f}")
