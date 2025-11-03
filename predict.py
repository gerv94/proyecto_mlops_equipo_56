import os
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
from mlops.models.evaluator import ModelEvaluator
from mlops.config import PREPROCESSED_CSV, CLEAN_CSV

# Tracking local y experimento (igual que en training)
mlflow.set_tracking_uri("file:./mlruns")
EXP_NAME = "student_performance_experiment_fase2"

client = MlflowClient()
exp = client.get_experiment_by_name(EXP_NAME)
assert exp is not None, f"Experimento no existe: {EXP_NAME}"

# Último run (más reciente)
runs = client.search_runs(
    experiment_ids=[exp.experiment_id],
    order_by=["attributes.end_time DESC"],
    max_results=1,
)
assert runs, f"No hay runs en el experimento {EXP_NAME}"
run_id = runs[0].info.run_id
print("Usando run_id:", run_id)

# Encontrar subcarpeta con archivo MLmodel (p.ej. student_model)
candidate_names = ["student_model", "model", "sklearn-model"]
artifact_subpath = None

for name in candidate_names:
    try:
        client.list_artifacts(run_id, name)
        artifact_subpath = name
        break
    except Exception:
        pass

if artifact_subpath is None:
    root_items = client.list_artifacts(run_id)
    for it in root_items:
        if it.is_dir:
            try:
                inner = client.list_artifacts(run_id, it.path)
                if any(os.path.basename(x.path) == "MLmodel" for x in inner):
                    artifact_subpath = it.path
                    break
            except Exception:
                continue

assert artifact_subpath is not None, "No encontré un artifact de modelo con archivo MLmodel en el run."

# Cargar por URI MLflow
model_uri = f"runs:/{run_id}/{artifact_subpath}"
print("Model URI:", model_uri)
loaded_model = mlflow.sklearn.load_model(model_uri)
print("✅ Modelo cargado correctamente")

X = pd.read_csv(PREPROCESSED_CSV)
y = pd.read_csv(CLEAN_CSV)["Performance"]

evaluator = ModelEvaluator()
y_pred = loaded_model.predict(X)
metrics = evaluator.evaluate_classification(y, y_pred)

print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"F1 (weighted): {metrics['f1_weighted']:.4f}")
