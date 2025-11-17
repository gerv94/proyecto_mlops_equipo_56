import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
import json
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_psi_categorical(expected, actual):
    """Calcula Population Stability Index (PSI) para variables categóricas"""
    # Obtener distribuciones
    expected_percents = expected.value_counts(normalize=True, sort=False)
    actual_percents = actual.value_counts(normalize=True, sort=False)
    
    # Alinear índices
    all_categories = set(expected_percents.index).union(set(actual_percents.index))
    expected_aligned = expected_percents.reindex(all_categories, fill_value=1e-6)
    actual_aligned = actual_percents.reindex(all_categories, fill_value=1e-6)
    
    # Calcular PSI
    psi_value = np.sum((actual_aligned - expected_aligned) * 
                       np.log(actual_aligned / expected_aligned))
    return psi_value

def evaluate_drift():
    logger.info("Evaluando drift en el modelo...")
    
    # Cargar modelo
    model = joblib.load("models/best_gridsearch_amplio.joblib")
    
    # Cargar datos
    baseline_data = pd.read_csv("data/interim/student_interim_clean.csv")
    drift_data = pd.read_csv("data/interim/student_drift_data.csv")
    
    # Separar features y target
    X_baseline = baseline_data.drop(columns=['Performance'])
    y_baseline = baseline_data['Performance']
    X_drift = drift_data.drop(columns=['Performance'])
    y_drift = drift_data['Performance']
    
    # Codificar target (el modelo predice números, no strings)
    label_encoder = LabelEncoder()
    label_encoder.fit(y_baseline)  # Fit en baseline
    y_baseline_encoded = label_encoder.transform(y_baseline)
    y_drift_encoded = label_encoder.transform(y_drift)
    
    logger.info(f"Clases detectadas: {label_encoder.classes_}")
    
    # Predicciones (el modelo ya devuelve números)
    y_pred_baseline = model.predict(X_baseline)
    y_pred_drift = model.predict(X_drift)
    
    # Métricas
    acc_baseline = accuracy_score(y_baseline_encoded, y_pred_baseline)
    acc_drift = accuracy_score(y_drift_encoded, y_pred_drift)
    f1_baseline = f1_score(y_baseline_encoded, y_pred_baseline, average='weighted', zero_division=0)
    f1_drift = f1_score(y_drift_encoded, y_pred_drift, average='weighted', zero_division=0)
    
    # Calcular drops
    acc_drop = (acc_baseline - acc_drift) / acc_baseline * 100 if acc_baseline > 0 else 0
    f1_drop = (f1_baseline - f1_drift) / f1_baseline * 100 if f1_baseline > 0 else 0
    
    # Calcular PSI para features categóricas
    categorical_cols = X_baseline.select_dtypes(include=['object']).columns
    psi_results = {}
    
    for col in categorical_cols:
        try:
            psi = calculate_psi_categorical(X_baseline[col], X_drift[col])
            psi_results[col] = psi
        except Exception as e:
            logger.warning(f"No se pudo calcular PSI para {col}: {e}")
            psi_results[col] = None
    
    # Generar reporte
    os.makedirs("reports/drift", exist_ok=True)
    
    report = f"""
=================================================================
REPORTE DE EVALUACIÓN DE DRIFT
=================================================================

CLASES DEL MODELO: {', '.join(label_encoder.classes_)}

MÉTRICAS BASELINE:
  - Accuracy: {acc_baseline:.4f}
  - F1-Score: {f1_baseline:.4f}

MÉTRICAS CON DRIFT:
  - Accuracy: {acc_drift:.4f}
  - F1-Score: {f1_drift:.4f}

DEGRADACIÓN:
  - Accuracy drop: {acc_drop:.2f}%
  - F1-Score drop: {f1_drop:.2f}%

PSI POR FEATURE (Population Stability Index):
"""
    
    for col, psi in psi_results.items():
        if psi is not None:
            status = "🟢 OK" if abs(psi) < 0.1 else ("🟡 MODERADO" if abs(psi) < 0.25 else "🔴 ALTO")
            report += f"  - {col}: {psi:.4f} {status}\n"
    
    report += "\nALERTAS:\n"
    
    if acc_drop > 10:
        report += "  🔴 ALERTA ROJA: Accuracy drop > 10% - RETRAINING URGENTE\n"
    elif acc_drop > 5:
        report += "  🟡 ALERTA AMARILLA: Accuracy drop > 5% - Revisión recomendada\n"
    else:
        report += "  🟢 OK: Drift dentro de umbrales aceptables\n"
    
    logger.info(report)
    
    with open("reports/drift/drift_evaluation_report.txt", "w") as f:
        f.write(report)
    
    # Visualizaciones
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Gráfico de métricas
    metrics = ['Accuracy', 'F1-Score']
    baseline_vals = [acc_baseline, f1_baseline]
    drift_vals = [acc_drift, f1_drift]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    axes[0].bar(x - width/2, baseline_vals, width, label='Baseline', color='green', alpha=0.7)
    axes[0].bar(x + width/2, drift_vals, width, label='Drift', color='red', alpha=0.7)
    axes[0].set_ylabel('Score')
    axes[0].set_title('Comparación de Métricas: Baseline vs Drift')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metrics)
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)
    
    # Distribución de predicciones (convertir a nombres de clases)
    y_pred_baseline_labels = label_encoder.inverse_transform(y_pred_baseline)
    y_pred_drift_labels = label_encoder.inverse_transform(y_pred_drift)
    
    pred_baseline_dist = pd.Series(y_pred_baseline_labels).value_counts(normalize=True)
    pred_drift_dist = pd.Series(y_pred_drift_labels).value_counts(normalize=True)
    
    classes = sorted(list(set(list(pred_baseline_dist.index) + list(pred_drift_dist.index))))
    baseline_probs = [pred_baseline_dist.get(c, 0) for c in classes]
    drift_probs = [pred_drift_dist.get(c, 0) for c in classes]
    
    x2 = np.arange(len(classes))
    axes[1].bar(x2 - width/2, baseline_probs, width, label='Baseline', color='blue', alpha=0.7)
    axes[1].bar(x2 + width/2, drift_probs, width, label='Drift', color='orange', alpha=0.7)
    axes[1].set_ylabel('Proporción')
    axes[1].set_title('Distribución de Predicciones')
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(classes, rotation=45)
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("reports/drift/drift_plots.png", dpi=150, bbox_inches='tight')
    logger.info("Gráficos guardados en reports/drift/drift_plots.png")
    
    return {
        "acc_baseline": float(acc_baseline),
        "acc_drift": float(acc_drift),
        "acc_drop_pct": float(acc_drop),
        "f1_baseline": float(f1_baseline),
        "f1_drift": float(f1_drift),
        "f1_drop_pct": float(f1_drop),
        "psi_features": {k: float(v) if v is not None else None for k, v in psi_results.items()}
    }

if __name__ == "__main__":
    results = evaluate_drift()
    print(json.dumps(results, indent=2))