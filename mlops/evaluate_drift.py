import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
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
    
    # Separar features y target (dataset completo)
    X_baseline_full = baseline_data.drop(columns=["Performance"])
    y_baseline_full = baseline_data["Performance"]
    X_drift_full = drift_data.drop(columns=["Performance"])
    y_drift_full = drift_data["Performance"]
    
    # Usar el mismo esquema de split que train_model_sre.py
    # test_size=0.2, random_state=42, stratify en y
    X_base_train, X_base_test, y_base_train, y_base_test = train_test_split(
        X_baseline_full,
        y_baseline_full,
        test_size=0.2,
        random_state=42,
        stratify=y_baseline_full,
    )
    
    # Mantener los mismos índices para baseline y drift (drift_data es copia de baseline)
    test_idx = X_base_test.index
    X_drift_test = X_drift_full.loc[test_idx]
    y_drift_test = y_drift_full.loc[test_idx]
    
    # Codificar target (el modelo predice números, no strings)
    label_encoder = LabelEncoder()
    label_encoder.fit(y_baseline_full)  # Fit en todos los datos baseline
    y_base_test_encoded = label_encoder.transform(y_base_test)
    y_drift_test_encoded = label_encoder.transform(y_drift_test)
    
    logger.info(f"Clases detectadas: {label_encoder.classes_}")
    
    # Predicciones (el modelo ya devuelve números)
    y_pred_baseline = model.predict(X_base_test)
    y_pred_drift = model.predict(X_drift_test)
    
    # Métricas sobre el mismo test set que train_model_sre.py
    acc_baseline = accuracy_score(y_base_test_encoded, y_pred_baseline)
    acc_drift = accuracy_score(y_drift_test_encoded, y_pred_drift)
    f1_baseline = f1_score(
        y_base_test_encoded, y_pred_baseline, average="weighted", zero_division=0
    )
    f1_drift = f1_score(
        y_drift_test_encoded, y_pred_drift, average="weighted", zero_division=0
    )
    
    # Calcular drops
    acc_drop = (acc_baseline - acc_drift) / acc_baseline * 100 if acc_baseline > 0 else 0
    f1_drop = (f1_baseline - f1_drift) / f1_baseline * 100 if f1_baseline > 0 else 0
    
    # Calcular PSI para features categóricas (usando TODO el dataset)
    categorical_cols = X_baseline_full.select_dtypes(include=["object"]).columns
    psi_results = {}
    
    for col in categorical_cols:
        try:
            psi = calculate_psi_categorical(X_baseline_full[col], X_drift_full[col])
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
NOTA: Dataset limpio sin registros 'none'

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
    
    with open("reports/drift/drift_evaluation_report.txt", "w", encoding="utf-8") as f:
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
    
    # Visualización adicional: distribuciones de features baseline vs drift
    num_features = len(categorical_cols)
    if num_features > 0:
        n_cols = 3
        n_rows = int(np.ceil(num_features / n_cols))
        fig_feat, axes_feat = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        axes_array = np.array(axes_feat).reshape(-1)
        
        for ax, col in zip(axes_array, categorical_cols):
            base_dist = X_baseline_full[col].value_counts(normalize=True, sort=False)
            drift_dist = X_drift_full[col].value_counts(normalize=True, sort=False)
            categories = sorted(set(base_dist.index).union(set(drift_dist.index)))
            base_vals = [base_dist.get(cat, 0.0) for cat in categories]
            drift_vals = [drift_dist.get(cat, 0.0) for cat in categories]
            x_vals = np.arange(len(categories))
            width_bar = 0.4
            
            ax.bar(
                x_vals - width_bar / 2,
                base_vals,
                width_bar,
                label="Baseline",
                color="steelblue",
                alpha=0.7,
            )
            ax.bar(
                x_vals + width_bar / 2,
                drift_vals,
                width_bar,
                label="Drift",
                color="orange",
                alpha=0.7,
            )
            ax.set_title(col)
            ax.set_xticks(x_vals)
            ax.set_xticklabels(categories, rotation=45, ha="right")
            ax.set_ylabel("Proporción")
            ax.grid(axis="y", alpha=0.3)
        
        # Ocultar ejes sobrantes si los hay
        for ax in axes_array[num_features:]:
            ax.axis("off")
        
        handles, labels = axes_array[0].get_legend_handles_labels()
        fig_feat.legend(handles, labels, loc="upper right")
        fig_feat.tight_layout()
        plt.savefig(
            "reports/drift/feature_distributions_baseline_vs_drift.png",
            dpi=150,
            bbox_inches="tight",
        )
        logger.info(
            "Gráfico de distribuciones baseline vs drift guardado en "
            "reports/drift/feature_distributions_baseline_vs_drift.png"
        )
    
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