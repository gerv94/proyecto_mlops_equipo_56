# ============================================================
# Grid Search Amplio - Optimización de Hiperparámetros
# RandomForestClassifier - Student Performance
# Partiendo del modelo base: n_estimators=20, max_depth=20, min_samples_split=15, random_state=888
# ============================================================

import os
import logging
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import mlflow
import mlflow.sklearn
import joblib

# ============================================================
# 1. CONFIGURACIÓN GENERAL
# ============================================================
os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)

logging.basicConfig(
    filename="gridsearch.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.info("==== Grid Search Amplio iniciado ====")

SEED = 42
np.random.seed(SEED)
logging.info(f"Semilla para split: {SEED}")

# ============================================================
# 2. CONFIGURACIÓN DE MLFLOW
# ============================================================
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("student_performance_gridsearch_amplio")
logging.info("Tracking MLflow configurado para Grid Search Amplio.")

# ============================================================
# 3. CARGA DE DATOS (IGUAL QUE train_model_sre.py)
# ============================================================
try:
    df = pd.read_csv("data/interim/student_interim_clean.csv")
    
    logging.info(f"Datos cargados. Dataset completo: {df.shape}")
    logging.info(f"Distribución de clases: {df['Performance'].value_counts().to_dict()}")
    
    # Separar features y target (IGUAL QUE EL NOTEBOOK)
    X = df.drop(columns=['Performance'])
    y = df['Performance']
    
    # Eliminar columna 'mixed_type_col' si existe
    if 'mixed_type_col' in X.columns:
        X = X.drop(columns=['mixed_type_col'])
        logging.info("Columna 'mixed_type_col' eliminada")
    
    # Codificar target con LabelEncoder ANTES del split (igual que el modelo actual)
    label_encoder = LabelEncoder()
    y_enc = label_encoder.fit_transform(y)
    class_names = label_encoder.classes_
    
    logging.info(f"Clases finales: {list(class_names)}")
    logging.info(f"Features: {X.shape}, Target: {len(y_enc)}")
    
    # Lista explícita de columnas categóricas (igual que train_model_sre.py)
    cat_cols_explicit = ['Gender', 'Caste', 'coaching', 'time', 'Class_ten_education', 
                         'twelve_education', 'medium', 'Class_ X_Percentage', 
                         'Class_XII_Percentage', 'Father_occupation', 'Mother_occupation']
    cat_cols_to_encode = [c for c in cat_cols_explicit if c in X.columns]
    logging.info(f"Columnas categóricas: {cat_cols_to_encode}")
    
except Exception as e:
    logging.error(f"Error al cargar datasets: {str(e)}")
    raise e

# ============================================================
# 4. DIVISIÓN TRAIN / TEST
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=SEED, stratify=y_enc
)
logging.info(f"Split hecho. Train: {X_train.shape}, Test: {X_test.shape}")

# ============================================================
# 5. PIPELINE DE PREPROCESAMIENTO (IGUAL QUE train_model_sre.py)
# ============================================================
preprocessor = ColumnTransformer(
    transformers=[('ohe', OneHotEncoder(handle_unknown='ignore'), cat_cols_to_encode)],
    remainder='drop'  # IMPORTANTE: igual que el modelo actual
)

# ============================================================
# 6. GRILLA DE BÚSQUEDA AMPLIA
# ============================================================
# Modelo base: n_estimators=20, max_depth=20, min_samples_split=15, random_state=888
# Exploraremos valores alrededor de estos parámetros

param_grid = {
    # Explorar número de árboles con pasos pequeños alrededor de 20
    'classifier__n_estimators': [10, 15, 18, 20, 22, 25, 30, 35],  # 8 valores (pasos de 2-5)
    
    # Explorar profundidad con pasos pequeños alrededor de 20
    'classifier__max_depth': [15, 18, 20, 22, 25, 28],  # 6 valores (pasos de 2-3)
    
    # Explorar min_samples_split con pasos pequeños alrededor de 15
    'classifier__min_samples_split': [8, 10, 12, 15, 18, 20],  # 6 valores (pasos de 2-3)
    
    # Explorar min_samples_leaf con más granularidad
    'classifier__min_samples_leaf': [1, 2, 3],  # 3 valores
    
    # Explorar max_features con más opciones
    'classifier__max_features': ['sqrt', 'log2', 0.6, 0.8],  # 4 valores
    
    # Criterio de split (ambos para comparar)
    'classifier__criterion': ['gini', 'entropy'],  # 2 valores
    
    # Class weight (sin balanced para acelerar)
    'classifier__class_weight': [None]  # 1 valor (el actual exitoso)
}

total_combinations = np.prod([len(v) for v in param_grid.values()])
logging.info(f"Grilla definida con {total_combinations} combinaciones")

print("\n" + "="*70)
print("GRID SEARCH GRANULAR - OPTIMIZACIÓN FINA DE HIPERPARÁMETROS")
print("="*70)
print(f"Modelo base: n_estimators=20, max_depth=20, min_samples_split=15")
print(f"Random state modelo: 888 (igual que el modelo actual exitoso)")
print(f"Total de combinaciones: {total_combinations}")
print(f"Tiempo estimado: ~30-40 minutos")
print(f"CV folds: 3 (reducido para acelerar)")
print(f"Estrategia: Pasos pequeños alrededor de parámetros exitosos")
print(f"Métrica principal: f1_weighted")
print(f"Preprocessing: OneHotEncoder (remainder='drop')")
print("="*70 + "\n")

# Pipeline completo
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=888, bootstrap=True))  # random_state=888
])

# ============================================================
# 7. CONFIGURAR Y EJECUTAR GRID SEARCH
# ============================================================
scoring = {
    'accuracy': 'accuracy',
    'f1_weighted': 'f1_weighted',
    'f1_macro': 'f1_macro',
    'precision_weighted': 'precision_weighted',
    'recall_weighted': 'recall_weighted'
}

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring=scoring,
    refit='f1_weighted',
    cv=3,  # Reducido de 5 a 3 para acelerar (~40% más rápido)
    verbose=2,
    n_jobs=-1,
    return_train_score=True
)

logging.info("Iniciando Grid Search...")
print("Iniciando búsqueda... (esto puede tomar varios minutos)\n")

grid_search.fit(X_train, y_train)

logging.info("Grid Search completado.")
logging.info(f"Mejor CV F1-Weighted: {grid_search.best_score_:.4f}")
logging.info(f"Mejores parámetros: {grid_search.best_params_}")

print("\n" + "="*70)
print("GRID SEARCH COMPLETADO")
print("="*70)
print(f"Mejor CV F1-Weighted: {grid_search.best_score_:.4f}")
print(f"\nMejores parámetros:")
for param, value in grid_search.best_params_.items():
    print(f"  {param.replace('classifier__', '')}: {value}")
print("="*70 + "\n")

# ============================================================
# 8. REGISTRAR TOP 10 MODELOS EN MLFLOW
# ============================================================
print("Registrando top 10 modelos en MLflow...\n")

# Ordenar resultados por f1_weighted
results_df = pd.DataFrame(grid_search.cv_results_)
results_df = results_df.sort_values('mean_test_f1_weighted', ascending=False)
top_10_indices = results_df.head(10).index

for rank, idx in enumerate(top_10_indices, 1):
    params = grid_search.cv_results_['params'][idx]
    
    with mlflow.start_run(run_name=f"gridsearch_rank_{rank}", nested=False):
        # Limpiar nombres de parámetros
        clean_params = {k.replace('classifier__', ''): v for k, v in params.items()}
        clean_params['random_state'] = 888
        
        # Reconstruir pipeline
        current_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(**clean_params))
        ])
        
        # Entrenar
        current_pipeline.fit(X_train, y_train)
        
        # Predicciones en test
        y_pred_test = current_pipeline.predict(X_test)
        
        # Métricas
        test_acc = accuracy_score(y_test, y_pred_test)
        test_f1_weighted = f1_score(y_test, y_pred_test, average='weighted')
        test_f1_macro = f1_score(y_test, y_pred_test, average='macro')
        test_precision = precision_score(y_test, y_pred_test, average='weighted', zero_division=0)
        test_recall = recall_score(y_test, y_pred_test, average='weighted', zero_division=0)
        
        # Log parámetros
        mlflow.log_params(clean_params)
        mlflow.log_param('preprocessing', 'OneHotEncoder_only_like_notebook')
        mlflow.log_param('rank', rank)
        
        # Log métricas
        metrics = {
            'cv_accuracy_mean': grid_search.cv_results_['mean_test_accuracy'][idx],
            'cv_accuracy_std': grid_search.cv_results_['std_test_accuracy'][idx],
            'cv_f1_weighted_mean': grid_search.cv_results_['mean_test_f1_weighted'][idx],
            'cv_f1_weighted_std': grid_search.cv_results_['std_test_f1_weighted'][idx],
            'cv_f1_macro_mean': grid_search.cv_results_['mean_test_f1_macro'][idx],
            'test_accuracy': test_acc,
            'test_f1_weighted': test_f1_weighted,
            'test_f1_macro': test_f1_macro,
            'test_precision_weighted': test_precision,
            'test_recall_weighted': test_recall
        }
        mlflow.log_metrics(metrics)
        
        # Marcar el mejor
        if rank == 1:
            mlflow.set_tag('best_model', 'True')
            logging.info(f"Mejor modelo: rank {rank}")
            
            print(f"\n{'='*70}")
            print(f"*** MEJOR MODELO (Rank {rank}) ***")
            print(f"{'='*70}")
            print(f"CV F1-Weighted: {metrics['cv_f1_weighted_mean']:.4f} ± {metrics['cv_f1_weighted_std']:.4f}")
            print(f"Test Accuracy:  {test_acc:.4f}")
            print(f"Test F1-Weighted: {test_f1_weighted:.4f}")
            print(f"Test F1-Macro:    {test_f1_macro:.4f}")
            print(f"\nParámetros:")
            for k, v in clean_params.items():
                if k != 'random_state':
                    print(f"  {k}: {v}")
            print(f"{'='*70}\n")
            
            # Guardar mejor modelo
            best_model_path = "models/best_gridsearch_amplio.joblib"
            joblib.dump(current_pipeline, best_model_path)
            mlflow.log_artifact(best_model_path)
            mlflow.sklearn.log_model(current_pipeline, "best_model")
        else:
            print(f"Rank {rank}: CV F1={metrics['cv_f1_weighted_mean']:.4f}, Test Acc={test_acc:.4f}, Test F1={test_f1_weighted:.4f}")

print(f"\nTop 10 modelos registrados en MLflow.")
logging.info("Top 10 modelos registrados en MLflow.")

print("\n" + "="*70)
print("GRID SEARCH FINALIZADO")
print("Revisa MLflow UI para comparar todos los modelos")
print("Experimento: student_performance_gridsearch_amplio")
print("="*70)