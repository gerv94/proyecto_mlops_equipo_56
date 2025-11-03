#!/usr/bin/env python3
"""
Enhanced Machine Learning Pipeline for Student Performance Classification

This module implements an enhanced version with:
1. Fixed XGBoost implementation with label encoding
2. Additional performance metrics
3. Feature importance analysis
4. Model persistence
5. Enhanced error handling

Following project structure and coding standards.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import pickle
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.model_selection import (
    train_test_split, 
    GridSearchCV, 
    cross_val_score,
    StratifiedKFold
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# Evaluation metrics
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

# MLflow
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from mlops.config import FIGURES, DATA_INTERIM, CLASSIFICATION_REPORTS

# -----------------------------------------------------------------------------
# CONSTANTES
# -----------------------------------------------------------------------------
RANDOM_STATE = 42
CV_FOLDS = 5
TEST_SIZE = 0.2

# -----------------------------------------------------------------------------
# FUNCIONES DE PREPROCESAMIENTO
# -----------------------------------------------------------------------------

def encode_target_variable(y_train, y_test):
    """
    Codifica las variables objetivo para compatibilidad con XGBoost.
    
    Args:
        y_train, y_test: Series con variables objetivo
        
    Returns:
        tuple: y_train_encoded, y_test_encoded, label_encoder
    """
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    
    print(f"[INFO] Target encoding mapping: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")
    
    return y_train_encoded, y_test_encoded, label_encoder

def load_and_prepare_data():
    """
    Carga y prepara los datos para el entrenamiento.
    
    Returns:
        tuple: X_train, X_test, y_train, y_test, y_train_encoded, y_test_encoded, label_encoder
    """
    # Load data
    X_path = DATA_INTERIM / "student_interim_preprocessed.csv"
    y_path = DATA_INTERIM / "student_interim_clean.csv"
    
    if not X_path.exists() or not y_path.exists():
        raise FileNotFoundError("Preprocessed data not found. Run EDA pipeline first.")
    
    X = pd.read_csv(X_path)
    y = pd.read_csv(y_path)["Performance"]
    
    print(f"[INFO] Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"[INFO] Target distribution:")
    print(y.value_counts())
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_STATE,
        stratify=y
    )
    
    print(f"[INFO] Train set: {X_train.shape}")
    print(f"[INFO] Test set: {X_test.shape}")
    
    # Encode target for XGBoost compatibility
    y_train_encoded, y_test_encoded, label_encoder = encode_target_variable(y_train, y_test)
    
    return X_train, X_test, y_train, y_test, y_train_encoded, y_test_encoded, label_encoder

# -----------------------------------------------------------------------------
# DEFINICIÓN DE MODELOS MEJORADA
# -----------------------------------------------------------------------------

def get_enhanced_model_configurations():
    """
    Define los modelos de ML con configuraciones optimizadas.
    
    Returns:
        dict: Configuraciones de modelos con parámetros para tuning
    """
    models_config = {
        'logistic_regression': {
            'model': LogisticRegression(random_state=RANDOM_STATE, max_iter=2000),
            'params': {
                'C': [0.1, 1, 10],
                'solver': ['liblinear', 'lbfgs']
            },
            'needs_encoding': False
        },
        'random_forest': {
            'model': RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
            'params': {
                'n_estimators': [100, 200],
                'max_depth': [10, None],
                'min_samples_split': [2, 5]
            },
            'needs_encoding': False
        },
        'gradient_boosting': {
            'model': GradientBoostingClassifier(random_state=RANDOM_STATE),
            'params': {
                'n_estimators': [100, 150],
                'learning_rate': [0.1, 0.2],
                'max_depth': [3, 5]
            },
            'needs_encoding': False
        },
        'svm': {
            'model': SVC(random_state=RANDOM_STATE, probability=True),
            'params': {
                'C': [1, 10],
                'kernel': ['rbf', 'linear']
            },
            'needs_encoding': False
        },
        'xgboost': {
            'model': XGBClassifier(random_state=RANDOM_STATE, eval_metric='mlogloss', verbosity=0),
            'params': {
                'n_estimators': [100, 200],
                'learning_rate': [0.1, 0.2],
                'max_depth': [3, 5]
            },
            'needs_encoding': True
        },
        'knn': {
            'model': KNeighborsClassifier(),
            'params': {
                'n_neighbors': [5, 7, 9],
                'weights': ['uniform', 'distance']
            },
            'needs_encoding': False
        },
        'decision_tree': {
            'model': DecisionTreeClassifier(random_state=RANDOM_STATE),
            'params': {
                'max_depth': [5, 10, None],
                'min_samples_split': [2, 5],
                'criterion': ['gini', 'entropy']
            },
            'needs_encoding': False
        },
        'naive_bayes': {
            'model': GaussianNB(),
            'params': {
                'var_smoothing': [1e-9, 1e-8]
            },
            'needs_encoding': False
        }
    }
    
    print(f"[INFO] Defined {len(models_config)} algorithms")
    return models_config

# -----------------------------------------------------------------------------
# FUNCIONES DE OPTIMIZACIÓN MEJORADA
# -----------------------------------------------------------------------------

def tune_enhanced_hyperparameters(models_config, X_train, y_train, y_train_encoded, cv_folds=CV_FOLDS):
    """
    Optimiza hiperparámetros con manejo especial para XGBoost.
    
    Args:
        models_config (dict): Configuraciones de modelos
        X_train (pd.DataFrame): Features de entrenamiento
        y_train (pd.Series): Target original
        y_train_encoded (np.array): Target codificado
        cv_folds (int): Número de folds para cross-validation
        
    Returns:
        dict: Mejores modelos optimizados
    """
    print("[INFO] Starting enhanced hyperparameter tuning...")
    
    best_models = {}
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    
    for name, model_config in models_config.items():
        print(f"[INFO] Tuning {name}...")
        
        with mlflow.start_run(run_name=f"{name}_enhanced_tuning"):
            try:
                # Choose appropriate target variable
                y_for_training = y_train_encoded if model_config['needs_encoding'] else y_train
                
                # GridSearchCV
                grid_search = GridSearchCV(
                    model_config['model'],
                    model_config['params'],
                    cv=cv,
                    scoring='accuracy',
                    n_jobs=-1,
                    verbose=0,
                    error_score='raise'
                )
                
                grid_search.fit(X_train, y_for_training)
                
                # Store best model
                best_models[name] = grid_search.best_estimator_
                
                # Log parameters and metrics to MLflow
                mlflow.log_params(grid_search.best_params_)
                mlflow.log_metric("best_cv_score", grid_search.best_score_)
                mlflow.log_metric("cv_std", grid_search.cv_results_['std_test_score'][grid_search.best_index_])
                mlflow.log_param("needs_encoding", model_config['needs_encoding'])
                
                print(f"[OK] Best CV score for {name}: {grid_search.best_score_:.4f}")
                print(f"[OK] Best parameters: {grid_search.best_params_}")
                
            except Exception as e:
                print(f"[ERROR] Error tuning {name}: {str(e)}")
                # Use default model if tuning fails
                best_models[name] = model_config['model']
                
    return best_models

# -----------------------------------------------------------------------------
# FUNCIONES DE EVALUACIÓN MEJORADA
# -----------------------------------------------------------------------------

def evaluate_enhanced_model(name, model, X_train, y_train, X_test, y_test, 
                           y_train_encoded, y_test_encoded, label_encoder, needs_encoding=False):
    """
    Evalúa un modelo individual con métricas comprehensivas mejoradas.
    
    Args:
        name (str): Nombre del modelo
        model: Modelo entrenado
        X_train, y_train, X_test, y_test: Datos originales
        y_train_encoded, y_test_encoded: Datos codificados
        label_encoder: Codificador de etiquetas
        needs_encoding (bool): Si el modelo necesita datos codificados
        
    Returns:
        dict: Métricas de evaluación
    """
    print(f"[INFO] Evaluating {name}...")
    
    with mlflow.start_run(run_name=f"{name}_enhanced_evaluation"):
        try:
            # Choose appropriate target variables for training
            y_train_for_cv = y_train_encoded if needs_encoding else y_train
            
            # Predictions
            if needs_encoding:
                y_pred_encoded = model.predict(X_test)
                y_pred = label_encoder.inverse_transform(y_pred_encoded)
            else:
                y_pred = model.predict(X_test)
            
            # Cross-validation scores
            cv_scores = cross_val_score(model, X_train, y_train_for_cv, cv=5, scoring='accuracy')
            
            # Metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision_macro': precision_score(y_test, y_pred, average='macro'),
                'precision_weighted': precision_score(y_test, y_pred, average='weighted'),
                'recall_macro': recall_score(y_test, y_pred, average='macro'),
                'recall_weighted': recall_score(y_test, y_pred, average='weighted'),
                'f1_macro': f1_score(y_test, y_pred, average='macro'),
                'f1_weighted': f1_score(y_test, y_pred, average='weighted'),
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            
            # Additional data for later use
            metrics['model'] = model
            metrics['y_pred'] = y_pred
            metrics['classification_report'] = classification_report(y_test, y_pred)
            metrics['needs_encoding'] = needs_encoding
            
            # Feature importance for tree-based models
            if hasattr(model, 'feature_importances_'):
                feature_importance = pd.DataFrame({
                    'feature': X_train.columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                # Log top 10 features
                top_features = feature_importance.head(10)
                for idx, row in top_features.iterrows():
                    mlflow.log_metric(f"feature_importance_{row['feature']}", row['importance'])
                
                metrics['feature_importance'] = feature_importance
            
            # Log metrics to MLflow
            mlflow_metrics = {k: v for k, v in metrics.items() 
                            if k not in ['model', 'y_pred', 'classification_report', 'feature_importance', 'needs_encoding']}
            mlflow.log_metrics(mlflow_metrics)
            mlflow.log_param("needs_encoding", needs_encoding)
            
            # Log model
            signature = infer_signature(X_test, y_pred)
            mlflow.sklearn.log_model(model, f"{name}_enhanced_model", signature=signature)
            
            # Ensure classification reports directory exists
            CLASSIFICATION_REPORTS.mkdir(parents=True, exist_ok=True)
            
            # Log classification report as artifact
            report_path = CLASSIFICATION_REPORTS / f"{name}_enhanced_classification_report.txt"
            with open(report_path, "w") as f:
                f.write(classification_report(y_test, y_pred))
            mlflow.log_artifact(str(report_path))
            
            print(f"[OK] {name} - Accuracy: {metrics['accuracy']:.4f}")
            print(f"[OK] {name} - F1 (weighted): {metrics['f1_weighted']:.4f}")
            print(f"[OK] {name} - CV Score: {metrics['cv_mean']:.4f} (±{metrics['cv_std']:.4f})")
            
            return metrics
            
        except Exception as e:
            print(f"[ERROR] Error evaluating {name}: {str(e)}")
            return None

def evaluate_all_enhanced_models(best_models, models_config, X_train, y_train, X_test, y_test, 
                                y_train_encoded, y_test_encoded, label_encoder):
    """
    Evalúa todos los modelos optimizados con manejo especial para modelos codificados.
    """
    print("[INFO] Evaluating all enhanced models...")
    
    results = {}
    for name, model in best_models.items():
        needs_encoding = models_config[name]['needs_encoding']
        result = evaluate_enhanced_model(
            name, model, X_train, y_train, X_test, y_test, 
            y_train_encoded, y_test_encoded, label_encoder, needs_encoding
        )
        if result is not None:
            results[name] = result
            
    return results

# -----------------------------------------------------------------------------
# FUNCIONES DE PERSISTENCIA
# -----------------------------------------------------------------------------

def save_best_model(results, models_config, label_encoder):
    """
    Guarda el mejor modelo y sus artefactos.
    
    Args:
        results (dict): Resultados de evaluación
        models_config (dict): Configuraciones de modelos
        label_encoder: Codificador de etiquetas
    """
    # Find best model
    best_model_name = max(results.keys(), key=lambda x: results[x]['f1_weighted'])
    best_model = results[best_model_name]['model']
    
    print(f"[INFO] Saving best model: {best_model_name}")
    
    # Create models directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Save model
    model_path = models_dir / f"best_model_{best_model_name}.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(best_model, f)
    
    # Save label encoder if needed
    if models_config[best_model_name]['needs_encoding']:
        encoder_path = models_dir / "label_encoder.pkl"
        with open(encoder_path, 'wb') as f:
            pickle.dump(label_encoder, f)
        print(f"[OK] Label encoder saved to: {encoder_path}")
    
    # Save model metadata
    metadata = {
        'model_name': best_model_name,
        'model_type': str(type(best_model)),
        'performance': results[best_model_name],
        'needs_encoding': models_config[best_model_name]['needs_encoding']
    }
    
    metadata_path = models_dir / "best_model_metadata.pkl"
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"[OK] Best model saved to: {model_path}")
    print(f"[OK] Model metadata saved to: {metadata_path}")

# -----------------------------------------------------------------------------
# FUNCIÓN PRINCIPAL MEJORADA
# -----------------------------------------------------------------------------

def main():
    """
    Ejecuta el pipeline completo mejorado de Machine Learning.
    """
    print("Starting Enhanced ML Pipeline for Student Performance Classification")
    print("="*80)
    
    # Setup MLflow
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("student_performance_enhanced_experiment")
    
    try:
        # 1. Load and prepare data
        X_train, X_test, y_train, y_test, y_train_encoded, y_test_encoded, label_encoder = load_and_prepare_data()
        
        # 2. Define enhanced models
        models_config = get_enhanced_model_configurations()
        
        # 3. Tune hyperparameters
        best_models = tune_enhanced_hyperparameters(models_config, X_train, y_train, y_train_encoded)
        
        # 4. Evaluate models
        results = evaluate_all_enhanced_models(
            best_models, models_config, X_train, y_train, X_test, y_test, 
            y_train_encoded, y_test_encoded, label_encoder
        )
        
        # 5. Save best model
        save_best_model(results, models_config, label_encoder)
        
        # 6. Print enhanced summary
        print_enhanced_summary(results)
        
        print("\nEnhanced pipeline completed successfully!")
        print("Check MLflow UI with: mlflow ui")
        print("Best model saved in: models/")
        
    except Exception as e:
        print(f"Enhanced pipeline failed: {str(e)}")
        raise

def print_enhanced_summary(results):
    """Imprime resumen mejorado de resultados."""
    print("\n" + "="*100)
    print("ENHANCED RESULTS SUMMARY")
    print("="*100)
    
    # Sort models by F1 weighted score
    sorted_models = sorted(results.items(), key=lambda x: x[1]['f1_weighted'], reverse=True)
    
    print(f"{'Rank':<5} {'Model':<20} {'Accuracy':<10} {'F1':<10} {'Precision':<12} {'Recall':<10} {'CV Score':<15} {'Encoding':<10}")
    print("-" * 100)
    
    for rank, (name, metrics) in enumerate(sorted_models, 1):
        encoding = "Yes" if metrics.get('needs_encoding', False) else "No"
        print(f"{rank:<5} {name:<20} {metrics['accuracy']:<10.4f} "
              f"{metrics['f1_weighted']:<10.4f} {metrics['precision_weighted']:<12.4f} "
              f"{metrics['recall_weighted']:<10.4f} {metrics['cv_mean']:.4f}±{metrics['cv_std']:.3f}    {encoding:<10}")
    
    # Best model details
    best_model_name, best_metrics = sorted_models[0]
    print(f"\nBEST MODEL: {best_model_name}")
    print(f"   Accuracy: {best_metrics['accuracy']:.4f}")
    print(f"   F1 Score (weighted): {best_metrics['f1_weighted']:.4f}")
    print(f"   Cross-validation: {best_metrics['cv_mean']:.4f} ± {best_metrics['cv_std']:.4f}")
    print(f"   Needs encoding: {best_metrics.get('needs_encoding', False)}")

if __name__ == "__main__":
    main()