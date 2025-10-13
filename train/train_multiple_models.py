#!/usr/bin/env python3
"""
Complete Machine Learning Pipeline for Student Performance Classification

This module implements:
1. Multiple ML algorithms comparison
2. Hyperparameter tuning with GridSearchCV
3. Comprehensive evaluation metrics
4. MLflow experiment tracking
5. Cross-validation and model selection

Following project structure and coding standards.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.model_selection import (
    train_test_split, 
    GridSearchCV, 
    RandomizedSearchCV,
    cross_val_score,
    StratifiedKFold
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
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
# FUNCIONES DE CARGA DE DATOS
# -----------------------------------------------------------------------------

def load_preprocessed_data():
    """
    Carga los datos preprocesados desde data/interim.
    
    Returns:
        tuple[pd.DataFrame, pd.Series]: Features (X) y target (y)
    """
    X_path = DATA_INTERIM / "student_interim_preprocessed.csv"
    y_path = DATA_INTERIM / "student_interim_clean.csv"
    
    if not X_path.exists() or not y_path.exists():
        raise FileNotFoundError("Preprocessed data not found. Run EDA pipeline first.")
    
    X = pd.read_csv(X_path)
    y = pd.read_csv(y_path)["Performance"]
    
    print(f"[INFO] Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"[INFO] Target distribution:")
    print(y.value_counts())
    
    return X, y

def split_data(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    """
    Divide los datos en conjuntos de entrenamiento y prueba.
    
    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Target variable
        test_size (float): Proporción del conjunto de prueba
        random_state (int): Semilla aleatoria
        
    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y
    )
    
    print(f"[INFO] Train set: {X_train.shape}")
    print(f"[INFO] Test set: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test

# -----------------------------------------------------------------------------
# DEFINICIÓN DE MODELOS Y PARÁMETROS
# -----------------------------------------------------------------------------

def get_model_configurations():
    """
    Define los modelos de ML y sus grids de parámetros.
    
    Returns:
        dict: Configuraciones de modelos con parámetros para tuning
    """
    models_config = {
        'logistic_regression': {
            'model': LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
            'params': {
                'C': [0.1, 1, 10, 100],
                'solver': ['liblinear', 'lbfgs']
            }
        },
        'random_forest': {
            'model': RandomForestClassifier(random_state=RANDOM_STATE),
            'params': {
                'n_estimators': [100, 200, 300],
                'max_depth': [5, 10, None],
                'min_samples_split': [2, 5, 10]
            }
        },
        'gradient_boosting': {
            'model': GradientBoostingClassifier(random_state=RANDOM_STATE),
            'params': {
                'n_estimators': [100, 200],
                'learning_rate': [0.05, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            }
        },
        'svm': {
            'model': SVC(random_state=RANDOM_STATE, probability=True),
            'params': {
                'C': [0.1, 1, 10],
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto']
            }
        },
        'xgboost': {
            'model': XGBClassifier(random_state=RANDOM_STATE, eval_metric='mlogloss'),
            'params': {
                'n_estimators': [100, 200],
                'learning_rate': [0.05, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            }
        },
        'knn': {
            'model': KNeighborsClassifier(),
            'params': {
                'n_neighbors': [3, 5, 7, 9],
                'weights': ['uniform', 'distance']
            }
        },
        'decision_tree': {
            'model': DecisionTreeClassifier(random_state=RANDOM_STATE),
            'params': {
                'max_depth': [5, 10, None],
                'min_samples_split': [2, 5, 10],
                'criterion': ['gini', 'entropy']
            }
        },
        'naive_bayes': {
            'model': GaussianNB(),
            'params': {
                'var_smoothing': [1e-9, 1e-8, 1e-7]
            }
        }
    }
    
    print(f"[INFO] Defined {len(models_config)} algorithms")
    return models_config

# -----------------------------------------------------------------------------
# FUNCIONES DE OPTIMIZACIÓN DE HIPERPARÁMETROS
# -----------------------------------------------------------------------------

def tune_hyperparameters(models_config, X_train, y_train, cv_folds=CV_FOLDS):
    """
    Optimiza hiperparámetros usando GridSearchCV.
    
    Args:
        models_config (dict): Configuraciones de modelos
        X_train (pd.DataFrame): Features de entrenamiento
        y_train (pd.Series): Target de entrenamiento
        cv_folds (int): Número de folds para cross-validation
        
    Returns:
        dict: Mejores modelos optimizados
    """
    print("[INFO] Starting hyperparameter tuning...")
    
    best_models = {}
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    
    for name, model_config in models_config.items():
        print(f"[INFO] Tuning {name}...")
        
        with mlflow.start_run(run_name=f"{name}_hyperparameter_tuning"):
            try:
                # GridSearchCV
                grid_search = GridSearchCV(
                    model_config['model'],
                    model_config['params'],
                    cv=cv,
                    scoring='accuracy',
                    n_jobs=-1,
                    verbose=0
                )
                
                grid_search.fit(X_train, y_train)
                
                # Store best model
                best_models[name] = grid_search.best_estimator_
                
                # Log parameters and metrics to MLflow
                mlflow.log_params(grid_search.best_params_)
                mlflow.log_metric("best_cv_score", grid_search.best_score_)
                mlflow.log_metric("cv_std", grid_search.cv_results_['std_test_score'][grid_search.best_index_])
                
                print(f"[OK] Best CV score for {name}: {grid_search.best_score_:.4f}")
                print(f"[OK] Best parameters: {grid_search.best_params_}")
                
            except Exception as e:
                print(f"[ERROR] Error tuning {name}: {str(e)}")
                # Use default model if tuning fails
                best_models[name] = model_config['model']
                
    return best_models

# -----------------------------------------------------------------------------
# FUNCIONES DE EVALUACIÓN
# -----------------------------------------------------------------------------

def evaluate_single_model(name, model, X_train, y_train, X_test, y_test):
    """
    Evalúa un modelo individual con métricas comprehensivas.
    
    Args:
        name (str): Nombre del modelo
        model: Modelo entrenado
        X_train, y_train: Datos de entrenamiento
        X_test, y_test: Datos de prueba
        
    Returns:
        dict: Métricas de evaluación
    """
    print(f"[INFO] Evaluating {name}...")
    
    with mlflow.start_run(run_name=f"{name}_final_evaluation"):
        try:
            # Predictions
            y_pred = model.predict(X_test)
            
            # Cross-validation scores
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            
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
            
            # Log metrics to MLflow
            mlflow_metrics = {k: v for k, v in metrics.items() 
                            if k not in ['model', 'y_pred', 'classification_report']}
            mlflow.log_metrics(mlflow_metrics)
            
            # Log model
            signature = infer_signature(X_test, y_pred)
            mlflow.sklearn.log_model(model, f"{name}_model", signature=signature)
            
            # Ensure classification reports directory exists
            CLASSIFICATION_REPORTS.mkdir(parents=True, exist_ok=True)
            
            # Log classification report as artifact
            report_path = CLASSIFICATION_REPORTS / f"{name}_classification_report.txt"
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

def evaluate_all_models(best_models, X_train, y_train, X_test, y_test):
    """
    Evalúa todos los modelos optimizados.
    
    Args:
        best_models (dict): Modelos optimizados
        X_train, y_train: Datos de entrenamiento
        X_test, y_test: Datos de prueba
        
    Returns:
        dict: Resultados de evaluación para todos los modelos
    """
    print("[INFO] Evaluating all models...")
    
    results = {}
    for name, model in best_models.items():
        result = evaluate_single_model(name, model, X_train, y_train, X_test, y_test)
        if result is not None:
            results[name] = result
            
    return results

# -----------------------------------------------------------------------------
# FUNCIONES DE VISUALIZACIÓN
# -----------------------------------------------------------------------------

def create_comparison_plots(results):
    """
    Crea gráficos de comparación entre modelos.
    
    Args:
        results (dict): Resultados de evaluación de todos los modelos
    """
    print("[INFO] Creating comparison plots...")
    
    # Prepare data for plotting
    model_names = list(results.keys())
    metrics_to_plot = ['accuracy', 'f1_weighted', 'precision_weighted', 'recall_weighted']
    
    # Create comparison dataframe
    comparison_data = []
    for name in model_names:
        row = [name] + [results[name][metric] for metric in metrics_to_plot]
        comparison_data.append(row)
        
    df_comparison = pd.DataFrame(comparison_data, columns=['Model'] + metrics_to_plot)
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
    
    for idx, metric in enumerate(metrics_to_plot):
        row, col = idx // 2, idx % 2
        ax = axes[row, col]
        
        # Sort by metric for better visualization
        df_sorted = df_comparison.sort_values(metric, ascending=True)
        
        bars = ax.barh(df_sorted['Model'], df_sorted[metric], 
                      color=plt.cm.viridis(np.linspace(0, 1, len(model_names))))
        
        ax.set_title(f'{metric.replace("_", " ").title()}', fontweight='bold')
        ax.set_xlabel('Score')
        
        # Add value labels on bars
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{width:.3f}', ha='left', va='center', fontweight='bold')
                   
    plt.tight_layout()
    
    # Save plot
    plot_path = FIGURES / "model_comparison_complete.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Comparison plot saved to: {plot_path}")
    
    plt.show()

def create_confusion_matrices(results, y_test, top_n=3):
    """
    Crea matrices de confusión para los mejores modelos.
    
    Args:
        results (dict): Resultados de evaluación
        y_test (pd.Series): Target de prueba
        top_n (int): Número de mejores modelos a mostrar
    """
    print("[INFO] Creating confusion matrices...")
    
    # Select top N models by F1 score
    top_models = sorted(results.items(), 
                       key=lambda x: x[1]['f1_weighted'], 
                       reverse=True)[:top_n]
    
    fig, axes = plt.subplots(1, top_n, figsize=(6*top_n, 5))
    if top_n == 1:
        axes = [axes]
        
    fig.suptitle(f'Confusion Matrices - Top {top_n} Models', fontsize=16, fontweight='bold')
    
    for idx, (name, result) in enumerate(top_models):
        cm = confusion_matrix(y_test, result['y_pred'])
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=np.unique(y_test),
                   yticklabels=np.unique(y_test),
                   ax=axes[idx])
        
        axes[idx].set_title(f'{name}\nF1: {result["f1_weighted"]:.3f}', fontweight='bold')
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('Actual')
        
    plt.tight_layout()
    
    # Save plot
    cm_path = FIGURES / f"confusion_matrices_top{top_n}.png"
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Confusion matrices saved to: {cm_path}")
    
    plt.show()

# -----------------------------------------------------------------------------
# FUNCIONES DE REPORTE
# -----------------------------------------------------------------------------

def print_results_summary(results):
    """
    Imprime resumen completo de resultados.
    
    Args:
        results (dict): Resultados de evaluación de todos los modelos
    """
    print("\n" + "="*90)
    print("FINAL RESULTS SUMMARY")
    print("="*90)
    
    # Sort models by F1 weighted score
    sorted_models = sorted(results.items(), 
                         key=lambda x: x[1]['f1_weighted'], 
                         reverse=True)
    
    print(f"{'Rank':<5} {'Model':<20} {'Accuracy':<10} {'F1':<10} {'Precision':<12} {'Recall':<10} {'CV Score':<15}")
    print("-" * 90)
    
    for rank, (name, metrics) in enumerate(sorted_models, 1):
        print(f"{rank:<5} {name:<20} {metrics['accuracy']:<10.4f} "
              f"{metrics['f1_weighted']:<10.4f} {metrics['precision_weighted']:<12.4f} "
              f"{metrics['recall_weighted']:<10.4f} {metrics['cv_mean']:.4f}±{metrics['cv_std']:.3f}")
    
    # Best model details
    best_model_name, best_metrics = sorted_models[0]
    print(f"\nBEST MODEL: {best_model_name}")
    print(f"   Accuracy: {best_metrics['accuracy']:.4f}")
    print(f"   F1 Score (weighted): {best_metrics['f1_weighted']:.4f}")
    print(f"   Cross-validation: {best_metrics['cv_mean']:.4f} ± {best_metrics['cv_std']:.4f}")
    
    print(f"\nDETAILED CLASSIFICATION REPORT (Best Model):")
    print(best_metrics['classification_report'])

# -----------------------------------------------------------------------------
# FUNCIÓN PRINCIPAL
# -----------------------------------------------------------------------------

def main():
    """
    Ejecuta el pipeline completo de Machine Learning.
    """
    print("Starting Complete ML Pipeline for Student Performance Classification")
    print("="*80)
    
    # Setup MLflow
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("student_performance_complete_experiment")
    
    try:
        # 1. Load data
        X, y = load_preprocessed_data()
        
        # 2. Split data
        X_train, X_test, y_train, y_test = split_data(X, y)
        
        # 3. Define models
        models_config = get_model_configurations()
        
        # 4. Tune hyperparameters
        best_models = tune_hyperparameters(models_config, X_train, y_train)
        
        # 5. Evaluate models
        results = evaluate_all_models(best_models, X_train, y_train, X_test, y_test)
        
        # 6. Create visualizations
        create_comparison_plots(results)
        create_confusion_matrices(results, y_test)
        
        # 7. Print summary
        print_results_summary(results)
        
        print("\nPipeline completed successfully!")
        print("Check MLflow UI with: mlflow ui")
        print("Plots saved in: reports/figures/")
        
    except Exception as e:
        print(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()