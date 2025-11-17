# ============================================================
# Grid Search Amplio - Optimización de Hiperparámetros
# RandomForestClassifier - Student Performance
# Partiendo del modelo base: n_estimators=20, max_depth=20, min_samples_split=15, random_state=888
# ============================================================

import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
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
from mlops.mlflow_config import setup_mlflow

# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

logging.basicConfig(
    filename="gridsearch.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

SEED = 42

# Columnas categóricas conocidas del dataset
CATEGORICAL_COLUMNS = [
    'Gender', 'Caste', 'coaching', 'time', 'Class_ten_education',
    'twelve_education', 'medium', 'Class_ X_Percentage',
    'Class_XII_Percentage', 'Father_occupation', 'Mother_occupation'
]


class GridSearchTrainer:
    """Encapsulates the grid search training pipeline for hyperparameter optimization."""

    def __init__(
        self,
        seed: int = SEED,
        data_path: str = "data/interim/student_interim_clean.csv",
        models_dir: str = "models",
        reports_dir: str = "reports",
        mlflow_experiment: str = "student_performance_gridsearch_amplio",
        model_random_state: int = 888,
        cv_folds: int = 5,
        top_n_models: int = 10,
    ) -> None:
        """Initialize the grid search trainer with configuration."""
        self.seed = seed
        self.data_path = Path(data_path)
        self.models_dir = Path(models_dir)
        self.reports_dir = Path(reports_dir)
        self.mlflow_experiment = mlflow_experiment
        self.model_random_state = model_random_state
        self.cv_folds = cv_folds
        self.top_n_models = top_n_models
        
        # Attributes to be set during training
        self.label_encoder: Optional[LabelEncoder] = None
        self.grid_search: Optional[GridSearchCV] = None
        self.class_names: Optional[np.ndarray] = None
        self.preprocessor: Optional[ColumnTransformer] = None

    def _ensure_directories(self) -> None:
        """Ensure required directories exist."""
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def _setup_mlflow(self) -> None:
        """Configure MLflow tracking."""
        setup_mlflow(self.mlflow_experiment)
        logging.info("Tracking MLflow configurado para Grid Search Amplio.")

    def load_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Load and prepare data from CSV file.
        
        Returns:
            Tuple of (features DataFrame, target Series)
        """
        df = pd.read_csv(self.data_path)
        
        logging.info(f"Datos cargados. Dataset completo: {df.shape}")
        logging.info(f"Distribución de clases: {df['Performance'].value_counts().to_dict()}")
        
        # Separar features y target (IGUAL QUE EL NOTEBOOK)
        X = df.drop(columns=['Performance'])
        y = df['Performance']
        
        return X, y

    def encode_target(self, y: pd.Series) -> np.ndarray:
        """Encode target variable with LabelEncoder.
        
        Args:
            y: Target series
            
        Returns:
            Encoded target array
        """
        self.label_encoder = LabelEncoder()
        y_enc = self.label_encoder.fit_transform(y)
        self.class_names = self.label_encoder.classes_
        
        logging.info(f"Clases finales: {list(self.class_names)}")
        logging.info(f"Target: {len(y_enc)}")
        
        return y_enc

    def get_categorical_columns(self, X: pd.DataFrame) -> List[str]:
        """Get list of categorical columns present in the DataFrame.
        
        Args:
            X: Features DataFrame
            
        Returns:
            List of categorical column names
        """
        cat_cols = [c for c in CATEGORICAL_COLUMNS if c in X.columns]
        logging.info(f"Columnas categóricas: {cat_cols}")
        return cat_cols

    def split_data(
        self, X: pd.DataFrame, y: np.ndarray
    ) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
        """Split data into train and test sets.
        
        Args:
            X: Features DataFrame
            y: Target array
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.seed, stratify=y
        )
        logging.info(f"Split hecho. Train: {X_train.shape}, Test: {X_test.shape}")
        return X_train, X_test, y_train, y_test

    def create_preprocessor(self, cat_cols: List[str]) -> ColumnTransformer:
        """Create preprocessing pipeline.
        
        Args:
            cat_cols: List of categorical column names
            
        Returns:
            ColumnTransformer for preprocessing
        """
        preprocessor = ColumnTransformer(
            transformers=[('ohe', OneHotEncoder(handle_unknown='ignore'), cat_cols)],
            remainder='drop'  # IMPORTANTE: igual que el modelo actual
        )
        self.preprocessor = preprocessor
        return preprocessor

    def get_param_grid(self) -> Dict[str, List[Any]]:
        """Get parameter grid for grid search.
        
        Returns:
            Dictionary with parameter grid
        """
        return {
            'classifier__n_estimators': [18, 20, 22, 25],
            'classifier__max_depth': [18, 20, 22],
            'classifier__min_samples_split': [12, 15, 18],
            'classifier__min_samples_leaf': [1, 2],
            'classifier__max_features': ['sqrt', 'log2'],
            'classifier__criterion': ['gini'],
            'classifier__class_weight': [None]
        }

    def create_pipeline(self, preprocessor: ColumnTransformer) -> Pipeline:
        """Create the base pipeline for grid search.
        
        Args:
            preprocessor: ColumnTransformer for preprocessing
            
        Returns:
            Base Pipeline with preprocessing and classifier
        """
        return Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(
                random_state=self.model_random_state,
                bootstrap=True
            ))
        ])

    def create_grid_search(self, pipeline: Pipeline, param_grid: Dict[str, List[Any]]) -> GridSearchCV:
        """Create GridSearchCV object.
        
        Args:
            pipeline: Base pipeline
            param_grid: Parameter grid for search
            
        Returns:
            Configured GridSearchCV object
        """
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
            cv=self.cv_folds,
            verbose=2,
            n_jobs=-1,
            return_train_score=True
        )
        
        return grid_search

    def run_grid_search(
        self, grid_search: GridSearchCV, X_train: pd.DataFrame, y_train: np.ndarray
    ) -> GridSearchCV:
        """Execute grid search.
        
        Args:
            grid_search: GridSearchCV object
            X_train: Training features
            y_train: Training target
            
        Returns:
            Fitted GridSearchCV object
        """
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
        
        return grid_search

    def get_top_models(self, grid_search: GridSearchCV) -> pd.Index:
        """Get indices of top N models from grid search results.
        
        Args:
            grid_search: Fitted GridSearchCV object
            
        Returns:
            Index of top N models
        """
        results_df = pd.DataFrame(grid_search.cv_results_)
        results_df = results_df.sort_values('mean_test_f1_weighted', ascending=False)
        top_indices = results_df.head(self.top_n_models).index
        return top_indices

    def log_model_to_mlflow(
        self,
        grid_search: GridSearchCV,
        idx: int,
        rank: int,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: np.ndarray,
        y_test: np.ndarray,
        preprocessor: ColumnTransformer,
    ) -> None:
        """Log a single model to MLflow.
        
        Args:
            grid_search: Fitted GridSearchCV object
            idx: Index of the model in grid_search results
            rank: Rank of the model (1 = best)
            X_train: Training features
            X_test: Test features
            y_test: Test target
            preprocessor: Preprocessor used
        """
        params = grid_search.cv_results_['params'][idx]
        
        # Limpiar nombres de parámetros
        clean_params = {k.replace('classifier__', ''): v for k, v in params.items()}
        clean_params['random_state'] = self.model_random_state
        
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
            best_model_path = self.models_dir / "best_gridsearch_amplio.joblib"
            joblib.dump(current_pipeline, best_model_path)
            mlflow.log_artifact(str(best_model_path))
            mlflow.sklearn.log_model(current_pipeline, "best_model")
        else:
            print(f"Rank {rank}: CV F1={metrics['cv_f1_weighted_mean']:.4f}, "
                  f"Test Acc={test_acc:.4f}, Test F1={test_f1_weighted:.4f}")

    def log_top_models_to_mlflow(
        self,
        grid_search: GridSearchCV,
        top_indices: pd.Index,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: np.ndarray,
        y_test: np.ndarray,
        preprocessor: ColumnTransformer,
    ) -> None:
        """Log top N models to MLflow.
        
        Args:
            grid_search: Fitted GridSearchCV object
            top_indices: Indices of top models
            X_train: Training features
            X_test: Test features
            y_train: Training target
            y_test: Test target
            preprocessor: Preprocessor used
        """
        print("Registrando top 10 modelos en MLflow...\n")
        
        for rank, idx in enumerate(top_indices, 1):
            with mlflow.start_run(run_name=f"gridsearch_rank_{rank}", nested=False):
                self.log_model_to_mlflow(
                    grid_search, idx, rank, X_train, X_test, y_train, y_test, preprocessor
                )
        
        print(f"\nTop {self.top_n_models} modelos registrados en MLflow.")
        logging.info(f"Top {self.top_n_models} modelos registrados en MLflow.")

    def print_grid_search_info(self, param_grid: Dict[str, List[Any]]) -> None:
        """Print information about the grid search configuration.
        
        Args:
            param_grid: Parameter grid dictionary
        """
        total_combinations = np.prod([len(v) for v in param_grid.values()])
        
        print("\n" + "="*70)
        print("GRID SEARCH FINO - BÚSQUEDA CERCANA AL MODELO BASE")
        print("="*70)
        print(f"Modelo base: n_estimators=20, max_depth=20, min_samples_split=15")
        print(f"Random state modelo: {self.model_random_state}")
        print(f"\nRangos de búsqueda:")
        print(f"  n_estimators: [18, 20, 22, 25]")
        print(f"  max_depth: [18, 20, 22]")
        print(f"  min_samples_split: [12, 15, 18]")
        print(f"  min_samples_leaf: [1, 2]")
        print(f"  max_features: ['sqrt', 'log2']")
        print(f"\nTotal de combinaciones: {total_combinations}")
        print(f"Tiempo estimado: ~15-20 minutos")
        print(f"CV folds: {self.cv_folds}")
        print(f"Métrica principal: f1_weighted")
        print(f"Preprocessing: OneHotEncoder (remainder='drop')")
        print("="*70 + "\n")
        
        logging.info(f"Grilla definida con {total_combinations} combinaciones")

    def run(self) -> None:
        """Execute the full grid search pipeline."""
        # Setup
        self._ensure_directories()
        np.random.seed(self.seed)
        logging.info("==== Grid Search Amplio iniciado ====")
        logging.info(f"Semilla para split: {self.seed}")
        
        self._setup_mlflow()
        
        # Load and prepare data
        X, y = self.load_data()
        y_enc = self.encode_target(y)
        cat_cols = self.get_categorical_columns(X)
        X_train, X_test, y_train, y_test = self.split_data(X, y_enc)
        
        # Create pipeline and grid search
        preprocessor = self.create_preprocessor(cat_cols)
        pipeline = self.create_pipeline(preprocessor)
        param_grid = self.get_param_grid()
        
        self.print_grid_search_info(param_grid)
        
        grid_search = self.create_grid_search(pipeline, param_grid)
        grid_search = self.run_grid_search(grid_search, X_train, y_train)
        self.grid_search = grid_search
        
        # Get top models and log to MLflow
        top_indices = self.get_top_models(grid_search)
        self.log_top_models_to_mlflow(
            grid_search, top_indices, X_train, X_test, y_train, y_test, preprocessor
        )
        
        print("\n" + "="*70)
        print("GRID SEARCH FINALIZADO")
        print("Revisa MLflow UI para comparar todos los modelos")
        print(f"Experimento: {self.mlflow_experiment}")
        print("="*70)


def main() -> None:
    """Main entry point for grid search script."""
    trainer = GridSearchTrainer()
    trainer.run()


if __name__ == "__main__":
    main()
