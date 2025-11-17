# ============================================================
# Grid Search para modelo ORDINAL (RandomForestRegressor)
# Target codificado como: average=0, good=1, vg=2, excellent=3
# Optimiza MAE y reporta métricas de clasificación redondeando predicciones
# ============================================================

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    mean_absolute_error,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

logging.basicConfig(
    filename="gridsearch_ordinal.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

SEED = 42

CATEGORICAL_COLUMNS = [
    "Gender",
    "Caste",
    "coaching",
    "time",
    "Class_ten_education",
    "twelve_education",
    "medium",
    "Class_ X_Percentage",
    "Class_XII_Percentage",
    "Father_occupation",
    "Mother_occupation",
]


class OrdinalGridSearchTrainer:
    """Grid Search para modelo ordinal con RandomForestRegressor."""

    def __init__(
        self,
        seed: int = SEED,
        data_path: str = "data/interim/student_interim_clean.csv",
        models_dir: str = "models",
        reports_dir: str = "reports",
        cv_folds: int = 3,
    ) -> None:
        self.seed = seed
        self.data_path = Path(data_path)
        self.models_dir = Path(models_dir)
        self.reports_dir = Path(reports_dir)
        self.cv_folds = cv_folds

        self.class_names: Optional[np.ndarray] = None
        self.target_mapping: Optional[Dict[str, int]] = None

    # ------------------------ utilidades básicas ------------------------

    def _ensure_dirs(self) -> None:
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def load_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        df = pd.read_csv(self.data_path)

        logging.info(f"Datos originales cargados. Dataset completo: {df.shape}")
        logging.info(
            "Distribución original de clases: "
            f"{df['Performance'].value_counts().to_dict()}"
        )

        # Limpieza de Performance
        initial_size = len(df)
        df = df[df["Performance"].notna()]
        df = df[~df["Performance"].astype(str).str.lower().isin(["none", "nan", ""])]
        removed_count = initial_size - len(df)
        if removed_count > 0:
            logging.info(
                f"✓ Eliminados {removed_count} registros con Performance='none' o vacío"
            )

        # Limpieza de features categóricas
        X = df.drop(columns=["Performance"])
        y = df["Performance"]

        valid_mask = pd.Series([True] * len(X), index=X.index)
        for col in X.columns:
            if X[col].dtype == "object":
                mask_invalid = (
                    X[col].isna()
                    | X[col].astype(str)
                    .str.strip()
                    .str.lower()
                    .isin(["none", "nan", ""])
                )
                if mask_invalid.any():
                    logging.info(
                        f"✓ Detectados {mask_invalid.sum()} valores inválidos en '{col}'"
                    )
                    valid_mask &= ~mask_invalid

        initial_feature_size = len(X)
        X = X[valid_mask]
        y = y[valid_mask]
        removed_feature_count = initial_feature_size - len(X)
        if removed_feature_count > 0:
            logging.info(
                f"✓ Eliminados {removed_feature_count} registros con valores inválidos en features"
            )

        logging.info(f"Dataset final: {X.shape}")
        logging.info(f"Distribución final de clases: {y.value_counts().to_dict()}")

        return X, y

    def encode_target(self, y: pd.Series) -> np.ndarray:
        ordered_classes = ["average", "good", "vg", "excellent"]
        self.class_names = np.array(ordered_classes)
        mapping = {cls: idx for idx, cls in enumerate(ordered_classes)}
        self.target_mapping = mapping

        y_norm = y.astype(str).str.strip().str.lower()
        unknown = sorted(set(y_norm.unique()) - set(ordered_classes))
        if unknown:
            raise ValueError(
                f"Valores de Performance no esperados para encoding ordinal: {unknown}"
            )

        y_enc = y_norm.map(mapping).to_numpy()

        logging.info(f"Clases (ordinal): {ordered_classes}")
        logging.info(f"Total muestras: {len(y_enc)}")
        return y_enc

    def get_categorical_columns(self, X: pd.DataFrame) -> List[str]:
        return [c for c in CATEGORICAL_COLUMNS if c in X.columns]

    def split_data(
        self, X: pd.DataFrame, y: np.ndarray
    ) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=self.seed,
            stratify=y,
        )
        logging.info(f"Split: Train={X_train.shape}, Test={X_test.shape}")
        return X_train, X_test, y_train, y_test

    def create_preprocessor(self, cat_cols: List[str]) -> ColumnTransformer:
        return ColumnTransformer(
            transformers=[("ohe", OneHotEncoder(handle_unknown="ignore"), cat_cols)],
            remainder="drop",
        )

    # ------------------------ grid search ------------------------

    def create_pipeline(self, preprocessor: ColumnTransformer) -> Pipeline:
        base_rf = RandomForestRegressor(random_state=SEED, bootstrap=True, oob_score=False)
        pipe = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("regressor", base_rf),
            ]
        )
        return pipe

    def get_param_grid(self) -> Dict[str, List]:
        """Definir ~2000 combinaciones para RandomForestRegressor."""
        param_grid = {
            # 10 valores
            "regressor__n_estimators": [40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
            # 6 valores
            "regressor__max_depth": [8, 12, 16, 20, 24, 28],
            # 5 valores
            "regressor__min_samples_split": [2, 4, 6, 8, 10],
            # 4 valores
            "regressor__min_samples_leaf": [1, 2, 3, 4],
            # 2 valores
            "regressor__max_features": ["sqrt", "log2"],
        }

        total = (
            len(param_grid["regressor__n_estimators"]) \
            * len(param_grid["regressor__max_depth"]) \
            * len(param_grid["regressor__min_samples_split"]) \
            * len(param_grid["regressor__min_samples_leaf"]) \
            * len(param_grid["regressor__max_features"])
        )
        logging.info(f"Total combinaciones en grilla: {total}")
        print(f"Total combinaciones en grilla: {total}")
        return param_grid

    def print_grid_info(self, param_grid: Dict[str, List]) -> None:
        total = (
            len(param_grid["regressor__n_estimators"]) \
            * len(param_grid["regressor__max_depth"]) \
            * len(param_grid["regressor__min_samples_split"]) \
            * len(param_grid["regressor__min_samples_leaf"]) \
            * len(param_grid["regressor__max_features"])
        )

        print("\n" + "=" * 70)
        print("GRID SEARCH ORDINAL - RandomForestRegressor")
        print("=" * 70)
        print(f"n_estimators: {param_grid['regressor__n_estimators']}")
        print(f"max_depth: {param_grid['regressor__max_depth']}")
        print(f"min_samples_split: {param_grid['regressor__min_samples_split']}")
        print(f"min_samples_leaf: {param_grid['regressor__min_samples_leaf']}")
        print(f"max_features: {param_grid['regressor__max_features']}")
        print(f"Total de combinaciones: {total:,}")
        print(f"Entrenamientos totales con CV={self.cv_folds}: {total * self.cv_folds:,}")
        print("Métrica principal (scoring): neg_mean_absolute_error")
        print("=" * 70 + "\n")

    def create_grid_search(
        self, pipeline: Pipeline, param_grid: Dict[str, List]
    ) -> GridSearchCV:
        gs = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring="neg_mean_absolute_error",
            refit="neg_mean_absolute_error",
            cv=self.cv_folds,
            n_jobs=-1,
            verbose=2,
        )
        return gs

    def create_confusion_matrix(
        self, y_test: np.ndarray, y_pred: np.ndarray, run_name: str
    ) -> Path:
        classes_ordered = list(self.class_names)
        mapping = self.target_mapping or {
            cls: idx for idx, cls in enumerate(classes_ordered)
        }
        labels_ordered = [mapping[cls] for cls in classes_ordered]

        cm = confusion_matrix(y_test, y_pred, labels=labels_ordered)

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=classes_ordered,
            yticklabels=classes_ordered,
            cbar_kws={"label": "Cantidad"},
        )
        plt.xlabel("Predicción", fontsize=13, fontweight="bold")
        plt.ylabel("Real", fontsize=13, fontweight="bold")
        plt.title("Matriz de confusión - GridSearch ORDINAL", fontsize=14, fontweight="bold")
        plt.xticks(rotation=0)
        plt.yticks(rotation=0)

        cm_path = self.models_dir / f"conf_matrix_ordinal_gridsearch_{run_name}.png"
        plt.savefig(cm_path, bbox_inches="tight", dpi=100)
        plt.close()

        logging.info(f"Matriz de confusión (ordinal gridsearch) guardada en {cm_path}")
        return cm_path

    def evaluate_best_model(
        self,
        grid_search: GridSearchCV,
        X_test: pd.DataFrame,
        y_test: np.ndarray,
    ) -> None:
        best_est = grid_search.best_estimator_

        # Predicción continua
        y_pred_cont = best_est.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred_cont)

        # Clases redondeadas 0-3
        y_pred_round = np.rint(y_pred_cont).astype(int)
        y_pred_round = np.clip(y_pred_round, 0, len(self.class_names) - 1)

        acc = accuracy_score(y_test, y_pred_round)
        f1_w = f1_score(y_test, y_pred_round, average="weighted", zero_division=0)
        prec_w = precision_score(y_test, y_pred_round, average="weighted", zero_division=0)
        rec_w = recall_score(y_test, y_pred_round, average="weighted", zero_division=0)

        print("\n" + "=" * 70)
        print("MEJOR MODELO - GRID SEARCH ORDINAL")
        print("=" * 70)
        print(f"Best CV neg_MAE: {grid_search.best_score_:.4f}")
        print(f"Test MAE (continuo): {mae:.4f}")
        print(f"Test Accuracy (redondeado): {acc:.4f}")
        print(f"Test F1-Weighted (redondeado): {f1_w:.4f}")
        print(f"Test Precision-Weighted: {prec_w:.4f}")
        print(f"Test Recall-Weighted: {rec_w:.4f}")
        print("\nMejores parámetros:")
        for k, v in grid_search.best_params_.items():
            print(f"  {k}: {v}")
        print("=" * 70 + "\n")

        classif_rep = classification_report(
            y_test,
            y_pred_round,
            target_names=list(self.class_names),
            zero_division=0,
        )

        report_path = self.reports_dir / "classification_report_ordinal_gridsearch.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(classif_rep)
            f.write("\n\n")
            f.write(f"Test MAE: {mae:.4f}\n")
            f.write(f"Test Accuracy: {acc:.4f}\n")
            f.write(f"Test F1-Weighted: {f1_w:.4f}\n")

        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        cm_path = self.create_confusion_matrix(y_test, y_pred_round, run_name)
        print(f"Reporte guardado en: {report_path}")
        print(f"Matriz de confusión guardada en: {cm_path}")

    # ------------------------ ejecución principal ------------------------

    def run(self) -> None:
        self._ensure_dirs()

        X, y = self.load_data()
        y_enc = self.encode_target(y)
        cat_cols = self.get_categorical_columns(X)
        X_train, X_test, y_train, y_test = self.split_data(X, y_enc)

        preprocessor = self.create_preprocessor(cat_cols)
        pipeline = self.create_pipeline(preprocessor)
        param_grid = self.get_param_grid()
        self.print_grid_info(param_grid)

        grid_search = self.create_grid_search(pipeline, param_grid)

        print("Iniciando Grid Search ORDINAL... (puede tardar varios minutos)\n")
        grid_search.fit(X_train, y_train)

        logging.info("Grid Search ORDINAL completado")

        self.evaluate_best_model(grid_search, X_test, y_test)


def main() -> None:
    trainer = OrdinalGridSearchTrainer()
    trainer.run()


if __name__ == "__main__":
    main()
