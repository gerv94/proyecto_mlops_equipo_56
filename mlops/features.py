from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import TABLES

# Ensure directory exists for derived tables
TABLES.mkdir(parents=True, exist_ok=True)

# Heuristic threshold: columns with <= 30 unique values are treated as categorical
CATEGORICAL_GUESS_MAX = 30


class FeatureEngineering:
    """Encapsulates feature engineering utilities (typing, cleaning, encoding)."""

    def __init__(self, categorical_guess_max: int = CATEGORICAL_GUESS_MAX) -> None:
        self.categorical_guess_max = categorical_guess_max

    # ------------------------------------------------------------------
    # Column classification
    # ------------------------------------------------------------------

    def split_num_cat(self, dataframe: pd.DataFrame) -> Tuple[List[str], List[str]]:
        numeric_columns = [
            column_name
            for column_name in dataframe.columns
            if pd.api.types.is_numeric_dtype(dataframe[column_name])
        ]

        categorical_columns = [
            column_name for column_name in dataframe.columns if column_name not in numeric_columns
        ]

        categorical_columns = list(
            {column_name for column_name in categorical_columns}
            | {
                column_name
                for column_name in dataframe.columns
                if dataframe[column_name].nunique(dropna=True) <= self.categorical_guess_max
            }
        )

        numeric_columns = [
            column_name for column_name in dataframe.columns if column_name not in categorical_columns
        ]
        return numeric_columns, categorical_columns

    # ------------------------------------------------------------------
    # Categorical cleaning
    # ------------------------------------------------------------------

    @staticmethod
    def normalize_categories(series: pd.Series) -> pd.Series:
        if series.dtype == object:
            return (
                series.astype(str)
                .str.strip()
                .str.replace(r"\s+", " ", regex=True)
                .str.lower()
            )
        return series

    def clean_categoricals(self, dataframe: pd.DataFrame, categorical_columns: List[str]) -> pd.DataFrame:
        cleaned_frame = dataframe.copy()
        for column_name in categorical_columns:
            cleaned_frame[column_name] = self.normalize_categories(cleaned_frame[column_name])
        return cleaned_frame

    # ------------------------------------------------------------------
    # Basic preprocessing for EDA / baseline
    # ------------------------------------------------------------------

    def minimal_preprocess(
        self,
        dataframe: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, List[str], List[str]]:
        numeric_columns, categorical_columns = self.split_num_cat(dataframe)
        processed_frame = dataframe.copy()

        for column_name in numeric_columns:
            processed_frame[column_name] = processed_frame[column_name].fillna(
                processed_frame[column_name].median()
            )

        for column_name in categorical_columns:
            processed_frame[column_name] = processed_frame[column_name].fillna(
                processed_frame[column_name].mode().iloc[0]
            )

        return processed_frame, numeric_columns, categorical_columns

    # ------------------------------------------------------------------
    # Advanced preprocessing (scaling + one-hot + PCA)
    # ------------------------------------------------------------------

    def preprocess_advanced(
        self,
        dataframe: pd.DataFrame,
        numeric_columns: List[str],
        categorical_columns: List[str],
        n_components: int = 3,
    ) -> pd.DataFrame:
        processed_frame = dataframe.copy()

        if numeric_columns:
            scaler = StandardScaler()
            processed_frame[numeric_columns] = scaler.fit_transform(processed_frame[numeric_columns])

        encoded_frame: pd.DataFrame | None = None
        if categorical_columns:
            try:
                encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            except TypeError:
                encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")

            encoded_values = encoder.fit_transform(processed_frame[categorical_columns])
            encoded_column_names = encoder.get_feature_names_out(categorical_columns)
            encoded_frame = pd.DataFrame(
                encoded_values,
                columns=encoded_column_names,
                index=processed_frame.index,
            )

        parts: List[pd.DataFrame] = []
        if numeric_columns:
            parts.append(processed_frame[numeric_columns])
        if encoded_frame is not None:
            parts.append(encoded_frame)

        if not parts:
            print("No se encontraron columnas numéricas ni categóricas para transformar.")
            return processed_frame

        features_matrix = pd.concat(parts, axis=1)

        if n_components and n_components > 0:
            max_components = min(n_components, features_matrix.shape[1])
            if max_components >= 1:
                pca = PCA(n_components=max_components, random_state=42)
                components = pca.fit_transform(features_matrix)
                pca_columns = [f"PC{i + 1}" for i in range(max_components)]
                pca_frame = pd.DataFrame(components, columns=pca_columns, index=features_matrix.index)
                print("Varianza explicada (PCA):", np.round(pca.explained_variance_ratio_, 3))
                features_matrix = pd.concat([features_matrix, pca_frame], axis=1)
            else:
                print("PCA omitido: menos columnas que componentes solicitados.")

        print("Preprocesamiento avanzado finalizado.")
        return features_matrix

    # ------------------------------------------------------------------
    # Training-oriented preprocessing (one-hot only)
    # ------------------------------------------------------------------

    def preprocess_for_training(
        self,
        dataframe: pd.DataFrame,
        numeric_columns: List[str],
        categorical_columns: List[str],
    ) -> pd.DataFrame:
        processed_frame = dataframe.copy()

        encoded_frame: pd.DataFrame | None = None
        if categorical_columns:
            try:
                encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            except TypeError:
                encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")

            encoded_values = encoder.fit_transform(processed_frame[categorical_columns])
            encoded_column_names = encoder.get_feature_names_out(categorical_columns)
            encoded_frame = pd.DataFrame(
                encoded_values,
                columns=encoded_column_names,
                index=processed_frame.index,
            )

        processed_column_names = set(numeric_columns) | set(categorical_columns)
        preserved_columns = [
            column_name for column_name in processed_frame.columns if column_name not in processed_column_names
        ]

        parts: List[pd.DataFrame] = []
        if numeric_columns:
            parts.append(processed_frame[numeric_columns])
        if encoded_frame is not None:
            parts.append(encoded_frame)
        if preserved_columns:
            parts.append(processed_frame[preserved_columns])

        if not parts:
            print("No se encontraron columnas numéricas ni categóricas para transformar.")
            return processed_frame

        features_matrix = pd.concat(parts, axis=1)

        print("Preprocesamiento para entrenamiento finalizado (OneHotEncoder, sin PCA ni escalado).")
        if preserved_columns:
            print(f"Columnas preservadas: {preserved_columns}")
        return features_matrix


# -----------------------------------------------------------------------------
# Backwards compatible functional API
# -----------------------------------------------------------------------------

_default_feature_engineer = FeatureEngineering()


def split_num_cat(df: pd.DataFrame):
    return _default_feature_engineer.split_num_cat(df)


def normalize_categories(s: pd.Series) -> pd.Series:
    return FeatureEngineering.normalize_categories(s)


def clean_categoricals(df: pd.DataFrame, cat_cols):
    return _default_feature_engineer.clean_categoricals(df, list(cat_cols))


def minimal_preprocess(df: pd.DataFrame):
    return _default_feature_engineer.minimal_preprocess(df)


def preprocess_advanced(
    df: pd.DataFrame,
    num_cols: list[str],
    cat_cols: list[str],
    n_components: int = 3,
) -> pd.DataFrame:
    return _default_feature_engineer.preprocess_advanced(df, num_cols, cat_cols, n_components)


def preprocess_for_training(
    df: pd.DataFrame,
    num_cols: list[str],
    cat_cols: list[str],
) -> pd.DataFrame:
    return _default_feature_engineer.preprocess_for_training(df, num_cols, cat_cols)
