import pandas as pd
import numpy as np
from .config import TABLES
from mlops.core.feature_engineering import FeatureEngineering
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA

TABLES.mkdir(parents=True, exist_ok=True)

_feature_eng = FeatureEngineering()
CATEGORICAL_GUESS_MAX = 30


# -----------------------------------------------------------------------------
# FUNCIONES PARA CLASIFICAR COLUMNAS NUMÉRICAS Y CATEGÓRICAS
# -----------------------------------------------------------------------------
def split_num_cat(dataframe: pd.DataFrame):
    """Split columns into numeric and categorical (delegates to FeatureEngineering)."""
    return _feature_eng.split_numeric_categorical(dataframe)


# -----------------------------------------------------------------------------
# FUNCIONES DE LIMPIEZA DE VARIABLES CATEGÓRICAS
# -----------------------------------------------------------------------------
def normalize_categories(series: pd.Series) -> pd.Series:
    """Normalize categorical text (delegates to FeatureEngineering)."""
    return _feature_eng.normalize_categories(series)

def clean_categoricals(dataframe: pd.DataFrame, categorical_cols):
    """Clean categorical columns (delegates to FeatureEngineering)."""
    return _feature_eng.clean_categoricals(dataframe, categorical_cols)


# -----------------------------------------------------------------------------
# PREPROCESAMIENTO BÁSICO (para EDA o baseline)
# -----------------------------------------------------------------------------
def minimal_preprocess(dataframe: pd.DataFrame):
    """Minimal imputation preprocessing (delegates to FeatureEngineering)."""
    numeric_cols, categorical_cols = split_num_cat(dataframe)
    imputed = _feature_eng.minimal_impute(dataframe)
    return imputed, numeric_cols, categorical_cols


# -----------------------------------------------------------------------------
# PREPROCESAMIENTO AVANZADO (ESCALADO + ONE-HOT + PCA)
# -----------------------------------------------------------------------------
def preprocess_advanced(
    dataframe: pd.DataFrame,
    num_cols: list[str],
    cat_cols: list[str],
    n_components: int = 3
) -> pd.DataFrame:
    """Advanced preprocessing with scaling, encoding, and PCA."""
    df_processed = dataframe.copy()
    parts = []
    
    if num_cols:
        scaled_df, _ = _feature_eng.scale_numerics(df_processed, num_cols)
        parts.append(scaled_df[num_cols])
    
    if cat_cols:
        encoded_frame, _ = _feature_eng.encode_categoricals(df_processed, cat_cols)
        if encoded_frame is not None:
            parts.append(encoded_frame)
    
    if not parts:
        print("No columns to transform")
        return df_processed
    
    combined = pd.concat(parts, axis=1)
    
    if n_components and n_components > 0:
        combined, _ = _feature_eng.apply_pca(combined, n_components)
    
    print("Advanced preprocessing complete")
    return combined
