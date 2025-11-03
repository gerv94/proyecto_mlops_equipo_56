from __future__ import annotations
import pandas as pd
from typing import Tuple
from mlops import dataset, features

def make_clean_interim() -> str:
    """
    Carga dataset modificado, tipifica y limpia categóricas (solo texto).
    NO hace imputación para evitar data leakage.
    Sigue la misma lógica que run_eda_2.py.
    Devuelve la ruta al CSV limpio intermedio guardado en data/interim/.
    """
    df = dataset.load_modified()
    df = dataset.basic_typing(df)

    # Normaliza texto en categóricas (limpieza determinística, no causa leakage)
    num_cols, cat_cols = features.split_num_cat(df)
    df_clean = features.clean_categoricals(df, cat_cols)

    # NOTA: No hacemos imputación aquí para evitar data leakage.
    # La imputación debe hacerse dentro del pipeline después de train/test split.

    out_clean = dataset.save_interim(df_clean, "student_interim_clean.csv")
    return out_clean

def make_preprocessed_from_clean() -> str:
    """
    Toma el limpio intermedio y aplica preprocesamiento para entrenamiento
    (OneHotEncoder sin PCA ni StandardScaler, alineado con el notebook).
    """
    # Carga el limpio intermedio recién generado
    df_clean = dataset.load_interim("student_interim_clean.csv")

    # Selección de columnas tras limpieza
    num_cols_clean, cat_cols_clean = features.split_num_cat(df_clean)
    
    # Excluir la columna target (Performance) de las categóricas si existe
    target_col = 'Performance'
    if target_col in cat_cols_clean:
        cat_cols_clean = [c for c in cat_cols_clean if c != target_col]

    # Aplica el preprocesamiento para entrenamiento (sin PCA, sin escalado)
    df_ready = features.preprocess_for_training(
        df_clean,
        num_cols=num_cols_clean,
        cat_cols=cat_cols_clean
    )

    out_ready = dataset.save_interim(df_ready, "student_entry_performance_original.csv")
    return out_ready

def run_all() -> Tuple[str, str]:
    clean_path = make_clean_interim()
    ready_path = make_preprocessed_from_clean()
    return clean_path, ready_path
