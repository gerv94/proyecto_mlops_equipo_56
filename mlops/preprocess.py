from __future__ import annotations
import pandas as pd
from typing import Tuple
from mlops import dataset, features

def make_clean_interim() -> str:
    """
    Carga dataset modificado, tipifica, limpia categóricas e imputa mínimos.
    Devuelve la ruta al CSV limpio intermedio guardado en data/interim/.
    """
    df = dataset.load_modified()
    df = dataset.basic_typing(df)

    # Normaliza texto en categóricas
    _, cat_cols = features.split_num_cat(df)
    df = features.clean_categoricals(df, cat_cols)

    # Imputación mínima (usando tu helper actual)
    df_clean, _, _ = features.minimal_preprocess(df)

    out_clean = dataset.save_interim(df_clean, "student_interim_clean.csv")
    return out_clean

def make_preprocessed_from_clean() -> str:
    """
    Toma el limpio intermedio y aplica preprocesamiento avanzado
    (escalado, One-Hot, PCA si así lo definieron dentro de features.preprocess_advanced).
    """
    # Carga el limpio intermedio recién generado
    df_clean = dataset.load_interim("student_interim_clean.csv")

    # Selección de columnas tras limpieza
    num_cols_clean, cat_cols_clean = features.split_num_cat(df_clean)

    # Aplica el pipeline avanzado ya existente
    df_ready = features.preprocess_advanced(
        df_clean,
        num_cols=num_cols_clean,
        cat_cols=cat_cols_clean,
        n_components=3  # parametrizable después
    )

    out_ready = dataset.save_interim(df_ready, "student_interim_preprocessed.csv")
    return out_ready

def run_all() -> Tuple[str, str]:
    clean_path = make_clean_interim()
    ready_path = make_preprocessed_from_clean()
    return clean_path, ready_path
