from __future__ import annotations
import pandas as pd
from typing import Tuple
from mlops import dataset, features

def make_clean_interim() -> str:
    """
    Carga dataset modificado, tipifica y limpia categóricas (solo texto).
    NO hace imputación para evitar data leakage.
    Solo limpieza determinística de texto (sin estadísticas globales).
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

def run_all() -> str:
    """
    Ejecuta el preprocesamiento completo.
    Retorna la ruta al archivo limpio que contiene features + target.
    """
    clean_path = make_clean_interim()
    return clean_path
