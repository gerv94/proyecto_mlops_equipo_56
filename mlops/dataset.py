from __future__ import annotations
import pandas as pd
from pathlib import Path
from .config import (MODIFIED_CSV, ORIGINAL_CSV, PREPROCESSED_CSV, DATA_INTERIM, DATA_PROCESSED,
                     TABLES)

# Aseguramos que las carpetas necesarias para el flujo de datos existan
DATA_INTERIM.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# FUNCIONES PRINCIPALES DE CARGA DE DATOS
# -----------------------------------------------------------------------------

def load_modified() -> pd.DataFrame:
    """
    Carga el dataset 'modificado' (limpio o preprocesado) desde la ruta
    especificada en MODIFIED_CSV.
    
    Returns:
        pd.DataFrame: DataFrame con el contenido del CSV modificado.
    """
    return pd.read_csv(MODIFIED_CSV)

def load_preprocessed() -> pd.DataFrame:
    """
    Carga el dataset 'preprocesado' (preprocesado) desde la ruta
    especificada en MODIFIED_CSV.
    
    Returns:
        pd.DataFrame: DataFrame con el contenido del CSV preprocesado.
    """
    return pd.read_csv(PREPROCESSED_CSV)

def load_original_if_exists() -> pd.DataFrame | None:
    """
    Carga el dataset 'original' (sin modificar) solo si el archivo existe.
    Esto es útil para comparar el dataset base con la versión modificada.

    Returns:
        pd.DataFrame | None: DataFrame con los datos originales, o None si no existe.
    """
    return pd.read_csv(ORIGINAL_CSV) if ORIGINAL_CSV.exists() else None

# -----------------------------------------------------------------------------
# FUNCIONES DE LIMPIEZA Y TIPIFICACIÓN
# -----------------------------------------------------------------------------

def coerce_numeric_col(s: pd.Series) -> pd.Series:
    """
    Intenta convertir una columna a formato numérico, eliminando símbolos comunes
    como comas, signos de dólar o porcentajes.

    Args:
        s (pd.Series): Columna individual del DataFrame.

    Returns:
        pd.Series: Columna con valores convertidos a tipo numérico si es posible.
    """
    # Si ya es numérica (int, float, etc.), se devuelve igual
    if s.dtype.kind in "biufc": 
        return s
    
    # Eliminamos caracteres no numéricos y normalizamos cadenas vacías
    cleaned = (s.astype(str)
                 .str.replace(r"[,%$]", "", regex=True)
                 .str.strip()
                 .replace({"": None, "nan": None, "None": None}))
    
    # Intentamos convertir a numérico (mantiene texto si no se puede convertir)
    return pd.to_numeric(cleaned, errors="ignore")

def basic_typing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica 'coerce_numeric_col' a todas las columnas del DataFrame para forzar
    la detección automática de tipos de datos.

    Args:
        df (pd.DataFrame): DataFrame de entrada.

    Returns:
        pd.DataFrame: Copia del DataFrame con columnas convertidas cuando es posible.
    """
    out = df.copy()
    for c in out.columns:
        out[c] = coerce_numeric_col(out[c])
    return out

# -----------------------------------------------------------------------------
# FUNCIONES DE GUARDADO DE DATOS
# -----------------------------------------------------------------------------

def save_interim(df: pd.DataFrame, name: str = "student_interim.csv") -> Path:
    """
    Guarda un DataFrame en la carpeta 'interim' (datos intermedios, aún no finales).

    Args:
        df (pd.DataFrame): DataFrame que se desea guardar.
        name (str): Nombre del archivo CSV de salida. Por defecto 'student_interim.csv'.

    Returns:
        Path: Ruta completa del archivo guardado.
    """
    p = DATA_INTERIM / name
    df.to_csv(p, index=False)
    return p


def save_processed(df: pd.DataFrame, name: str = "student_processed.csv") -> Path:
    """
    Guarda un DataFrame procesado (ya limpio o listo para modelar)
    en la carpeta 'processed'.

    Args:
        df (pd.DataFrame): DataFrame procesado que se desea guardar.
        name (str): Nombre del archivo CSV de salida. Por defecto 'student_processed.csv'.

    Returns:
        Path: Ruta completa del archivo guardado.
    """
    p = DATA_PROCESSED / name
    df.to_csv(p, index=False)
    return p
def load_interim(name: str) -> pd.DataFrame:
    """Carga un CSV desde data/interim/ por nombre de archivo."""
    p = DATA_INTERIM / name
    return pd.read_csv(p)
