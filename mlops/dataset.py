from __future__ import annotations
import pandas as pd
from pathlib import Path
from .config import (MODIFIED_CSV, ORIGINAL_CSV, PREPROCESSED_CSV, DATA_INTERIM, DATA_PROCESSED, TABLES)
from mlops.core.data_manager import DataManager

_data_manager = DataManager()

DATA_INTERIM.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# FUNCIONES PRINCIPALES DE CARGA DE DATOS
# -----------------------------------------------------------------------------

def load_modified() -> pd.DataFrame:
    """Load modified dataset."""
    return _data_manager.load(MODIFIED_CSV)

def load_preprocessed() -> pd.DataFrame:
    """Load preprocessed dataset."""
    return _data_manager.load(PREPROCESSED_CSV)

def load_original_if_exists() -> pd.DataFrame | None:
    """Load original dataset if exists."""
    return _data_manager.load(ORIGINAL_CSV) if ORIGINAL_CSV.exists() else None

# -----------------------------------------------------------------------------
# FUNCIONES DE LIMPIEZA Y TIPIFICACIÓN
# -----------------------------------------------------------------------------

def coerce_numeric_col(series: pd.Series) -> pd.Series:
    """Coerce column to numeric (delegates to DataManager)."""
    return _data_manager.coerce_numeric_column(series)

def basic_typing(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Apply type coercion to all columns (delegates to DataManager)."""
    return _data_manager.basic_typing(dataframe)

# -----------------------------------------------------------------------------
# FUNCIONES DE GUARDADO DE DATOS
# -----------------------------------------------------------------------------

def save_interim(dataframe: pd.DataFrame, name: str = "student_interim.csv") -> Path:
    """Save DataFrame to interim folder."""
    path = DATA_INTERIM / name
    _data_manager.save(dataframe, path)
    return path

def save_processed(dataframe: pd.DataFrame, name: str = "student_processed.csv") -> Path:
    """Save DataFrame to processed folder."""
    path = DATA_PROCESSED / name
    _data_manager.save(dataframe, path)
    return path

def load_interim(name: str) -> pd.DataFrame:
    """Load CSV from data/interim/ by filename."""
    return _data_manager.load(DATA_INTERIM / name)
