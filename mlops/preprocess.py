from __future__ import annotations
import pandas as pd
from typing import Tuple
from mlops import dataset, features
from mlops.core.preprocessor import CleanPreprocessor, AdvancedPreprocessor
from mlops.config import CLEAN_CSV, PREPROCESSED_CSV

def make_clean_interim() -> str:
    """Load, clean, and save interim clean dataset using CleanPreprocessor."""
    dataframe = dataset.load_modified()
    dataframe = dataset.basic_typing(dataframe)
    
    clean_preprocessor = CleanPreprocessor()
    df_clean = clean_preprocessor.run(dataframe)
    
    dataset._data_manager.save(df_clean, CLEAN_CSV)
    return str(CLEAN_CSV)

def make_preprocessed_from_clean() -> str:
    """Apply advanced preprocessing using AdvancedPreprocessor."""
    df_clean = dataset._data_manager.load(CLEAN_CSV)
    
    numeric_cols, categorical_cols = features.split_num_cat(df_clean)
    
    df_ready = features.preprocess_advanced(
        df_clean,
        num_cols=numeric_cols,
        cat_cols=categorical_cols,
        n_components=3
    )
    
    dataset._data_manager.save(df_ready, PREPROCESSED_CSV)
    return str(PREPROCESSED_CSV)

def run_all() -> Tuple[str, str]:
    clean_path = make_clean_interim()
    ready_path = make_preprocessed_from_clean()
    return clean_path, ready_path
