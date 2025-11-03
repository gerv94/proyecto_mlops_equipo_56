from __future__ import annotations
import os
import re
import time
import random
from functools import wraps
from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd


def ensure_dir(path: str | Path) -> None:
    """Ensure directory exists, creating it if necessary."""
    if not path:
        return
    Path(path).mkdir(parents=True, exist_ok=True)


def slugify(text: str) -> str:
    """Convert text to URL-friendly slug."""
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", str(text)).strip("_").lower()


def set_random_seed(seed: int) -> None:
    """Set random seed for reproducibility across numpy and Python random."""
    random.seed(seed)
    np.random.seed(seed)


def guess_target(dataframe: pd.DataFrame, explicit_target: str | None = None) -> str | None:
    """
    Heuristically detect target column in DataFrame.
    
    Args:
        dataframe: Input DataFrame
        explicit_target: If provided, return this directly
        
    Returns:
        Target column name, or None if not found
    """
    if explicit_target and explicit_target in dataframe.columns:
        return explicit_target
    
    candidates = ["Performance", "performance", "target", "label", "Target", "Label", "y", "class"]
    for candidate in candidates:
        if candidate in dataframe.columns:
            return candidate
    
    categorical_candidates = [
        (column, dataframe[column].nunique(dropna=True)) 
        for column in dataframe.columns
    ]
    viable = [column for column, nunique in categorical_candidates if 2 <= nunique <= 10]
    
    return viable[0] if viable else None


def split_features_target(dataframe: pd.DataFrame, target: str | None = None) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split DataFrame into features and target.
    
    Args:
        dataframe: Input DataFrame
        target: Target column name (auto-detected if None)
        
    Returns:
        Tuple of (features_df, target_series)
    """
    target_column = guess_target(dataframe, target)
    if not target_column:
        raise ValueError("Could not detect target column")
    
    features = dataframe.drop(columns=[target_column])
    target_series = dataframe[target_column]
    return features, target_series


def timeit(function):
    """Decorator to measure and print function execution time."""
    @wraps(function)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = function(*args, **kwargs)
        elapsed_ms = (time.time() - start_time) * 1000
        print(f"{function.__name__} completed: {elapsed_ms:.1f} ms")
        return result
    return wrapper


def safe_write_text(path: str | Path, text: str, encoding: str = "utf-8") -> Path:
    """Safely write text to file, creating parent directories if needed."""
    file_path = Path(path)
    ensure_dir(file_path.parent)
    file_path.write_text(text, encoding=encoding)
    return file_path
