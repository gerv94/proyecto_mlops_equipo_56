from __future__ import annotations
import os
from pathlib import Path
from typing import Any, Dict, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from mlops.utils.common import ensure_dir


class DataManager:
    """Manages data loading, saving, typification, and splits."""
    
    def load(self, path: str | Path, dtype_map: Dict[str, Any] | None = None) -> pd.DataFrame:
        """
        Load data from CSV or Parquet.
        
        Args:
            path: File path to load
            dtype_map: Optional dtype mapping for columns
            
        Returns:
            Loaded DataFrame
        """
        path_str = str(path)
        if path_str.endswith(".parquet"):
            return pd.read_parquet(path_str)
        return pd.read_csv(path_str, dtype=dtype_map)
    
    def save(self, data: pd.DataFrame, path: str | Path, format_type: str = "csv") -> str:
        """
        Save DataFrame to file.
        
        Args:
            data: DataFrame to save
            path: Output file path
            format_type: File format ('csv' or 'parquet')
            
        Returns:
            Path to saved file
        """
        path_obj = Path(path)
        ensure_dir(path_obj.parent)
        
        if format_type.lower() == "parquet":
            data.to_parquet(path, index=False)
        else:
            data.to_csv(path, index=False)
        
        return str(path)
    
    def coerce_numeric_column(self, series: pd.Series) -> pd.Series:
        """
        Attempt to convert a column to numeric, cleaning common symbols.
        
        Args:
            series: Input Series
            
        Returns:
            Converted Series (numeric if possible, otherwise original)
        """
        if series.dtype.kind in "biufc":
            return series
        
        cleaned = (
            series.astype(str)
            .str.replace(r"[,%$]", "", regex=True)
            .str.strip()
            .replace({"": None, "nan": None, "None": None})
        )
        
        try:
            return pd.to_numeric(cleaned)
        except (ValueError, TypeError):
            return cleaned
    
    def basic_typing(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Apply numeric coercion to all columns.
        
        Args:
            dataframe: Input DataFrame
            
        Returns:
            DataFrame with coerced types
        """
        result = dataframe.copy()
        for column in result.columns:
            result[column] = self.coerce_numeric_column(result[column])
        return result
    
    def split_train_test(
        self, 
        dataframe: pd.DataFrame, 
        test_size: float = 0.2, 
        random_state: int = 42,
        stratify_column: pd.Series | None = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split DataFrame into train and test sets.
        
        Args:
            dataframe: Input DataFrame
            test_size: Proportion for test set
            random_state: Random seed
            stratify_column: Column for stratified splitting
            
        Returns:
            Tuple of (train_df, test_df)
        """
        return train_test_split(
            dataframe, 
            test_size=test_size, 
            random_state=random_state,
            stratify=stratify_column
        )
    
    def infer_dtypes(self, dataframe: pd.DataFrame) -> Dict[str, Any]:
        """Get dtype mapping for DataFrame columns."""
        return {name: series.dtype for name, series in dataframe.items()}
    
    def cast_dtypes(self, dataframe: pd.DataFrame, dtype_map: Dict[str, Any]) -> pd.DataFrame:
        """Cast DataFrame columns to specified dtypes."""
        return dataframe.astype(dtype_map)
