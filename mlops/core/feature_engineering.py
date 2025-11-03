from __future__ import annotations
from typing import Tuple, List
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA


class FeatureEngineering:
    """Feature engineering operations: encoding, scaling, and transformations."""
    
    CATEGORICAL_GUESS_MAX = 30
    
    def split_numeric_categorical(self, dataframe: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """
        Split columns into numeric and categorical based on type and cardinality.
        
        Args:
            dataframe: Input DataFrame
            
        Returns:
            Tuple of (numeric_columns, categorical_columns)
        """
        numeric_columns = [
            column for column in dataframe.columns 
            if pd.api.types.is_numeric_dtype(dataframe[column])
        ]
        
        categorical_columns = [
            column for column in dataframe.columns 
            if column not in numeric_columns
        ]
        
        low_cardinality = {
            column for column in dataframe.columns 
            if dataframe[column].nunique(dropna=True) <= self.CATEGORICAL_GUESS_MAX
        }
        
        categorical_columns = list(set(categorical_columns) | low_cardinality)
        numeric_columns = [column for column in dataframe.columns if column not in categorical_columns]
        
        return numeric_columns, categorical_columns
    
    def normalize_categories(self, series: pd.Series) -> pd.Series:
        """
        Normalize categorical text: lowercase, trim, collapse spaces.
        
        Args:
            series: Input Series
            
        Returns:
            Normalized Series
        """
        if series.dtype == object:
            return (
                series.astype(str)
                .str.strip()
                .str.replace(r"\s+", " ", regex=True)
                .str.lower()
            )
        return series
    
    def clean_categoricals(self, dataframe: pd.DataFrame, categorical_columns: List[str]) -> pd.DataFrame:
        """
        Apply category normalization to categorical columns.
        
        Args:
            dataframe: Input DataFrame
            categorical_columns: List of categorical column names
            
        Returns:
            DataFrame with normalized categoricals
        """
        result = dataframe.copy()
        for column in categorical_columns:
            result[column] = self.normalize_categories(result[column])
        return result
    
    def encode_categoricals(
        self, 
        dataframe: pd.DataFrame, 
        categorical_columns: List[str],
        strategy: str = "onehot"
    ) -> Tuple[pd.DataFrame, OneHotEncoder | None]:
        """
        Encode categorical columns.
        
        Args:
            dataframe: Input DataFrame
            categorical_columns: List of categorical column names
            strategy: Encoding strategy ('onehot' supported)
            
        Returns:
            Tuple of (encoded_df, encoder)
        """
        if not categorical_columns or strategy != "onehot":
            return dataframe, None
        
        try:
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        except TypeError:
            encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        
        encoded = encoder.fit_transform(dataframe[categorical_columns])
        encoded_columns = encoder.get_feature_names_out(categorical_columns)
        encoded_frame = pd.DataFrame(encoded, columns=encoded_columns, index=dataframe.index)
        
        return encoded_frame, encoder
    
    def scale_numerics(
        self, 
        dataframe: pd.DataFrame, 
        numeric_columns: List[str]
    ) -> Tuple[pd.DataFrame, StandardScaler | None]:
        """
        Scale numeric columns using StandardScaler.
        
        Args:
            dataframe: Input DataFrame
            numeric_columns: List of numeric column names
            
        Returns:
            Tuple of (scaled_df, scaler)
        """
        if not numeric_columns:
            return dataframe, None
        
        scaler = StandardScaler()
        scaled_values = scaler.fit_transform(dataframe[numeric_columns])
        
        result = dataframe.copy()
        result[numeric_columns] = scaled_values
        return result, scaler
    
    def apply_pca(
        self, 
        dataframe: pd.DataFrame, 
        n_components: int = 3,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, PCA | None]:
        """
        Apply PCA transformation and append components to DataFrame.
        
        Args:
            dataframe: Input DataFrame
            n_components: Number of principal components
            random_state: Random seed
            
        Returns:
            Tuple of (df_with_pca, pca_model)
        """
        if n_components <= 0:
            return dataframe, None
        
        max_components = min(n_components, dataframe.shape[1])
        if max_components < 1:
            return dataframe, None
        
        pca = PCA(n_components=max_components, random_state=random_state)
        components = pca.fit_transform(dataframe)
        
        pca_columns = [f"PC{i+1}" for i in range(max_components)]
        pca_frame = pd.DataFrame(components, columns=pca_columns, index=dataframe.index)
        
        print(f"PCA variance explained: {np.round(pca.explained_variance_ratio_, 3)}")
        
        return pd.concat([dataframe, pca_frame], axis=1), pca
    
    def minimal_impute(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Simple imputation: median for numeric, mode for categorical.
        
        Args:
            dataframe: Input DataFrame
            
        Returns:
            Imputed DataFrame
        """
        numeric_columns, categorical_columns = self.split_numeric_categorical(dataframe)
        result = dataframe.copy()
        
        for column in numeric_columns:
            result[column] = result[column].fillna(result[column].median())
        
        for column in categorical_columns:
            mode_value = result[column].mode()
            if not mode_value.empty:
                result[column] = result[column].fillna(mode_value.iloc[0])
        
        return result
