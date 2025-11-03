from __future__ import annotations
import pandas as pd
from mlops.core.feature_engineering import FeatureEngineering


class BasePreprocessor:
    """Base preprocessor with Template Method pattern: validate -> transform -> finalize."""
    
    def __init__(self):
        self.feature_eng = FeatureEngineering()
    
    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        """Main preprocessing pipeline."""
        self.validate(data)
        processed = self.transform(data.copy())
        return self.finalize(processed)
    
    def validate(self, data: pd.DataFrame) -> None:
        """Optional validation step."""
        if data.empty:
            raise ValueError("Cannot preprocess empty DataFrame")
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform data - to be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement transform()")
    
    def finalize(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optional finalization step."""
        return data


class CleanPreprocessor(BasePreprocessor):
    """Preprocessor for basic cleaning and imputation."""
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean and impute data."""
        numeric_cols, categorical_cols = self.feature_eng.split_numeric_categorical(data)
        data = self.feature_eng.clean_categoricals(data, categorical_cols)
        data = self.feature_eng.minimal_impute(data)
        return data


class AdvancedPreprocessor(BasePreprocessor):
    """Preprocessor for advanced transformations: scaling, encoding, PCA."""
    
    def __init__(self, n_components: int = 3):
        super().__init__()
        self.n_components = n_components
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply advanced preprocessing."""
        numeric_cols, categorical_cols = self.feature_eng.split_numeric_categorical(data)
        
        parts = []
        
        if numeric_cols:
            scaled_data, _ = self.feature_eng.scale_numerics(data, numeric_cols)
            parts.append(scaled_data[numeric_cols])
        
        if categorical_cols:
            encoded_frame, _ = self.feature_eng.encode_categoricals(data, categorical_cols)
            if encoded_frame is not None:
                parts.append(encoded_frame)
        
        if not parts:
            print("No columns to transform")
            return data
        
        combined = pd.concat(parts, axis=1)
        
        if self.n_components > 0:
            combined, _ = self.feature_eng.apply_pca(combined, self.n_components)
        
        print("Advanced preprocessing complete")
        return combined
