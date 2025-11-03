from __future__ import annotations
import pandas as pd
import numpy as np
from mlops.core.feature_engineering import FeatureEngineering
from mlops.utils.common import guess_target


class InsightGenerator:
    """Generate automated insights for EDA reports."""
    
    def __init__(self):
        self.feature_eng = FeatureEngineering()
    
    def global_overview(self, dataframe: pd.DataFrame, target: str | None = None) -> str:
        """Generate global dataset overview insight."""
        rows, cols = dataframe.shape
        numeric_cols, categorical_cols = self.feature_eng.split_numeric_categorical(dataframe)
        n_numeric, n_categorical = len(numeric_cols), len(categorical_cols)
        
        null_pct = float(dataframe.isna().mean().mean() * 100)
        nulls_by_col = (dataframe.isna().mean() * 100).sort_values(ascending=False)
        high_nulls = nulls_by_col[nulls_by_col > 10].index.tolist()
        
        duplicates = int(dataframe.duplicated().sum())
        mem_mb = float(dataframe.memory_usage(deep=True).sum() / 1e6)
        
        target_text = f"Target: {target}." if target else "No target detected."
        nulls_text = f"{null_pct:.2f}% nulls avg"
        if high_nulls:
            nulls_text += f"; columns with >10% nulls: {', '.join(high_nulls)}"
        
        return (
            f"Dataset: <b>{rows}</b> rows × <b>{cols}</b> columns "
            f"({n_numeric} numeric, {n_categorical} categorical). "
            f"{target_text} {nulls_text}. "
            f"Duplicates: {duplicates}. Memory: {mem_mb:.2f} MB."
        )
    
    def target_analysis(self, dataframe: pd.DataFrame, target: str) -> str:
        """Generate target variable insight."""
        if not target or target not in dataframe.columns:
            return "No target column found."
        
        value_counts = dataframe[target].value_counts(normalize=True, dropna=False).sort_values(ascending=False)
        
        if value_counts.empty:
            return "Target column has no data."
        
        top_pct = value_counts.iloc[0] * 100.0
        balance_status = "balanced" if value_counts.max() < 0.6 else "imbalanced"
        
        return (
            f"Target <b>{target}</b> is {balance_status}. "
            f"Most frequent class: {top_pct:.1f}%."
        )
    
    def missingness_summary(self, dataframe: pd.DataFrame) -> str:
        """Generate missingness insight."""
        nulls_by_col = (dataframe.isna().mean() * 100).sort_values(ascending=False)
        
        if nulls_by_col.max() == 0:
            return "No missing values detected."
        
        top_3 = nulls_by_col.head(3)
        items = [f"{col} ({val:.1f}%)" for col, val in top_3.items()]
        return "Top missing columns: " + ", ".join(items) + "."
    
    def correlation_summary(self, dataframe: pd.DataFrame) -> str:
        """Generate correlation insight."""
        numeric_data = dataframe.select_dtypes(include=[np.number])
        
        if numeric_data.shape[1] < 2:
            return "Insufficient numeric columns for correlation."
        
        corr = numeric_data.corr(numeric_only=True).copy()
        np.fill_diagonal(corr.values, 0.0)
        
        max_pair, max_val = None, 0.0
        for idx, col1 in enumerate(corr.columns):
            for jdx, col2 in enumerate(corr.columns):
                if jdx > idx:
                    val = abs(float(corr.loc[col1, col2]))
                    if val > max_val:
                        max_val, max_pair = val, (col1, col2)
        
        if not max_pair:
            return "No notable correlations found."
        
        return f"Highest correlation: {max_pair[0]}–{max_pair[1]} (|r|={max_val:.2f})."
    
    def numeric_insight(self, dataframe: pd.DataFrame, column: str) -> str:
        """Generate insight for numeric column."""
        series = dataframe[column].dropna()
        
        if series.empty:
            return f"{column}: no data."
        
        mean_val = float(series.mean())
        std_val = float(series.std(ddof=0))
        skew_val = float(series.skew() if hasattr(series, "skew") else 0.0)
        
        skew_desc = "symmetric" if abs(skew_val) < 0.3 else ("right-skewed" if skew_val > 0 else "left-skewed")
        
        return f"{column}: mean={mean_val:.2f}, sd={std_val:.2f}, {skew_desc}."
    
    def categorical_insight(self, dataframe: pd.DataFrame, column: str) -> str:
        """Generate insight for categorical column."""
        value_counts = dataframe[column].value_counts(dropna=False, normalize=True)
        
        if value_counts.empty:
            return f"{column}: no data."
        
        n_unique = len(value_counts)
        categories = [str(cat) for cat in value_counts.index.tolist()]
        categories_str = ", ".join(categories[:15]) + (", ..." if len(categories) > 15 else "")
        
        top_category = value_counts.index[0]
        top_pct = value_counts.iloc[0] * 100
        
        return (
            f"<b>{column}</b>: {n_unique} categories ({categories_str}). "
            f"Most frequent: '{top_category}' ({top_pct:.1f}%)."
        )
