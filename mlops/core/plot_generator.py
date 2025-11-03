from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from mlops.utils.common import ensure_dir


class PlotGenerator:
    """Unified plotting for EDA and model evaluation."""
    
    def __init__(self, output_dir: str | Path | None = None, dpi: int = 300):
        """
        Initialize plot generator.
        
        Args:
            output_dir: Directory for saving plots
            dpi: Resolution for saved images
        """
        self.output_dir = Path(output_dir) if output_dir else None
        self.dpi = dpi
    
    def _get_output_path(self, filename: str) -> Path:
        """Get full output path for a plot."""
        if self.output_dir:
            ensure_dir(self.output_dir)
            return self.output_dir / filename
        return Path(filename)
    
    def _save_and_close(self, path: Path) -> str:
        """Save current figure and close it."""
        plt.tight_layout()
        plt.savefig(path, dpi=self.dpi)
        plt.close()
        return str(path)
    
    def target_distribution(self, data: pd.DataFrame, target_col: str, filename: str = "target_distribution.png") -> str:
        """Plot target variable distribution."""
        output_path = self._get_output_path(filename)
        
        plt.figure(figsize=(6, 4))
        value_counts = data[target_col].value_counts()
        sns.countplot(x=data[target_col], order=value_counts.index)
        plt.title(f"Distribution: {target_col}")
        plt.xlabel(target_col)
        plt.ylabel("Count")
        
        return self._save_and_close(output_path)
    
    def missingness(self, data: pd.DataFrame, filename: str = "missingness_by_column.png") -> str:
        """Plot missing values by column."""
        output_path = self._get_output_path(filename)
        
        plt.figure(figsize=(8, max(4, 0.25 * len(data.columns))))
        data.isna().mean().sort_values().plot(kind="barh", color="#4A7486")
        plt.title("Missing values by column")
        plt.xlabel("Proportion missing")
        
        return self._save_and_close(output_path)
    
    def numeric_distribution(self, data: pd.DataFrame, column: str, filename_prefix: str) -> tuple[str, str]:
        """Plot histogram and boxplot for numeric column."""
        hist_path = self._get_output_path(f"{filename_prefix}_hist.png")
        box_path = self._get_output_path(f"{filename_prefix}_box.png")
        
        plt.figure(figsize=(5, 3))
        sns.histplot(data[column].dropna(), bins=30, kde=True, color="#0E3A4B")
        plt.title(f"Histogram: {column}")
        hist_result = self._save_and_close(hist_path)
        
        plt.figure(figsize=(5, 2.8))
        sns.boxplot(x=data[column], orient="h", color="#AFC4B2")
        plt.title(f"Boxplot: {column}")
        box_result = self._save_and_close(box_path)
        
        return hist_result, box_result
    
    def categorical_distribution(self, data: pd.DataFrame, column: str, filename: str, max_categories: int = 20) -> str | None:
        """Plot categorical variable distribution."""
        n_unique = data[column].nunique(dropna=False)
        
        if not (2 <= n_unique <= max_categories):
            return None
        
        output_path = self._get_output_path(filename)
        
        plt.figure(figsize=(7, 3))
        value_order = data[column].value_counts(dropna=False).index
        sns.countplot(y=data[column], order=value_order, palette="crest")
        plt.title(f"Category: {column} (n={n_unique})")
        plt.xlabel("Count")
        plt.ylabel(column)
        
        return self._save_and_close(output_path)
    
    def correlation_heatmap(self, data: pd.DataFrame, filename: str = "correlation_matrix.png") -> str | None:
        """Plot correlation heatmap for numeric columns."""
        numeric_data = data.select_dtypes(include=["number"])
        
        if numeric_data.shape[1] < 2:
            return None
        
        output_path = self._get_output_path(filename)
        
        plt.figure(figsize=(10, 8))
        corr = numeric_data.corr(numeric_only=True)
        sns.heatmap(corr, annot=False, cmap="coolwarm", center=0, vmin=-1, vmax=1)
        plt.title("Correlation Matrix")
        
        return self._save_and_close(output_path)
    
    def confusion_matrix(self, y_true, y_pred, labels, filename: str = "confusion_matrix.png") -> str:
        """Plot confusion matrix."""
        from sklearn.metrics import confusion_matrix as compute_cm
        
        output_path = self._get_output_path(filename)
        
        confusion_mat = compute_cm(y_true, y_pred)
        
        plt.figure(figsize=(6, 4))
        sns.heatmap(confusion_mat, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        
        return self._save_and_close(output_path)
