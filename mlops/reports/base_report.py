from __future__ import annotations
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from mlops.reports.insights import InsightGenerator
from mlops.core.feature_engineering import FeatureEngineering
from mlops.utils.common import guess_target, safe_write_text


class BaseReportGenerator:
    """Base class for HTML report generation with Plotly visualizations."""
    
    PALETTE = ["#041B25", "#0E3A4B", "#4A7486", "#AFC4B2", "#EFF5DE"]
    
    def __init__(self):
        self.insight_gen = InsightGenerator()
        self.feature_eng = FeatureEngineering()
    
    def generate(self, dataframe: pd.DataFrame, target: str | None, output_path: str | Path) -> Path:
        """
        Generate HTML report.
        
        Args:
            dataframe: Data for report
            target: Target column name
            output_path: Output file path
            
        Returns:
            Path to generated HTML
        """
        target = target or guess_target(dataframe)
        html_content = self.build_html(dataframe, target)
        return safe_write_text(output_path, html_content)
    
    def build_html(self, dataframe: pd.DataFrame, target: str | None) -> str:
        """Build complete HTML content."""
        raise NotImplementedError("Subclasses must implement build_html()")
    
    def fig_target_distribution(self, dataframe: pd.DataFrame, target: str):
        """Generate target distribution figure."""
        if not target or target not in dataframe.columns:
            return None
        
        value_counts = dataframe[target].value_counts(dropna=False).reset_index()
        value_counts.columns = [target, "count"]
        
        fig = px.bar(
            value_counts, 
            x=target, 
            y="count",
            title=f"Distribution: {target}",
            color_discrete_sequence=[self.PALETTE[0]]
        )
        fig.update_layout(height=360)
        return fig
    
    def fig_missing_heatmap(self, dataframe: pd.DataFrame):
        """Generate missing values heatmap."""
        missing_matrix = dataframe.isna().astype(int).T
        
        fig = px.imshow(
            missing_matrix,
            color_continuous_scale=[self.PALETTE[4], self.PALETTE[1]],
            aspect="auto",
            origin="lower"
        )
        fig.update_layout(title="Missing values (1 = missing)", height=420, coloraxis_colorbar_title="NA")
        fig.update_yaxes(tickmode="array", ticktext=list(dataframe.columns), tickvals=list(range(len(dataframe.columns))))
        return fig
    
    def fig_corr_heatmap(self, dataframe: pd.DataFrame):
        """Generate correlation heatmap."""
        numeric_data = dataframe.select_dtypes(include=[np.number])
        
        if numeric_data.shape[1] < 2:
            return None
        
        corr = numeric_data.corr(numeric_only=True)
        
        fig = px.imshow(
            corr,
            text_auto=False,
            color_continuous_scale="RdBu",
            zmin=-1,
            zmax=1,
            origin="lower"
        )
        fig.update_layout(title="Correlation matrix", height=520)
        return fig
    
    def figs_numeric(self, dataframe: pd.DataFrame):
        """Generate histogram + boxplot figures for numeric columns."""
        numeric_cols, _ = self.feature_eng.split_numeric_categorical(dataframe)
        figures = []
        
        for column in numeric_cols:
            fig = px.histogram(
                dataframe,
                x=column,
                nbins=30,
                marginal="box",
                title=f"Histogram + Box: {column}",
                color_discrete_sequence=[self.PALETTE[2]]
            )
            fig.update_layout(height=340)
            insight = self.insight_gen.numeric_insight(dataframe, column)
            figures.append((fig, insight))
        
        return figures
    
    def figs_categorical(self, dataframe: pd.DataFrame, max_bars: int = 20):
        """Generate bar charts for categorical columns."""
        _, categorical_cols = self.feature_eng.split_numeric_categorical(dataframe)
        figures = []
        
        for column in categorical_cols:
            n_unique = dataframe[column].nunique(dropna=False)
            
            if 2 <= n_unique <= max_bars:
                value_counts = dataframe[column].value_counts(dropna=False).reset_index()
                value_counts.columns = [column, "count"]
                
                fig = px.bar(
                    value_counts,
                    x=column,
                    y="count",
                    title=f"Distribution: {column}",
                    color_discrete_sequence=[self.PALETTE[1]]
                )
                fig.update_layout(height=340, xaxis_tickangle=-30)
                insight = self.insight_gen.categorical_insight(dataframe, column)
                figures.append((fig, insight))
        
        return figures
