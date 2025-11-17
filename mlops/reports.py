# -----------------------------------------------------------------------------
# Sistema de Reportes HTML Interactivos (Orientado a Objetos)
# -----------------------------------------------------------------------------
# Arquitectura MLOps: Clases base y especializadas para diferentes tipos
# de reportes (EDA, Preprocessed, Models) con código común reutilizable.
# -----------------------------------------------------------------------------

from pathlib import Path
from abc import ABC, abstractmethod
from typing import Optional, Dict, List, Tuple
import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import mlflow

from .config import REPORTS, PREPROCESSED_CSV
from . import features

# -----------------------------------------------------------------------------
# Configuración Global
# -----------------------------------------------------------------------------

# Paleta de colores (Ocean Serenity) para reportes EDA
PALETTE_EDA = ["#041B25", "#0E3A4B", "#4A7486", "#AFC4B2", "#EFF5DE"]

# Paleta de colores moderna para reportes de modelos
PALETTE_MODELS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", 
                  "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]


# -----------------------------------------------------------------------------
# Clase Base Abstracta
# -----------------------------------------------------------------------------

class ReportBase(ABC):
    """
    Clase base abstracta para todos los reportes HTML.
    Define la interfaz común y métodos compartidos.
    """
    
    def __init__(self, output_dir: Path, palette: List[str] = None):
        """
        Args:
            output_dir: Directorio donde se guardarán los reportes
            palette: Lista de colores para las visualizaciones
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.palette = palette or PALETTE_EDA
    
    @abstractmethod
    def generate(self, *args, **kwargs) -> Path:
        """
        Método abstracto que debe ser implementado por cada clase hija.
        Genera el reporte HTML y retorna la ruta del archivo.
        """
        pass
    
    @staticmethod
    def guess_target(df: pd.DataFrame) -> Optional[str]:
        """
        Intenta detectar automáticamente la columna objetivo (target).
        """
        for c in ["Performance", "performance", "target", "label", "Target", "Label"]:
            if c in df.columns:
                return c
        # Fallback: busca columnas categóricas con pocas clases (2–10)
        cands = [(c, df[c].nunique(dropna=True)) for c in df.columns]
        cands = [c for c, n in cands if 2 <= n <= 10]
        return cands[0] if cands else None
    
    @staticmethod
    def compute_summary_metrics(df: pd.DataFrame) -> dict:
        """
        Calcula métricas descriptivas generales sobre el DataFrame.
        """
        rows, cols = df.shape
        num_cols, cat_cols = features.split_num_cat(df)
        n_num, n_cat = len(num_cols), len(cat_cols)
        null_pct_mean = float(df.isna().mean().mean() * 100)
        dups = int(df.duplicated().sum())
        mem_mb = float(df.memory_usage(deep=True).sum() / 1e6)
        
        # Cardinalidad por columna
        cardinality = df.nunique(dropna=True).sort_values(ascending=False)
        cardinality_mean = float(cardinality.mean()) if len(cardinality) else 0.0
        
        return {
            "rows": rows,
            "cols": cols,
            "n_num": n_num,
            "n_cat": n_cat,
            "null_pct_mean": round(null_pct_mean, 2),
            "dups": dups,
            "mem_mb": round(mem_mb, 2),
            "cardinality": cardinality,
            "cardinality_mean": round(cardinality_mean, 2),
            "num_cols": num_cols,
            "cat_cols": cat_cols,
        }
    
    @staticmethod
    def _get_base_html_template(title: str, content: str, palette: List[str] = None) -> str:
        """
        Genera el template HTML base con estilos consistentes.
        """
        palette = palette or PALETTE_EDA
        return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
 body {{ font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; margin: 10px 22px; }}
 h1 {{ color: {palette[0]}; }}
 h2 {{ color: {palette[1]}; margin-top: 28px; }}
 .card {{ background: {palette[4] if len(palette) > 4 else '#f0f0f0'}; padding: 12px 14px; border-left: 4px solid {palette[1]}; }}
 table {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
 th, td {{ border: 1px solid #e9e9e9; padding: 6px 8px; }}
 hr {{ border: none; height: 1px; background: #eaeaea; margin: 20px 0; }}
</style>
</head>
<body>
{content}
</body>
</html>"""


# -----------------------------------------------------------------------------
# Clase EDAReport: Reportes de Análisis Exploratorio de Datos
# -----------------------------------------------------------------------------

class EDAReport(ReportBase):
    """
    Genera reportes EDA interactivos en HTML utilizando Plotly.
    Soporta múltiples variantes: base, clean, y con datos originales.
    """
    
    def __init__(self, variant: str = "base", output_dir: Path = None):
        """
        Args:
            variant: Tipo de reporte EDA ("base", "clean")
            output_dir: Directorio de salida (default: reports/eda_html)
        """
        if output_dir is None:
            output_dir = REPORTS / "eda_html"
        super().__init__(output_dir, PALETTE_EDA)
        self.variant = variant
        self.apply_cleaning = (variant == "clean")
    
    def generate(self, df_modified: pd.DataFrame, df_original: pd.DataFrame = None,
                 filename: str = None) -> Path:
        """
        Genera el reporte EDA.
        
        Args:
            df_modified: DataFrame modificado/limpio
            df_original: DataFrame original (opcional, para comparación)
            filename: Nombre del archivo HTML (default: auto-generado)
            
        Returns:
            Path: Ruta del archivo HTML generado
        """
        # Aplicar limpieza si es necesario
        if self.apply_cleaning:
            num_cols, cat_cols = features.split_num_cat(df_modified)
            df_modified = features.clean_categoricals(df_modified, cat_cols)
            df_modified, num_cols, cat_cols = features.minimal_preprocess(df_modified)
        
        # Calcular métricas y detectar target
        target = self.guess_target(df_modified)
        metrics = self.compute_summary_metrics(df_modified)
        
        # Generar visualizaciones
        figs_blocks = self._generate_figures(df_modified, df_original, target, metrics)
        
        # Construir HTML
        html_content = self._build_html_content(df_modified, df_original, target, metrics, figs_blocks)
        
        # Guardar archivo
        if filename is None:
            suffix = "_clean" if self.apply_cleaning else ""
            filename = f"eda_modified_plotly{suffix}.html"
        
        output_path = self.output_dir / filename
        output_path.write_text(html_content, encoding="utf-8")
        return output_path
    
    def _generate_figures(self, df: pd.DataFrame, df_original: pd.DataFrame,
                          target: str, metrics: dict) -> List[Tuple[str, go.Figure, str]]:
        """
        Genera todas las figuras del reporte EDA.
        """
        figs_blocks = []
        
        # Distribución del target
        td_fig = self._fig_target_distribution(df, target)
        if td_fig:
            figs_blocks.append(("Distribución del objetivo", td_fig, self._insight_target(df, target)))
        
        # Mapa de nulos
        miss_fig = self._fig_missing_heatmap(df)
        figs_blocks.append(("Mapa de nulos", miss_fig, self._insight_missing(df)))
        
        # Matriz de correlación
        corr_fig = self._fig_corr_heatmap(df)
        if corr_fig:
            figs_blocks.append(("Matriz de correlación", corr_fig, self._insight_corr(df)))
        
        # Cardinalidad
        card_fig = self._fig_cardinality(df)
        card_insight = (
            f"Cardinalidad media: {metrics['cardinality_mean']}. "
            f"Columnas con mayor cardinalidad: "
            + ", ".join([f"{c} ({int(n)})" for c, n in metrics["cardinality"].head(3).items()])
        )
        figs_blocks.append(("Cardinalidad por columna", card_fig, card_insight))
        
        # Visualizaciones numéricas y categóricas
        for f, ins in self._figs_numeric(df):
            figs_blocks.append(("Distribución numérica", f, ins))
        for f, ins in self._figs_categorical(df):
            figs_blocks.append(("Distribución categórica", f, ins))
        
        # Comparativa con dataset original
        if df_original is not None:
            figs_blocks.insert(1, ("Original vs Modified (tamaños)", 
                                  self._fig_meta_compare(df, df_original), 
                                  "Comparativa de dimensiones entre datasets."))
        
        return figs_blocks
    
    def _build_html_content(self, df: pd.DataFrame, df_original: pd.DataFrame,
                           target: str, metrics: dict, figs_blocks: List[Tuple]) -> str:
        """
        Construye el contenido HTML completo del reporte.
        """
        # Tablas rápidas
        head_html = df.head().to_html(index=False)
        dtypes_html = pd.DataFrame({"col": df.columns, "dtype": df.dtypes.astype(str)}).to_html(index=False)
        
        # Renderizar figuras
        plots_html = []
        for title, fig, insight in figs_blocks:
            plots_html.append(f"<h2>{title}</h2>")
            plots_html.append(fig.to_html(full_html=False, include_plotlyjs='cdn'))
            plots_html.append(f"<p><em>Insight:</em> {insight}</p>")
            plots_html.append("<hr/>")
        plots_html = "\n".join(plots_html)
        
        # Métricas y resumen global
        metrics_html = f"""
<div class="card">
  <b>Filas×Columnas:</b> {metrics['rows']} × {metrics['cols']} &nbsp; | &nbsp;
  <b>Numéricas/Categóricas:</b> {metrics['n_num']} / {metrics['n_cat']} &nbsp; | &nbsp;
  <b>% nulos promedio:</b> {metrics['null_pct_mean']}% &nbsp; | &nbsp;
  <b>Duplicados:</b> {metrics['dups']} &nbsp; | &nbsp;
  <b>Memoria:</b> {metrics['mem_mb']} MB &nbsp; | &nbsp;
  <b>Cardinalidad media:</b> {metrics['cardinality_mean']}
</div>
"""
        if self.apply_cleaning:
            metrics_html += "<p><em>Nota:</em> Este reporte utiliza los datos ya <b>limpios y normalizados</b> con las funciones <code>clean_categoricals()</code> y <code>minimal_preprocess()</code> de <b>features.py</b>.</p>"
        
        metrics_html += f"<p>{self._insight_global(df, target, metrics)}</p>"
        
        # Análisis del target
        target_final = ""
        if target and target in df.columns:
            vc = df[target].value_counts(dropna=False).reset_index()
            vc.columns = [target, "count"]
            vc["pct"] = (vc["count"] / len(df) * 100).round(2)
            table_html = vc.to_html(index=False)
            target_final = f"""
<h2>Análisis del target</h2>
<p>{self._insight_target(df, target)}</p>
{table_html}
"""
        
        # Construir contenido completo
        content = f"""
<h1>EDA – Student Entrance Performance (Modified)</h1>

{metrics_html}

<h2>Vista rápida (head)</h2>
{head_html}

<h2>Tipos de datos</h2>
{dtypes_html}

<hr/>
{plots_html}

{target_final}
"""
        
        return self._get_base_html_template("EDA – Student Entrance Performance", content, self.palette)
    
    # -------------------------------------------------------------------------
    # Métodos de visualización (figuras Plotly)
    # -------------------------------------------------------------------------
    
    def _fig_target_distribution(self, df: pd.DataFrame, target: str) -> Optional[go.Figure]:
        """Gráfico de barras que muestra la distribución del target."""
        if not target or target not in df.columns:
            return None
        vc = df[target].value_counts(dropna=False).reset_index()
        vc.columns = [target, "count"]
        fig = px.bar(vc, x=target, y="count",
                     title=f"Distribución del objetivo: {target}",
                     color_discrete_sequence=[self.palette[0]])
        fig.update_layout(height=360)
        return fig
    
    def _fig_missing_heatmap(self, df: pd.DataFrame) -> go.Figure:
        """Mapa de calor de valores nulos."""
        m = df.isna().astype(int).T
        fig = px.imshow(m, color_continuous_scale=[self.palette[4] if len(self.palette) > 4 else '#f0f0f0', self.palette[1]],
                        aspect="auto", origin="lower")
        fig.update_layout(title="Mapa de nulos (1 = nulo)", height=420,
                          coloraxis_colorbar_title="NA")
        fig.update_yaxes(tickmode="array",
                         ticktext=list(df.columns),
                         tickvals=list(range(len(df.columns))))
        return fig
    
    def _fig_corr_heatmap(self, df: pd.DataFrame) -> Optional[go.Figure]:
        """Matriz de correlación entre variables numéricas."""
        num = df.select_dtypes(include=[np.number])
        if num.shape[1] < 2:
            return None
        corr = num.corr(numeric_only=True)
        fig = px.imshow(corr, text_auto=False, color_continuous_scale="RdBu",
                        zmin=-1, zmax=1, origin="lower")
        fig.update_layout(title="Matriz de correlación (numéricas)", height=520)
        return fig
    
    def _fig_cardinality(self, df: pd.DataFrame) -> go.Figure:
        """Gráfico horizontal que muestra la cardinalidad de cada columna."""
        card = df.nunique(dropna=True).sort_values(ascending=True)
        tmp = card.reset_index()
        tmp.columns = ["column", "n_unique"]
        fig = px.bar(tmp, x="n_unique", y="column", orientation="h",
                     title="Cardinalidad por columna",
                     color_discrete_sequence=[self.palette[1]])
        fig.update_layout(height=max(350, 18 * len(tmp)))
        return fig
    
    def _figs_numeric(self, df: pd.DataFrame) -> List[Tuple[go.Figure, str]]:
        """Genera histogramas con boxplots integrados para cada variable numérica."""
        figs = []
        num_cols, _ = features.split_num_cat(df)
        for c in num_cols:
            f = px.histogram(df, x=c, nbins=30, marginal="box",
                             title=f"Histograma + Box: {c}",
                             color_discrete_sequence=[self.palette[2]])
            f.update_layout(height=340)
            figs.append((f, self._insight_numeric(df, c)))
        return figs
    
    def _figs_categorical(self, df: pd.DataFrame, max_bars=20) -> List[Tuple[go.Figure, str]]:
        """Genera gráficos de barras para cada variable categórica."""
        figs = []
        num_cols, cat_cols = features.split_num_cat(df)
        for c in cat_cols:
            n = df[c].nunique(dropna=False)
            if 2 <= n <= max_bars:
                vc = df[c].value_counts(dropna=False).reset_index()
                vc.columns = [c, "count"]
                f = px.bar(vc, x=c, y="count",
                           title=f"Distribución categórica: {c}",
                           color_discrete_sequence=[self.palette[1]])
                f.update_layout(height=340, xaxis_tickangle=-30)
                figs.append((f, self._insight_categorical(df, c)))
        return figs
    
    def _fig_meta_compare(self, dfm: pd.DataFrame, dfo: pd.DataFrame) -> go.Figure:
        """Comparativa de dimensiones entre datasets."""
        meta = {
            "rows_modified": dfm.shape[0], "cols_modified": dfm.shape[1],
            "rows_original": dfo.shape[0], "cols_original": dfo.shape[1]
        }
        m = pd.DataFrame([meta]).melt(var_name="metric", value_name="value")
        fig = px.bar(m, x="metric", y="value",
                     title="Original vs Modified (tamaños)",
                     color_discrete_sequence=[self.palette[3] if len(self.palette) > 3 else self.palette[1]])
        fig.update_layout(height=320)
        return fig
    
    # -------------------------------------------------------------------------
    # Métodos de insights (texto automático)
    # -------------------------------------------------------------------------
    
    def _insight_global(self, df: pd.DataFrame, target: str, m: dict) -> str:
        """Genera una descripción general del dataset."""
        nulls_by_col = (df.isna().mean() * 100).sort_values(ascending=False)
        high_nulls = nulls_by_col[nulls_by_col > 10].index.tolist()
        
        txt_target = f"Target estimado: {target}." if target else "No se detectó target automáticamente."
        txt_nulls = (
            f"{m['null_pct_mean']}% nulos promedio; columnas con >10% nulos: {', '.join(high_nulls)}."
            if len(high_nulls)
            else f"{m['null_pct_mean']}% nulos promedio; sin columnas con >10% nulos."
        )
        
        return (
            f"El dataset contiene <b>{m['rows']}</b> filas y <b>{m['cols']}</b> columnas "
            f"({m['n_num']} numéricas, {m['n_cat']} categóricas). "
            f"{txt_target} {txt_nulls} "
            f"Duplicados: {m['dups']}. Memoria estimada: {m['mem_mb']} MB. "
            f"Cardinalidad media por columna: {m['cardinality_mean']}."
        )
    
    def _insight_target(self, df: pd.DataFrame, target: str) -> str:
        """Describe la distribución del target."""
        if not target or target not in df.columns:
            return "No se generó insight del target porque no se detectó columna objetivo."
        vc = df[target].value_counts(normalize=True, dropna=False).sort_values(ascending=False)
        top = (vc.iloc[0] * 100.0) if len(vc) else 0.0
        balance = "balanceado" if vc.max() < 0.6 else "desbalanceado"
        return f"El target <b>{target}</b> está {balance}. La clase más frecuente representa {top:.1f}%."
    
    def _insight_missing(self, df: pd.DataFrame) -> str:
        """Resume las columnas con mayor porcentaje de valores nulos."""
        nulls_by_col = (df.isna().mean() * 100).sort_values(ascending=False)
        if nulls_by_col.max() == 0:
            return "No se detectaron valores nulos."
        top_cols = nulls_by_col.head(3)
        return "Columnas con mayor porcentaje de nulos: " + ", ".join([f"{c} ({v:.1f}%)" for c, v in top_cols.items()]) + "."
    
    def _insight_corr(self, df: pd.DataFrame) -> str:
        """Identifica el par de variables numéricas más correlacionadas."""
        num = df.select_dtypes(include=[np.number])
        if num.shape[1] < 2:
            return "No hay suficientes variables numéricas para correlación."
        corr = num.corr(numeric_only=True).copy()
        np.fill_diagonal(corr.values, 0.0)
        max_pair, max_val = None, 0.0
        for i, c1 in enumerate(corr.columns):
            for j, c2 in enumerate(corr.columns):
                v = abs(float(corr.loc[c1, c2]))
                if j > i and v > max_val:
                    max_val, max_pair = v, (c1, c2)
        if not max_pair:
            return "No se identificaron correlaciones destacables."
        return f"Correlación más alta: {max_pair[0]}–{max_pair[1]} (|r|={max_val:.2f})."
    
    def _insight_numeric(self, df: pd.DataFrame, col: str) -> str:
        """Genera insight automático para una variable numérica."""
        s = df[col].dropna()
        if len(s) == 0:
            return f"{col}: sin datos."
        mean, std = float(s.mean()), float(s.std(ddof=0))
        skew = float((s.skew() if hasattr(s, "skew") else 0.0) or 0.0)
        skew_txt = "simétrica" if abs(skew) < 0.3 else ("sesgada a la derecha" if skew > 0 else "sesgada a la izquierda")
        return f"{col}: media={mean:.2f}, sd={std:.2f}, distribución {skew_txt}."
    
    def _insight_categorical(self, df: pd.DataFrame, col: str) -> str:
        """Genera insight automático para variables categóricas."""
        vc = df[col].value_counts(dropna=False, normalize=True)
        if vc.empty:
            return f"{col}: sin datos."
        
        categorias = [str(cat) for cat in vc.index.tolist()]
        categorias_str = ", ".join(categorias[:15]) + (", ..." if len(categorias) > 15 else "")
        return (
            f"La variable <b>{col}</b> contiene {len(vc)} categorías únicas: "
            f"{categorias_str}. "
            f"La más frecuente es '{vc.index[0]}' con {vc.iloc[0]*100:.1f}% de los registros."
        )


# -----------------------------------------------------------------------------
# Clase PreprocessedReport: Reportes de Datos Preprocesados
# -----------------------------------------------------------------------------

class PreprocessedReport(ReportBase):
    """
    Genera reportes EDA específicos para datasets preprocesados.
    Incluye análisis de correlación sin PCA, varianza por columna, etc.
    """
    
    def __init__(self, output_dir: Path = None):
        """
        Args:
            output_dir: Directorio de salida (default: reports/eda_html)
        """
        if output_dir is None:
            output_dir = REPORTS / "eda_html"
        super().__init__(output_dir, PALETTE_EDA)
    
    def generate(self, filename: str = "eda_preprocessed_plotly.html") -> Path:
        """
        Genera el reporte EDA para datos preprocesados.
        
        Args:
            filename: Nombre del archivo HTML
            
        Returns:
            Path: Ruta del archivo HTML generado
        """
        # Cargar datos preprocesados
        if not PREPROCESSED_CSV.exists():
            raise FileNotFoundError(f"No se encontró el archivo preprocesado: {PREPROCESSED_CSV}")
        
        df = pd.read_csv(PREPROCESSED_CSV)
        metrics = self.compute_summary_metrics(df)
        
        # Generar visualizaciones
        figs_blocks = []
        
        # Correlación (sin PCA)
        corr_fig = self._fig_corr_heatmap_preprocessed(df)
        if corr_fig:
            figs_blocks.append((
                "Correlación – Variables (sin PCA, sin 'mixed_type_col')",
                corr_fig,
                self._insight_corr_preprocessed(df)
            ))
        
        # Varianza por columna
        var_fig = self._fig_feature_variance(df, top_k=30)
        if var_fig:
            figs_blocks.append(("Top columnas por varianza", var_fig,
                                "Muestra qué columnas concentran mayor variabilidad tras el preprocesamiento."))
        
        # Scatter de PCs (si existen)
        for fig in self._fig_scatter_pcs(df):
            figs_blocks.append(("Dispersión de componentes principales", fig,
                                "Visualiza estructura en el espacio de las primeras PCs."))
        
        # Construir HTML
        html_content = self._build_html_content(df, metrics, figs_blocks)
        
        # Guardar archivo
        output_path = self.output_dir / filename
        output_path.write_text(html_content, encoding="utf-8")
        return output_path
    
    def _build_html_content(self, df: pd.DataFrame, metrics: dict, 
                           figs_blocks: List[Tuple]) -> str:
        """Construye el contenido HTML completo del reporte."""
        head_html = df.head().to_html(index=False)
        
        # Renderizar figuras
        plots_html = []
        for title, fig, insight in figs_blocks:
            plots_html.append(f"<h2>{title}</h2>")
            plots_html.append(fig.to_html(full_html=False, include_plotlyjs='cdn'))
            plots_html.append(f"<p><em>Insight:</em> {insight}</p>")
            plots_html.append("<hr/>")
        plots_html = "\n".join(plots_html)
        
        # Métricas y resumen global
        has_pca = any(str(c).lower().startswith(("pc", "pca_")) for c in df.columns)
        pca_txt = "Se detectaron componentes PCA; la correlación fuera de la diagonal debería ser cercana a 0." \
                  if has_pca else \
                  "No se detectaron columnas PCA; las correlaciones reflejan interacciones entre variables derivadas."
        
        metrics_html = f"""
<div class="card">
  <b>Filas×Columnas:</b> {metrics['rows']} × {metrics['cols']} &nbsp; | &nbsp;
  <b>Numéricas/Categóricas:</b> {metrics['n_num']} / {metrics['n_cat']} &nbsp; | &nbsp;
  <b>% nulos promedio:</b> {metrics['null_pct_mean']}% &nbsp; | &nbsp;
  <b>Duplicados:</b> {metrics['dups']} &nbsp; | &nbsp;
  <b>Memoria:</b> {metrics['mem_mb']} MB
</div>
<p>Dataset preprocesado con <b>{metrics['rows']}</b> filas y <b>{metrics['cols']}</b> columnas 
({metrics['n_num']} numéricas, {metrics['n_cat']} categóricas). 
Nulos promedio: {metrics['null_pct_mean']}%. Duplicados: {metrics['dups']}. 
Memoria: {metrics['mem_mb']} MB. {pca_txt}</p>
"""
        
        content = f"""
<h1>EDA – Dataset PREPROCESADO</h1>

{metrics_html}

<h2>Vista rápida (head)</h2>
{head_html}

<hr/>
{plots_html}
"""
        
        return self._get_base_html_template("EDA – Dataset PREPROCESADO", content, self.palette)
    
    def _select_corr_columns(self, num_df: pd.DataFrame, max_cols: int = 60) -> pd.DataFrame:
        """Selecciona columnas para el heatmap de correlación."""
        cols = num_df.columns
        
        def is_pca(c: str) -> bool:
            lc = str(c).lower()
            return lc.startswith("pc") or lc.startswith("pca_")
        
        def is_mixed(c: str) -> bool:
            return "mixed_type_col" in str(c).lower()
        
        filtered_cols = [c for c in cols if not is_pca(c) and not is_mixed(c)]
        
        if len(filtered_cols) == 0:
            filtered_cols = list(cols)
        
        sub = num_df[filtered_cols]
        
        if sub.shape[1] > max_cols:
            var = sub.var(numeric_only=True).sort_values(ascending=False)
            keep = var.index[:max_cols]
            sub = sub[keep]
        
        return sub
    
    def _fig_corr_heatmap_preprocessed(self, df: pd.DataFrame) -> Optional[go.Figure]:
        """Matriz de correlación para datos preprocesados."""
        num = df.select_dtypes(include=[np.number])
        if num.shape[1] < 2:
            return None
        num_sel = self._select_corr_columns(num, max_cols=60)
        corr = num_sel.corr(numeric_only=True)
        fig = px.imshow(
            corr, text_auto=False, color_continuous_scale="RdBu",
            zmin=-1, zmax=1, origin="lower",
            title="Matriz de correlación (sin PCA, sin 'mixed_type_col')"
        )
        fig.update_layout(height=max(520, 12 * len(num_sel.columns)))
        return fig
    
    def _insight_corr_preprocessed(self, df: pd.DataFrame) -> str:
        """Insight sobre correlación en datos preprocesados."""
        num = df.select_dtypes(include=[np.number])
        if num.shape[1] < 2:
            return "No hay suficientes columnas numéricas para calcular correlación."
        
        def is_pca(c: str) -> bool:
            lc = str(c).lower()
            return lc.startswith("pc") or lc.startswith("pca_")
        
        def is_mixed(c: str) -> bool:
            return "mixed_type_col" in str(c).lower()
        
        total = len(num.columns)
        filtradas = [c for c in num.columns if not is_pca(c) and not is_mixed(c)]
        return (f"La matriz usa {len(filtradas)} de {total} columnas numéricas "
                f"(excluyendo PCA y 'mixed_type_col'); si había demasiadas, se seleccionaron "
                f"las de mayor varianza para mejorar la legibilidad.")
    
    def _fig_feature_variance(self, df: pd.DataFrame, top_k: int = 30) -> Optional[go.Figure]:
        """Grafica la varianza de las columnas numéricas."""
        num = df.select_dtypes(include=[np.number])
        if num.empty:
            return None
        var = num.var(numeric_only=True).sort_values(ascending=False)
        top = var.head(top_k).reset_index()
        top.columns = ["feature", "variance"]
        fig = px.bar(
            top, x="variance", y="feature", orientation="h",
            title=f"Top {min(top_k, len(top))} columnas por varianza",
            color_discrete_sequence=[self.palette[1]]
        )
        fig.update_layout(height=max(400, 18 * len(top)))
        return fig
    
    def _fig_scatter_pcs(self, df: pd.DataFrame) -> List[go.Figure]:
        """Scatter de primeras componentes principales."""
        cols = df.columns
        pc1 = next((c for c in cols if str(c).lower() in ("pc1", "pca_1")), None)
        pc2 = next((c for c in cols if str(c).lower() in ("pc2", "pca_2")), None)
        pc3 = next((c for c in cols if str(c).lower() in ("pc3", "pca_3")), None)
        
        figs = []
        if pc1 and pc2:
            f12 = px.scatter(df, x=pc1, y=pc2,
                             title=f"Scatter {pc1} vs {pc2}",
                             color_discrete_sequence=[self.palette[2]])
            f12.update_layout(height=420)
            figs.append(f12)
        if pc1 and pc3:
            f13 = px.scatter(df, x=pc1, y=pc3,
                             title=f"Scatter {pc1} vs {pc3}",
                             color_discrete_sequence=[self.palette[2]])
            f13.update_layout(height=420)
            figs.append(f13)
        return figs


# -----------------------------------------------------------------------------
# Clase ModelsReport: Reportes de Comparación de Modelos
# -----------------------------------------------------------------------------

class ModelsReport(ReportBase):
    """
    Genera reportes HTML interactivos de comparación de modelos
    usando Plotly, basado en resultados de MLflow.
    """
    
    def __init__(self, output_dir: Path = None):
        """
        Args:
            output_dir: Directorio de salida (default: reports/experiments_html)
        """
        if output_dir is None:
            output_dir = REPORTS / "experiments_html"
        super().__init__(output_dir, PALETTE_MODELS)
    
    def generate(self, experiment_name: str = None, tracking_uri: str = None,
                 filename: str = None) -> Path:
        """
        Genera el reporte de comparación de modelos.
        
        Args:
            experiment_name: Nombre del experimento en MLflow (None = todos)
            tracking_uri: URI del tracking server de MLflow
            filename: Nombre del archivo HTML (default: auto-generado)
            
        Returns:
            Path: Ruta del archivo HTML generado
        """
        # Cargar resultados desde MLflow
        results_dict = self._load_results_from_mlflow(experiment_name, tracking_uri)
        
        if not results_dict:
            raise ValueError("No se encontraron resultados en MLflow para generar el reporte.")
        
        # Formatear datos
        df = self._format_metrics_dict(results_dict)
        
        if df.empty:
            raise ValueError("No se encontraron métricas válidas para mostrar.")
        
        # Crear visualizaciones
        bar_fig = self._create_metrics_comparison_bar(df)
        radar_fig = self._create_radar_chart(df, top_n=5)
        ranking_html = self._create_ranking_table(df)
        
        # Determinar mejor modelo
        best_model = df.iloc[0]
        
        # Generar insights
        insights = self._generate_insights(df, best_model)
        
        # Construir HTML
        html_content = self._build_html_content(best_model, insights, bar_fig, radar_fig, ranking_html, experiment_name)
        
        # Guardar archivo
        if filename is None:
            if experiment_name:
                safe_name = experiment_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
                filename = f"models_comparison_{safe_name}.html"
            else:
                filename = "models_comparison_report.html"
        
        output_path = self.output_dir / filename
        output_path.write_text(html_content, encoding='utf-8')
        
        print(f"[OK] Model comparison report saved to: {output_path}")
        return output_path
    
    def _load_results_from_mlflow(self, experiment_name: str = None, tracking_uri: str = None) -> Dict:
        """Carga resultados de experimentos desde MLflow."""
        from mlops.mlflow_config import MLFLOW_TRACKING_URI
        
        if tracking_uri is None:
            tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", MLFLOW_TRACKING_URI)
        
        mlflow.set_tracking_uri(tracking_uri)
        
        try:
            # Si no se especifica experimento, combinar todos los experimentos
            if experiment_name is None:
                all_experiments = mlflow.search_experiments()
                if not all_experiments:
                    print("[WARNING] No experiments found in MLflow.")
                    return {}
                
                print(f"[INFO] Combinando {len(all_experiments)} experimento(s) (no se especificó --experiment):")
                experiment_ids = []
                for exp in all_experiments:
                    runs_count = len(mlflow.search_runs([exp.experiment_id]))
                    print(f"  - {exp.name} ({runs_count} runs)")
                    experiment_ids.append(exp.experiment_id)
                
                runs = mlflow.search_runs(experiment_ids=experiment_ids)
            else:
                experiment = mlflow.get_experiment_by_name(experiment_name)
                if experiment is None:
                    print(f"[WARNING] Experiment '{experiment_name}' not found.")
                    all_experiments = mlflow.search_experiments()
                    if all_experiments:
                        print(f"[INFO] Experimentos disponibles en MLflow:")
                        for exp in all_experiments:
                            runs_count = len(mlflow.search_runs([exp.experiment_id]))
                            print(f"  - {exp.name} ({runs_count} runs)")
                    return {}
                
                runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
            
            if runs.empty:
                exp_name = experiment_name if experiment_name else "experimentos combinados"
                print(f"[WARNING] No runs found in '{exp_name}'.")
                return {}
            
            # Agrupar por modelo
            results = {}
            exp_names = {}
            if experiment_name is None:
                all_experiments = mlflow.search_experiments()
                for exp in all_experiments:
                    exp_names[exp.experiment_id] = exp.name
            
            for idx, row in runs.iterrows():
                run_name = row.get('tags.mlflow.runName', 'unknown')
                exp_id = row.get('experiment_id', 'unknown')
                exp_name = exp_names.get(exp_id, '')
                
                if 'GridSearch' in run_name or 'grid' in run_name.lower():
                    base_model_name = run_name
                else:
                    base_model_name = run_name
                
                if experiment_name is None and exp_name:
                    model_name = f"{exp_name} - {base_model_name}"
                else:
                    model_name = base_model_name
                
                if model_name in results:
                    current_f1 = row.get('metrics.test_f1_weighted', row.get('metrics.f1_weighted', 0))
                    best_f1 = results[model_name].get('metrics.test_f1_weighted', results[model_name].get('metrics.f1_weighted', 0))
                    if current_f1 > best_f1:
                        results[model_name] = row.to_dict()
                else:
                    results[model_name] = row.to_dict()
            
            print(f"[OK] Loaded {len(results)} models from MLflow.")
            return results
            
        except Exception as e:
            print(f"[ERROR] Error loading from MLflow: {str(e)}")
            return {}
    
    def _format_metrics_dict(self, results_dict: Dict) -> pd.DataFrame:
        """Formatea resultados de MLflow en un formato estándar."""
        formatted_data = []
        
        for model_name, data in results_dict.items():
            if isinstance(data, dict):
                if 'accuracy' in data:
                    # Formato directo de train_multiple_models.py
                    formatted_data.append({
                        'model': model_name,
                        'accuracy': data.get('accuracy', 0),
                        'f1_weighted': data.get('f1_weighted', 0),
                        'precision_weighted': data.get('precision_weighted', 0),
                        'recall_weighted': data.get('recall_weighted', 0),
                        'cv_mean': data.get('cv_mean', 0),
                        'cv_std': data.get('cv_std', 0)
                    })
                elif any(k.startswith('metrics.') for k in data.keys()):
                    # Formato de MLflow
                    formatted_data.append({
                        'model': model_name,
                        'test_acc': data.get('metrics.test_acc', data.get('metrics.acc_test', 0)),
                        'train_acc': data.get('metrics.acc_train', 0),
                        'test_f1_weighted': data.get('metrics.test_f1_weighted', data.get('metrics.f1_weighted', 0)),
                        'test_f1_macro': data.get('metrics.test_f1_macro', data.get('metrics.f1_macro', 0)),
                        'test_f1_micro': data.get('metrics.f1_micro', 0),
                        'cv_f1_weighted_mean': data.get('metrics.cv_f1_weighted_mean', 0),
                        'cv_f1_weighted_std': data.get('metrics.cv_f1_weighted_std', 0)
                    })
        
        df = pd.DataFrame(formatted_data)
        
        if df.empty:
            print("[WARNING] No valid metrics found in results.")
            return df
        
        # Ordenar por F1 score descendente
        sort_col = 'test_f1_weighted' if 'test_f1_weighted' in df.columns else ('f1_weighted' if 'f1_weighted' in df.columns else df.columns[1])
        df = df.sort_values(sort_col, ascending=False).reset_index(drop=True)
        df['rank'] = range(1, len(df) + 1)
        
        return df
    
    def _create_metrics_comparison_bar(self, df: pd.DataFrame) -> go.Figure:
        """Crea gráfico de barras horizontal comparando métricas principales."""
        fig = go.Figure()
        
        available_metrics = []
        metric_mapping = {
            'test_acc': 'Test Accuracy',
            'test_f1_weighted': 'Test F1 Weighted',
            'test_f1_macro': 'Test F1 Macro',
            'cv_f1_weighted_mean': 'CV F1 Mean',
            'accuracy': 'Accuracy',
            'f1_weighted': 'F1 Weighted'
        }
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        for col in df.columns:
            if col in metric_mapping and col != 'model' and col != 'rank':
                available_metrics.append(col)
        
        metrics_to_plot = available_metrics[:len(colors)]
        
        for metric, color in zip(metrics_to_plot, colors[:len(metrics_to_plot)]):
            fig.add_trace(go.Bar(
                name=metric_mapping.get(metric, metric.replace('_', ' ').title()),
                x=df['model'],
                y=df[metric],
                marker_color=color,
                text=[f'{val:.3f}' for val in df[metric]],
                textposition='outside'
            ))
        
        fig.update_layout(
            title='Comparación de Métricas por Modelo',
            xaxis_title='Modelo',
            yaxis_title='Score',
            barmode='group',
            height=500,
            yaxis=dict(range=[0, 1.05]),
            hovermode='x unified'
        )
        
        return fig
    
    def _create_radar_chart(self, df: pd.DataFrame, top_n: int = 5) -> go.Figure:
        """Crea gráfico radar para los mejores modelos."""
        top_models = df.head(top_n)
        
        available_metrics = []
        metric_labels = {
            'test_acc': 'Test Acc',
            'test_f1_weighted': 'F1 Weighted',
            'test_f1_macro': 'F1 Macro',
            'cv_f1_weighted_mean': 'CV Mean',
            'accuracy': 'Accuracy',
            'f1_weighted': 'F1 Weighted'
        }
        
        for col in df.columns:
            if col in metric_labels and col != 'model' and col != 'rank':
                available_metrics.append(col)
        
        metrics_to_plot = available_metrics[:4]
        
        fig = go.Figure()
        
        for idx, row in top_models.iterrows():
            fig.add_trace(go.Scatterpolar(
                r=[row[m] for m in metrics_to_plot],
                theta=[metric_labels.get(m, m.replace('_', ' ').title()) for m in metrics_to_plot],
                fill='toself',
                name=row['model']
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(range=[0, 1], visible=True)
            ),
            showlegend=True,
            title=f'Radar Chart - Top {top_n} Modelos',
            height=600
        )
        
        return fig
    
    def _create_ranking_table(self, df: pd.DataFrame) -> str:
        """Crea tabla HTML con el ranking de modelos."""
        html = """
    <div class="ranking-table">
        <h3>Ranking de Modelos</h3>
        <table>
            <thead>
                <tr>
                    <th>Rank</th>
                    <th>Modelo</th>
                    <th>Test Acc</th>
                    <th>Test F1 (weighted)</th>
                    <th>Test F1 (macro)</th>
                    <th>CV F1 Mean</th>
                    <th>CV F1 Std</th>
                </tr>
            </thead>
            <tbody>
    """
        
        for _, row in df.iterrows():
            badge = "🥇" if row['rank'] == 1 else "🥈" if row['rank'] == 2 else "🥉" if row['rank'] == 3 else ""
            test_acc = row.get('test_acc', row.get('accuracy', 0))
            test_f1_w = row.get('test_f1_weighted', row.get('f1_weighted', 0))
            test_f1_m = row.get('test_f1_macro', 0)
            cv_mean = row.get('cv_f1_weighted_mean', row.get('cv_mean', 0))
            cv_std = row.get('cv_f1_weighted_std', row.get('cv_std', 0))
            
            html += f"""
                <tr>
                    <td><strong>{badge} {int(row['rank'])}</strong></td>
                    <td><strong>{row['model']}</strong></td>
                    <td>{test_acc:.4f}</td>
                    <td>{test_f1_w:.4f}</td>
                    <td>{test_f1_m:.4f}</td>
                    <td>{cv_mean:.4f} ± {cv_std:.4f}</td>
                    <td>{cv_std:.4f}</td>
                </tr>
        """
        
        html += """
            </tbody>
        </table>
    </div>
    """
        
        return html
    
    def _generate_insights(self, df: pd.DataFrame, best_model: pd.Series) -> str:
        """Genera insights automáticos basados en los resultados."""
        num_models = len(df)
        best_f1_col = 'test_f1_weighted' if 'test_f1_weighted' in df.columns else 'f1_weighted'
        best_score = best_model.get(best_f1_col, best_model.get('f1_weighted', 0))
        avg_score = df[best_f1_col].mean() if best_f1_col in df.columns else df.get('f1_weighted', pd.Series([0])).mean()
        score_range = df[best_f1_col].max() - df[best_f1_col].min() if best_f1_col in df.columns else 0
        
        insights_html = "<ul>"
        insights_html += f"<li>✅ Se evaluaron <strong>{num_models} modelos distintos</strong></li>"
        insights_html += f"<li>🏆 El mejor modelo ({best_model['model']}) alcanzó un <strong>F1-score (weighted) de {best_score:.4f}</strong></li>"
        insights_html += f"<li>📊 El rendimiento promedio del conjunto de modelos es <strong>{avg_score:.4f}</strong></li>"
        insights_html += f"<li>📈 La diferencia entre el mejor y peor modelo es <strong>{score_range:.4f}</strong> puntos</li>"
        
        cv_std = best_model.get('cv_f1_weighted_std', best_model.get('cv_std', 0))
        if cv_std < 0.01:
            insights_html += "<li>✨ El mejor modelo muestra <strong>alta estabilidad</strong> en validación cruzada</li>"
        elif cv_std < 0.05:
            insights_html += "<li>✓ El mejor modelo muestra <strong>estabilidad moderada</strong> en validación cruzada</li>"
        else:
            insights_html += "<li>⚠️ El mejor modelo muestra <strong>cierta variabilidad</strong> en validación cruzada</li>"
        
        if best_score > avg_score + 0.1:
            insights_html += "<li>🚀 El mejor modelo <strong>destaca significativamente</strong> sobre el promedio</li>"
        
        insights_html += "</ul>"
        
        return insights_html
    
    def _build_html_content(self, best_model: pd.Series, insights: str, 
                           bar_fig: go.Figure, radar_fig: go.Figure, 
                           ranking_html: str, experiment_name: str = None) -> str:
        """Construye el contenido HTML completo del reporte."""
        exp_title = f" - {experiment_name}" if experiment_name else ""
        
        html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Reporte de Comparación de Modelos - Student Performance{exp_title}</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 20px;
                color: #333;
            }}
            .container {{
                max-width: 1400px;
                margin: 0 auto;
                background: white;
                border-radius: 15px;
                box-shadow: 0 10px 40px rgba(0,0,0,0.2);
                overflow: hidden;
            }}
            .header {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 30px;
                text-align: center;
            }}
            .header h1 {{
                font-size: 2.5em;
                margin-bottom: 10px;
            }}
            .header p {{
                font-size: 1.1em;
                opacity: 0.9;
            }}
            .content {{
                padding: 30px;
            }}
            .insights {{
                background: #f8f9fa;
                border-left: 5px solid #667eea;
                padding: 20px;
                margin: 20px 0;
                border-radius: 5px;
            }}
            .insights h3 {{
                color: #667eea;
                margin-bottom: 15px;
            }}
            .insights ul {{
                list-style-type: none;
                padding-left: 0;
            }}
            .insights li {{
                padding: 8px 0;
                border-bottom: 1px solid #e0e0e0;
            }}
            .insights li:last-child {{
                border-bottom: none;
            }}
            .plot-container {{
                margin: 30px 0;
                padding: 20px;
                background: #f8f9fa;
                border-radius: 10px;
            }}
            .ranking-table {{
                margin: 20px 0;
            }}
            .ranking-table table {{
                width: 100%;
                border-collapse: collapse;
                margin-top: 15px;
            }}
            .ranking-table th, .ranking-table td {{
                padding: 12px;
                text-align: left;
                border-bottom: 2px solid #ddd;
            }}
            .ranking-table th {{
                background: #667eea;
                color: white;
                font-weight: bold;
            }}
            .ranking-table tr:hover {{
                background: #f5f5f5;
            }}
            .footer {{
                text-align: center;
                padding: 20px;
                background: #f8f9fa;
                color: #666;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>📊 Reporte de Comparación de Modelos</h1>
                <p>Proyecto: Student Performance on an Entrance Examination{exp_title}</p>
                <p><small>Equipo 56 - MLOps Fase 2 | Generado automáticamente</small></p>
            </div>
            
            <div class="content">
                <div class="insights">
                    <h3>🎯 Mejor Modelo: {best_model['model']}</h3>
                    <p><strong>Métricas:</strong> Test Acc: {best_model.get('test_acc', best_model.get('accuracy', 0)):.4f} | 
                       Test F1 (weighted): {best_model.get('test_f1_weighted', best_model.get('f1_weighted', 0)):.4f} | 
                       Test F1 (macro): {best_model.get('test_f1_macro', 0):.4f} | 
                       CV F1 Mean: {best_model.get('cv_f1_weighted_mean', best_model.get('cv_mean', 0)):.4f} ± {best_model.get('cv_f1_weighted_std', best_model.get('cv_std', 0)):.4f}</p>
                </div>
                
                <div class="insights">
                    <h3>🔍 Insights Principales</h3>
                    {insights}
                </div>
                
                <div class="plot-container">
                    <div id="bar-chart" style="width: 100%; height: 500px;"></div>
                </div>
                
                <div class="plot-container">
                    <div id="radar-chart" style="width: 100%; height: 600px;"></div>
                </div>
                
                {ranking_html}
            </div>
            
            <div class="footer">
                <p>📌 Este reporte se actualiza automáticamente con cada nuevo entrenamiento</p>
                <p>💡 Para ver más detalles, consulta MLflow UI: <code>mlflow ui</code></p>
            </div>
        </div>
        
        <script>
            var barData = {bar_fig.to_json()};
            var radarData = {radar_fig.to_json()};
            
            Plotly.newPlot('bar-chart', barData.data, barData.layout);
            Plotly.newPlot('radar-chart', radarData.data, radarData.layout);
        </script>
    </body>
    </html>
    """
        
        return html_content


# -----------------------------------------------------------------------------
# Factory Function para facilitar la creación de reportes
# -----------------------------------------------------------------------------

def create_report(report_type: str, **kwargs) -> ReportBase:
    """
    Factory function para crear instancias de reportes.
    
    Args:
        report_type: Tipo de reporte ("eda_base", "eda_clean", "preprocessed", "models")
        **kwargs: Argumentos adicionales para el constructor del reporte
        
    Returns:
        Instancia del reporte solicitado
    """
    report_map = {
        "eda_base": lambda: EDAReport(variant="base", **kwargs),
        "eda_clean": lambda: EDAReport(variant="clean", **kwargs),
        "preprocessed": lambda: PreprocessedReport(**kwargs),
        "models": lambda: ModelsReport(**kwargs),
    }
    
    if report_type not in report_map:
        raise ValueError(f"Tipo de reporte desconocido: {report_type}. "
                        f"Opciones: {list(report_map.keys())}")
    
    return report_map[report_type]()

