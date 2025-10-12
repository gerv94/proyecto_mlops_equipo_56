# -----------------------------------------------------------------------------
# Reporte EDA (solo PREPROCESADO) – Plotly + HTML
# Lee exclusivamente el CSV preprocesado (one-hot, escalado, PCA) y genera:
# - Métricas resumidas
# - Matriz de correlación (con selección de columnas)
# - Varianza por columna
# - Scatter de primeras PCs (si existen)
# -----------------------------------------------------------------------------

from pathlib import Path
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from .config import REPORTS, PREPROCESSED_CSV
# Si necesitas utilidades comunes, puedes importar features:
# from . import features

# Carpeta de salida para el HTML
REPORTS_HTML = (REPORTS / "eda_html")
REPORTS_HTML.mkdir(parents=True, exist_ok=True)

# Paleta (Ocean Serenity) para consistencia visual
PALETTE = ["#041B25", "#0E3A4B", "#4A7486", "#AFC4B2", "#EFF5DE"]


# -----------------------------------------------------------------------------
# Carga + Métricas
# -----------------------------------------------------------------------------
def load_preprocessed_df() -> pd.DataFrame:
    """
    Carga el dataset preprocesado definido en PREPROCESSED_CSV.
    Lanza error si no existe (este reporte es SOLO para preprocesado).
    """
    if not PREPROCESSED_CSV.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo preprocesado: {PREPROCESSED_CSV}"
        )
    return pd.read_csv(PREPROCESSED_CSV)


def compute_summary_metrics(df: pd.DataFrame) -> dict:
    """
    Métricas generales del dataset preprocesado.
    (Al ser preprocesado, esperamos todo numérico/One-Hot/PCAs; pero el código
    soporta mixto por seguridad.)
    """
    rows, cols = df.shape
    null_pct_mean = float(df.isna().mean().mean() * 100)
    dups = int(df.duplicated().sum())
    mem_mb = float(df.memory_usage(deep=True).sum() / 1e6)
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in df.columns if c not in num_cols]
    cardinality = df.nunique(dropna=True).sort_values(ascending=False)
    cardinality_mean = float(cardinality.mean()) if len(cardinality) else 0.0

    return {
        "rows": rows,
        "cols": cols,
        "n_num": len(num_cols),
        "n_cat": len(cat_cols),
        "null_pct_mean": round(null_pct_mean, 2),
        "dups": dups,
        "mem_mb": round(mem_mb, 2),
        "cardinality": cardinality,
        "cardinality_mean": round(cardinality_mean, 2),
        "num_cols": num_cols,
        "cat_cols": cat_cols,
    }


# -----------------------------------------------------------------------------
# Insights breves
# -----------------------------------------------------------------------------
def insight_global(df: pd.DataFrame, m: dict) -> str:
    """
    Mensaje general para el dataset preprocesado.
    """
    has_pca = any(str(c).lower().startswith(("pc", "pca_")) for c in df.columns)
    pca_txt = "Se detectaron componentes PCA; la correlación fuera de la diagonal debería ser cercana a 0." \
              if has_pca else \
              "No se detectaron columnas PCA; las correlaciones reflejan interacciones entre variables derivadas."
    return (
        f"Dataset preprocesado con <b>{m['rows']}</b> filas y <b>{m['cols']}</b> columnas "
        f"({m['n_num']} numéricas, {m['n_cat']} categóricas). "
        f"Nulos promedio: {m['null_pct_mean']}%. Duplicados: {m['dups']}. "
        f"Memoria: {m['mem_mb']} MB. {pca_txt}"
    )


# -----------------------------------------------------------------------------
# Selección de columnas para correlación (evitar heatmaps inmanejables)
# -----------------------------------------------------------------------------
def _select_corr_columns(num_df: pd.DataFrame, max_cols: int = 60) -> pd.DataFrame:
    """
    Selecciona columnas para el heatmap de correlación:
      - EXCLUYE columnas PCA (prefijos 'pc', 'pca_').
      - EXCLUYE columnas que contengan 'mixed_type_col' (no aportan).
      - Si quedan demasiadas columnas, toma las de mayor varianza (top max_cols).
    """
    # filtro base: numéricas ya vienen en num_df
    cols = num_df.columns

    def is_pca(c: str) -> bool:
        lc = str(c).lower()
        return lc.startswith("pc") or lc.startswith("pca_")

    def is_mixed(c: str) -> bool:
        return "mixed_type_col" in str(c).lower()

    # 1) excluir PCA y 'mixed_type_col'
    filtered_cols = [c for c in cols if not is_pca(c) and not is_mixed(c)]

    # 2) si no queda nada, regresamos todas numéricas como fallback
    if len(filtered_cols) == 0:
        filtered_cols = list(cols)

    sub = num_df[filtered_cols]

    # 3) si hay muchísimas (ej. por one-hot), quedarnos con mayor varianza
    if sub.shape[1] > max_cols:
        var = sub.var(numeric_only=True).sort_values(ascending=False)
        keep = var.index[:max_cols]
        sub = sub[keep]

    return sub

# -----------------------------------------------------------------------------
# Figuras
# -----------------------------------------------------------------------------
def fig_corr_heatmap_preprocessed(df: pd.DataFrame,
                                  title="Matriz de correlación (sin PCA, sin 'mixed_type_col')"):
    num = df.select_dtypes(include=[np.number])
    if num.shape[1] < 2:
        return None
    num_sel = _select_corr_columns(num, max_cols=60)
    corr = num_sel.corr(numeric_only=True)
    fig = px.imshow(
        corr, text_auto=False, color_continuous_scale="RdBu",
        zmin=-1, zmax=1, origin="lower", title=title
    )
    fig.update_layout(height=max(520, 12 * len(num_sel.columns)))
    return fig

def insight_corr_preprocessed(df: pd.DataFrame) -> str:
    """
    Ahora la correlación se calcula sobre variables (no PCs),
    excluyendo columnas que contengan 'mixed_type_col'. Si hay
    demasiadas columnas (por OHE), se toma el top por varianza.
    """
    num = df.select_dtypes(include=[np.number])
    if num.shape[1] < 2:
        return "No hay suficientes columnas numéricas para calcular correlación."

    # Contar cuántas quedaron tras el filtro para informar
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

def fig_feature_variance(df: pd.DataFrame, top_k: int = 30):
    """
    Grafica la varianza de las columnas numéricas (top_k) para entender
    cuáles contribuyen más tras OHE/escala/PCA.
    """
    num = df.select_dtypes(include=[np.number])
    if num.empty:
        return None
    var = num.var(numeric_only=True).sort_values(ascending=False)
    top = var.head(top_k).reset_index()
    top.columns = ["feature", "variance"]
    fig = px.bar(
        top, x="variance", y="feature", orientation="h",
        title=f"Top {min(top_k, len(top))} columnas por varianza",
        color_discrete_sequence=[PALETTE[1]]
    )
    fig.update_layout(height=max(400, 18 * len(top)))
    return fig


def fig_scatter_pcs(df: pd.DataFrame):
    """
    Scatter de primeras componentes principales (si existen).
    Intenta PC1 vs PC2; si existe PC3, añade otro scatter.
    """
    cols = df.columns
    pc1 = next((c for c in cols if str(c).lower() in ("pc1", "pca_1")), None)
    pc2 = next((c for c in cols if str(c).lower() in ("pc2", "pca_2")), None)
    pc3 = next((c for c in cols if str(c).lower() in ("pc3", "pca_3")), None)

    figs = []
    if pc1 and pc2:
        f12 = px.scatter(df, x=pc1, y=pc2,
                         title=f"Scatter {pc1} vs {pc2}",
                         color_discrete_sequence=[PALETTE[2]])
        f12.update_layout(height=420)
        figs.append(f12)
    if pc1 and pc3:
        f13 = px.scatter(df, x=pc1, y=pc3,
                         title=f"Scatter {pc1} vs {pc3}",
                         color_discrete_sequence=[PALETTE[2]])
        f13.update_layout(height=420)
        figs.append(f13)
    return figs


# -----------------------------------------------------------------------------
# HTML builder
# -----------------------------------------------------------------------------
def build_html_preprocessed(file_name: str = "eda_preprocessed_plotly.html") -> Path:
    """
    Construye un reporte HTML SOLO para el dataset preprocesado.
    Devuelve la ruta del HTML generado.
    """
    df = load_preprocessed_df()
    metrics = compute_summary_metrics(df)

    head_html = df.head().to_html(index=False)

    # Secciones de figuras
    figs_blocks = []

    # (A) Correlación (preprocesado)
    corr_fig = fig_corr_heatmap_preprocessed(df)
    if corr_fig:
        figs_blocks.append((
            "Correlación – Variables (sin PCA, sin 'mixed_type_col')",
            corr_fig,
            insight_corr_preprocessed(df)
        ))
    
    # (B) Varianza por columna
    var_fig = fig_feature_variance(df, top_k=30)
    if var_fig:
        figs_blocks.append(("Top columnas por varianza", var_fig,
                            "Muestra qué columnas concentran mayor variabilidad tras el preprocesamiento."))

    # (C) Scatter de PCs (si existen)
    for fig in fig_scatter_pcs(df):
        figs_blocks.append(("Dispersión de componentes principales", fig,
                            "Visualiza estructura en el espacio de las primeras PCs."))

    # Render secciones
    plots_html = []
    for title, fig, insight in figs_blocks:
        plots_html.append(f"<h2>{title}</h2>")
        plots_html.append(fig.to_html(full_html=False, include_plotlyjs='cdn'))
        plots_html.append(f"<p><em>Insight:</em> {insight}</p>")
        plots_html.append("<hr/>")
    plots_html = "\n".join(plots_html)

    # Métricas + Insight global
    metrics_html = f"""
<div class="card">
  <b>Filas×Columnas:</b> {metrics['rows']} × {metrics['cols']} &nbsp; | &nbsp;
  <b>Numéricas/Categóricas:</b> {metrics['n_num']} / {metrics['n_cat']} &nbsp; | &nbsp;
  <b>% nulos promedio:</b> {metrics['null_pct_mean']}% &nbsp; | &nbsp;
  <b>Duplicados:</b> {metrics['dups']} &nbsp; | &nbsp;
  <b>Memoria:</b> {metrics['mem_mb']} MB
</div>
<p>{insight_global(df, metrics)}</p>
"""

    # HTML final
    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>EDA – Dataset PREPROCESADO</title>
<style>
 body {{ font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; margin: 10px 22px; }}
 h1 {{ color: {PALETTE[0]}; }}
 h2 {{ color: {PALETTE[1]}; margin-top: 28px; }}
 .card {{ background: {PALETTE[4]}; padding: 12px 14px; border-left: 4px solid {PALETTE[1]}; }}
 table {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
 th, td {{ border: 1px solid #e9e9e9; padding: 6px 8px; }}
 hr {{ border: none; height: 1px; background: #eaeaea; margin: 20px 0; }}
</style>
</head>
<body>
<h1>EDA – Dataset PREPROCESADO</h1>

{metrics_html}

<h2>Vista rápida (head)</h2>
{head_html}

<hr/>
{plots_html}

</body>
</html>"""

    out = REPORTS_HTML / file_name
    out.write_text(html, encoding="utf-8")
    return out
