# -----------------------------------------------------------------------------
# Genera un reporte EDA (Exploratory Data Analysis) interactivo en HTML
# utilizando Plotly, basado en los datasets original y modificado.
# Incluye métricas automáticas, visualizaciones y "insights" descriptivos.
# -----------------------------------------------------------------------------

from pathlib import Path
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from .config import REPORTS
from . import features  # Reutiliza utilidades como split_num_cat()

# Carpeta de salida para los reportes HTML
REPORTS_HTML = (REPORTS / "eda_html")
REPORTS_HTML.mkdir(parents=True, exist_ok=True)

# Paleta de colores (Ocean Serenity)
PALETTE = ["#041B25", "#0E3A4B", "#4A7486", "#AFC4B2", "#EFF5DE"]


# -----------------------------------------------------------------------------
# Funciones auxiliares (detección y análisis de datos)
# -----------------------------------------------------------------------------
def guess_target(df: pd.DataFrame):
    """
    Intenta detectar automáticamente la columna objetivo (target) 
    a partir de nombres comunes o heurística basada en cardinalidad.
    """
    for c in ["Performance", "performance", "target", "label", "Target", "Label"]:
        if c in df.columns:
            return c
    # Fallback: busca columnas categóricas con pocas clases (2–10)
    cands = [(c, df[c].nunique(dropna=True)) for c in df.columns]
    cands = [c for c, n in cands if 2 <= n <= 10]
    return cands[0] if cands else None


def compute_summary_metrics(df: pd.DataFrame):
    """
    Calcula métricas descriptivas generales sobre el DataFrame:
    - Tamaño, % nulos, duplicados, uso de memoria, cardinalidad, etc.
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


# -----------------------------------------------------------------------------
# Generadores de "insights" (texto automático contextual)
# -----------------------------------------------------------------------------
def insight_global(df: pd.DataFrame, target: str, m: dict) -> str:
    """
    Genera una descripción general del dataset combinando métricas y contexto.
    """
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


def insight_target(df: pd.DataFrame, target: str) -> str:
    """
    Describe la distribución del target y si el dataset está balanceado.
    """
    if not target or target not in df.columns:
        return "No se generó insight del target porque no se detectó columna objetivo."
    vc = df[target].value_counts(normalize=True, dropna=False).sort_values(ascending=False)
    top = (vc.iloc[0] * 100.0) if len(vc) else 0.0
    balance = "balanceado" if vc.max() < 0.6 else "desbalanceado"
    return f"El target <b>{target}</b> está {balance}. La clase más frecuente representa {top:.1f}%."


def insight_missing(df: pd.DataFrame) -> str:
    """
    Resume las columnas con mayor porcentaje de valores nulos.
    """
    nulls_by_col = (df.isna().mean() * 100).sort_values(ascending=False)
    if nulls_by_col.max() == 0:
        return "No se detectaron valores nulos."
    top_cols = nulls_by_col.head(3)
    return "Columnas con mayor porcentaje de nulos: " + ", ".join([f"{c} ({v:.1f}%)" for c, v in top_cols.items()]) + "."


def insight_corr(df: pd.DataFrame) -> str:
    """
    Identifica el par de variables numéricas más correlacionadas (en valor absoluto).
    """
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


def insight_numeric(df: pd.DataFrame, col: str) -> str:
    """
    Genera insight automático para una variable numérica.
    Describe media, desviación y sesgo de la distribución.
    """
    s = df[col].dropna()
    if len(s) == 0:
        return f"{col}: sin datos."
    mean, std = float(s.mean()), float(s.std(ddof=0))
    skew = float((s.skew() if hasattr(s, "skew") else 0.0) or 0.0)
    skew_txt = "simétrica" if abs(skew) < 0.3 else ("sesgada a la derecha" if skew > 0 else "sesgada a la izquierda")
    return f"{col}: media={mean:.2f}, sd={std:.2f}, distribución {skew_txt}."


def insight_categorical(df: pd.DataFrame, col: str) -> str:
    """
    Genera insight automático para variables categóricas.
    Lista todas las categorías (o hasta 15) y destaca la más frecuente.
    """
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
# Funciones de generación de figuras Plotly
# -----------------------------------------------------------------------------
# (se omiten comentarios repetitivos para brevedad, pero cada una crea
# una visualización específica: correlación, nulos, target, etc.)
# -----------------------------------------------------------------------------
# Ejemplo: fig_target_distribution(df, target) → gráfico de barras del target
# fig_missing_heatmap(df) → mapa de calor de valores nulos
# figs_numeric(df) → histogramas + boxplots con insight automático
# figs_categorical(df) → gráficas de barras con insight por categoría
# -----------------------------------------------------------------------------

def fig_target_distribution(df: pd.DataFrame, target: str):
    """Gráfico de barras que muestra la distribución del target."""
    if not target or target not in df.columns:
        return None
    vc = df[target].value_counts(dropna=False).reset_index()
    vc.columns = [target, "count"]
    fig = px.bar(vc, x=target, y="count",
                 title=f"Distribución del objetivo: {target}",
                 color_discrete_sequence=[PALETTE[0]])
    fig.update_layout(height=360)
    return fig


def fig_missing_heatmap(df: pd.DataFrame):
    """Mapa de calor de valores nulos (1 = nulo, 0 = no nulo)."""
    m = df.isna().astype(int).T
    fig = px.imshow(m, color_continuous_scale=[PALETTE[4], PALETTE[1]],
                    aspect="auto", origin="lower")
    fig.update_layout(title="Mapa de nulos (1 = nulo)", height=420,
                      coloraxis_colorbar_title="NA")
    fig.update_yaxes(tickmode="array",
                     ticktext=list(df.columns),
                     tickvals=list(range(len(df.columns))))
    return fig


def fig_corr_heatmap(df: pd.DataFrame):
    """Matriz de correlación entre variables numéricas."""
    num = df.select_dtypes(include=[np.number])
    if num.shape[1] < 2:
        return None
    corr = num.corr(numeric_only=True)
    fig = px.imshow(corr, text_auto=False, color_continuous_scale="RdBu",
                    zmin=-1, zmax=1, origin="lower")
    fig.update_layout(title="Matriz de correlación (numéricas)", height=520)
    return fig


def fig_cardinality(df: pd.DataFrame):
    """Gráfico horizontal que muestra la cardinalidad de cada columna."""
    card = df.nunique(dropna=True).sort_values(ascending=True)
    tmp = card.reset_index()
    tmp.columns = ["column", "n_unique"]
    fig = px.bar(tmp, x="n_unique", y="column", orientation="h",
                 title="Cardinalidad por columna",
                 color_discrete_sequence=[PALETTE[1]])
    fig.update_layout(height=max(350, 18 * len(tmp)))
    return fig


def figs_numeric(df: pd.DataFrame):
    """Genera histogramas con boxplots integrados para cada variable numérica."""
    figs = []
    num_cols, _ = features.split_num_cat(df)
    for c in num_cols:
        f = px.histogram(df, x=c, nbins=30, marginal="box",
                         title=f"Histograma + Box: {c}",
                         color_discrete_sequence=[PALETTE[2]])
        f.update_layout(height=340)
        figs.append((f, insight_numeric(df, c)))  # insight automático
    return figs


def figs_categorical(df: pd.DataFrame, max_bars=20):
    """Genera gráficos de barras para cada variable categórica (limitado por max_bars)."""
    figs = []
    num_cols, cat_cols = features.split_num_cat(df)
    for c in cat_cols:
        n = df[c].nunique(dropna=False)
        if 2 <= n <= max_bars:
            vc = df[c].value_counts(dropna=False).reset_index()
            vc.columns = [c, "count"]
            f = px.bar(vc, x=c, y="count",
                       title=f"Distribución categórica: {c}",
                       color_discrete_sequence=[PALETTE[1]])
            f.update_layout(height=340, xaxis_tickangle=-30)
            figs.append((f, insight_categorical(df, c)))
    return figs

def fig_meta_compare(dfm: pd.DataFrame, dfo: pd.DataFrame):
    meta = {
        "rows_modified": dfm.shape[0], "cols_modified": dfm.shape[1],
        "rows_original": dfo.shape[0], "cols_original": dfo.shape[1]
    }
    m = pd.DataFrame([meta]).melt(var_name="metric", value_name="value")
    fig = px.bar(m, x="metric", y="value",
                 title="Original vs Modified (tamaños)",
                 color_discrete_sequence=[PALETTE[3]])
    fig.update_layout(height=320)
    return fig


# -----------------------------------------------------------------------------
# Función principal: construir el HTML final
# -----------------------------------------------------------------------------
def build_html(dfm: pd.DataFrame, dfo: pd.DataFrame | None,
               file_name: str = "eda_modified_plotly_clean.html") -> Path:
    """
    Construye un reporte EDA completo e interactivo en formato HTML.
    Aplica las mismas transformaciones de limpieza y normalización
    definidas en features.py antes de generar las visualizaciones.
    """
    # 🔹 APLICAR LIMPIEZA Y NORMALIZACIÓN (MISMAS FUNCIONES DEL PIPELINE)
    num_cols, cat_cols = features.split_num_cat(dfm)
    dfm = features.clean_categoricals(dfm, cat_cols)
    dfm, num_cols, cat_cols = features.minimal_preprocess(dfm)

    # 🔹 DETECTAR TARGET Y CALCULAR MÉTRICAS CON EL DATASET LIMPIO
    target = guess_target(dfm)
    metrics = compute_summary_metrics(dfm)

    # ✅ 4) Tablas rápidas del DF limpio
    head_html = dfm.head().to_html(index=False)
    dtypes_html = pd.DataFrame({"col": dfm.columns,
                                "dtype": dfm.dtypes.astype(str)}).to_html(index=False)
    # Sección de figuras e insights
    figs_blocks = []

    # Bloques principales (target, nulos, correlaciones, cardinalidad, etc.)
    td_fig = fig_target_distribution(dfm, target)
    if td_fig:
        figs_blocks.append(("Distribución del objetivo", td_fig, insight_target(dfm, target)))

    miss_fig = fig_missing_heatmap(dfm)
    figs_blocks.append(("Mapa de nulos", miss_fig, insight_missing(dfm)))

    corr_fig = fig_corr_heatmap(dfm)
    if corr_fig:
        figs_blocks.append(("Matriz de correlación", corr_fig, insight_corr(dfm)))

    card_fig = fig_cardinality(dfm)
    card_insight = (
        f"Cardinalidad media: {metrics['cardinality_mean']}. "
        f"Columnas con mayor cardinalidad: "
        + ", ".join([f"{c} ({int(n)})" for c, n in metrics["cardinality"].head(3).items()])
    )
    figs_blocks.append(("Cardinalidad por columna", card_fig, card_insight))

    # Visualizaciones numéricas y categóricas
    for f, ins in figs_numeric(dfm):
        figs_blocks.append(("Distribución numérica", f, ins))
    for f, ins in figs_categorical(dfm):
        figs_blocks.append(("Distribución categórica", f, ins))

    # Comparativa con dataset original (si aplica)
    if dfo is not None:
        figs_blocks.insert(1, ("Original vs Modified (tamaños)", fig_meta_compare(dfm, dfo), "Comparativa de dimensiones entre datasets."))

    # Renderiza cada bloque como sección HTML
    plots_html = []
    for title, fig, insight in figs_blocks:
        plots_html.append(f"<h2>{title}</h2>")
        plots_html.append(fig.to_html(full_html=False, include_plotlyjs='cdn'))
        plots_html.append(f"<p><em>Insight:</em> {insight}</p>")
        plots_html.append("<hr/>")
    plots_html = "\n".join(plots_html)

    # Sección inicial de métricas + resumen global
    metrics_html = f"""
<div class="card">
  <b>Filas×Columnas:</b> {metrics['rows']} × {metrics['cols']} &nbsp; | &nbsp;
  <b>Numéricas/Categóricas:</b> {metrics['n_num']} / {metrics['n_cat']} &nbsp; | &nbsp;
  <b>% nulos promedio:</b> {metrics['null_pct_mean']}% &nbsp; | &nbsp;
  <b>Duplicados:</b> {metrics['dups']} &nbsp; | &nbsp;
  <b>Memoria:</b> {metrics['mem_mb']} MB
</div>
<p><em>Nota:</em> Este reporte utiliza los datos ya <b>limpios y normalizados</b> con las funciones 
<code>clean_categoricals()</code> y <code>minimal_preprocess()</code> de <b>features.py</b>.</p>
<p>{insight_global(dfm, target, metrics)}</p>
"""

    # Análisis final del target
    target_final = ""
    if target and target in dfm.columns:
        vc = dfm[target].value_counts(dropna=False).reset_index()
        vc.columns = [target, "count"]
        vc["pct"] = (vc["count"] / len(dfm) * 100).round(2)
        table_html = vc.to_html(index=False)
        target_final = f"""
<h2>Análisis del target</h2>
<p>{insight_target(dfm, target)}</p>
{table_html}
"""

    # Construcción del HTML completo
    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>EDA – Student Entrance Performance (Modified)</title>
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
<h1>EDA – Student Entrance Performance (Modified)</h1>

{metrics_html}

<h2>Vista rápida (head)</h2>
{head_html}

<h2>Tipos de datos</h2>
{dtypes_html}

<hr/>
{plots_html}

{target_final}

</body>
</html>"""

    # Guarda el reporte y devuelve la ruta
    out = REPORTS_HTML / file_name
    out.write_text(html, encoding="utf-8")
    return out
