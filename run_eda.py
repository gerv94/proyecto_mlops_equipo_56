import pandas as pd
from mlops.config import TABLES
from mlops import dataset, features, plots

# -----------------------------------------------------------------------------
# FUNCIONES AUXILIARES
# -----------------------------------------------------------------------------

#prueba uno git

def guess_target(df: pd.DataFrame):
    """
    Intenta detectar automáticamente cuál columna del DataFrame
    representa la variable objetivo (target) del modelo.

    - Busca nombres comunes como 'Performance', 'target', 'label', etc.
    - Si no encuentra coincidencias exactas, elige la primera columna
      categórica con entre 2 y 10 valores únicos.

    Args:
        df (pd.DataFrame): DataFrame con los datos cargados.

    Returns:
        str | None: Nombre de la columna objetivo si se detecta, o None.
    """
    for c in ["Performance", "performance", "target", "label", "Target", "Label"]:
        if c in df.columns:
            return c
    # fallback: columna categórica con pocas clases
    cands = [(c, df[c].nunique(dropna=True)) for c in df.columns]
    cands = [c for c, n in cands if 2 <= n <= 10]
    return cands[0] if cands else None


# -----------------------------------------------------------------------------
# FUNCIÓN PRINCIPAL: ORQUESTA EL ANÁLISIS EXPLORATORIO (EDA)
# -----------------------------------------------------------------------------
def main():
    """
    Pasos:
    1. Cargar dataset modificado y tipificar columnas.
    2. Normalizar texto en categóricas.
    3. Visualizaciones básicas (target, nulos, numéricas, categóricas).
    4. Imputación mínima (mediana/moda) y guardado limpio intermedio.
    5. Preprocesamiento avanzado (escalado, One-Hot, PCA) y guardado para modelado.
    """
    # -------------------------------------------------------------------------
    # (1) CARGA Y TIPIFICADO
    # -------------------------------------------------------------------------
    df = dataset.load_modified()
    df = dataset.basic_typing(df)

    # -------------------------------------------------------------------------
    # (2) NORMALIZACIÓN DE TEXTO EN CATEGÓRICAS (ANTES DE GRAFICAR)
    # -------------------------------------------------------------------------
    num_cols, cat_cols = features.split_num_cat(df)
    df = features.clean_categoricals(df, cat_cols)

    # -------------------------------------------------------------------------
    # (3) VISUALIZACIONES BÁSICAS
    # -------------------------------------------------------------------------
    # Recalcular listas por si cambió algo tras la normalización
    num_cols, cat_cols = features.split_num_cat(df)

    target = guess_target(df)
    print(f"[INFO] target: {target}")

    if target is not None:
        plots.plot_target_distribution(df, target)
    plots.plot_missingness(df)
    plots.plot_numerics(df, num_cols)
    plots.plot_categoricals(df, cat_cols)

    # -------------------------------------------------------------------------
    # (4) LIMPIEZA MÍNIMA (IMPUTACIÓN) Y GUARDADO LIMPIO
    # -------------------------------------------------------------------------
    df_clean, _, _ = features.minimal_preprocess(df)
    out_clean = dataset.save_interim(df_clean, "student_interim_clean.csv")
    print(f"[OK] interim guardado en: {out_clean}")

    # -------------------------------------------------------------------------
    # (5) PREPROCESAMIENTO AVANZADO PARA MODELADO (NO PARA EL HTML)
    # -------------------------------------------------------------------------
    num_cols_clean, cat_cols_clean = features.split_num_cat(df_clean)
    df_ready = features.preprocess_advanced(
        df_clean,
        num_cols=num_cols_clean,
        cat_cols=cat_cols_clean,
        n_components=3
    )
    out_ready = dataset.save_interim(df_ready, "student_interim_preprocessed.csv")
    print(f"[OK] preprocessed guardado en: {out_ready}")


# -----------------------------------------------------------------------------
# EJECUCIÓN DIRECTA DEL SCRIPT
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()