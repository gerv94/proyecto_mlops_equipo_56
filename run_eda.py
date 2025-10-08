import pandas as pd
from mlops.config import TABLES
from mlops import dataset, features, plots

# -----------------------------------------------------------------------------
# FUNCIONES AUXILIARES
# -----------------------------------------------------------------------------

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
    Punto de entrada principal del análisis exploratorio (EDA).

    Pasos:
    1. Cargar dataset 'modificado' desde mlops/dataset.py.
    2. Asegurar que las columnas tengan tipos de datos consistentes.
    3. Detectar la variable objetivo automáticamente.
    4. Generar gráficos básicos (target, nulos, numéricos, categóricos).
    5. Normalizar y limpiar las variables categóricas.
    6. Aplicar preprocesamiento mínimo e imputación.
    7. Guardar el dataset intermedio limpio en data/interim/.
    """
    # -------------------------------------------------------------------------
    # (1) CARGA DE DATOS
    # -------------------------------------------------------------------------
    df = dataset.load_modified()         # lee student_entry_performance_modified.csv
    df = dataset.basic_typing(df)        # fuerza detección de tipos numéricos y texto

    # -------------------------------------------------------------------------
    # (2) DETECCIÓN DEL TARGET
    # -------------------------------------------------------------------------
    target = guess_target(df)
    print(f"[INFO] target: {target}")

    # -------------------------------------------------------------------------
    # (3) VISUALIZACIONES BÁSICAS (EDA RÁPIDO)
    # -------------------------------------------------------------------------
    # Distribución de la variable objetivo
    plots.plot_target_distribution(df, target)
    # Porcentaje de nulos por columna
    plots.plot_missingness(df)

    # Histogramas / boxplots de numéricas y categóricas
    num_cols, cat_cols = features.split_num_cat(df)
    plots.plot_numerics(df, num_cols)
    plots.plot_categoricals(df, cat_cols)

    # -------------------------------------------------------------------------
    # (4) LIMPIEZA MÍNIMA Y GUARDADO INTERMEDIO
    # -------------------------------------------------------------------------
    # Normalización de texto en categóricas (minusculas, espacios, etc.)
    df_norm = features.clean_categoricals(df, cat_cols)
    # Imputación de nulos: medianas y modas
    df_clean, _, _ = features.minimal_preprocess(df_norm)
    # Guarda versión limpia intermedia
    out = dataset.save_interim(df_clean, "student_interim_clean.csv")
    print(f"[OK] interim guardado en: {out}")


# -----------------------------------------------------------------------------
# EJECUCIÓN DIRECTA DEL SCRIPT
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
