import pandas as pd
# Asumo que las importaciones de mlops siguen siendo válidas
from mlops.config import TABLES 
from mlops import dataset, features, plots
# Importamos el módulo train_test_split para la recomendación, aunque no se usa en el EDA
from sklearn.model_selection import train_test_split 

# -----------------------------------------------------------------------------
# FUNCIONES AUXILIARES (SE MANTIENEN IGUAL)
# -----------------------------------------------------------------------------

def guess_target(df: pd.DataFrame):
    """
    Intenta detectar automáticamente cuál columna del DataFrame
    representa la variable objetivo (target) del modelo.
    ...
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
    Pasos (Corregidos para evitar Data Leakage):
    1. Cargar dataset modificado y tipificar columnas.
    2. Normalizar texto en categóricas (limpieza determinística).
    3. Visualizaciones básicas (target, nulos, numéricas, categóricas).
    4. **Eliminar pasos de preprocesamiento estadístico global.**
    """
    # -------------------------------------------------------------------------
    # (1) CARGA Y TIPIFICADO
    # -------------------------------------------------------------------------
    df = dataset.load_modified()
    df = dataset.basic_typing(df)

    # -------------------------------------------------------------------------
    # (2) NORMALIZACIÓN DE TEXTO EN CATEGÓRICAS (Única limpieza permitida globalmente)
    # -------------------------------------------------------------------------
    num_cols, cat_cols = features.split_num_cat(df)
    # features.clean_categoricals solo hace limpieza de strings (determinística, no causa leakage)
    df_eda = features.clean_categoricals(df, cat_cols) 

    # -------------------------------------------------------------------------
    # (3) VISUALIZACIONES BÁSICAS
    # -------------------------------------------------------------------------
    num_cols_eda, cat_cols_eda = features.split_num_cat(df_eda)

    target = guess_target(df_eda)
    print(f"[INFO] target: {target}")

    if target is not None:
        plots.plot_target_distribution(df_eda, target)
    plots.plot_missingness(df_eda)
    plots.plot_numerics(df_eda, num_cols_eda)
    plots.plot_categoricals(df_eda, cat_cols_eda)
    
    print(f"[OK] EDA y visualizaciones completadas. El DataFrame está listo para ser guardado como 'interim' si es necesario.")
    
    # -------------------------------------------------------------------------
    # (4) ELIMINACIÓN DE PASOS DE LEAKAGE (ANTERIORES PASOS 4 Y 5)
    # -------------------------------------------------------------------------
    
    # NOTA IMPORTANTE: Los pasos de imputación (mediana/moda), escalado (StandardScaler) 
    # y PCA deben realizarse DESPUÉS de dividir los datos en train/test, y deben 
    # estar encapsulados en un sklearn.Pipeline para evitar Data Leakage.
    # Por lo tanto, se eliminan del script de EDA.
    # -------------------------------------------------------------------------
    
    # Opcional: Guardar la versión solo con limpieza de texto para el modelado
    out_clean = dataset.save_interim(df_eda, "student_interim_clean_for_model_2.csv")
    print(f"[OK] interim (solo limpieza de texto) guardado en: {out_clean}")


# -----------------------------------------------------------------------------
# EJECUCIÓN DIRECTA DEL SCRIPT
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()