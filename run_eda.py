import pandas as pd
from mlops.config import TABLES
from mlops import dataset, features, plots
from mlops.utils.common import guess_target


# -----------------------------------------------------------------------------
# FUNCIÓN PRINCIPAL: ORQUESTA EL ANÁLISIS EXPLORATORIO (EDA)
# -----------------------------------------------------------------------------
def main():
    """Main EDA pipeline with OOP refactored classes."""
    dataframe = dataset.load_modified()
    dataframe = dataset.basic_typing(dataframe)
    
    numeric_cols, categorical_cols = features.split_num_cat(dataframe)
    dataframe = features.clean_categoricals(dataframe, categorical_cols)
    
    numeric_cols, categorical_cols = features.split_num_cat(dataframe)
    target = guess_target(dataframe)
    print(f"[INFO] Target detected: {target}")
    
    if target:
        plots.plot_target_distribution(dataframe, target)
    plots.plot_missingness(dataframe)
    plots.plot_numerics(dataframe, numeric_cols)
    plots.plot_categoricals(dataframe, categorical_cols)
    
    df_clean, _, _ = features.minimal_preprocess(dataframe)
    out_clean = dataset.save_interim(df_clean, "student_interim_clean.csv")
    print(f"[OK] Clean data saved: {out_clean}")
    
    numeric_cols_clean, categorical_cols_clean = features.split_num_cat(df_clean)
    df_ready = features.preprocess_advanced(
        df_clean,
        num_cols=numeric_cols_clean,
        cat_cols=categorical_cols_clean,
        n_components=3
    )
    out_ready = dataset.save_interim(df_ready, "student_interim_preprocessed.csv")
    print(f"[OK] Preprocessed data saved: {out_ready}")


# -----------------------------------------------------------------------------
# EJECUCIÓN DIRECTA DEL SCRIPT
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()