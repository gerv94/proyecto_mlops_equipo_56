import pandas as pd
from mlops.config import TABLES, CLEAN_CSV, PREPROCESSED_CSV
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
    dataset._data_manager.save(df_clean, CLEAN_CSV)
    print(f"[OK] Clean data saved: {CLEAN_CSV}")
    
    numeric_cols_clean, categorical_cols_clean = features.split_num_cat(df_clean)
    df_ready = features.preprocess_advanced(
        df_clean,
        num_cols=numeric_cols_clean,
        cat_cols=categorical_cols_clean,
        n_components=3
    )
    dataset._data_manager.save(df_ready, PREPROCESSED_CSV)
    print(f"[OK] Preprocessed data saved: {PREPROCESSED_CSV}")


# -----------------------------------------------------------------------------
# EJECUCIÓN DIRECTA DEL SCRIPT
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()