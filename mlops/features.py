import pandas as pd
from .config import TABLES

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Asegura que exista la carpeta TABLES (almacenará tablas o artefactos derivados)
TABLES.mkdir(parents=True, exist_ok=True)

# Umbral heurístico: columnas con <= 30 valores únicos se tratarán como categóricas
CATEGORICAL_GUESS_MAX = 30


# -----------------------------------------------------------------------------
# FUNCIONES PARA CLASIFICAR COLUMNAS NUMÉRICAS Y CATEGÓRICAS
# -----------------------------------------------------------------------------
def split_num_cat(df: pd.DataFrame):
    """
    Divide las columnas de un DataFrame entre numéricas y categóricas usando
    una heurística basada en el tipo de datos y la cardinalidad.

    1. Detecta tipos numéricos directamente con `pandas.api.types.is_numeric_dtype`.
    2. Cualquier columna con <= 30 valores únicos se considera categórica
       (esto cubre columnas tipo "score bands", "levels", etc.).
    3. Se ajustan listas finales para evitar duplicados.

    Args:
        df (pd.DataFrame): DataFrame de entrada.

    Returns:
        tuple[list[str], list[str]]: 
            - Lista de columnas numéricas.
            - Lista de columnas categóricas.
    """
    # Columnas que pandas detecta como numéricas
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    # Inicialmente, las demás se marcan como categóricas
    cat_cols = [c for c in df.columns if c not in num_cols]

    # Heurística adicional: columnas con pocos valores únicos son categóricas
    cat_cols = list(
        {c for c in cat_cols} |
        {c for c in df.columns if df[c].nunique(dropna=True) <= CATEGORICAL_GUESS_MAX}
    )

    # Refinamos numéricas para eliminar las categóricas detectadas
    num_cols = [c for c in df.columns if c not in cat_cols]
    return num_cols, cat_cols


# -----------------------------------------------------------------------------
# FUNCIONES DE LIMPIEZA DE VARIABLES CATEGÓRICAS
# -----------------------------------------------------------------------------
def normalize_categories(s: pd.Series) -> pd.Series:
    """
    Normaliza texto en columnas categóricas para evitar duplicados debidos a formato.

    Estandariza el texto:
        - Convierte todo a minúsculas.
        - Elimina espacios extra al inicio, final y duplicados entre palabras.
    
    Args:
        s (pd.Series): Columna del DataFrame.

    Returns:
        pd.Series: Columna normalizada si es tipo 'object'; caso contrario, se devuelve igual.
    """
    if s.dtype == object:
        return (
            s.astype(str)
             .str.strip()                    # elimina espacios en extremos
             .str.replace(r"\s+", " ", regex=True)  # reemplaza múltiples espacios por uno
             .str.lower()                    # pasa todo a minúsculas
        )
    return s


def clean_categoricals(df: pd.DataFrame, cat_cols):
    """
    Aplica `normalize_categories` a todas las columnas categóricas del DataFrame.

    Args:
        df (pd.DataFrame): DataFrame original.
        cat_cols (list[str]): Lista de columnas categóricas.

    Returns:
        pd.DataFrame: Copia del DataFrame con valores categóricos normalizados.
    """
    out = df.copy()
    for c in cat_cols:
        out[c] = normalize_categories(out[c])
    return out


# -----------------------------------------------------------------------------
# PREPROCESAMIENTO BÁSICO (para EDA o baseline)
# -----------------------------------------------------------------------------
def minimal_preprocess(df: pd.DataFrame):
    """
    Realiza un preprocesamiento mínimo para preparar los datos antes del modelado o EDA.

    - Imputa valores faltantes:
        * Numéricas → mediana.
        * Categóricas → moda (valor más frecuente).
    - Retorna tanto el DataFrame transformado como las listas de columnas numéricas y categóricas.

    Nota:
        Este enfoque es deliberadamente simple (no usa sklearn.Pipeline)
        y está pensado para análisis exploratorio rápido.

    Args:
        df (pd.DataFrame): DataFrame de entrada.

    Returns:
        tuple[pd.DataFrame, list[str], list[str]]:
            - DataFrame preprocesado.
            - Lista de columnas numéricas.
            - Lista de columnas categóricas.
    """
    num_cols, cat_cols = split_num_cat(df)
    out = df.copy()

    # Imputación simple por tipo de variable
    for c in num_cols:
        out[c] = out[c].fillna(out[c].median())
    for c in cat_cols:
        out[c] = out[c].fillna(out[c].mode().iloc[0])

    return out, num_cols, cat_cols


# -----------------------------------------------------------------------------
# PREPROCESAMIENTO AVANZADO (ESCALADO + ONE-HOT + PCA)
# -----------------------------------------------------------------------------
def preprocess_advanced(
    df: pd.DataFrame,
    num_cols: list[str],
    cat_cols: list[str],
    n_components: int = 3
) -> pd.DataFrame:
    """
    Preprocesamiento para modelado:
      - Escalado de numéricas (StandardScaler)
      - Codificación One-Hot de categóricas (handle_unknown='ignore')
      - PCA opcional (añade columnas PC1..PCk)

    Reglas:
      - Si no hay numéricas, no escala.
      - Si no hay categóricas, no codifica.
      - PCA solo si hay columnas suficientes tras combinar numéricas + One-Hot.
    """
    df_proc = df.copy()

    # Escalado de numéricas
    if len(num_cols) > 0:
        scaler = StandardScaler()
        df_proc[num_cols] = scaler.fit_transform(df_proc[num_cols])

    # Codificación One-Hot (compatibilidad de versiones)
    df_encoded = None
    if len(cat_cols) > 0:
        try:
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        except TypeError:
            encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        encoded = encoder.fit_transform(df_proc[cat_cols])
        encoded_cols = encoder.get_feature_names_out(cat_cols)
        df_encoded = pd.DataFrame(encoded, columns=encoded_cols, index=df_proc.index)

    # Unir partes
    parts = []
    if len(num_cols) > 0:
        parts.append(df_proc[num_cols])
    if df_encoded is not None:
        parts.append(df_encoded)

    if not parts:
        print("No se encontraron columnas numéricas ni categóricas para transformar.")
        return df_proc

    X = pd.concat(parts, axis=1)

    # PCA (Opcional)
    if n_components and n_components > 0:
        max_components = min(n_components, X.shape[1])
        if max_components >= 1:
            pca = PCA(n_components=max_components, random_state=42)
            components = pca.fit_transform(X)
            pca_cols = [f"PC{i+1}" for i in range(max_components)]
            df_pca = pd.DataFrame(components, columns=pca_cols, index=X.index)
            print("Varianza explicada (PCA):", np.round(pca.explained_variance_ratio_, 3))
            X = pd.concat([X, df_pca], axis=1)
        else:
            print("PCA omitido: menos columnas que componentes solicitados.")

    print("Preprocesamiento avanzado finalizado.")
    return X


# -----------------------------------------------------------------------------
# PREPROCESAMIENTO PARA ENTRENAMIENTO (ONE-HOT SIN PCA NI ESCALADO)
# -----------------------------------------------------------------------------
def preprocess_for_training(
    df: pd.DataFrame,
    num_cols: list[str],
    cat_cols: list[str]
) -> pd.DataFrame:
    """
    Preprocesamiento para modelado alineado con el pipeline de entrenamiento:
      - Mantiene numéricas sin escalar (como en el notebook)
      - Codificación One-Hot de categóricas (handle_unknown='ignore')
      - NO aplica PCA
      - NO aplica StandardScaler
    
    Esta función es compatible con el pipeline usado en train_model_randomfores_dvc_3.ipynb
    que usa OneHotEncoder y LabelEncoder (este último solo para el target, se aplica en training).

    Args:
        df (pd.DataFrame): DataFrame limpio (ya imputado)
        num_cols (list[str]): Lista de columnas numéricas
        cat_cols (list[str]): Lista de columnas categóricas

    Returns:
        pd.DataFrame: DataFrame preprocesado con OneHot encoding en categóricas
    """
    df_proc = df.copy()

    # Codificación One-Hot (compatibilidad de versiones)
    df_encoded = None
    if len(cat_cols) > 0:
        try:
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        except TypeError:
            encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        encoded = encoder.fit_transform(df_proc[cat_cols])
        encoded_cols = encoder.get_feature_names_out(cat_cols)
        df_encoded = pd.DataFrame(encoded, columns=encoded_cols, index=df_proc.index)

    # Identificar columnas a preservar (target y otras que no sean numéricas ni categóricas)
    all_processed_cols = set(num_cols) | set(cat_cols)
    preserve_cols = [c for c in df_proc.columns if c not in all_processed_cols]
    
    # Unir partes: numéricas (sin escalar) + categóricas codificadas + columnas preservadas
    parts = []
    if len(num_cols) > 0:
        parts.append(df_proc[num_cols])
    if df_encoded is not None:
        parts.append(df_encoded)
    if len(preserve_cols) > 0:
        parts.append(df_proc[preserve_cols])

    if not parts:
        print("No se encontraron columnas numéricas ni categóricas para transformar.")
        return df_proc

    X = pd.concat(parts, axis=1)

    print("Preprocesamiento para entrenamiento finalizado (OneHotEncoder, sin PCA ni escalado).")
    if preserve_cols:
        print(f"Columnas preservadas: {preserve_cols}")
    return X