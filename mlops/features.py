import pandas as pd
from .config import TABLES

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
