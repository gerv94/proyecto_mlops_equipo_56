import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from .config import FIGURES, TABLES, MAX_CATS_TO_PLOT

# -------------------------------------------------------------------------
# Configuración inicial de directorios y estilos
# -------------------------------------------------------------------------

# Crea las carpetas donde se guardarán las figuras y tablas si no existen.
FIGURES.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------------
# Funciones de soporte
# -------------------------------------------------------------------------
def savefig(name, dpi=300):
    """
    Guarda la figura actual de Matplotlib en formato PNG dentro de la carpeta FIGURES.

    Args:
        name (str): Nombre base del archivo sin extensión.
        dpi (int): Resolución de la imagen en puntos por pulgada (default=300).

    Returns:
        Path: Ruta completa al archivo guardado.
    """
    plt.tight_layout()  # Ajusta márgenes para evitar cortes de etiquetas
    p = FIGURES / f"{name}.png"
    plt.savefig(p, dpi=dpi)
    plt.close()  # Cierra la figura para liberar memoria
    return p


# -------------------------------------------------------------------------
# Gráficas principales del EDA
# -------------------------------------------------------------------------
def plot_target_distribution(df, target: str):
    """
    Genera una gráfica de barras que muestra la distribución de la variable objetivo.

    Args:
        df (pd.DataFrame): DataFrame de entrada.
        target (str): Nombre de la columna objetivo.

    Returns:
        Path | None: Ruta de la imagen generada, o None si la columna no existe.
    """
    if target not in df.columns:
        return None

    plt.figure(figsize=(6, 4))
    sns.countplot(x=df[target], order=df[target].value_counts().index)
    plt.title(f"Distribución de {target}")
    plt.xlabel(target)
    plt.ylabel("Frecuencia")
    return savefig("target_distribution")


def plot_missingness(df):
    """
    Muestra el porcentaje de valores nulos por columna en forma de gráfico horizontal.

    Args:
        df (pd.DataFrame): DataFrame de entrada.

    Returns:
        Path: Ruta de la figura guardada en PNG.
    """
    plt.figure(figsize=(8, max(4, 0.25 * len(df.columns))))
    df.isna().mean().sort_values().plot(kind="barh", color="#4A7486")
    plt.title("% de nulos por columna")
    plt.xlabel("Porcentaje de valores nulos")
    return savefig("missingness_by_column")


def plot_numerics(df, numeric_cols):
    """
    Genera histogramas y boxplots para cada columna numérica del DataFrame.

    Args:
        df (pd.DataFrame): DataFrame de entrada.
        numeric_cols (list[str]): Lista de nombres de columnas numéricas.

    Returns:
        None: Las imágenes se guardan automáticamente en la carpeta FIGURES.
    """
    for c in numeric_cols:
        # Histograma con KDE (distribución de densidad)
        plt.figure(figsize=(5, 3))
        sns.histplot(df[c].dropna(), bins=30, kde=True, color="#0E3A4B")
        plt.title(f"Histograma: {c}")
        savefig(f"hist_{c}")

        # Boxplot para detectar outliers
        plt.figure(figsize=(5, 2.8))
        sns.boxplot(x=df[c], orient="h", color="#AFC4B2")
        plt.title(f"Boxplot: {c}")
        savefig(f"box_{c}")


def plot_categoricals(df, categorical_cols):
    """
    Genera gráficos de barras horizontales para las variables categóricas
    con pocas categorías (controlado por MAX_CATS_TO_PLOT).

    Args:
        df (pd.DataFrame): DataFrame de entrada.
        categorical_cols (list[str]): Lista de nombres de columnas categóricas.

    Returns:
        None: Las imágenes se guardan automáticamente en la carpeta FIGURES.
    """
    for c in categorical_cols:
        n = df[c].nunique(dropna=False)

        # Solo se grafican si el número de categorías es razonable
        if 2 <= n <= MAX_CATS_TO_PLOT:
            plt.figure(figsize=(7, 3))
            order = df[c].value_counts(dropna=False).index
            sns.countplot(y=df[c], order=order, palette="crest")
            plt.title(f"Categoría: {c} (n={n})")
            plt.xlabel("Frecuencia")
            plt.ylabel(c)
            savefig(f"cat_{c}")
