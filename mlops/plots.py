import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from .config import FIGURES, TABLES, MAX_CATS_TO_PLOT
from mlops.core.plot_generator import PlotGenerator

FIGURES.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)

_plot_gen = PlotGenerator(output_dir=FIGURES)

# -------------------------------------------------------------------------
# Funciones de soporte
# -------------------------------------------------------------------------
def savefig(name, dpi=300):
    """Save current matplotlib figure to FIGURES folder."""
    plt.tight_layout()
    path = FIGURES / f"{name}.png"
    plt.savefig(path, dpi=dpi)
    plt.close()
    return path


# -------------------------------------------------------------------------
# Gráficas principales del EDA
# -------------------------------------------------------------------------
def plot_target_distribution(dataframe, target: str):
    """Plot target distribution (delegates to PlotGenerator)."""
    if target not in dataframe.columns:
        return None
    return _plot_gen.target_distribution(dataframe, target, "target_distribution.png")


def plot_missingness(dataframe):
    """Plot missing values by column (delegates to PlotGenerator)."""
    return _plot_gen.missingness(dataframe, "missingness_by_column.png")


def plot_numerics(dataframe, numeric_cols):
    """Generate histogram and boxplot for numeric columns (delegates to PlotGenerator)."""
    for column in numeric_cols:
        _plot_gen.numeric_distribution(dataframe, column, column)


def plot_categoricals(dataframe, categorical_cols):
    """Generate bar charts for categorical columns (delegates to PlotGenerator)."""
    for column in categorical_cols:
        _plot_gen.categorical_distribution(dataframe, column, f"cat_{column}.png", MAX_CATS_TO_PLOT)
