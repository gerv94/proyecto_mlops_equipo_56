from __future__ import annotations

from typing import Tuple

import pandas as pd

from mlops import dataset, features
from mlops.dataset import DatasetRepository
from mlops.features import FeatureEngineering


class PreprocessPipeline:
    """High-level preprocessing pipeline used by the DVC stage.

    The goal is to keep all pre-processing logic encapsulated in this
    class while ``make_clean_interim`` and ``run_all`` remain thin
    functional wrappers for backwards compatibility.
    """

    def __init__(
        self,
        dataset_repository: DatasetRepository | None = None,
        feature_engineer: FeatureEngineering | None = None,
    ) -> None:
        self.dataset_repository = dataset_repository or DatasetRepository()
        self.feature_engineer = feature_engineer or FeatureEngineering()

    def make_clean_interim(self) -> str:
        """Generate the cleaned intermediate CSV without imputation.

        This method:
        - Loads the modified dataset.
        - Applies basic typing to coerce numeric columns.
        - Normalises categorical text values.
        - Removes 'mixed_type_col' if it exists (generated during cleaning).
        - Saves the cleaned dataset under ``data/interim``.
        """

        raw_dataframe = self.dataset_repository.load_modified()
        typed_dataframe = self.dataset_repository.basic_typing(raw_dataframe)

        numeric_columns, categorical_columns = self.feature_engineer.split_num_cat(typed_dataframe)
        cleaned_dataframe = self.feature_engineer.clean_categoricals(
            typed_dataframe, categorical_columns
        )

        # Eliminar columna 'mixed_type_col' si existe (generada durante limpieza)
        if "mixed_type_col" in cleaned_dataframe.columns:
            cleaned_dataframe = cleaned_dataframe.drop(columns=["mixed_type_col"])

        output_path = self.dataset_repository.save_interim(
            cleaned_dataframe, "student_interim_clean.csv"
        )
        return str(output_path)

    def run_all(self) -> str:
        """Entry point used by DVC: run the whole preprocessing flow."""

        return self.make_clean_interim()


# -----------------------------------------------------------------------------
# Backwards compatible functional API
# -----------------------------------------------------------------------------

_default_pipeline = PreprocessPipeline()


def make_clean_interim() -> str:
    """Backwards compatible wrapper around ``PreprocessPipeline.make_clean_interim``."""

    return _default_pipeline.make_clean_interim()


def run_all() -> str:
    """Backwards compatible wrapper around ``PreprocessPipeline.run_all``."""

    return _default_pipeline.run_all()
