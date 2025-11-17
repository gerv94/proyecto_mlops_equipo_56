from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from .config import (
    DATA_INTERIM,
    DATA_PROCESSED,
    MODIFIED_CSV,
    ORIGINAL_CSV,
    PREPROCESSED_CSV,
    TABLES,
)


class DatasetRepository:
    """High-level data access layer for the project datasets.

    This class encapsulates all filesystem interaction related to the
    student performance datasets. It centralizes paths and guarantees
    that required directories exist before any read/write operations.
    """

    def __init__(
        self,
        modified_csv: Path = MODIFIED_CSV,
        original_csv: Path = ORIGINAL_CSV,
        preprocessed_csv: Path = PREPROCESSED_CSV,
        interim_dir: Path = DATA_INTERIM,
        processed_dir: Path = DATA_PROCESSED,
        tables_dir: Path = TABLES,
    ) -> None:
        self.modified_csv = modified_csv
        self.original_csv = original_csv
        self.preprocessed_csv = preprocessed_csv
        self.interim_dir = interim_dir
        self.processed_dir = processed_dir
        self.tables_dir = tables_dir

        # Ensure required directories exist for the data flow
        self.interim_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.tables_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load methods
    # ------------------------------------------------------------------

    def load_modified(self) -> pd.DataFrame:
        """Load the cleaned/modified dataset from disk."""
        return pd.read_csv(self.modified_csv)

    def load_preprocessed(self) -> pd.DataFrame:
        """Load the already preprocessed dataset from disk."""
        return pd.read_csv(self.preprocessed_csv)

    def load_original_if_exists(self) -> Optional[pd.DataFrame]:
        """Load the original (raw) dataset only if it exists."""
        if self.original_csv.exists():
            return pd.read_csv(self.original_csv)
        return None

    def load_interim(self, name: str) -> pd.DataFrame:
        """Load an arbitrary CSV from the interim data directory."""
        interim_path = self.interim_dir / name
        return pd.read_csv(interim_path)

    # ------------------------------------------------------------------
    # Save methods
    # ------------------------------------------------------------------

    def save_interim(self, dataframe: pd.DataFrame, name: str = "student_interim.csv") -> Path:
        """Persist an intermediate DataFrame under data/interim."""
        output_path = self.interim_dir / name
        dataframe.to_csv(output_path, index=False)
        return output_path

    def save_processed(self, dataframe: pd.DataFrame, name: str = "student_processed.csv") -> Path:
        """Persist a processed (model-ready) DataFrame under data/processed."""
        output_path = self.processed_dir / name
        dataframe.to_csv(output_path, index=False)
        return output_path

    # ------------------------------------------------------------------
    # Typing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def coerce_numeric_column(series: pd.Series) -> pd.Series:
        """Attempt to coerce a Series to numeric, cleaning common symbols."""
        if series.dtype.kind in "biufc":
            return series
        cleaned = (
            series.astype(str)
            .str.replace(r"[,%$]", "", regex=True)
            .str.strip()
            .replace({"": None, "nan": None, "None": None})
        )
        try:
            return pd.to_numeric(cleaned)
        except (ValueError, TypeError):
            return cleaned

    @classmethod
    def basic_typing(cls, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Apply numeric coercion to every column of the DataFrame."""
        typed_frame = dataframe.copy()
        for column_name in typed_frame.columns:
            typed_frame[column_name] = cls.coerce_numeric_column(typed_frame[column_name])
        return typed_frame


# -----------------------------------------------------------------------------
# Backwards compatible functional API
# -----------------------------------------------------------------------------

_default_repository = DatasetRepository()


def load_modified() -> pd.DataFrame:
    return _default_repository.load_modified()


def load_preprocessed() -> pd.DataFrame:
    return _default_repository.load_preprocessed()


def load_original_if_exists() -> Optional[pd.DataFrame]:
    return _default_repository.load_original_if_exists()


def coerce_numeric_col(series: pd.Series) -> pd.Series:
    return DatasetRepository.coerce_numeric_column(series)


def basic_typing(dataframe: pd.DataFrame) -> pd.DataFrame:
    return DatasetRepository.basic_typing(dataframe)


def save_interim(dataframe: pd.DataFrame, name: str = "student_interim.csv") -> Path:
    return _default_repository.save_interim(dataframe, name)


def save_processed(dataframe: pd.DataFrame, name: str = "student_processed.csv") -> Path:
    return _default_repository.save_processed(dataframe, name)


def load_interim(name: str) -> pd.DataFrame:
    return _default_repository.load_interim(name)
