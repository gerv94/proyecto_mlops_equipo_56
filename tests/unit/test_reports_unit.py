import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil

from mlops.reports import (
    ReportBase,
    EDAReport,
    PreprocessedReport,
    ModelsReport,
    create_report,
    PALETTE_EDA,
    PALETTE_MODELS,
)


# -----------------------------------------------------------------------------
# Fixtures and helpers for unit tests
# -----------------------------------------------------------------------------

@pytest.fixture
def sample_dataframe():
    """Return a sample DataFrame for unit tests."""
    np.random.seed(42)
    data = {
        "numeric_col_1": np.random.randn(100),
        "numeric_col_2": np.random.randn(100),
        "categorical_col_1": np.random.choice(["A", "B", "C"], 100),
        "categorical_col_2": np.random.choice(["X", "Y"], 100),
        "Performance": np.random.choice(["Low", "Medium", "High"], 100),
        "target": np.random.choice([0, 1], 100),
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_dataframe_with_nulls():
    """Return a DataFrame with null values for metric tests."""
    np.random.seed(42)
    data = {
        "numeric_col_1": np.random.randn(50),
        "numeric_col_2": np.random.randn(50),
        "categorical_col_1": np.random.choice(["A", "B", "C"], 50),
    }
    df = pd.DataFrame(data)
    df.loc[0:5, "numeric_col_1"] = np.nan
    df.loc[10:12, "categorical_col_1"] = None
    return df


@pytest.fixture
def temp_output_dir():
    """Create a temporary output directory for unit tests."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


# -----------------------------------------------------------------------------
# Unit tests for ReportBase static utilities
# -----------------------------------------------------------------------------


class TestReportBase:
    """Unit tests for the base report utilities."""

    def test_guess_target_finds_performance_column(self, sample_dataframe):
        result = ReportBase.guess_target(sample_dataframe)
        assert result == "Performance"

    def test_guess_target_finds_target_column(self):
        df = pd.DataFrame({"target": [0, 1, 0, 1], "other": [1, 2, 3, 4]})
        result = ReportBase.guess_target(df)
        assert result == "target"

    def test_guess_target_finds_categorical_with_few_classes(self):
        n_rows = 20
        few_classes = ["A", "B", "C"] * ((n_rows // 3) + 1)
        numeric = [1.0, 2.0, 3.0] * ((n_rows // 3) + 1)
        df = pd.DataFrame(
            {
                "many_classes": list(range(n_rows)),
                "few_classes": few_classes[:n_rows],
                "numeric": numeric[:n_rows],
            }
        )
        result = ReportBase.guess_target(df)
        assert result == "few_classes"

    def test_guess_target_returns_none_when_no_target(self):
        n_rows = 20
        df = pd.DataFrame(
            {
                "numeric_1": [float(i) for i in range(n_rows)],
                "numeric_2": [float(i * 2) for i in range(n_rows)],
                "many_classes": list(range(n_rows)),
            }
        )
        result = ReportBase.guess_target(df)
        assert result is None

    def test_compute_summary_metrics_basic(self, sample_dataframe):
        metrics = ReportBase.compute_summary_metrics(sample_dataframe)

        assert metrics["rows"] == 100
        assert metrics["cols"] == 6
        assert metrics["n_num"] == 2
        assert metrics["n_cat"] == 4
        assert isinstance(metrics["null_pct_mean"], float)
        assert isinstance(metrics["dups"], int)
        assert isinstance(metrics["mem_mb"], float)
        assert "num_cols" in metrics
        assert "cat_cols" in metrics

    def test_compute_summary_metrics_with_nulls(self, sample_dataframe_with_nulls):
        metrics = ReportBase.compute_summary_metrics(sample_dataframe_with_nulls)

        assert metrics["rows"] == 50
        assert metrics["null_pct_mean"] > 0
        assert isinstance(metrics["null_pct_mean"], float)

    def test_compute_summary_metrics_empty_dataframe(self):
        df = pd.DataFrame()
        metrics = ReportBase.compute_summary_metrics(df)

        assert metrics["rows"] == 0
        assert metrics["cols"] == 0
        assert metrics["n_num"] == 0
        assert metrics["n_cat"] == 0


# -----------------------------------------------------------------------------
# Unit tests for EDAReport
# -----------------------------------------------------------------------------


class TestEDAReport:
    """Unit tests for EDAReport configuration and basic generation."""

    def test_eda_report_initialization_default(self):
        report = EDAReport()
        assert report.variant == "base"
        assert report.output_dir.exists()
        assert report.palette == PALETTE_EDA

    def test_eda_report_initialization_custom_variant(self):
        report = EDAReport(variant="clean")
        assert report.variant == "clean"

    def test_eda_report_initialization_custom_output_dir(self, temp_output_dir):
        report = EDAReport(output_dir=temp_output_dir)
        assert report.output_dir == temp_output_dir
        assert report.output_dir.exists()

    def test_eda_report_generate_creates_file(self, sample_dataframe, temp_output_dir):
        report = EDAReport(variant="base", output_dir=temp_output_dir)

        comparison_dataframe = sample_dataframe.copy()
        comparison_dataframe["new_col"] = range(len(comparison_dataframe))

        output_path = report.generate(sample_dataframe, comparison_dataframe)

        assert output_path.exists()
        assert output_path.suffix == ".html"
        assert output_path.stat().st_size > 0

        content = output_path.read_text(encoding="utf-8")
        assert "<html>" in content
        assert "</html>" in content


# -----------------------------------------------------------------------------
# Unit tests for PreprocessedReport
# -----------------------------------------------------------------------------


class TestPreprocessedReport:
    """Unit tests for PreprocessedReport basic behaviour."""

    def test_preprocessed_report_initialization(self):
        report = PreprocessedReport()
        assert report.output_dir.exists()
        assert isinstance(report, ReportBase)

    def test_preprocessed_report_initialization_custom_dir(self, temp_output_dir):
        report = PreprocessedReport(output_dir=temp_output_dir)
        assert report.output_dir == temp_output_dir


# -----------------------------------------------------------------------------
# Unit tests for ModelsReport
# -----------------------------------------------------------------------------


class TestModelsReport:
    """Unit tests for ModelsReport basic behaviour."""

    def test_models_report_initialization(self):
        report = ModelsReport()
        assert report.output_dir.exists()
        assert report.palette == PALETTE_MODELS
        assert isinstance(report, ReportBase)

    def test_models_report_initialization_custom_dir(self, temp_output_dir):
        report = ModelsReport(output_dir=temp_output_dir)
        assert report.output_dir == temp_output_dir


# -----------------------------------------------------------------------------
# Unit tests for create_report factory
# -----------------------------------------------------------------------------


class TestCreateReport:
    """Unit tests for the create_report factory function."""

    def test_create_report_eda_base(self):
        report = create_report("eda_base")
        assert isinstance(report, EDAReport)
        assert report.variant == "base"

    def test_create_report_eda_clean(self):
        report = create_report("eda_clean")
        assert isinstance(report, EDAReport)
        assert report.variant == "clean"

    def test_create_report_preprocessed(self):
        report = create_report("preprocessed")
        assert isinstance(report, PreprocessedReport)

    def test_create_report_models(self):
        report = create_report("models")
        assert isinstance(report, ModelsReport)

    def test_create_report_invalid_type(self):
        with pytest.raises(ValueError, match="Tipo de reporte desconocido"):
            create_report("invalid_type")
