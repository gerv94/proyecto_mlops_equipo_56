import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil

from mlops.reports import EDAReport


@pytest.fixture
def sample_dataframe():
    """Return a sample DataFrame for integration-style report tests."""
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
def temp_output_dir():
    """Create a temporary directory for integration outputs."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.mark.integration
class TestReportsIntegration:
    """Basic integration test for the EDA HTML report generation."""

    def test_eda_report_full_workflow(self, sample_dataframe, temp_output_dir):
        report = EDAReport(variant="base", output_dir=temp_output_dir)
        comparison_dataframe = sample_dataframe.copy()

        output_path = report.generate(sample_dataframe, comparison_dataframe)

        assert output_path.exists()
        assert output_path.is_file()

        content = output_path.read_text(encoding="utf-8")
        assert "EDA" in content or "Exploratory" in content or "Análisis" in content
