# tests/test_reports.py
# -----------------------------------------------------------------------------
# Pruebas unitarias para el módulo mlops.reports
# -----------------------------------------------------------------------------

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
    PALETTE_MODELS
)


# -----------------------------------------------------------------------------
# Fixtures y utilidades de prueba
# -----------------------------------------------------------------------------

@pytest.fixture
def sample_dataframe():
    """Fixture que retorna un DataFrame de ejemplo para pruebas."""
    np.random.seed(42)
    data = {
        'numeric_col_1': np.random.randn(100),
        'numeric_col_2': np.random.randn(100),
        'categorical_col_1': np.random.choice(['A', 'B', 'C'], 100),
        'categorical_col_2': np.random.choice(['X', 'Y'], 100),
        'Performance': np.random.choice(['Low', 'Medium', 'High'], 100),
        'target': np.random.choice([0, 1], 100)
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_dataframe_with_nulls():
    """Fixture que retorna un DataFrame con valores nulos."""
    np.random.seed(42)
    data = {
        'numeric_col_1': np.random.randn(50),
        'numeric_col_2': np.random.randn(50),
        'categorical_col_1': np.random.choice(['A', 'B', 'C'], 50),
    }
    df = pd.DataFrame(data)
    # Agregar algunos valores nulos
    df.loc[0:5, 'numeric_col_1'] = np.nan
    df.loc[10:12, 'categorical_col_1'] = None
    return df


@pytest.fixture
def temp_output_dir():
    """Fixture que crea un directorio temporal para pruebas."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


# -----------------------------------------------------------------------------
# Pruebas para ReportBase - Métodos estáticos
# -----------------------------------------------------------------------------

class TestReportBase:
    """Pruebas para la clase base ReportBase."""
    
    def test_guess_target_finds_performance_column(self, sample_dataframe):
        """Test que guess_target encuentra la columna 'Performance'."""
        result = ReportBase.guess_target(sample_dataframe)
        assert result == 'Performance'
    
    def test_guess_target_finds_target_column(self):
        """Test que guess_target encuentra la columna 'target'."""
        df = pd.DataFrame({'target': [0, 1, 0, 1], 'other': [1, 2, 3, 4]})
        result = ReportBase.guess_target(df)
        assert result == 'target'
    
    def test_guess_target_finds_categorical_with_few_classes(self):
        """Test que guess_target encuentra columnas categóricas con pocas clases."""
        n = 20
        # Crear arrays de la misma longitud
        few_classes = ['A', 'B', 'C'] * ((n // 3) + 1)
        numeric = [1.0, 2.0, 3.0] * ((n // 3) + 1)
        df = pd.DataFrame({
            'many_classes': list(range(n)),  # Demasiadas clases
            'few_classes': few_classes[:n],  # Pocas clases
            'numeric': numeric[:n]
        })
        result = ReportBase.guess_target(df)
        assert result == 'few_classes'
    
    def test_guess_target_returns_none_when_no_target(self):
        """Test que guess_target retorna None cuando no hay target obvio."""
        n = 20
        # Crear DataFrame con solo columnas numéricas y con muchas clases únicas
        df = pd.DataFrame({
            'numeric_1': [float(i) for i in range(n)],  # n valores únicos (demasiados)
            'numeric_2': [float(i*2) for i in range(n)],  # n valores únicos
            'many_classes': list(range(n))  # n valores únicos (demasiadas clases)
        })
        result = ReportBase.guess_target(df)
        # Como todas las columnas tienen muchas clases únicas (>10), no debería encontrar target
        assert result is None
    
    def test_compute_summary_metrics_basic(self, sample_dataframe):
        """Test que compute_summary_metrics calcula métricas básicas correctamente."""
        metrics = ReportBase.compute_summary_metrics(sample_dataframe)
        
        assert metrics['rows'] == 100
        assert metrics['cols'] == 6
        assert metrics['n_num'] == 2  # numeric_col_1, numeric_col_2
        assert metrics['n_cat'] == 4  # categorical_col_1, categorical_col_2, Performance, target
        assert isinstance(metrics['null_pct_mean'], float)
        assert isinstance(metrics['dups'], int)
        assert isinstance(metrics['mem_mb'], float)
        assert 'num_cols' in metrics
        assert 'cat_cols' in metrics
    
    def test_compute_summary_metrics_with_nulls(self, sample_dataframe_with_nulls):
        """Test que compute_summary_metrics maneja valores nulos correctamente."""
        metrics = ReportBase.compute_summary_metrics(sample_dataframe_with_nulls)
        
        assert metrics['rows'] == 50
        assert metrics['null_pct_mean'] > 0  # Debe detectar valores nulos
        assert isinstance(metrics['null_pct_mean'], float)
    
    def test_compute_summary_metrics_empty_dataframe(self):
        """Test que compute_summary_metrics maneja DataFrames vacíos."""
        df = pd.DataFrame()
        metrics = ReportBase.compute_summary_metrics(df)
        
        assert metrics['rows'] == 0
        assert metrics['cols'] == 0
        assert metrics['n_num'] == 0
        assert metrics['n_cat'] == 0


# -----------------------------------------------------------------------------
# Pruebas para EDAReport
# -----------------------------------------------------------------------------

class TestEDAReport:
    """Pruebas para la clase EDAReport."""
    
    def test_eda_report_initialization_default(self):
        """Test que EDAReport se inicializa con valores por defecto."""
        report = EDAReport()
        assert report.variant == "base"
        assert report.output_dir.exists()
        assert report.palette == PALETTE_EDA
    
    def test_eda_report_initialization_custom_variant(self):
        """Test que EDAReport acepta variante personalizada."""
        report = EDAReport(variant="clean")
        assert report.variant == "clean"
    
    def test_eda_report_initialization_custom_output_dir(self, temp_output_dir):
        """Test que EDAReport acepta directorio de salida personalizado."""
        report = EDAReport(output_dir=temp_output_dir)
        assert report.output_dir == temp_output_dir
        assert report.output_dir.exists()
    
    def test_eda_report_generate_creates_file(self, sample_dataframe, temp_output_dir):
        """Test que generate() crea un archivo HTML."""
        report = EDAReport(variant="base", output_dir=temp_output_dir)
        
        # Crear un segundo DataFrame para comparación
        df2 = sample_dataframe.copy()
        df2['new_col'] = range(len(df2))
        
        output_path = report.generate(sample_dataframe, df2)
        
        assert output_path.exists()
        assert output_path.suffix == '.html'
        assert output_path.stat().st_size > 0  # El archivo no está vacío
        
        # Verificar que el contenido HTML es válido
        content = output_path.read_text(encoding='utf-8')
        assert '<html>' in content
        assert '</html>' in content


# -----------------------------------------------------------------------------
# Pruebas para PreprocessedReport
# -----------------------------------------------------------------------------

class TestPreprocessedReport:
    """Pruebas para la clase PreprocessedReport."""
    
    def test_preprocessed_report_initialization(self):
        """Test que PreprocessedReport se inicializa correctamente."""
        report = PreprocessedReport()
        assert report.output_dir.exists()
        assert isinstance(report, ReportBase)
    
    def test_preprocessed_report_initialization_custom_dir(self, temp_output_dir):
        """Test que PreprocessedReport acepta directorio personalizado."""
        report = PreprocessedReport(output_dir=temp_output_dir)
        assert report.output_dir == temp_output_dir


# -----------------------------------------------------------------------------
# Pruebas para ModelsReport
# -----------------------------------------------------------------------------

class TestModelsReport:
    """Pruebas para la clase ModelsReport."""
    
    def test_models_report_initialization(self):
        """Test que ModelsReport se inicializa correctamente."""
        report = ModelsReport()
        assert report.output_dir.exists()
        assert report.palette == PALETTE_MODELS
        assert isinstance(report, ReportBase)
    
    def test_models_report_initialization_custom_dir(self, temp_output_dir):
        """Test que ModelsReport acepta directorio personalizado."""
        report = ModelsReport(output_dir=temp_output_dir)
        assert report.output_dir == temp_output_dir


# -----------------------------------------------------------------------------
# Pruebas para create_report (Factory function)
# -----------------------------------------------------------------------------

class TestCreateReport:
    """Pruebas para la función factory create_report."""
    
    def test_create_report_eda_base(self):
        """Test que create_report crea un EDAReport base."""
        report = create_report("eda_base")
        assert isinstance(report, EDAReport)
        assert report.variant == "base"
    
    def test_create_report_eda_clean(self):
        """Test que create_report crea un EDAReport clean."""
        report = create_report("eda_clean")
        assert isinstance(report, EDAReport)
        assert report.variant == "clean"
    
    def test_create_report_preprocessed(self):
        """Test que create_report crea un PreprocessedReport."""
        report = create_report("preprocessed")
        assert isinstance(report, PreprocessedReport)
    
    def test_create_report_models(self):
        """Test que create_report crea un ModelsReport."""
        report = create_report("models")
        assert isinstance(report, ModelsReport)
    
    def test_create_report_invalid_type(self):
        """Test que create_report lanza error con tipo inválido."""
        with pytest.raises(ValueError, match="Tipo de reporte desconocido"):
            create_report("invalid_type")


# -----------------------------------------------------------------------------
# Pruebas de integración (básicas)
# -----------------------------------------------------------------------------

@pytest.mark.integration
class TestReportsIntegration:
    """Pruebas de integración para el módulo de reportes."""
    
    def test_eda_report_full_workflow(self, sample_dataframe, temp_output_dir):
        """Test del flujo completo de generación de reporte EDA."""
        report = EDAReport(variant="base", output_dir=temp_output_dir)
        df2 = sample_dataframe.copy()
        
        output_path = report.generate(sample_dataframe, df2)
        
        # Verificar que el archivo se generó correctamente
        assert output_path.exists()
        assert output_path.is_file()
        
        # Verificar contenido básico del HTML
        content = output_path.read_text(encoding='utf-8')
        assert 'EDA' in content or 'Exploratory' in content or 'Análisis' in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

