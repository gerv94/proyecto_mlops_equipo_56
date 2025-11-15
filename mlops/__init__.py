# mlops package
from . import config, dataset, features, plots

# Reportes HTML (sistema orientado a objetos)
try:
    from . import reports
    from .reports import ReportBase, EDAReport, PreprocessedReport, ModelsReport, create_report
    __all__ = [
        'config',
        'dataset', 
        'features',
        'plots',
        'reports',
        'ReportBase',
        'EDAReport',
        'PreprocessedReport',
        'ModelsReport',
        'create_report'
    ]
except ImportError:
    __all__ = ['config', 'dataset', 'features', 'plots']

