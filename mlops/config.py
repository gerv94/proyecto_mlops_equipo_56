from pathlib import Path

# Raíz del proyecto (modificar en caso de ser otra)
ROOT = Path(__file__).resolve().parents[1]

DATA_RAW       = ROOT / "data" / "raw"
DATA_INTERIM   = ROOT / "data" / "interim"
DATA_PROCESSED = ROOT / "data" / "processed"

REPORTS        = ROOT / "reports"
FIGURES        = REPORTS / "figures"
TABLES         = REPORTS / "tables"

# Archivos
MODIFIED_CSV = DATA_RAW / "student_entry_performance_modified.csv"
ORIGINAL_CSV = DATA_RAW / "student_entry_performance_original.csv"  # opcional

# Parámetros EDA
MAX_CATS_TO_PLOT = 20
RANDOM_STATE = 42