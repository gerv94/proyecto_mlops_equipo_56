import sys
from pathlib import Path

# Agregar la raíz del proyecto al PYTHONPATH usando la misma lógica que config.py
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from mlops.preprocess import run_all

if __name__ == "__main__":
    clean_path = run_all()
    print(f"[OK] Datos limpios generados: {clean_path}")
