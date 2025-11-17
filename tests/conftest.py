"""
conftest.py - Configuración compartida para pytest

Este archivo se ejecuta automáticamente por pytest antes de cualquier test.
Aquí configuramos el PYTHONPATH para que pytest pueda importar el módulo mlops.
"""

import sys
from pathlib import Path

# Agregar el directorio raíz del proyecto al PYTHONPATH
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

