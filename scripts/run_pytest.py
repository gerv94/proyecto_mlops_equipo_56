#!/usr/bin/env python
"""
Script helper para ejecutar pytest usando el entorno virtual .venv
"""

import subprocess
import sys
from pathlib import Path


def main():
    """Ejecuta pytest con los argumentos proporcionados."""
    # Determinar la ruta al pytest del .venv
    project_root = Path(__file__).resolve().parent.parent
    venv_pytest = project_root / ".venv" / "Scripts" / "pytest.exe"
    
    # Fallback si no existe en Windows, probar formato Unix
    if not venv_pytest.exists():
        venv_pytest = project_root / ".venv" / "bin" / "pytest"
    
    # Si no existe pytest.exe, usar python -m pytest del venv
    if not venv_pytest.exists():
        venv_python = project_root / ".venv" / "Scripts" / "python.exe"
        if not venv_python.exists():
            venv_python = project_root / ".venv" / "bin" / "python"
        
        if venv_python.exists():
            cmd = [str(venv_python), "-m", "pytest"] + sys.argv[1:]
        else:
            print("ERROR: No se encontró el entorno virtual .venv", file=sys.stderr)
            print("Por favor, créalo con: python -m venv .venv", file=sys.stderr)
            sys.exit(1)
    else:
        cmd = [str(venv_pytest)] + sys.argv[1:]
    
    # Ejecutar pytest
    print(f"Ejecutando pytest desde: {cmd[0]}")
    print("-" * 80)
    sys.exit(subprocess.run(cmd).returncode)


if __name__ == '__main__':
    main()

