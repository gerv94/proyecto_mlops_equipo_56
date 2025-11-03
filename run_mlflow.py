# ===============================================================
# Script: run_mlflow.py
# Descripción:
#   Lanza la interfaz gráfica de MLflow (MLflow UI) utilizando
#   el entorno virtual del proyecto (.venv) y el backend local
#   de experimentos (mlruns/).
# ===============================================================

import subprocess
import sys
import os

# ---------------------------------------------------------------
# 1. Detección automática del entorno virtual (.venv)
# ---------------------------------------------------------------
# En Windows, los binarios están en ".venv/Scripts/"
# En Linux/macOS (o code-server), están en ".venv/bin/"
# ---------------------------------------------------------------
if os.name == "nt":  # nt = Windows
    venv_python = os.path.join(".venv", "Scripts", "python.exe")
else:
    venv_python = os.path.join(".venv", "bin", "python")

# ---------------------------------------------------------------
# 2. Comando de ejecución de MLflow
# ---------------------------------------------------------------
# Se lanza el servidor UI de MLflow, apuntando al backend local.
# Cambia el puerto si ya hay otro servicio usando el 5000.
# En code-server o entornos remotos, usa "--host 0.0.0.0".
# ---------------------------------------------------------------
cmd = [
    venv_python, "-m", "mlflow", "ui",
    "--backend-store-uri", "file:./mlruns",
    "--port", "5001"
]

def run_mlflow_ui():
    """Launch MLflow UI server."""
    print("===============================================================")
    print("  Starting MLflow UI at http://127.0.0.1:5001")
    print("  Press Ctrl+C to stop server.")
    print("===============================================================")
    subprocess.run(cmd)

if __name__ == "__main__":
    run_mlflow_ui()
