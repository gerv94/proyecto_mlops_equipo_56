# ===============================================================
# Script: run_mlflow.py
# ... (descripción) ...
# ===============================================================

import subprocess
import sys
import os

# ---------------------------------------------------------------
# 1. Detección automática del entorno virtual (.venv)
# ---------------------------------------------------------------

# Define el prefijo de la ruta para retroceder un nivel
VENV_PREFIX = os.path.join("..", ".venv")

# En Windows, los binarios están en ".venv/Scripts/"
if os.name == "nt":
    venv_python_relative = os.path.join(VENV_PREFIX, "Scripts", "python.exe")
else:
    venv_python_relative = os.path.join(VENV_PREFIX, "bin", "python")

# --- ¡CORRECCIÓN CRUCIAL AQUÍ! ---
# Convertir la ruta a absoluta para que subprocess.run() la pueda encontrar en Windows.
venv_python = os.path.abspath(venv_python_relative) 
# ----------------------------------

# ---------------------------------------------------------------
# 2. Comando de ejecución de MLflow
# ---------------------------------------------------------------
cmd = [venv_python, "-m", "mlflow", "ui","--backend-store-uri", "file:../mlruns","--port", "5001"] # El backend sigue siendo relativo a la raíz

# ---------------------------------------------------------------
# 3. Ejecución del proceso
# ---------------------------------------------------------------
print("===============================================================")
print(f" Usando Python: {venv_python}") # Añadido para verificar la ruta
print(" Iniciando MLflow UI en http://127.0.0.1:5001")
print(" Presiona Ctrl+C para detener el servidor.")
print("===============================================================")

# Intentar ejecutar
try:
    subprocess.run(cmd, check=True)
except FileNotFoundError:
    print("\nERROR CRÍTICO: No se encontró el ejecutable de Python en la ruta absoluta.")
    print("Por favor, verifica que la carpeta .venv exista y que el entorno virtual haya sido creado correctamente.")
    sys.exit(1)
