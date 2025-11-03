"""Wrapper script for backward compatibility. Delegates to run_mlflow.py"""
import warnings
import subprocess
import sys
import os

warnings.warn(
    "run_mlflow_2.py is deprecated. Use run_mlflow.py instead.",
    DeprecationWarning,
    stacklevel=2
)

if __name__ == "__main__":
    print("[DEPRECATION] run_mlflow_2.py is deprecated. Use run_mlflow.py")
    from run_mlflow import run_mlflow_ui
    run_mlflow_ui()
