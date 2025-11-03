"""Wrapper script for backward compatibility. Delegates to run_eda.py"""
import warnings
from run_eda import main as run_eda_main

warnings.warn(
    "run_eda_2.py is deprecated and will be removed in future versions. Use run_eda.py instead.",
    DeprecationWarning,
    stacklevel=2
)

if __name__ == "__main__":
    print("[DEPRECATION] run_eda_2.py is deprecated. Please use run_eda.py instead.")
    run_eda_main()
