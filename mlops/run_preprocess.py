import sys
from pathlib import Path

# Agregar la raíz del proyecto al PYTHONPATH usando la misma lógica que config.py
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from mlops.preprocess import PreprocessPipeline, run_all


class PreprocessRunner:
    """Command-line entrypoint wrapper for the preprocessing pipeline."""

    def __init__(self) -> None:
        self.pipeline = PreprocessPipeline()

    def run(self) -> str:
        """Execute the preprocessing pipeline and return the output path."""

        clean_path = self.pipeline.run_all()
        print(f"[OK] Datos limpios generados: {clean_path}")
        return clean_path


def main() -> None:
    """Module-level entrypoint kept for backward compatibility."""

    runner = PreprocessRunner()
    runner.run()


if __name__ == "__main__":
    main()
