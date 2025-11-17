# run_reports.py
# -----------------------------------------------------------------------------
# Script unificado para generar reportes HTML interactivos
# Permite elegir qué tipo de reporte generar:
#   - eda:       reportes de exploración de datos
#   - models:    reporte comparativo de modelos
#   - all:       genera todos los reportes (default)
# -----------------------------------------------------------------------------

import argparse
import os
from pathlib import Path

import mlflow

from mlops import dataset
from mlops.dataset import load_original_if_exists


class ReportsRunner:
    """High-level orchestrator for HTML report generation."""

    def __init__(self) -> None:
        self.dataset_module = dataset

    # ------------------------------------------------------------------
    # EDA reports
    # ------------------------------------------------------------------

    @staticmethod
    def build_eda_base(df_modified, df_original):
        """Genera reporte EDA base (original)."""

        from mlops.reports import EDAReport

        report = EDAReport(variant="base")
        return report.generate(df_modified, df_original)

    @staticmethod
    def build_eda_clean(df_modified, df_original):
        """Genera reporte EDA con datos limpios."""

        from mlops.reports import EDAReport

        report = EDAReport(variant="clean")
        return report.generate(df_modified, df_original)

    @staticmethod
    def build_eda_preprocessed():
        """Genera reporte EDA de datos preprocesados."""

        from mlops.reports import PreprocessedReport

        report = PreprocessedReport()
        return report.generate()

    def build_all_eda(self, df_modified, df_original):
        """Genera todos los reportes EDA y devuelve sus rutas."""

        outputs = []

        print("\n[1/3] Generando reporte EDA base...")
        out = self.build_eda_base(df_modified, df_original)
        print(f"[OK] Reporte BASE: {out}")
        outputs.append(out)

        print("\n[2/3] Generando reporte EDA clean...")
        out = self.build_eda_clean(df_modified, df_original)
        print(f"[OK] Reporte CLEAN: {out}")
        outputs.append(out)

        print("\n[3/3] Generando reporte EDA preprocessed...")
        out = self.build_eda_preprocessed()
        print(f"[OK] Reporte PREPROCESSED: {out}")
        outputs.append(out)

        return outputs

    # ------------------------------------------------------------------
    # Models reports
    # ------------------------------------------------------------------

    @staticmethod
    def build_models_report(
        experiment_name: str = "student_performance_complete_experiment",
        tracking_uri: str | None = None,
    ):
        """Genera reporte comparativo de modelos."""

        from mlops.reports import ModelsReport

        report = ModelsReport()
        return report.generate(experiment_name=experiment_name, tracking_uri=tracking_uri)

    def run(self, args: argparse.Namespace) -> None:
        """Execute the reports generation logic according to CLI args."""

        print("=" * 80)
        print("  Generador de Reportes HTML Interactivos")
        print("=" * 80)

        all_outputs: list[Path] = []

        # EDA reports
        if args.type in ("eda", "all"):
            print("\n" + "-" * 80)
            print("  Generando reportes EDA")
            print("-" * 80)

            try:
                df_modified = self.dataset_module.basic_typing(self.dataset_module.load_modified())
                df_original = load_original_if_exists()

                outputs = self.build_all_eda(df_modified, df_original)
                all_outputs.extend(outputs)

                print("\n[OK] Reportes EDA generados exitosamente")

            except Exception as exc:
                print(f"\n[ERROR] Error generando reportes EDA: {str(exc)}")
                print("[INFO] Asegúrate de tener los datos en data/raw/")

        # Models reports
        if args.type in ("models", "all"):
            print("\n" + "-" * 80)
            print("  Generando reporte de modelos")
            print("-" * 80)

            try:
                tracking_uri = args.mlflow_tracking_uri or os.environ.get(
                    "MLFLOW_TRACKING_URI", "file:./mlruns"
                )
                mlflow.set_tracking_uri(tracking_uri)

                if args.experiment is None:
                    all_experiments = mlflow.search_experiments()

                    if not all_experiments:
                        print("\n[WARNING] No se encontraron experimentos en MLflow.")
                    else:
                        print(
                            f"\n[INFO] Generando reportes separados para {len(all_experiments)} experimento(s):"
                        )
                        for experiment in all_experiments:
                            runs_count = len(mlflow.search_runs([experiment.experiment_id]))
                            print(f"  - {experiment.name} ({runs_count} runs)")

                        for experiment in all_experiments:
                            print(f"\n[INFO] Generando reporte para: {experiment.name}")
                            out = self.build_models_report(experiment.name, tracking_uri)
                            if out:
                                all_outputs.append(out)
                                print(f"[OK] Reporte generado: {out}")
                            else:
                                print(
                                    f"[WARNING] No se pudo generar reporte para {experiment.name}"
                                )
                else:
                    out = self.build_models_report(args.experiment, tracking_uri)
                    if out:
                        all_outputs.append(out)
                        print(f"\n[OK] Reporte de modelos: {out}")
                    else:
                        print("\n[WARNING] No se encontraron modelos. Ejecuta primero:")
                        print("   python train/train_multiple_models.py")

            except Exception as exc:
                print(f"\n[ERROR] Error generando reporte de modelos: {str(exc)}")
                import traceback

                traceback.print_exc()

        # Summary
        print("\n" + "=" * 80)
        print("  RESUMEN")
        print("=" * 80)

        if all_outputs:
            print(f"\n[OK] Se generaron {len(all_outputs)} reporte(s):")
            for output in all_outputs:
                print(f"   - {output}")

            print("\n[INFO] Abre los archivos HTML en tu navegador para visualizarlos")
            print("       O usa: python run_mlflow.py para ver experimentos en MLflow UI")
        else:
            print("\n[WARNING] No se generaron reportes. Revisa los errores arriba.")

        print("=" * 80)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Genera reportes HTML interactivos (EDA o Models).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python -m mlops.run_reports                    # Genera todos los reportes
  python -m mlops.run_reports --type eda         # Solo reportes EDA
  python -m mlops.run_reports --type models      # Solo reporte de modelos
  python -m mlops.run_reports --type models --mlflow-tracking-uri http://server:5001  # Con servidor remoto
  python -m mlops.run_reports --type all         # Todos (equivalente a sin argumentos)
        """,
    )

    parser.add_argument(
        "--type",
        choices=["eda", "models", "all"],
        default="all",
        help="Tipo de reporte a generar: eda, models, o all (default: all)",
    )

    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Nombre del experimento en MLflow para reporte de modelos (default: usa el primero disponible)",
    )

    parser.add_argument(
        "--mlflow-tracking-uri",
        type=str,
        default=None,
        help="URI del servidor MLflow (default: lee de MLFLOW_TRACKING_URI o usa local)",
    )

    return parser


def main() -> None:
    """Module-level entrypoint kept for backward compatibility."""

    parser = build_argument_parser()
    args = parser.parse_args()

    runner = ReportsRunner()
    runner.run(args)


if __name__ == "__main__":
    main()

