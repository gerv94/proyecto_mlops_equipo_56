# run_reports.py
# -----------------------------------------------------------------------------
# Script unificado para generar reportes HTML interactivos
# Permite elegir qué tipo de reporte generar:
#   - eda:       reportes de exploración de datos
#   - models:    reporte comparativo de modelos
#   - all:       genera todos los reportes (default)
# -----------------------------------------------------------------------------

import argparse
from pathlib import Path
from mlops import dataset
from mlops.dataset import load_original_if_exists


# -----------------------------------------------------------------------------
# Builders de reportes EDA
# -----------------------------------------------------------------------------

def build_eda_base(dfm, dfo):
    """Genera reporte EDA base (original)"""
    from mlops.report_html import build_html as build_html_base
    return build_html_base(dfm, dfo)


def build_eda_clean(dfm, dfo):
    """Genera reporte EDA con datos limpios"""
    from mlops.report_html_clean import build_html as build_html_clean
    return build_html_clean(dfm, dfo)


def build_eda_preprocessed():
    """Genera reporte EDA de datos preprocesados"""
    from mlops.report_html_preprocessed import build_html_preprocessed
    return build_html_preprocessed()


def build_all_eda(dfm, dfo):
    """Genera todos los reportes EDA"""
    outputs = []
    
    print("\n[1/3] Generando reporte EDA base...")
    out = build_eda_base(dfm, dfo)
    print(f"[OK] Reporte BASE: {out}")
    outputs.append(out)
    
    print("\n[2/3] Generando reporte EDA clean...")
    out = build_eda_clean(dfm, dfo)
    print(f"[OK] Reporte CLEAN: {out}")
    outputs.append(out)
    
    print("\n[3/3] Generando reporte EDA preprocessed...")
    out = build_eda_preprocessed()
    print(f"[OK] Reporte PREPROCESSED: {out}")
    outputs.append(out)
    
    return outputs


# -----------------------------------------------------------------------------
# Builder de reporte de modelos
# -----------------------------------------------------------------------------

def build_models_report(experiment_name="student_performance_complete_experiment", tracking_uri=None):
    """Genera reporte comparativo de modelos"""
    from mlops.report_html_models import build_html as build_html_models
    return build_html_models(experiment_name=experiment_name, tracking_uri=tracking_uri)


# -----------------------------------------------------------------------------
# Función principal
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Genera reportes HTML interactivos (EDA o Models).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python run_reports.py                          # Genera todos los reportes
  python run_reports.py --type eda               # Solo reportes EDA
  python run_reports.py --type models            # Solo reporte de modelos
  python run_reports.py --type models --mlflow-tracking-uri http://server:5000  # Con servidor remoto
  python run_reports.py --type all               # Todos (equivalente a sin argumentos)
        """
    )
    
    parser.add_argument(
        "--type",
        choices=["eda", "models", "all"],
        default="all",
        help="Tipo de reporte a generar: eda, models, o all (default: all)"
    )
    
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Nombre del experimento en MLflow para reporte de modelos (default: usa el primero disponible)"
    )
    
    parser.add_argument(
        "--mlflow-tracking-uri",
        type=str,
        default=None,
        help="URI del servidor MLflow (default: lee de MLFLOW_TRACKING_URI o usa local)"
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("  Generador de Reportes HTML Interactivos")
    print("="*80)
    
    all_outputs = []
    
    # Generar reportes EDA
    if args.type in ("eda", "all"):
        print("\n" + "-"*80)
        print("  Generando reportes EDA")
        print("-"*80)
        
        try:
            dfm = dataset.basic_typing(dataset.load_modified())
            dfo = load_original_if_exists()
            
            outputs = build_all_eda(dfm, dfo)
            all_outputs.extend(outputs)
            
            print("\n[OK] Reportes EDA generados exitosamente")
            
        except Exception as e:
            print(f"\n[ERROR] Error generando reportes EDA: {str(e)}")
            print("[INFO] Asegúrate de tener los datos en data/raw/")
    
    # Generar reporte de modelos
    if args.type in ("models", "all"):
        print("\n" + "-"*80)
        print("  Generando reporte de modelos")
        print("-"*80)
        
        try:
            # Si no se especifica experimento, generar reportes separados para cada uno
            if args.experiment is None:
                import mlflow
                import os
                
                tracking_uri = args.mlflow_tracking_uri
                if tracking_uri is None:
                    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns")
                
                mlflow.set_tracking_uri(tracking_uri)
                all_experiments = mlflow.search_experiments()
                
                if not all_experiments:
                    print("\n[WARNING] No se encontraron experimentos en MLflow.")
                else:
                    print(f"\n[INFO] Generando reportes separados para {len(all_experiments)} experimento(s):")
                    for exp in all_experiments:
                        runs_count = len(mlflow.search_runs([exp.experiment_id]))
                        print(f"  - {exp.name} ({runs_count} runs)")
                    
                    # Generar un reporte separado para cada experimento
                    for exp in all_experiments:
                        print(f"\n[INFO] Generando reporte para: {exp.name}")
                        out = build_models_report(exp.name, args.mlflow_tracking_uri)
                        if out:
                            all_outputs.append(out)
                            print(f"[OK] Reporte generado: {out}")
                        else:
                            print(f"[WARNING] No se pudo generar reporte para {exp.name}")
            else:
                # Generar reporte para el experimento específico
                out = build_models_report(args.experiment, args.mlflow_tracking_uri)
                
                if out:
                    all_outputs.append(out)
                    print(f"\n[OK] Reporte de modelos: {out}")
                else:
                    print("\n[WARNING] No se encontraron modelos. Ejecuta primero:")
                    print("   python train/train_multiple_models.py")
            
        except Exception as e:
            print(f"\n[ERROR] Error generando reporte de modelos: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Resumen final
    print("\n" + "="*80)
    print("  RESUMEN")
    print("="*80)
    
    if all_outputs:
        print(f"\n[OK] Se generaron {len(all_outputs)} reporte(s):")
        for output in all_outputs:
            print(f"   - {output}")
        
        print("\n[INFO] Abre los archivos HTML en tu navegador para visualizarlos")
        print("       O usa: python run_mlflow.py para ver experimentos en MLflow UI")
    else:
        print("\n[WARNING] No se generaron reportes. Revisa los errores arriba.")
    
    print("="*80)


if __name__ == "__main__":
    main()

