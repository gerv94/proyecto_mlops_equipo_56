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

def build_models_report(experiment_name="student_performance_complete_experiment"):
    """Genera reporte comparativo de modelos"""
    from mlops.report_html_models import build_html as build_html_models
    return build_html_models(experiment_name=experiment_name)


# -----------------------------------------------------------------------------
# Función principal
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Genera reportes HTML interactivos (EDA o Models).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python run_reports.py                  # Genera todos los reportes
  python run_reports.py --type eda       # Solo reportes EDA
  python run_reports.py --type models    # Solo reporte de modelos
  python run_reports.py --type all       # Todos (equivalente a sin argumentos)
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
        default="student_performance_complete_experiment",
        help="Nombre del experimento en MLflow para reporte de modelos (default: student_performance_complete_experiment)"
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
            out = build_models_report(args.experiment)
            
            if out:
                all_outputs.append(out)
                print(f"\n[OK] Reporte de modelos: {out}")
            else:
                print("\n[WARNING] No se encontraron modelos. Ejecuta primero:")
                print("   python train/train_multiple_models.py")
            
        except Exception as e:
            print(f"\n[ERROR] Error generando reporte de modelos: {str(e)}")
    
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

