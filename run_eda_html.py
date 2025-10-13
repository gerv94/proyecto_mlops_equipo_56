# run_eda_html.py
# -----------------------------------------------------------------------------
# Genera reportes HTML interactivos del EDA.
# Permite elegir:
#   - base:   reporte original (mlops/report_html.py)
#   - clean:  reporte que aplica limpieza/normalización (mlops/report_html_clean.py)
#   - pre:    reporte SOLO del dataset preprocesado (mlops/report_html_preprocessed.py)
#   - all:    genera los tres (por defecto)
# -----------------------------------------------------------------------------

import argparse
from mlops import dataset
from mlops.dataset import load_original_if_exists

def build_base(dfm, dfo):
    # Import tardío para no requerir el módulo si no se usa
    from mlops.report_html import build_html as build_html_base
    return build_html_base(dfm, dfo)  # genera reports/eda_html/eda_modified_plotly.html

def build_clean(dfm, dfo):
    from mlops.report_html_clean import build_html as build_html_clean
    return build_html_clean(dfm, dfo)  # genera reports/eda_html/eda_modified_clean_plotly.html (o similar)

def build_preprocessed():
    from mlops.report_html_preprocessed import build_html_preprocessed
    return build_html_preprocessed()   # genera reports/eda_html/eda_preprocessed_plotly.html

def main():
    parser = argparse.ArgumentParser(
        description="Genera reportes HTML del EDA (base / clean / pre / all)."
    )
    parser.add_argument(
        "--which",
        choices=["base", "clean", "pre", "all"],
        default="all",
        help="Selecciona el/los reportes a generar (default: all)."
    )
    args = parser.parse_args()

    # Carga común (para base/clean). El de 'pre' carga su propio CSV preprocesado.
    dfm = dataset.basic_typing(dataset.load_modified())
    dfo = load_original_if_exists()

    outputs = []

    if args.which in ("base", "all"):
        out = build_base(dfm, dfo)
        print(f"[OK] Reporte BASE: {out}")
        outputs.append(out)

    if args.which in ("clean", "all"):
        out = build_clean(dfm, dfo)
        print(f"[OK] Reporte CLEAN: {out}")
        outputs.append(out)

    if args.which in ("pre", "all"):
        out = build_preprocessed()
        print(f"[OK] Reporte PREPROCESSED: {out}")
        outputs.append(out)

    # Resumen
    if outputs:
        print("\n=== Reportes generados ===")
        for o in outputs:
            print(f"- {o}")

if __name__ == "__main__":
    main()
