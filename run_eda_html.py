# -----------------------------------------------------------------------------
# Este script ejecuta el flujo de análisis exploratorio (EDA) para generar
# un reporte interactivo en HTML utilizando Plotly, con base en los módulos:
# - dataset.py       → carga y tipificación de datos
# - report_html.py   → construcción del reporte HTML con métricas e insights
# -----------------------------------------------------------------------------

from mlops import dataset
from mlops.report_html import build_html
from mlops.dataset import load_original_if_exists


def main():
    """
    Función principal que orquesta la generación del reporte HTML EDA.

    Flujo:
    1. Carga el dataset "modificado" desde data/raw/ y aplica tipificación básica.
    2. Carga (opcionalmente) el dataset original si existe para comparación.
    3. Genera el reporte interactivo (EDA) con build_html() del módulo report_html.py.
    4. Imprime la ruta final del reporte generado.

    Este flujo no guarda CSVs intermedios ni genera figuras PNG — solo
    produce un archivo HTML en la carpeta /reports/eda_html/.
    """
    # -------------------------------------------------------------------------
    # 1️⃣ CARGA DE DATOS
    # -------------------------------------------------------------------------
    # dataset.basic_typing() convierte columnas a tipo numérico o string
    dfm = dataset.basic_typing(dataset.load_modified())   # Dataset modificado
    dfo = load_original_if_exists()                       # Dataset original (opcional)

    # -------------------------------------------------------------------------
    # 2️⃣ GENERACIÓN DEL REPORTE INTERACTIVO
    # -------------------------------------------------------------------------
    out = build_html(dfm, dfo)

    # -------------------------------------------------------------------------
    # 3️⃣ CONFIRMACIÓN
    # -------------------------------------------------------------------------
    print(f"[OK] Reporte HTML generado en: {out}")


# -----------------------------------------------------------------------------
# EJECUCIÓN DIRECTA
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
