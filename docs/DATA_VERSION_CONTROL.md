# DATA_VERSION_CONTROL.md

> Control de versiones de datos y trazabilidad del EDA/preprocesamiento  
> Proyecto: **MLOps – Student Performance on an Entrance Examination**  
> Autor del registro: **Luis Felipe Neri Alvarado Fregoso**  
> Fecha de ejecución: **12/10/2025**  
> Branch de trabajo: **`neri_sre`**

---

## 0) Resumen ejecutivo

- Se ejecutó el pipeline de EDA y preprocesamiento sobre el dataset *Student Entrance Performance*.  
- Se generaron dos artefactos de datos en `data/interim/`:
  - `student_interim_clean.csv` (datos limpios e imputados).
  - `student_interim_preprocessed.csv` (datos listos para modelado: escala, OHE y PCA).
- PCA (3 componentes) reportó varianza explicada aproximada: **[0.132, 0.079, 0.074]**.  
- Se emitieron reportes HTML interactivos en `reports/eda_html/`.  
- Se documentan rutas, scripts y comandos para reproducibilidad.

---

## 1) Inventario de datos (estado actual del repo local)

```
data/
├── raw/
│   ├── student_entry_performance_original.csv      # dataset original (UCI)
│   └── student_entry_performance_modified.csv      # versión modificada para ejercicio EDA/limpieza
└── interim/
    ├── student_interim_clean.csv                   # limpio + imputación simple
    └── student_interim_preprocessed.csv            # escalado + OHE + PCA
```

**Notas de contexto**  
- `*_modified.csv`: provisto para ejercitar limpieza y comparar distribución respecto al original.  
- `*_original.csv`: referencia del dataset bruto de UCI (en `data/raw/`).  

---

## 2) Entorno y dependencias

### 2.1 Entorno
- **SO:** Windows 11 x64  
- **Python:** `3.14.0` (venv activado en `.venv/`)  
- **Intérprete VS Code:** `~\proyecto_mlops_equipo_56\.venv\Scripts\python.exe`

### 2.2 Paquetes utilizados (relevantes)
```
pandas 2.3.3
numpy 2.3.3
matplotlib 3.10.7
seaborn 0.13.2
plotly 6.3.1
scikit-learn 1.7.2
scipy 1.16.2
```

> Para instalar: `pip install -r requirements.txt` (o usar la lista anterior en el venv).

---

## 3) Scripts y funciones que transforman datos

### 3.1 Orquestación de Preprocesamiento
- **`mlops/run_preprocess.py`**
  - Carga modificado → tipificación básica (`dataset.basic_typing`)
  - Normaliza categóricas (`features.clean_categoricals`) - solo limpieza de texto (sin data leakage)
  - Preprocesamiento para entrenamiento (`features.preprocess_for_training`) → guarda `interim_preprocessed`
  - NO aplica imputación (debe hacerse en el pipeline después de train/test split)
  - NO aplica PCA ni StandardScaler (alineado con notebook)

### 3.2 Construcción de reportes HTML
- **`run_reports.py`** (reemplazo de run_eda_html.py) con opciones:
  - `--which base` → usa `mlops/report_html.py`
  - `--which clean` → usa `mlops/report_html_clean.py` (aplica limpieza/normalización antes del EDA)
  - `--which pre` → usa `mlops/report_html_preprocessed.py` (solo CSV preprocesado)
  - `--which all` (por defecto) → genera los tres

### 3.3 Utilidades de features (resumen)
- `split_num_cat(df)`: separa numéricas/categóricas (tipo + cardinalidad ≤ 30).  
- `clean_categoricals(df, cat_cols)`: `strip`, normaliza espacios, minúsculas.  
- `minimal_preprocess(df)`: imputación simple  
  - Numéricas → mediana  
  - Categóricas → moda  
- `preprocess_advanced(df, num_cols, cat_cols, n_components=3)`:
  - **StandardScaler** en numéricas  
  - **OneHotEncoder** (handle_unknown='ignore') en categóricas  
  - **PCA** (k=3) añade columnas `PC1..PC3` (varianza explicada mostrada en consola)

### 3.4 Utilidades de dataset
- `load_modified()` / `load_original_if_exists()`  
- `basic_typing(df)` → coerción numérica segura  
- `save_interim(df, name)` / `save_processed(df, name)`

---

## 4) Artefactos generados en esta corrida (11/10/2025)

### 4.1 Datos intermedios
- ✅ `data/interim/student_interim_clean.csv`  
  - Resultado de `minimal_preprocess` (imputación) tras normalización de categóricas.
- ✅ `data/interim/student_interim_preprocessed.csv`  
  - Escalado + OHE + **PCA(3)**.  
  - **Varianza explicada** (aprox.): `[0.132, 0.079, 0.074]`.

### 4.2 Reportes
- ✅ `reports/eda_html/eda_modified_plotly.html` (base)  
- ✅ `reports/eda_html/eda_modified_plotly_clean.html` (clean)  
- ✅ `reports/eda_html/eda_preprocessed_plotly.html` (preprocesado)

> Las figuras PNG del EDA tradicional se guardan en `reports/figures/`.

---

## 5) Trazabilidad (comandos para reproducir)

> Ejecutar desde la raíz del repo y con el **venv activado**.

```bash
# 1) Activar entorno (PowerShell)
.\.venv\Scripts\Activate

# 2) Instalar dependencias
pip install -r requirements.txt

# 3) Preprocesamiento (CSV limpio + CSV preprocesado)
python mlops/run_preprocess.py

# 4) Reportes HTML
#   a) Todos los reportes (EDA + Models)
python run_reports.py
#   b) Solo reportes EDA
python run_reports.py --type eda
#   c) Solo reporte de modelos
python run_reports.py --type models
```

---

## 6) Control de versiones (Git) — buenas prácticas aplicadas

- **Ramas**: trabajo en `neri_sre`; sincronización previa con `main` (`git fetch`, `git merge origin/main`).  
- **Commits**: mensajes descriptivos (ej.: *“SRE: Diagnóstico entorno y ejecución exitosa del EDA”*).  
- **`.gitignore`**: incluye `.venv/`, `data/processed/`, `reports/eda_html/*.html`, `reports/figures/*.png`, `mlruns/`, etc., para evitar versionar artefactos pesados o reproducibles.

> Para subir cambios de datos/artefactos que **sí** deben quedar, evaluar si conviene versionarlos directamente o registrar solo el **log** y el **script** que los genera (lo recomendado).  

---

## 7) Métricas y observaciones del EDA

- **Target**: `Performance` (detección heurística).  
- **Nulos**: mapa de nulos y % por columna generados en los reportes.  
- **Categóricas**: normalizadas (minúsculas, espacios).  
- **Numéricas**: imputación por mediana; histogramas/boxplots generados.  
- **Correlación**: matriz en reportes (numéricas); en preprocesado se excluyen columnas PCA en la vista principal para legibilidad.  
- **Cardinalidad**: se incluye gráfico y resumen en HTML.

---

## 8) Checklist de reproducibilidad

- [x] Estructura de carpetas creada automáticamente (`mlops/config.py`).  
- [x] Scripts orquestadores (`mlops/run_preprocess.py`, `run_reports.py`) listos para ejecutar.  
- [x] Entorno aislado (`.venv`) + `requirements.txt`.  
- [x] Artefactos y reportes generados con rutas deterministas.  
- [x] Este documento de control de versiones de datos.

---

## 9) Próximos pasos sugeridos (SRE / MLOps)

1. **Etiquetado de corridas**: anotar hash de commit + timestamp en nombre de artefactos o en un `RUN_LOG.md`.  
2. **MLflow/DAG** (opcional): registrar pipeline y parámetros (ya existe carpeta `mlruns/` por los experimentos de modelado).  
3. **Validación de datos**: agregar checks (esquema/tipos/rangos) previos a guardar `interim`.  
4. **Contenerización**: Dockerfile + `make preprocess` para automatizar entorno/ejecución.  
5. **(Opcional)** DVC o similar para versionado de datasets si el tamaño/flujo lo amerita.

---

## 10) Apéndice — comandos Git útiles

```bash
# Traer cambios del main y fusionar en tu rama
git fetch origin
git checkout neri_sre
git merge origin/main

# Añadir y subir cambios
git add <rutas>
git commit -m "feat(eda): descripción breve"
git push origin neri_sre
```

---

**Fin del registro (12/10/2025).**