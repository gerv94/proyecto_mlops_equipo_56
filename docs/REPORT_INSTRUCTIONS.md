# Instrucciones para Generar Reportes
## Proyecto MLOps – Student Performance

---

## 📋 Descripción General

Este proyecto realiza **Análisis Exploratorio de Datos (EDA)** y **Comparación de Modelos** sobre el dataset *Student Performance on an Entrance Examination* (UCI), siguiendo una estructura modular basada en MLOps.

### Objetivos

- Analizar y limpiar los datos (versión modificada y original)
- Generar visualizaciones y reportes automáticos
- Producir reportes interactivos en formato HTML con insights automáticos
- Comparar y evaluar múltiples modelos de Machine Learning

---

## 📁 Estructura del Proyecto

```
proyecto_mlops_equipo_56/
│
├── mlops/                        # Módulos principales
│   ├── config.py                # Rutas, parámetros y constantes globales
│   ├── dataset.py               # Carga y guarda datasets (raw/interim)
│   ├── features.py              # Clasifica columnas, limpia y prepara datos
│   ├── plots.py                 # Crea figuras estáticas (PNG) con Seaborn
│   ├── report_html.py           # Genera reporte EDA interactivo
│   ├── report_html_clean.py     # Genera reporte EDA de datos limpios
│   ├── report_html_preprocessed.py # Genera reporte EDA preprocesado
│   └── report_html_models.py    # Genera reporte comparativo de modelos
│
├── data/
│   ├── raw/                     # Datos originales (CSVs de UCI)
│   └── interim/                 # Datos intermedios (versionados con DVC)
│
├── reports/
│   ├── eda_html/                # Reportes EDA interactivos (HTML)
│   └── experiments_html/        # Reporte comparativo de modelos (HTML)
│
├── train/                       # Scripts de entrenamiento
├── docs/                        # Documentación técnica
│
├── run_reports.py               # Generador unificado de reportes HTML
├── run_mlflow.py                # Lanza MLflow UI
├── requirements.txt             # Dependencias del proyecto
└── dvc.yaml                     # Definición del pipeline
```

---

## 🔧 Preparación del Entorno

### Requisitos

- **Python:** 3.12.6 o superior
- **Sistema:** Windows 10+, Linux, o macOS

### Instalación de Dependencias

```bash
# Instalar todas las dependencias
pip install -r requirements.txt

# O instalarlas manualmente
pip install pandas numpy matplotlib seaborn plotly scikit-learn mlflow dvc
```

> **Nota:** El archivo `requirements.txt` contiene todas las versiones específicas validadas para reproducibilidad.

---

## 📊 Ubicación de los Datasets

Coloca los archivos en esta ruta:

```
data/raw/
├── student_entry_performance_original.csv
└── student_entry_performance_modified.csv
```

> **Nota:** Si solo tienes un archivo, el sistema usará el modificado como fuente principal.

---

## 🚀 Ejecución del Análisis

### Generación de Reportes HTML Interactivos

Genera reportes interactivos con Plotly:

```bash
# Todos los reportes (EDA + Models)
python run_reports.py

# Solo reportes EDA
python run_reports.py --type eda

# Solo reporte de modelos (local)
python run_reports.py --type models

# Reporte de modelos desde servidor MLflow remoto
python run_reports.py --type models \
  --mlflow-tracking-uri http://servidor:5000 \
  --experiment nombre_experimento

# Ver ayuda
python run_reports.py --help
```

**Reportes EDA generados:**
- `reports/eda_html/eda_modified_plotly.html` → Dataset original
- `reports/eda_html/eda_modified_plotly_clean.html` → Datos limpios
- `reports/eda_html/eda_preprocessed_plotly.html` → Datos preprocesados

**Reporte de Modelos:**
- `reports/experiments_html/models_comparison_report.html` → Comparación de modelos

### Opción 3: Pipeline Automatizado (DVC)

Ejecuta todo el pipeline automáticamente:

```bash
# Ejecutar pipeline completo
dvc repro

# Verificar estado del pipeline
dvc status

# Visualizar dependencias
dvc dag
```

---

## 📈 Contenido de los Reportes HTML

### Reportes EDA

Cada reporte EDA incluye:

- **Información General**: filas, columnas, nulos, duplicados, memoria
- **Distribución del Target**: análisis de la variable objetivo (Performance)
- **Mapa de Valores Faltantes**: visualización de nulos por columna
- **Matriz de Correlación**: relaciones entre variables numéricas
- **Análisis Numérico**: histogramas y boxplots
- **Análisis Categórico**: gráficos de barras y distribución
- **Cardinalidad**: conteo de valores únicos por variable
- **Insights Automáticos**: resúmenes textuales de hallazgos

### Reporte de Modelos

El reporte comparativo incluye:

- **Mejor Modelo Identificado**: con métricas destacadas
- **Ranking Automático**: ordenado por F1-score
- **Gráficos Comparativos**: barras comparando métricas principales
- **Radar Chart**: visualización multi-dimensional de los mejores modelos
- **Tabla de Métricas**: accuracy, precision, recall, F1, CV score
- **Insights Automáticos**: análisis de estabilidad y rendimiento

---

## 🎯 Flujo de Trabajo Recomendado

### Paso 1: Preparación de Datos

```bash
# Ejecutar preprocesamiento
python mlops/run_preprocess.py

# Verificar que se generaron los datos intermedios
ls data/interim/
```

### Paso 2: Entrenamiento de Modelos

```bash
# Entrenar modelos individuales
python train/train_model_sre.py

# O entrenar múltiples modelos comparativos
python train/train_multiple_models.py
```

### Paso 3: Generación de Reportes

```bash
# Generar todos los reportes
python run_reports.py

# Abrir reportes en navegador
# - reports/eda_html/*.html
# - reports/experiments_html/models_comparison_report.html
```

### Paso 4: Visualización en MLflow

```bash
# Levantar MLflow UI
python run_mlflow.py

# O manualmente
mlflow ui

# Abrir en navegador: http://127.0.0.1:5000
```

---

## ⚠️ Notas Importantes

### Dependencias

- **No modifiques** el código dentro de `mlops/` directamente
- Los únicos scripts que debes ejecutar son `mlops/run_preprocess.py` y `run_reports.py`
- Si falta alguna librería, instálala con: `pip install <nombre_libreria>`

### Datos y Versionado

- Los archivos en `data/raw/` deben estar presentes antes de ejecutar
- Los datos en `data/interim/` están versionados con DVC
- Los reportes HTML se regeneran cada vez que ejecutas los scripts

### Visualización

- Los reportes HTML se abren directamente en cualquier navegador
- Recomendado: Chrome, Edge, Firefox
- Los reportes son interactivos: puedes hacer zoom, hover, filtrar

---

## 🔍 Solución de Problemas

### Error: "No module named 'mlflow'"

```bash
pip install mlflow
```

### Error: "Datos no encontrados"

Verifica que existan los archivos en `data/raw/`:
```bash
ls data/raw/
```

### Error: "FileNotFoundError: preprocessed data not found"

Ejecuta primero el pipeline de preprocesamiento:
```bash
python mlops/run_preprocess.py
```

### Error: "No experiments found in MLflow"

**Para modelos locales:**
Entrena modelos primero:
```bash
python train/train_multiple_models.py
```

**Para servidor MLflow remoto:**
Si los experimentos están en un servidor remoto, especifica la URI:
```bash
python run_reports.py --type models \
  --mlflow-tracking-uri http://servidor:puerto \
  --experiment nombre_del_experimento
```

O configura la variable de entorno:
```bash
# Windows
set MLFLOW_TRACKING_URI=http://servidor:puerto

# Linux/macOS
export MLFLOW_TRACKING_URI=http://servidor:puerto

# Luego ejecuta
python run_reports.py --type models
```

---

## 📚 Documentación Adicional

### Documentos Técnicos

- **`docs/model_comparison_report.md`**: Análisis comparativo de modelos
- **`docs/architecture_diagram.md`**: Arquitectura del pipeline
- **`docs/informe_sre_fase2.md`**: Reporte SRE y reproducibilidad
- **`docs/DATA_VERSION_CONTROL.md`**: Control de versiones de datos

### README Principal

Consulta `README.md` en la raíz del proyecto para:
- Visión general del proyecto
- Tecnologías utilizadas
- Guía de instalación
- Ejemplos de uso

---

## 🤝 Soporte y Contribución

**Equipo:** MLOps Equipo 56  
**Repositorio:** https://github.com/gerv94/proyecto_mlops_equipo_56

### Convenciones de Commits

```bash
DS:  # Data Scientist - reportes y visualizaciones
SRE: # Site Reliability Engineer - reproducibilidad
MODEL: # ML Engineer - entrenamiento y tracking
DATA: # Data Engineer - preprocesamiento
PIPELINE: # Software Engineer - integración
```

---

**Última actualización:** 2025-01

