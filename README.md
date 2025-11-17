# Proyecto MLOps – Exploración, Modelado y Reproducibilidad
## Equipo 56 – *Student Performance on an Entrance Examination*

---

## Descripción general

Este proyecto forma parte del flujo de trabajo integral de **MLOps** desarrollado por el **Equipo 56**, en el marco de la Fase 2 del curso.  
El objetivo es analizar y modelar el dataset *Student Performance on an Entrance Examination*, garantizando la **reproducibilidad del entorno**, la **trazabilidad de los experimentos** y la **automatización del pipeline de entrenamiento** mediante **MLflow** y **DVC**.

**Fuente de datos:**  
[UCI Machine Learning Repository – Student Performance on an Entrance Examination](https://archive.ics.uci.edu/dataset/582/student+performance+on+an+entrance+examination)

---

## 1. Manipulación y preparación de datos

La preparación de datos en este proyecto fue diseñada como un flujo **totalmente reproducible**, separando el análisis exploratorio del preprocesamiento utilizado por el modelo.  
Las transformaciones se implementaron en los módulos:

- `mlops/dataset.py`
- `mlops/features.py`
- `mlops/preprocess.py`

El flujo completo consta de **dos etapas principales**, controladas por DVC:

### **1.1 Limpieza y tipificación (`student_interim_clean.csv`)**
Incluye:
- Inferencia automática de variables numéricas y categóricas mediante heurística de cardinalidad.
- Estandarización de texto en columnas categóricas (lowercase, trimming, normalización de espacios).
- Conversión segura de tipos (coerción a numérico donde aplica).
- Imputación mínima:
  - Numéricas → mediana
  - Categóricas → moda

### **1.2 Preprocesamiento avanzado (`student_interim_preprocessed.csv`)**
Incluye:
- Escalado de características numéricas con **StandardScaler**.
- Codificación One-Hot robusta con `handle_unknown="ignore"`.
- Integración opcional de **PCA** (3 componentes) para enriquecer representaciones.
- Unificación de variables transformadas en un dataset final listo para entrenamiento.

Ambos artefactos generados se guardan en `data/interim/` y son consumidos por la etapa de entrenamiento.

### **1.3 Ejecución del preprocesamiento**
El pipeline ejecuta de manera determinística:

```bash
dvc repro
```

---

## 2. Exploración y análisis de datos (EDA)

El análisis exploratorio se realiza con **Plotly**, generando reportes HTML interactivos con visualizaciones descriptivas y distribuciones de variables.  

```bash
python -m mlops.run_reports      # Reportes HTML interactivos (Plotly)
```

Los resultados se exportan a `reports/eda_html/` y `reports/experiments_html/`.

### 2.1 Reportes HTML Interactivos

El script **`mlops/run_reports.py`** es un generador unificado de reportes que permite elegir qué tipo de reporte generar:

#### Opciones disponibles:

```bash
# Generar TODOS los reportes (default)
python -m mlops.run_reports

# Solo reportes EDA (exploración de datos)
python -m mlops.run_reports --type eda

# Solo reporte de modelos (comparación)
# Nota: Si no se especifica --experiment, genera reportes separados para TODOS los experimentos en MLflow
python -m mlops.run_reports --type models

# Todos los reportes (explícito)
python -m mlops.run_reports --type all

# Personalizar experimento MLflow específico
python -m mlops.run_reports --type models --experiment mi_experimento

# Especificar servidor MLflow remoto
python -m mlops.run_reports --type models --mlflow-tracking-uri http://server:5001

# Combinar opciones: servidor remoto + experimento específico
python -m mlops.run_reports --type models --experiment mi_experimento --mlflow-tracking-uri http://server:5001
```

#### Tipos de reportes generados:

**EDA Reports:**
- `base`: Dataset original vs modificado
- `clean`: Datos limpios y normalizados
- `preprocessed`: Datos preprocesados para modelado

**Models Report:**
- Comparación visual de modelos entrenados desde MLflow
- Gráficos interactivos (barras, radar)
- Ranking automático por rendimiento
- Genera un reporte separado por cada experimento cuando no se especifica `--experiment`
- Los archivos se guardan como: `reports/experiments_html/models_comparison_{experiment_name}.html`
- Soporta servidores MLflow remotos mediante `--mlflow-tracking-uri`
- Se actualiza automáticamente con cada nuevo entrenamiento

#### Ejemplo de flujo completo:

```bash
# 1. Entrenar modelos
python train/train_multiple_models.py

# 2. Generar reportes
python -m mlops.run_reports --type models

# 3. Ver en navegador
# Abre: reports/experiments_html/models_comparison_{experiment_name}.html
# (Se genera un HTML separado por cada experimento encontrado en MLflow)
```

---

## 3. Modelado y evaluación

El modelo principal se entrena desde la carpeta `train/` con el siguiente comando:

```bash
python train/train_model_sre.py
```

El script realiza las siguientes tareas:
1. Carga los datos procesados desde `data/interim/`.  
2. Entrena un **RandomForestClassifier** con parámetros reproducibles.  
3. Calcula métricas de desempeño (*accuracy*, *precision*, *recall*, *f1-score*).  
4. Guarda artefactos en `models/` y reportes en `reports/`.  
5. Registra automáticamente parámetros, métricas y artefactos en **MLflow**.

**Ejemplo de resultados:**

| Métrica | Valor |
|----------|--------|
| Accuracy | 0.9926 |
| Precision (weighted) | 0.99 |
| Recall (weighted) | 0.99 |
| F1-score (weighted) | 0.99 |

El entrenamiento consume directamente los artefactos generados en la etapa de preprocesamiento:
	•	data/interim/student_interim_clean.csv
	•	data/interim/student_interim_preprocessed.csv

Esto garantiza que el modelo siempre se entrene con los mismos datos transformados de forma determinística, manteniendo la reproducibilidad del pipeline.

---

## 4. Seguimiento de experimentos con MLflow

MLflow se emplea para registrar, comparar y reproducir ejecuciones del modelo:

```bash
mlflow ui --host 127.0.0.1 --port 5001
```

Desde la interfaz se pueden consultar:
- Parámetros y métricas de cada *run*  
- Modelos registrados  
- Artefactos y reportes generados  
- Gráficos de desempeño y evolución de experimentos

---

## 5. Pipeline de procesamiento y entrenamiento (DVC)

El flujo de datos y entrenamiento fue definido mediante **DVC (Data Version Control)**, garantizando la reproducibilidad completa del proyecto.  
Cada *stage* define sus dependencias (`-d`) y salidas (`-o`), de modo que cualquier cambio en los archivos fuente desencadena la ejecución del pipeline completo o parcial.

### Estructura del pipeline

```bash
dvc.yaml
├── stages:
│   ├── preprocessing:
│   │   deps:
│   │     - mlops/dataset.py
│   │     - mlops/features.py
│   │     - mlops/preprocess.py
│   │     - mlops/run_preprocess.py
│   │   outs:
│   │     - data/interim/student_interim_clean.csv
│   │     - data/interim/student_interim_preprocessed.csv
│   │   cmd: python mlops/run_preprocess.py
│   │
│   └── training:
│       deps:
│         - data/interim/student_interim_preprocessed.csv
│         - data/interim/student_interim_clean.csv
│       outs:
│         - models/model_latest.joblib
│         - reports/classification_report_latest.txt
│       cmd: python train/train_model_sre.py

```

El pipeline completo fue estandarizado mediante DVC, garantizando que los datos limpios, los datos procesados y el modelo final puedan regenerarse desde cero con un solo comando.

### Ejecución del pipeline

```bash
# Ejecuta todas las etapas necesarias
dvc repro

# Verifica el estado del pipeline
dvc status

# Visualiza el flujo de dependencias
dvc dag
```

### Resultados esperados

- **Etapa preprocessing:** genera los datos limpios y preprocesados en `data/interim/`.  
- **Etapa training:** entrena el modelo, genera métricas y guarda artefactos en `models/` y `reports/`.  
- **DVC + Git:** registran automáticamente los cambios en código, datos y salidas.

---

## 6. Reproducibilidad y control de versiones

El entorno reproducible se define mediante:

```bash
Python 3.12.6
pip install -r requirements.txt
```

### Ejecuta el pipeline completo (preprocessing + training)
dvc repro

### O si se quiere entrenar forzado manualmente
python train/train_model_sre.py

### Validación rápida del modelo estable
python predict_joblib.py

Cada ejecución de `dvc repro` garantiza la regeneración exacta del pipeline y los resultados, asegurando reproducibilidad total.

**Entidades controladas:**
- **Datos:** mediante DVC (`data/interim/`)  
- **Modelos:** versión `.joblib` en `models/`  
- **Reportes:** métricas y reportes clasificados en `reports/`  
- **Código:** mediante Git y GitHub (`main` limpio y sincronizado)

---

## 7. Estructura actual del proyecto

```bash
proyecto_mlops_equipo_56/
│
├── data/
│   ├── raw/                # Datos originales
│   ├── interim/            # Datos limpios y preprocesados
│
├── docs/
│   ├── informe_sre_fase2.md
│   ├── model_comparison_report.md
│   ├── architecture_diagram.md
│   └── DATA_VERSION_CONTROL.md
│
├── models/                 # Modelos serializados (.joblib)
├── reports/                # Reportes, métricas y figuras
│   ├── eda.html
│   ├── figures/
│   └── classification_report_rf_train_*.txt
│
├── train/
│   ├── train_model_sre.py
│   ├── train_model.py
│   └── train_multiple_models.py
│
├── requirements.txt
├── dvc.yaml
├── .dvc/
├── .gitignore
└── README.md
```

---

## 8. Columnas numéricas, categóricas y objetivo

A continuación se presentan las columnas detectadas automáticamente por el módulo de Data Engineering, clasificadas con base en tipo y cardinalidad:

| Columna                 | Tipo        | Descripción |
|------------------------|-------------|-------------|
| **Performance**        | Target      | Variable objetivo del modelo. |
| Mother_occupation      | Categórica  | Ocupación de la madre. |
| Father_occupation      | Categórica  | Ocupación del padre. |
| Gender                 | Categórica  | Género del estudiante. |
| Caste                  | Categórica  | Grupo social asociado al estudiante. |
| medium                 | Categórica  | Medio o idioma de instrucción. |
| coaching               | Categórica  | Participación en programas de coaching. |
| Class_ten_education    | Categórica  | Nivel educativo previo (10°). |
| Class_XII_Percentage   | Categórica* | Porcentaje de calificaciones en 12°. |
| Class_ X_Percentage    | Categórica* | Porcentaje de calificaciones en 10°. |
| twelve_education       | Categórica  | Educación previa en 12°. |
| time                   | Categórica  | Tiempo dedicado al estudio. |
| mixed_type_col         | Categórica  | Columna detectada como híbrida; convertida a categórica. |

Estas columnas se clasificaron como categóricas debido a su cardinalidad y formato textual, aun cuando representan porcentajes.

## 9. Tecnologías y librerías clave

| Categoría | Herramientas |
|------------|--------------|
| Lenguaje base | Python 3.12.6 |
| Procesamiento de datos | Pandas, NumPy, Scikit-learn |
| Visualización | Matplotlib, Seaborn, Plotly |
| MLOps / Reproducibilidad | MLflow, DVC |
| Servidor / API | Flask, FastAPI, Waitress |
| Control de versiones | Git + GitHub |
| Testing | pytest, pytest-cov |

---

## 10. Pruebas unitarias y de integración

El proyecto incluye pruebas automatizadas para validar componentes críticos y asegurar la estabilidad del sistema.

### 10.1 Ejecución de pruebas

**⚠️ Importante:** Asegúrate de usar el pytest del entorno virtual `.venv` para garantizar que uses las dependencias correctas del proyecto.

#### Opción 1: Usar el script helper (Recomendado)
```bash
# Ejecutar todas las pruebas usando el .venv automáticamente
python run_tests.py -q

# Ejecutar con salida detallada
python run_tests.py -v

# Ejecutar pruebas específicas
python run_tests.py tests/test_reports.py
```

#### Opción 2: Activar el entorno virtual primero
```bash
# En PowerShell
.venv\Scripts\Activate.ps1

# Ahora pytest usará el del .venv
pytest -q
pytest -v
pytest tests/test_reports.py
```

#### Opción 3: Usar el Python del .venv directamente
```bash
# Ejecutar todas las pruebas
.venv\Scripts\python.exe -m pytest -q

# Con salida detallada
.venv\Scripts\python.exe -m pytest -v

# Ejecutar pruebas específicas
.venv\Scripts\python.exe -m pytest tests/test_reports.py

# Ejecutar pruebas con cobertura
.venv\Scripts\python.exe -m pytest --cov=mlops --cov-report=html

# Ejecutar solo pruebas unitarias
.venv\Scripts\python.exe -m pytest -m unit

# Ejecutar solo pruebas de integración
.venv\Scripts\python.exe -m pytest -m integration
```

#### Opción 4: Usar Makefile (Linux/Mac)
```bash
make test
```

### 10.2 Estructura de pruebas

```
tests/
├── __init__.py
├── test_reports.py        # Pruebas para el módulo de reportes
└── (futuros: test_features.py, test_preprocess.py, etc.)
```

### 10.3 Cobertura de pruebas

Actualmente el proyecto incluye pruebas para:
- **Módulo de reportes** (`mlops/reports.py`):
  - Métodos estáticos de `ReportBase` (`guess_target`, `compute_summary_metrics`)
  - Inicialización y configuración de `EDAReport`, `PreprocessedReport`, `ModelsReport`
  - Función factory `create_report`
  - Flujos de generación de reportes HTML

### 10.4 Agregar nuevas pruebas

Para agregar nuevas pruebas:
1. Crea un archivo `tests/test_<modulo>.py`
2. Importa el módulo a probar
3. Define clases de prueba que hereden de `unittest.TestCase` o usen funciones con `pytest`
4. Ejecuta con `pytest` para validar

Ejemplo de prueba:
```python
import pytest
from mlops.reports import ReportBase

def test_guess_target():
    df = pd.DataFrame({'Performance': ['A', 'B', 'C']})
    result = ReportBase.guess_target(df)
    assert result == 'Performance'
```

---

## 11. Control de versiones y convenciones de commits

El versionado de código se gestiona en GitHub bajo el repositorio:

```
https://github.com/gerv94/proyecto_mlops_equipo_56
```

**Convenciones de commits:**

```
SRE: entorno reproducible y documentación
DATA: limpieza y preprocesamiento de datos
EDA: análisis exploratorio y visualizaciones
MODEL: entrenamiento y evaluación de modelos
PIPELINE: integración DVC y MLflow
```

---

## 12. Documentación técnica

El proyecto cuenta con documentación técnica comprehensiva en `/docs/`:

### 12.1 Informe SRE
```
/docs/informe_sre_fase2.md
```
Incluye:
- Configuración del entorno reproducible.  
- Integración y validación del pipeline DVC.  
- Registro y trazabilidad de experimentos en MLflow.  
- Resultados y métricas de desempeño del modelo.  
- Conclusiones del Ingeniero de Confiabilidad (SRE).

### 12.2 Reporte de Comparación de Modelos
```
/docs/model_comparison_report.md
```
Incluye:
- Evaluación de 8 algoritmos distintos (Logistic Regression, Random Forest, XGBoost, SVM, KNN, Decision Tree, Naive Bayes, Gradient Boosting).
- Análisis de trade-offs (complejidad, precisión, interpretabilidad).
- Justificación de la selección del mejor modelo.
- Conclusiones y lecciones aprendidas del Data Scientist.

### 12.3 Diagrama de Arquitectura
```
/docs/architecture_diagram.md
```
Incluye:
- Visualización del pipeline completo (data flow).
- Flujo de trabajo por fases.
- Herramientas y versionado multi-capa.
- Responsabilidades por rol del equipo.
- Decisiones de arquitectura y mejoras futuras.

### 12.4 Control de Versión de Datos
```
/docs/DATA_VERSION_CONTROL.md
```
Incluye:
- Trazabilidad de transformaciones de datos.
- Versionado de datasets con DVC.
- Ejecuciones y resultados del pipeline.
