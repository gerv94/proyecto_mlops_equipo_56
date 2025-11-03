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

La etapa de preparación de datos se realiza ahora como preprocesamiento productivo real separado del EDA.
Este proceso se encuentra implementado en **mlops/preprocess.py** y es ejecutado automáticamente por el pipeline de DVC mediante **mlops/run_preprocess.py**.

Aquí se tipifican columnas, se normalizan categóricas, se imputan valores faltantes y se aplica preprocesamiento avanzado (escalado, One-Hot Encoding y PCA).
Este flujo es el que alimenta directamente la fase de entrenamiento y asegura reproducibilidad en cualquier entorno.

Los datasets intermedios generados por este preprocesamiento se guardan automáticamente en data/interim/ y son consumidos después por el proceso de entrenamiento del modelo.

---

## 2. Exploración y análisis de datos (EDA)

El análisis exploratorio se realiza con **Plotly**, generando reportes HTML interactivos con visualizaciones descriptivas y distribuciones de variables.  

```bash
python run_reports.py      # Reportes HTML interactivos (Plotly)
```

Los resultados se exportan a `reports/eda_html/` y `reports/experiments_html/`.

### 2.1 Reportes HTML Interactivos

El script **`run_reports.py`** es un generador unificado de reportes que permite elegir qué tipo de reporte generar:

#### Opciones disponibles:

```bash
# Generar TODOS los reportes (default)
python run_reports.py

# Solo reportes EDA (exploración de datos)
python run_reports.py --type eda

# Solo reporte de modelos (comparación)
python run_reports.py --type models

# Todos los reportes (explícito)
python run_reports.py --type all

# Personalizar experimento MLflow para reporte de modelos
python run_reports.py --type models --experiment mi_experimento
```

#### Tipos de reportes generados:

**EDA Reports:**
- `base`: Dataset original vs modificado
- `clean`: Datos limpios y normalizados
- `preprocessed`: Datos preprocesados para modelado

**Models Report:**
- Comparación visual de modelos entrenados
- Gráficos interactivos (barras, radar)
- Ranking automático por rendimiento
- Se actualiza automáticamente con cada nuevo entrenamiento
- Visible en: `reports/experiments_html/models_comparison_report.html`

#### Ejemplo de flujo completo:

```bash
# 1. Entrenar modelos
python train/train_multiple_models.py

# 2. Generar reportes
python run_reports.py --type models

# 3. Ver en navegador
# Abre: reports/experiments_html/models_comparison_report.html
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
│   │     - data/raw/
│   │   outs:
│   │     - data/interim/student_interim_clean.csv
│   │     - data/interim/student_interim_preprocessed.csv
│   │   cmd: python -m mlops.run_preprocess
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

## 8. Tecnologías y librerías clave

| Categoría | Herramientas |
|------------|--------------|
| Lenguaje base | Python 3.12.6 |
| Procesamiento de datos | Pandas, NumPy, Scikit-learn |
| Visualización | Matplotlib, Seaborn, Plotly |
| MLOps / Reproducibilidad | MLflow, DVC |
| Servidor / API | Flask, FastAPI, Waitress |
| Control de versiones | Git + GitHub |

---

## 9. Control de versiones y convenciones de commits

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

## 10. Documentación técnica

El proyecto cuenta con documentación técnica comprehensiva en `/docs/`:

### 10.1 Informe SRE
```
/docs/informe_sre_fase2.md
```
Incluye:
- Configuración del entorno reproducible.  
- Integración y validación del pipeline DVC.  
- Registro y trazabilidad de experimentos en MLflow.  
- Resultados y métricas de desempeño del modelo.  
- Conclusiones del Ingeniero de Confiabilidad (SRE).

### 10.2 Reporte de Comparación de Modelos
```
/docs/model_comparison_report.md
```
Incluye:
- Evaluación de 8 algoritmos distintos (Logistic Regression, Random Forest, XGBoost, SVM, KNN, Decision Tree, Naive Bayes, Gradient Boosting).
- Análisis de trade-offs (complejidad, precisión, interpretabilidad).
- Justificación de la selección del mejor modelo.
- Conclusiones y lecciones aprendidas del Data Scientist.

### 10.3 Diagrama de Arquitectura
```
/docs/architecture_diagram.md
```
Incluye:
- Visualización del pipeline completo (data flow).
- Flujo de trabajo por fases.
- Herramientas y versionado multi-capa.
- Responsabilidades por rol del equipo.
- Decisiones de arquitectura y mejoras futuras.

### 10.4 Control de Versión de Datos
```
/docs/DATA_VERSION_CONTROL.md
```
Incluye:
- Trazabilidad de transformaciones de datos.
- Versionado de datasets con DVC.
- Ejecuciones y resultados del pipeline.

---

## 11. Próximos pasos – Fase 3

- Integrar orquestación con **DVC pipelines** o **MLflow Projects**.  
- Implementar despliegue del modelo como API (FastAPI / Docker).  
- Automatizar evaluación continua (CI/CD).  
- Monitorear métricas en tiempo real (Prometheus / Grafana).
