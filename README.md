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

La etapa de preparación utiliza **Pandas** y **Scikit-learn** para limpiar, transformar y normalizar los datos.  
Se eliminan valores nulos, duplicados y atípicos, garantizando la consistencia de las variables numéricas y categóricas.  

Los datasets intermedios se almacenan en `data/interim/`, mientras que los datos finales preprocesados se utilizan en el entrenamiento del modelo.

---

## 2. Exploración y análisis de datos (EDA)

El análisis exploratorio se realiza con **Matplotlib**, **Seaborn** y **Plotly**, generando visualizaciones descriptivas y distribuciones de variables.  
Este proceso puede ejecutarse de dos formas:

```bash
python run_eda.py          # EDA clásico con gráficos estáticos
python run_eda_html.py     # EDA interactivo (Plotly)
```

Los resultados se exportan a `reports/eda.html` y `reports/figures/`.

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

## 5. Reproducibilidad y control de versiones

El entorno reproducible se define mediante:

```bash
Python 3.12.6
pip install -r requirements.txt
```

El proyecto utiliza **DVC** para el versionado de datos y artefactos de entrenamiento:

```bash
dvc add data/interim/student_interim_preprocessed.csv
dvc push
dvc repro
```

Cada ejecución de `dvc repro` garantiza la regeneración exacta del pipeline y los resultados, asegurando reproducibilidad total.

**Entidades controladas:**
- **Datos:** mediante DVC (`data/interim/`)  
- **Modelos:** versión `.joblib` en `models/`  
- **Reportes:** métricas y reportes clasificados en `reports/`  
- **Código:** mediante Git y GitHub (`main` limpio y sincronizado)

---

## 6. Estructura actual del proyecto

```bash
proyecto_mlops_equipo_56/
│
├── data/
│   ├── raw/                # Datos originales
│   ├── interim/            # Datos limpios y preprocesados
│
├── docs/
│   └── informe_sre_fase2.md
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

## 7. Tecnologías y librerías clave

| Categoría | Herramientas |
|------------|--------------|
| Lenguaje base | Python 3.12.6 |
| Procesamiento de datos | Pandas, NumPy, Scikit-learn |
| Visualización | Matplotlib, Seaborn, Plotly |
| MLOps / Reproducibilidad | MLflow, DVC |
| Servidor / API | Flask, FastAPI, Waitress |
| Control de versiones | Git + GitHub |

---

## 8. Control de versiones y convenciones de commits

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

## 9. Documentación técnica (SRE)

El documento principal de la Fase 2 se encuentra en:

```
/docs/informe_sre_fase2.md
```

Incluye:
- Configuración del entorno reproducible.  
- Integración y validación del pipeline DVC.  
- Registro y trazabilidad de experimentos en MLflow.  
- Resultados y métricas de desempeño del modelo.  
- Conclusiones del Ingeniero de Confiabilidad (SRE).

---

## 10. Próximos pasos – Fase 3

- Integrar orquestación con **DVC pipelines** o **MLflow Projects**.  
- Implementar despliegue del modelo como API (FastAPI / Docker).  
- Automatizar evaluación continua (CI/CD).  
- Monitorear métricas en tiempo real (Prometheus / Grafana).
