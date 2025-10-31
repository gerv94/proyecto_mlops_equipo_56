# Proyecto MLOps – Exploración, Modelado y Reproducibilidad  
## Equipo 56 – *Student Performance on an Entrance Examination*

---

## Descripción general

Este proyecto forma parte del flujo de trabajo integral de **MLOps** desarrollado por el **Equipo 56**.  
El objetivo es analizar y modelar el dataset *Student Performance on an Entrance Examination*, garantizando la **reproducibilidad del entorno**, la **trazabilidad de los experimentos** y la **automatización del flujo de entrenamiento** mediante **MLflow**.

**Dataset original:**  
[UCI Machine Learning Repository – Student Performance on an Entrance Examination](https://archive.ics.uci.edu/dataset/582/student+performance+on+an+entrance+examination)

---

## 1. Manipulación y preparación de datos

En esta etapa se realiza la limpieza inicial del dataset utilizando **Pandas** y **Scikit-learn**.  
Se eliminan valores nulos, duplicados y atípicos, garantizando la coherencia de las variables.  
Los datos intermedios se almacenan en `data/interim/` para ser utilizados por el pipeline de modelado.

---

## 2. Exploración y análisis de datos (EDA)

Mediante herramientas como **Matplotlib**, **Seaborn** y **Plotly**, se genera un análisis descriptivo y visual de las variables.  
Este proceso se puede ejecutar en dos modalidades:

```bash
python run_eda.py          # EDA clásico con gráficos estáticos
python run_eda_html.py     # EDA interactivo (Plotly)
```

Los resultados se guardan en `reports/eda.html` y `reports/figures/`.

---

## 3. Modelado y evaluación

Los modelos de Machine Learning son entrenados y evaluados desde la carpeta `train/`.

Ejemplo de ejecución del pipeline principal:

```bash
python train/train_model_sre.py
```

Este script:
1. Carga los datos preprocesados desde `data/interim/`.  
2. Entrena un **RandomForestClassifier**.  
3. Calcula métricas (accuracy, precision, recall, f1-score).  
4. Guarda reportes y artefactos en `reports/` y `models/`.  
5. Registra todo en **MLflow** para su trazabilidad.

---

## 4. Seguimiento de experimentos (MLflow)

Para visualizar los resultados del entrenamiento:

```bash
mlflow ui --host 127.0.0.1 --port 5001 --workers 1
```

Desde la interfaz podrás revisar:
- Parámetros de cada *run*  
- Métricas de desempeño  
- Modelos registrados  
- Artefactos generados (reportes, figuras, logs)

**Ejemplo de resultados:**
| Métrica | Valor |
|----------|--------|
| Accuracy | 0.9926 |
| Precision (weighted) | 0.99 |
| Recall (weighted) | 0.99 |
| F1-score (weighted) | 0.99 |

---

## 5. Reproducibilidad y control de versiones

El entorno reproducible está definido mediante:

```bash
Python 3.12.6
pip install -r requirements.txt
```

Se validó la ejecución completa en un entorno limpio (`test_env`), confirmando la consistencia total de dependencias.  
El archivo `.gitignore` excluye entornos virtuales, artefactos y datos temporales.

---

## 6. Estructura actual del proyecto

```bash
PROYECTO_MLOPS_EQUIPO_56/
│
├── data/
│   ├── raw/                # Datos originales
│   ├── interim/            # Datos limpios/preprocesados
│
├── docs/
│   └── informe_sre_fase2.md
│
├── models/                 # Modelos serializados (.joblib)
├── reports/                # Reportes y métricas
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
├── .gitignore
└── README.md
```

---

## 7. Tecnologías utilizadas

| Categoría | Librerías |
|------------|------------|
| Core | Python 3.12.6, NumPy, Pandas, SciPy |
| ML | Scikit-learn, Joblib |
| Visualización | Matplotlib, Seaborn, Plotly |
| MLOps | MLflow, Alembic, FastAPI |
| Utilidades | Flask, Waitress, PyYAML |

---

## 8. Versionado y control de código

El proyecto se gestiona con GitHub.  
Convenciones de commits empleadas:
```
SRE: entorno reproducible y documentación
DATA: limpieza y preprocesamiento de datos
EDA: visualizaciones y análisis exploratorio
MODEL: entrenamiento y evaluación de modelos
```

---

## 9. Documentación técnica

El detalle técnico de la fase SRE se encuentra en:
```
/docs/informe_sre_fase2.md
```
Este documento describe:
- La configuración reproducible del entorno.  
- El proceso de validación del pipeline.  
- El registro y seguimiento de experimentos con MLflow.

---

## 10. Próximos pasos (Fase 3)

- Integrar orquestación del pipeline con **DVC o MLflow Projects**.  
- Desplegar el modelo como API (FastAPI / Docker).  
- Monitorear métricas en tiempo real.

---

## Autores
**Equipo 56 – MLOps**  
**Rol SRE:** Neri Felipe  
