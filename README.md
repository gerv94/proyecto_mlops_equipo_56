# Proyecto MLOps – Exploración y Modelado de Datos
# Equipo 56 - Student Performance on an Entrance Examination

## Descripción general
Este proyecto forma parte del proceso de MLOps aplicado al análisis exploratorio y modelado de datos.  
El objetivo es realizar una exploración profunda del dataset **Student Performance on an Entrance Examination**, aplicar limpieza, preprocesamiento, versionado y posteriormente construir modelos de aprendizaje automático que permitan identificar patrones y relaciones significativas.

Dataset original:  
https://archive.ics.uci.edu/dataset/582/student+performance+on+an+entrance+examination

## 1. Manipulación y preparación de datos
En esta etapa se lleva a cabo la limpieza y organización inicial del dataset utilizando bibliotecas como Pandas, Scikit-learn y DVC. Se eliminan valores nulos, duplicados y atípicos, asegurando la coherencia de las variables. Además, se documentan las métricas empleadas y se gestionan versiones de los datos para mantener un registro reproducible. Esta fase es esencial para garantizar la calidad de la información que se empleará en las siguientes etapas del proceso analítico.

## 2. Exploración y preprocesamiento de datos
Aquí se analizan los datos a fondo para comprender sus relaciones, distribuciones y comportamientos generales. Se aplican técnicas de visualización y estadística descriptiva, junto con procesos de preprocesamiento como la normalización, la codificación de variables categóricas y la reducción de dimensionalidad mediante PCA. Este análisis permite detectar patrones significativos y preparar un conjunto de datos optimizado para el modelado posterior.

## 3. Versionado de datos
En esta fase se implementa un sistema de control de versiones con DVC para registrar los cambios realizados en los datasets a lo largo del proyecto. Esto permite rastrear cada modificación, mantener la trazabilidad y asegurar la reproducibilidad de los experimentos. El versionado de datos resulta fundamental para trabajar de forma colaborativa y garantizar la consistencia entre las distintas etapas del flujo de trabajo.

## 4. Construcción, ajuste y evaluación de modelos de Machine Learning
Finalmente, se desarrollan y ajustan modelos de aprendizaje automático basados en los datos preprocesados. Se prueban distintos algoritmos —como árboles de decisión, regresión logística o Random Forest— optimizando sus hiperparámetros y evaluando su desempeño mediante métricas como accuracy, precisión, recall y F1-score. Esta fase busca identificar el modelo más eficiente y estable, capaz de capturar de forma confiable los patrones del conjunto de datos.

## Tecnologías utilizadas
- Python 3.10+
- Pandas
- NumPy
- Matplotlib / Seaborn / Plotly
- Scikit-learn
- DVC (Data Version Control)

## Estructura del proyecto
mlops/
├── config.py # Parámetros globales y rutas
├── dataset.py # Carga y guardado de datasets
├── features.py # Limpieza, normalización y preprocesamiento
├── plots.py # Visualizaciones (EDA)
├── report_html.py # Reporte EDA interactivo (Plotly + HTML)
data/
├── raw/ # Datasets originales
├── interim/ # Datasets limpios o intermedios
└── processed/ # Datasets finales para modelado
run_eda.py # Ejecución del EDA tradicional
run_eda_html.py # Ejecución del EDA interactivo


