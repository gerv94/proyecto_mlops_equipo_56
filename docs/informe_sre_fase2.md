# Informe Técnico – Rol SRE  
## Fase 2: Implementación de un entorno reproducible con MLflow  
**Proyecto:** Student Performance  
**Equipo:** MLOps – Grupo 56  
**Rol:** Site Reliability Engineer (SRE)  
**Autor:** Luis Felipe Neri Alvarado Fregoso

---

## 1. Objetivo del rol SRE en la Fase 2

El objetivo principal de esta fase fue garantizar la **reproducibilidad, trazabilidad y estabilidad** del ciclo de entrenamiento del modelo de Machine Learning, asegurando que los resultados puedan ser replicados en cualquier entorno, bajo las mismas condiciones y dependencias.

El rol SRE se encargó de:
- Estandarizar la estructura del repositorio para separar código, datos y reportes.  
- Definir y controlar el entorno virtual (`.venv`).  
- Implementar y registrar experimentos mediante **MLflow**.  
- Generar evidencia de reproducibilidad a través de reportes y artefactos controlados por Git.  

---

## 2. Configuración del entorno reproducible

Se utilizó un entorno virtual basado en **Python 3.12.6**, dentro del cual se instalaron todas las dependencias exactas utilizadas para el entrenamiento y seguimiento del modelo.

### 2.1 Requisitos técnicos
```bash
Python 3.12.6
pip install -r requirements.txt
```

### 2.2 Principales librerías instaladas
| Librería | Versión | Propósito |
|-----------|----------|-----------|
| numpy | 2.3.4 | Operaciones numéricas vectorizadas |
| pandas | 2.3.3 | Manipulación y limpieza de datos |
| scikit-learn | 1.7.2 | Entrenamiento y evaluación de modelos |
| matplotlib / seaborn / plotly | 3.10.7 / 0.13.2 / 6.3.1 | Visualización |
| mlflow | 3.5.1 | Registro de experimentos y artefactos |
| pyarrow | 21.0.0 | Lectura y escritura de datos en formato parquet |
| Flask + waitress | 3.1.2 / 3.0.2 | Despliegue del MLflow UI local |

---

## 3. Estructura del proyecto

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
├── reports/                # Reportes generados
│   └── classification_report_rf_train_*.txt
│
├── train/
│   ├── train_model_sre.py  # Script principal del pipeline
│   ├── train_model.py
│   └── train_multiple_models.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 4. Ejecución del pipeline

### 4.1 Entrenamiento del modelo
El modelo se entrena y registra automáticamente con el siguiente comando:

```bash
python train/train_model_sre.py
```

El script realiza las siguientes tareas:
1. Carga los datasets preprocesados (`data/interim/`).
2. Divide los datos en entrenamiento y prueba con `train_test_split()`.
3. Entrena un modelo `RandomForestClassifier`.
4. Calcula métricas de desempeño (`accuracy`, `precision`, `recall`, `f1`).
5. Genera y guarda:
   - Reporte de clasificación `.txt`
   - Matriz de confusión `.png`
   - Modelo serializado `.joblib`
6. Registra todos los artefactos en **MLflow** bajo el experimento:
   ```
   student_performance_experiment_fase2
   ```

---

## 5. Registro y seguimiento con MLflow

El rastreo de experimentos se habilita ejecutando:
```bash
mlflow ui --host 127.0.0.1 --port 5001 --workers 1
```

Esto levanta una interfaz local donde se pueden visualizar:
- Los *runs* del experimento.  
- Parámetros y métricas asociados a cada ejecución.  
- Artefactos generados (modelo, reportes y figuras).

### Ejemplo de métricas registradas
| Métrica | Valor |
|----------|--------|
| Accuracy | 0.9926 |
| Precision (weighted) | 0.99 |
| Recall (weighted) | 0.99 |
| F1-score (weighted) | 0.99 |

---

## 6. Control de versiones

El proyecto se mantuvo bajo control de versiones Git, con un `.gitignore` configurado para excluir entornos virtuales, artefactos y datos temporales, permitiendo que el repositorio contenga únicamente el código fuente, documentación y evidencias necesarias.

### Estructura de commits principales
```bash
SRE: added reproducible environment setup and technical report (Fase 2)
SRE: finalized reproducible requirements for MLflow environment
SRE: integración completa Fase 2 (entorno reproducible, MLflow tracking, requirements y documentación)
```

---

## 7. Verificación de reproducibilidad

Se realizó una prueba de ejecución en un entorno limpio (`test_env`) siguiendo el proceso:

```bash
git clone https://github.com/<usuario>/proyecto_mlops_equipo_56.git
cd proyecto_mlops_equipo_56
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python train/train_model_sre.py
```

Resultado:
```
Entrenamiento completado. Accuracy: 0.9926
```

Con esto se validó que el pipeline puede ejecutarse exitosamente en cualquier entorno a partir de `requirements.txt`, garantizando **reproducibilidad total**.

---

## 8. Conclusiones

El rol SRE cumplió su propósito al:
- Estandarizar la estructura del repositorio y sus componentes.
- Garantizar la reproducibilidad total del entorno.  
- Integrar el seguimiento de experimentos mediante MLflow.  
- Implementar buenas prácticas de control de versiones y documentación.

El pipeline resultante permite replicar entrenamientos, registrar métricas, y almacenar artefactos de manera controlada, sentando las bases para una fase posterior de **orquestación y despliegue (Fase 3)**.

---

## 9. Referencias

- Treveil, A., et al. *Introducing MLOps: How to Scale Machine Learning in the Enterprise*. O’Reilly, 2020.  
- Google Cloud. *Machine Learning Design Patterns*. O’Reilly, 2021.  
- Lauchande, R. *ML Engineering with MLflow*. O’Reilly, 2023.
