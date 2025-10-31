# Informe Técnico – SRE (Fase 2 MLOps)
**Proyecto:** Student Performance  
**Rol:** Site Reliability Engineer (SRE)  
**Autor:** Neri [Equipo 56, MLOps – Fase 2]  
**Fecha:** 30 de octubre de 2025

## Resumen
Este informe documenta el trabajo del rol SRE en la Fase 2 del proyecto *Student Performance*. Se presenta el diseño de un entorno reproducible (control de dependencias, versionamiento y estructura de proyecto), la gestión de experimentos con MLflow (parámetros, métricas, artefactos), la automatización del pipeline (scripts/Makefile) y un esquema inicial de monitoreo y logging. Se justifica cada decisión con buenas prácticas de MLOps (reproducibilidad, separación de entornos, control de riesgo de modelos y preparación para producción). El objetivo es habilitar que el modelo del equipo se ejecute de forma consistente desde Visual Studio Code, facilitando iteración, trazabilidad y posterior despliegue controlado.

**Palabras clave:** SRE, MLOps, reproducibilidad, MLflow, automatización, monitoreo, Student Performance.

---

## 1. Introducción y alcance del rol SRE
El rol SRE en MLOps garantiza confiabilidad, reproducibilidad y operabilidad del ciclo de vida de ML: desde el entorno y las dependencias, pasando por el seguimiento de experimentos, hasta la automatización y observabilidad inicial. En la Fase 2, el foco no es un despliegue productivo definitivo, sino dejar listo el andamiaje técnico para ejecutar el modelo del equipo de manera consistente, auditable y repetible, y preparar el terreno para validación y futura integración continua.

Objetivos de esta fase:
- Estandarizar el entorno de ejecución y las dependencias.
- Implementar tracking de experimentos (MLflow).
- Automatizar ejecución de pipeline (entrenamiento, registro de artefactos).
- Habilitar logging y evidencias mínimas de monitoreo local.
- Documentar riesgos y controles (reproducibilidad, training–serving skew, data drift).

---

## 2. Contexto del proyecto y dataset
El proyecto *Student Performance* busca entrenar un modelo supervisado para predecir el rendimiento de estudiantes (métrica principal por confirmar con el equipo: por ejemplo *accuracy* o *F1*). La Fase 2 exige una versión inicial operativa del pipeline y evidencias de control de versiones y documentación técnica. Como SRE, se habilita un flujo reproducible que el resto de roles puede usar y auditar desde VS Code.

---

## 3. Entorno reproducible (VS Code)

### 3.1 Estructura del proyecto
Se adopta una estructura mínima, portable y clara:

```
student-performance/
├─ data/                       # (no versionar datos sensibles)
├─ notebooks/                  # exploración/E2E rápido (opcional)
├─ src/
│  ├─ train_model.py           # entrenamiento y registro MLflow
│  ├─ utils_logging.py         # configuración de logging
│  └─ __init__.py
├─ models/                     # salidas (artefactos locales)
├─ mlruns/                     # tracking MLflow local
├─ requirements.txt            # dependencias fijadas
├─ run_pipeline.sh             # pipeline reproducible (bash)
├─ Makefile                    # alternativa cross-OS a run_pipeline.sh
├─ README.md
└─ .gitignore
```

### 3.2 Dependencias y fijado de versiones
Para garantizar resultados repetibles, se fijan versiones en `requirements.txt`:

```
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.5.0
mlflow==2.13.1
matplotlib==3.9.0
joblib==1.3.2
```

La fijación evita derivas de entorno entre máquinas.

### 3.3 Entorno virtual
En **VS Code (Terminal integrado):**

```
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

Registrar versión de Python y hash del archivo de dependencias como evidencia de entornos equivalentes.

---

## 4. Gestión de experimentos con MLflow
Se activa **MLflow Tracking** local para:
- Registrar parámetros (algoritmo, random_state, test_size, features).
- Registrar métricas (accuracy, precision, recall, f1, AUC, según tarea).
- Guardar artefactos (modelo `.pkl` o `.joblib`, matriz de confusión `.png`, reporte de clasificación).
- Versionar cada run con etiquetas y commits asociados.

Buenas prácticas:
- Definir un `experiment_name` por iteración.
- Añadir tags: `role=SRE`, `phase=2`, `dataset=v1`.
- Mantener `mlruns/` o documentar si se excluye del repositorio.

---

## 5. Automatización del pipeline
Se implementa un pipeline reproducible con script `run_pipeline.sh` y Makefile.

### 5.1 Script `run_pipeline.sh`
Ejemplo de automatización en bash:

```
#!/usr/bin/env bash
set -e
echo "[SRE] Activando entorno..."
source .venv/bin/activate

echo "[SRE] Entrenando y registrando en MLflow..."
python -u src/train_model.py 2>&1 | tee logs_training.txt

echo "[SRE] Pipeline finalizado OK."
```

### 5.2 Makefile
Alternativa más portable:

```
init:
	python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt

train:
	source .venv/bin/activate && python -u src/train_model.py

all: init train
```

---

## 6. Monitoreo y logging
Se configura el módulo `logging` en Python:

```python
import logging
logging.basicConfig(
    filename='training.log',
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)
logging.info("Entrenamiento iniciado correctamente")
```

El registro de logs permite trazabilidad y detección temprana de fallos. En fases posteriores se ampliará con validaciones automáticas y monitoreo de drift.

---

## 7. Control de versiones (Git)
Buenas prácticas de versionamiento:
- Rama principal `main` y rama de trabajo `sre/setup-phase2`.
- `.gitignore` debe excluir: `.venv/`, `__pycache__/`, `*.pyc`, `mlruns/` (opcional), `data/` si contiene datos crudos.

Cada commit debe describir cambios específicos de infraestructura, dependencias o scripts.

---

## 8. Riesgos y mitigaciones
1. **Reproducibilidad:** mitigada con entornos virtuales, fijado de dependencias y MLflow.  
2. **Training–Serving Skew:** separación clara de inputs y features, registro de transformaciones.  
3. **Data Drift:** preparar scripts para comparar distribuciones entre datasets.  
4. **Errores del pipeline:** control de fallos con `set -e`, logging estructurado y automatización progresiva.

---

## 9. Evidencias esperadas
- Archivo `requirements.txt` y captura de `pip list`.  
- Ejecución exitosa de `run_pipeline.sh` o `make train`.  
- Registro en MLflow con parámetros y métricas.  
- Artefactos guardados en `models/`.  
- Historial de commits documentando cada avance.

---

## 10. Conclusiones
La contribución SRE en la Fase 2 deja un entorno reproducible, un pipeline automatizado y tracking de experimentos que habilitan al equipo a iterar de forma confiable desde VS Code y presentar evidencias de ingeniería MLOps. Con ello, se reduce el riesgo operativo, se facilita la auditoría de resultados y se prepara el terreno para validación avanzada, monitoreo de drift y CI/CD en fases siguientes.

---

## Referencias

- Breck, E., Polyzotis, N., Zinkevich, M., et al. (2020). *Machine Learning Design Patterns*. O’Reilly Media.  
- Kleppmann, M. (2017). *Designing Data-Intensive Applications*. O’Reilly Media.  
- Kazil, J., & Jaworski, K. (2016). *Data Wrangling with Python*. O’Reilly Media.  
- Bruce, P., Bruce, A., & Gedeck, P. (2020). *Practical Statistics for Data Scientists (2nd ed.)*. O’Reilly Media.  
- Treveil, M., et al. (2020). *Introducing MLOps*. O’Reilly Media.
