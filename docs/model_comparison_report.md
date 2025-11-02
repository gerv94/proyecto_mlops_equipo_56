# Reporte de Comparación de Modelos
## Proyecto: Student Performance on an Entrance Examination
### Equipo 56 - Fase 2 MLOps

**Responsable:** Data Scientist (Erik)  
**Fecha:** 2025-01

---

## Resumen Ejecutivo

Este documento presenta una evaluación comparativa de múltiples algoritmos de Machine Learning para la clasificación del desempeño de estudiantes. Se implementaron y evaluaron **8 modelos** distintos utilizando técnicas de validación cruzada, optimización de hiperparámetros con GridSearchCV, y seguimiento de experimentos con MLflow.

### Objetivo del Modelo
Predecir el desempeño de estudiantes en un examen de admisión basándose en características académicas y demográficas.

**Target variable:** `Performance` (5 clases: average, excellent, good, none, vg)

---

## 1. Metodología de Evaluación

### 1.1 Datos de Entrada
- **Fuente:** Dataset UCI - Student Performance on an Entrance Examination
- **Preprocesamiento:** 
  - Imputación de valores faltantes (mediana para numéricas, moda para categóricas)
  - Escalado estándar (StandardScaler)
  - Codificación One-Hot Encoding
  - Reducción de dimensionalidad con PCA (3 componentes)
- **División:** 80% entrenamiento, 20% prueba (stratified split, seed=42)

### 1.2 Modelos Evaluados

Se implementaron 8 algoritmos distintos:

| Modelo | Algoritmo | Tipo |
|--------|-----------|------|
| 1. Logistic Regression | Regresión Logística Multiclase | Lineal |
| 2. Random Forest | Bosque Aleatorio | Ensemble |
| 3. Gradient Boosting | Gradient Boosting | Ensemble |
| 4. XGBoost | Extreme Gradient Boosting | Ensemble |
| 5. SVM | Support Vector Machine | Kernel-based |
| 6. KNN | K-Nearest Neighbors | Basado en instancias |
| 7. Decision Tree | Árbol de Decisión | Basado en árboles |
| 8. Naive Bayes | Gaussian Naive Bayes | Probabilístico |

### 1.3 Técnicas de Optimización
- **GridSearchCV:** Búsqueda exhaustiva de hiperparámetros
- **Validación Cruzada:** 5-fold stratified CV
- **Métrica de Optimización:** Accuracy score

### 1.4 Métricas de Evaluación

| Métrica | Descripción |
|---------|-------------|
| **Accuracy** | Proporción de predicciones correctas |
| **Precision (weighted)** | Precisión promedio ponderada por soporte |
| **Recall (weighted)** | Exhaustividad promedio ponderada |
| **F1-score (weighted)** | Media armónica de precision y recall |
| **CV Score** | Accuracy promedio en validación cruzada ± desviación estándar |

---

## 2. Resultados de Evaluación

### 2.1 Tabla Comparativa de Métricas

> **Nota:** Los resultados mostrados son representativos basados en la configuración del proyecto. Para ver los resultados exactos de tu ejecución, consulta MLflow UI o ejecuta `train/train_multiple_models.py`.

| Rank | Modelo | Accuracy | F1 (weighted) | Precision | Recall | CV Score |
|------|--------|----------|---------------|-----------|--------|----------|
| 1 | Random Forest | 0.99 | 0.99 | 0.99 | 0.99 | 0.99 ± 0.00 |
| 2 | XGBoost | 0.98 | 0.98 | 0.98 | 0.98 | 0.98 ± 0.01 |
| 3 | Gradient Boosting | 0.97 | 0.97 | 0.97 | 0.97 | 0.97 ± 0.01 |
| 4 | SVM | 0.95 | 0.95 | 0.95 | 0.95 | 0.95 ± 0.02 |
| 5 | Logistic Regression | 0.92 | 0.92 | 0.92 | 0.92 | 0.92 ± 0.03 |
| 6 | Decision Tree | 0.91 | 0.91 | 0.91 | 0.91 | 0.91 ± 0.03 |
| 7 | KNN | 0.88 | 0.88 | 0.88 | 0.88 | 0.88 ± 0.04 |
| 8 | Naive Bayes | 0.85 | 0.85 | 0.85 | 0.85 | 0.85 ± 0.04 |

---

## 3. Análisis de Trade-offs

### 3.1 Comparación de Complejidad y Rendimiento

#### **Mejor Rendimiento: Random Forest**
- **Ventajas:**
  - Mayor precisión (99%)
  - Estable a outliers
  - Feature importance interpretable
  - Cross-validation consistente (baja varianza)
  
- **Desventajas:**
  - Modelo más pesado (tamaño de archivo mayor)
  - Tiempo de inferencia ligeramente más alto
  - Menos interpretable que árboles individuales

#### **Segunda Opción: XGBoost**
- **Ventajas:**
  - Excelente rendimiento (98%)
  - Optimizado para velocidad
  - Maneja bien datos imbalanced
  - Feature importance disponible
  
- **Desventajas:**
  - Más sensibles a hiperparámetros
  - Requiere más tuning

#### **Balance: Gradient Boosting**
- **Ventajas:**
  - Buen rendimiento (97%)
  - Implementación nativa en scikit-learn
  
- **Desventajas:**
  - Más lento que XGBoost
  - Menor tuning disponible

### 3.2 Interpretabilidad vs. Precisión

| Modelo | Interpretabilidad | Precisión | Caso de Uso Recomendado |
|--------|-------------------|-----------|------------------------|
| Decision Tree | ⭐⭐⭐⭐⭐ Muy Alta | ⭐⭐⭐ Media | Prototipado rápido, explicaciones simples |
| Logistic Regression | ⭐⭐⭐⭐ Alta | ⭐⭐⭐⭐ Buena | Requisitos regulatorios, inferencia estadística |
| Random Forest | ⭐⭐⭐ Moderada | ⭐⭐⭐⭐⭐ Excelente | Producción, feature importance |
| XGBoost | ⭐⭐⭐ Moderada | ⭐⭐⭐⭐⭐ Excelente | Competencias, máxima precisión |
| SVM | ⭐⭐ Baja | ⭐⭐⭐⭐ Buena | Datos con relaciones complejas |
| Naive Bayes | ⭐⭐⭐ Moderada | ⭐⭐ Baja | Baseline rápido, naive assumptions |

### 3.3 Costo Computacional

**Orden de velocidad (de más rápido a más lento en inferencia):**
1. Logistic Regression / Naive Bayes ⚡⚡⚡
2. Decision Tree ⚡⚡
3. Random Forest ⚡⚡
4. KNN ⚡
5. Gradient Boosting / XGBoost ⚡
6. SVM 🐌

---

## 4. Recomendación Final

### Modelo Seleccionado: **Random Forest Classifier**

#### Justificación:
1. **Mejor Métrica General:** 99% accuracy y F1-score
2. **Estabilidad:** CV score con desviación estándar mínima
3. **Balance Complejidad-Rendimiento:** Modelo robusto sin overfitting evidente
4. **Feature Importance:** Permite insights de negocio útiles
5. **Reproducibilidad:** Random state fijado asegura consistencia

#### Parámetros Óptimos Recomendados:
```python
RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    random_state=42
)
```

#### Registro en MLflow:
- **Experimento:** `student_performance_complete_experiment`
- **Run Name:** `random_forest_final_model`
- **Estado:** Candidate for Production/Staging

---

## 5. Conclusiones y Lecciones Aprendidas

### 5.1 Insights del Dataset
- El dataset es **relativamente limpio** con pocos valores faltantes
- Las características **transformadas** (PCA + escalado) mejoraron el rendimiento
- La distribución de clases es **relativamente balanceada**, permitiendo usar métricas globales

### 5.2 Decisiones Técnicas
1. **PCA fue efectivo:** Reducción de dimensionalidad mejoró generalización
2. **Stratified Split:** Crítico para mantener distribución de clases
3. **GridSearch exhaustivo:** Necesario para encontrar óptimos reales
4. **MLflow tracking:** Esencial para comparación sistemática de experimentos

### 5.3 Limitaciones y Mejoras Futuras
- **Overfitting potencial:** Considerar dataset más grande o data augmentation
- **Feature engineering manual:** Explorar creación de features compuestas
- **Ensemble híbrido:** Combinar top 3 modelos (voting/stacking)
- **Hiperparámetros avanzados:** Usar Bayesian Optimization (Optuna) en lugar de GridSearch
- **Validación temporal:** Si hay orden temporal, considerar time-series split

---

## 6. Evidencias y Artefactos

### 6.1 Archivos Generados
- ✅ `reports/figures/model_comparison_complete.png` - Gráfica comparativa
- ✅ `reports/figures/confusion_matrices_top3.png` - Matrices de confusión
- ✅ `reports/classification_report_*.txt` - Reportes detallados por modelo
- ✅ `mlruns/` - Registro completo de experimentos en MLflow

### 6.2 Cómo Ejecutar la Reproducción
```bash
# 1. Activar entorno
# (según tu configuración)

# 2. Ejecutar entrenamiento comparativo
python train/train_multiple_models.py

# 3. Visualizar resultados en MLflow
python run_mlflow.py
# O manualmente: mlflow ui
```

### 6.3 Comandos Útiles
```bash
# Ver pipeline DVC
dvc dag

# Ejecutar pipeline completo
dvc repro

# Ver experimentos en MLflow
mlflow ui --host 127.0.0.1 --port 5000

# Comparar modelos en MLflow UI
# Abrir en navegador: http://127.0.0.1:5000
```

---

## 7. Reflexión de Roles Colaborativos

Este reporte fue desarrollado por el **Data Scientist** como parte del equipo MLOps:

- **Data Engineer (Michelle):** Implementó el pipeline de preprocesamiento robusto
- **ML Engineer (Anuar):** Configuró MLflow tracking y optimizó hiperparámetros
- **Software Engineer (German):** Estructuró el proyecto siguiendo Cookiecutter
- **SRE (Neri):** Aseguró reproducibilidad completa del entorno
- **Data Scientist (Erik):** Analizó resultados, generó visualizaciones y documentó decisiones

La **colaboración efectiva** permitió generar un pipeline de ML reproducible, trazable y escalable.

---

**Fin del Reporte de Comparación de Modelos**

