import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import mlflow
import mlflow.sklearn

#Prueba 

# --- Cargar datos ---
df = pd.read_csv('data/raw/student_interim_preprocessed.csv')

X = df.drop(columns=['Performance']) 
y = df['Performance'] 
cat_cols = ['Gender','Caste','coaching','time','Class_ten_education','twelve_education','medium',
            'Class_ X_Percentage','Class_XII_Percentage','Father_occupation','Mother_occupation']

# Label encode target
le = LabelEncoder()
y_enc = le.fit_transform(y)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, stratify=y_enc, random_state=42
)

# Preprocesamiento
ohe = OneHotEncoder(handle_unknown='ignore')
preprocessor = ColumnTransformer(
    transformers=[('ohe', ohe, cat_cols)],
    remainder='drop'
)

# Pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('clf', RandomForestClassifier(
        n_estimators=60, random_state=42, n_jobs=-1, class_weight='balanced'
    ))
])

# --- MLflow Tracking ---
mlflow.set_experiment("student_performance_classification")

with mlflow.start_run() as run:
    # Entrenamiento
    pipeline.fit(X_train, y_train)
    
    # Predicciones
    y_pred_train = pipeline.predict(X_train)
    y_pred_test = pipeline.predict(X_test)
    
    # Métricas
    acc_train = accuracy_score(y_train, y_pred_train)
    acc_test = accuracy_score(y_test, y_pred_test)
    
    print("Acc train:", acc_train)
    print("Acc test :", acc_test)
    print("\nClassification report (test):\n", classification_report(y_test, y_pred_test, target_names=le.classes_))
    
    # Registrar hiperparámetros
    mlflow.log_param("n_estimators", 60)
    mlflow.log_param("random_state", 42)
    mlflow.log_param("class_weight", "balanced")
    
    # Registrar métricas
    mlflow.log_metric("accuracy_train", acc_train)
    mlflow.log_metric("accuracy_test", acc_test)
    
    # Guardar modelo
    mlflow.sklearn.log_model(pipeline, "random_forest_pipeline")
    
    print("Modelo y métricas registrados en MLflow con run_id:", run.info.run_id)