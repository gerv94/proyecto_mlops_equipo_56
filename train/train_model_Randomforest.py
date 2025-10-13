import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


df = pd.read_csv('data/raw/student_entry_performance_original.csv')
df.head()

X = df.drop(columns=['Performance']) 
y = df['Performance'] 
cat_cols=['Gender','Caste','coaching','time','Class_ten_education','twelve_education','medium','Class_ X_Percentage','Class_XII_Percentage','Father_occupation','Mother_occupation']
# Label encode target (opcional; sklearn acepta strings pero conviene estandarizar) 
le = LabelEncoder()
y_enc = le.fit_transform(y) # guarda le.classes_ para interpretar

#split , se uso un muestreo estratificado al 80 y 20%
X_train, X_test, y_train, y_test = train_test_split( X, y_enc, test_size=0.2, stratify=y_enc, random_state=42 )

#ColumnTransformer: OneHot para categóricas 
ohe = OneHotEncoder(handle_unknown='ignore') # sparse=False para DataFrame fácil 

preprocessor = ColumnTransformer( transformers=[ ('ohe', ohe, cat_cols) ], remainder='drop' )# si todo es categórico, ok. Si hay num, usar passthrough  
# pipeline con RandomForest 
                                  
pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('clf', RandomForestClassifier(n_estimators=60, random_state=42, n_jobs=-1, class_weight='balanced')) ]) 

# entrenar 
pipeline.fit(X_train, y_train) 

# predecir 
y_pred_train = pipeline.predict(X_train) 
y_pred_test = pipeline.predict(X_test)
print("Acc train:", accuracy_score(y_train, y_pred_train))
print("Acc test :", accuracy_score(y_test, y_pred_test))
print("\nClassification report (test):\n", classification_report(y_test, y_pred_test, target_names=le.classes_))