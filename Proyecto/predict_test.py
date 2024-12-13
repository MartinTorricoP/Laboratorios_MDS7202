from zipfile import ZipFile
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
import os


class DateTransformer(BaseEstimator, TransformerMixin):
    """
    Transformador para convertir columnas de timestamp en componentes de fecha (año, mes, día).
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        transformed = pd.DataFrame()  # Crear un dataframe temporal
        for col in X.columns:
            if col in X:  # Verificar que la columna existe
                transformed[f"{col}_year"] = pd.to_datetime(X[col], unit="s").dt.year
                transformed[f"{col}_month"] = pd.to_datetime(X[col], unit="s").dt.month
                transformed[f"{col}_day"] = pd.to_datetime(X[col], unit="s").dt.day
        return transformed.values


# Cargar datos
X_t1 = pd.read_parquet("data/raw/X_t3.parquet").reset_index(drop=True)
X_t2 = pd.read_parquet("data/raw/X_t2.parquet")
y_t2 = pd.read_parquet("data/raw/y_t2.parquet")["target"]

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X_t2, y_t2, test_size=0.3, random_state=42, stratify=y_t2
)

# Eliminar columnas innecesarias si existen
columns_to_remove = ["wallet_address"]
X_train = X_train.drop(columns=[col for col in columns_to_remove if col in X_train], errors="ignore")
X_test = X_test.drop(columns=[col for col in columns_to_remove if col in X_test], errors="ignore")
X_t1 = X_t1.drop(columns=[col for col in columns_to_remove if col in X_t1], errors="ignore")

# Definir columnas categóricas, de fechas y numéricas
timestamp_columns = [
    col for col in [
        "borrow_timestamp",
        "first_tx_timestamp",
        "last_tx_timestamp",
        "risky_first_tx_timestamp",
        "risky_last_tx_timestamp",
    ] if col in X_train.columns
]

numerical_columns = [col for col in X_train.columns if col not in timestamp_columns]

# Crear el preprocesador
preprocessor = ColumnTransformer(
    transformers=[
        ("date", DateTransformer(), timestamp_columns),  # Transformar columnas de fecha
        ("num", Pipeline(steps=[("scaler", StandardScaler())]), numerical_columns),  # Escalar columnas numéricas
    ],
    remainder="passthrough",
)

# Definir hiperparámetros del modelo XGBoost
best_params_xgb = {
    'n_estimators': 416,
    'max_depth': 10,
    'learning_rate': 0.08080810343490429,
    'subsample': 0.7328453030465105,
    'colsample_bytree': 0.5725079926214207,
    'gamma': 0.0903831051321515,
    'reg_alpha': 0.2101342999347895,
    'reg_lambda': 9.537189783402264
}

seed=42

# Envolver el modelo XGBoost
class WrappedXGBClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, **kwargs):
        self.model = XGBClassifier(**kwargs)

    def fit(self, X, y, **kwargs):
        return self.model.fit(X, y, **kwargs)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def get_params(self, deep=True):
        return self.model.get_params(deep)

    def set_params(self, **params):
        return self.model.set_params(**params)


# Crear modelo
model_xgb = WrappedXGBClassifier(
    n_estimators=best_params_xgb['n_estimators'],
    max_depth=best_params_xgb['max_depth'],
    learning_rate=best_params_xgb['learning_rate'],
    subsample=best_params_xgb['subsample'],
    colsample_bytree=best_params_xgb['colsample_bytree'],
    gamma=best_params_xgb['gamma'],
    reg_alpha=best_params_xgb['reg_alpha'],
    reg_lambda=best_params_xgb['reg_lambda'],
    random_state=seed,
    use_label_encoder=False,  # Evitar warnings en XGBoost
    eval_metric='logloss'  # Métrica interna del modelo
)

# Crear pipeline
pipeline_a = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", model_xgb)
])

# Entrenar el pipeline
pipeline_a.fit(X_train, y_train)
print("Pipeline entrenado correctamente.")

# Predecir sobre X_t1
y_pred_clf = pipeline_a.predict_proba(X_t1)[:, 1]

# Generar archivos de predicción
def generateFiles(predict_data, clf_pipe):
    """Genera los archivos a subir en CodaLab"""
    y_pred_clf = clf_pipe.predict_proba(predict_data)[:, 1]
    with open("./predictions.txt", "w") as f:
        for item in y_pred_clf:
            f.write("%s\n" % item)

    with ZipFile("codalab/predictions.zip", "w") as zipObj:
        zipObj.write("predictions.txt")
    os.remove("predictions.txt")


generateFiles(X_t1, pipeline_a)
print("Archivos generados correctamente.")
