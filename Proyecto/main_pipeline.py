import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.dummy import DummyClassifier
import mlflow
import optuna
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from src.mlflow_tracking.tracking import configure_mlflow
from src.mlflow_tracking.model_logger import log_model_with_mlflow
from src.mlflow_tracking.model_loader import load_best_model_by_metric
from src.mlflow_tracking.artifact_logger import log_data_to_mlflow, log_artifact_to_mlflow
from src.optimization.optuna_logger import optimize_model_with_optuna
from src.utils.check_data import check_or_create_processed_data
from src.utils.params import get_param_distributions
from src.utils.predict import get_predictions

from src.mlflow_tracking.interpretability import log_shap_interpretation

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# modificar path raw 
raw_path= "data/raw/X_t2.parquet"
path_y = "data/raw/y_t2.parquet"

processed_path = "data/processed/clean_data.csv"

experiment_name = "BCI Riesgo2"
n_trials = 2

def main():
    # 1. Configurar MLFlow
    configure_mlflow(experiment_name)

    # 2. Cargar datos
    check_or_create_processed_data(raw_path, processed_path)

    print("Cargando datos procesados...")
    log_artifact_to_mlflow(file_path=processed_path, artifact_path="data/processed")
    df = pd.read_csv(processed_path)
    target = pd.read_parquet(path_y)

    X = df.copy()
    y = target['target']

    # Dividir datos
    print("Dividiendo datos en entrenamiento y prueba...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Aplicar SMOTE en el conjunto de entrenamiento
    smote = SMOTE(random_state=42)

    # Ajustar y transformar el conjunto de entrenamiento (X_train, y_train)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    # Registrar datos divididos como artefactos
    log_data_to_mlflow(pd.DataFrame(X_train), "X_train.csv", "data/splits")
    log_data_to_mlflow(pd.DataFrame(X_test), "X_test.csv", "data/splits")
    log_data_to_mlflow(pd.DataFrame(y_train), "y_train.csv", "data/splits")
    log_data_to_mlflow(pd.DataFrame(y_test), "y_test.csv", "data/splits")

    # 3. Entrenar baseline 
    print("Entrenando baseline...")
    params_base = {}
    model_base = DummyClassifier(strategy='stratified')
    log_model_with_mlflow(
        model=model_base,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        run_name="Baseline Model",
        params=params_base
    )

    # 4. Modelos de Machine Learning
    print("Entrenando modelos de Machine Learning...")
    seed = 42
    params_ml = {}
    models_to_evaluate = {
        'Logistic Regression': LogisticRegression(random_state=seed),
        'KNN': KNeighborsClassifier(),
        'Decision Tree': DecisionTreeClassifier(random_state=seed),
        'Random Forest': RandomForestClassifier(random_state=seed),
        'LightGBM': LGBMClassifier(random_state=seed),
        'XGBoost': XGBClassifier(random_state=seed)
    }

    # Iterar sobre los modelos y evaluar los pipelines
    for model_name, model in models_to_evaluate.items():
        log_model_with_mlflow(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            run_name=model_name,
            params=params_ml
        )

    print("Cargando el mejor modelo desde MLFlow...")
    best_model, best_run_id = load_best_model_by_metric(
        experiment_name=experiment_name,
        metric_name="f1_score",
        maximize=True
    )
    print(f"El tipo del mejor modelo es: {type(best_model)}")
    print(f"Mejor modelo cargado desde el run: {best_run_id}")
    get_predictions(best_model, X_test, y_test)

    # 5. Optimizar modelo con Optuna
    print("Optimización de hiperparámetros con Optuna...")
    param_distributions = get_param_distributions(best_model)

    study_op = optimize_model_with_optuna(
        model_class=best_model,
        param_distributions=param_distributions,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        n_trials=n_trials,
    )

    # 6. Entrenar el mejor modelo optimizado y registrarlo
    print("Entrenando modelo optimizado...")
    best_params = study_op.best_trial.params
    model_optimized = best_model.set_params(**best_params)
    log_model_with_mlflow(
        model=model_optimized,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        run_name="Optimized Model",
        params=best_params,
        register_model=True
    )

    # 8. Realizar predicciones con el mejor modelo
    model_name = type(model_optimized).__name__
    get_predictions(model_optimized, X_test, y_test)
    log_shap_interpretation(model_name=model_name, dataset=df)
    print("Pipeline completado con éxito.")

if __name__ == "__main__":
    main()
