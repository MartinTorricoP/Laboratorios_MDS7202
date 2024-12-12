import optuna
import mlflow
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score

from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


def optimize_model_with_optuna(
    model_class, 
    param_distributions, 
    X_train, 
    y_train, 
    X_test, 
    y_test, 
    n_trials=10, 
    metric=f1_score
):
    """
    Optimiza un modelo usando Optuna y registra los resultados en MLFlow.

    Args:
        model_class: Clase del modelo (e.g., RandomForestClassifier, XGBClassifier).
        param_distributions (dict): Diccionario de distribuciones de parámetros a optimizar.
        X_train, y_train: Datos de entrenamiento.
        X_test, y_test: Datos de prueba.
        n_trials (int): Número de trials para Optuna.
        metric (callable): Métrica de evaluación para optimizar (por defecto: f1_score).

    Returns:
        study: Objeto de estudio 
    """
    def objective(trial):
        # hiperparametros sugeridos
        params = {key: trial._suggest(key, value) for key, value in param_distributions.items()}

        #model_type = get_model_class(model_class)

        # se crea la instancia del modelo con los parametros 
        model = model_class.set_params(**params, random_state=42)

        pipeline = Pipeline([
        # ('preprocessor', preprocessor),  
            ('classifier', model)
        ])

        # validacion cruzada para calcular el f1-score
        score = cross_val_score(pipeline, X_train, y_train, n_jobs=-1, cv=3, scoring='f1_weighted').mean()
        return score
    
    # ahora se crea un estudio para maximizar el f1-score
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)  

    with mlflow.start_run(run_name=f"{str(model_class)} Optimization"):
        # Mejor trial
        best_trial = study.best_trial

        # Registrar hiperparámetros y métrica
        mlflow.log_params(best_trial.params)
        mlflow.log_metric("best_value", best_trial.value)

        # Registrar gráfico de optimización como artefacto
        fig = optuna.visualization.matplotlib.plot_optimization_history(study)
        fig.savefig("artifacts/optuna_history.png")
        mlflow.log_artifact("artifacts/optuna_history.png")

        print("Estudio de Optuna registrado en MLFlow.")

    return study





