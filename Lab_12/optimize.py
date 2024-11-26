import os
import pickle
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import optuna
from optuna.visualization.matplotlib import plot_optimization_history
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import mlflow.xgboost
import pkg_resources
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def get_best_model(experiment_id):
    runs = mlflow.search_runs(experiment_ids=[experiment_id])
    best_model_id = runs.sort_values("metrics.valid_f1", ascending=False)["run_id"].iloc[0]
    best_model = mlflow.xgboost.load_model(f"runs:/{best_model_id}/model")
    return best_model

def get_best_model_global():
    # Buscar todas las ejecuciones en todos los experimentos
    runs = mlflow.search_runs()

    # Verificar si existe la métrica 'metrics.valid_f1'
    if "metrics.valid_f1" in runs.columns:
        # Filtrar ejecuciones con valores no nulos para 'valid_f1'
        valid_runs = runs[runs["metrics.valid_f1"].notnull()]

        if not valid_runs.empty:
            # Ordenar por 'valid_f1' de forma descendente y seleccionar la mejor
            best_run = valid_runs.loc[valid_runs["metrics.valid_f1"].idxmax()]
            best_model_id = best_run["run_id"]
            best_model = mlflow.xgboost.load_model(f"runs:/{best_model_id}/model")
            best_f1_score = best_run["metrics.valid_f1"]
            best_experiment_name = best_run["experiment_id"]

            print(f"Mejor modelo encontrado en el experimento ID: {best_experiment_name}")
            print(f"F1-Score: {best_f1_score}")
            
            return best_model, best_f1_score

def optimize_model(learning_rate_values):
    
    # Crear carpetas necesarias
    os.makedirs("plots", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # Cargar el dataset
    df = pd.read_csv("water_potability.csv")

    # Eliminamos las entradas con valores nulos
    df = df.dropna()

    # Dividir el dataset
    X = df.drop(columns=["Potability"])
    y = df["Potability"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)

    for learning_rate in learning_rate_values:
        # Configurar experimento de MLflow para este learning_rate
        experiment_name = f"XGBoost_LR_{learning_rate}"
        mlflow.set_experiment(experiment_name)

        # Crear un estudio para optimización
        study = optuna.create_study(direction="maximize")

        def objective(trial):
            # Finalizar cualquier run activo
            if mlflow.active_run() is not None:
                mlflow.end_run()
            
            # Definir los hiperparámetros que Optuna optimizará
            param_grid = {
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'learning_rate': learning_rate,  # Valor fijo para este experimento
                'max_depth': trial.suggest_int('max_depth', 2, 10),
                'min_child_weight': trial.suggest_loguniform('min_child_weight', 0.1, 10),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'lambda': trial.suggest_loguniform('lambda', 1e-3, 10.0),
                'alpha': trial.suggest_loguniform('alpha', 1e-3, 10.0),
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            }

            # Crear el modelo con los hiperparámetros sugeridos
            model = xgb.XGBClassifier(**param_grid, random_state=123)
            
            # Entrenar el modelo
            model.fit(X_train, y_train)
            
            # Realizar predicciones
            y_pred = model.predict(X_test)
            
            # Calcular el F1-Score
            f1 = f1_score(y_test, y_pred)

            # Registrar en un nuevo run de MLflow
            with mlflow.start_run(run_name=f"XGBoost_Run_LR_{learning_rate:.2f}_Trial_{trial.number}"):
                mlflow.log_params(param_grid)
                mlflow.log_metric("valid_f1", f1)
                mlflow.xgboost.log_model(model, "model")
            
            return f1  # Optuna buscará maximizar este valor

        # Optimizar hiperparámetros para este learning_rate
        study.optimize(objective, n_trials=25)

        # Generar el gráfico de Optuna (Optimization History)
        fig1 = plot_optimization_history(study)

        # Ajustar el tamaño del gráfico
        fig1.figure.set_size_inches(12, 6)

        # Guardar el gráfico como archivo PNG
        fig1.figure.savefig(f"plots/optimization_history_lr_{learning_rate}_.png", bbox_inches="tight")

        # Registrar el archivo en MLflow
        mlflow.log_artifact(f"plots/optimization_history_lr_{learning_rate}_.png")

        # Devolver y guardar el mejor modelo para este experimento
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
        best_model = get_best_model(experiment_id)
        
        # Guardar el modelo con pickle
        with open(f"models/best_model_lr_{learning_rate}.pkl", "wb") as f:
            pickle.dump(best_model.get_booster(), f)

        # Guardar la importancia de las variables
        importance = best_model.get_booster().get_score(importance_type="weight")
        importance_df = pd.DataFrame(importance.items(), columns=["Feature", "Importance"])
        # Ajustar el tamaño del gráfico
        plt.figure(figsize=(12, 6))  # Aumentar el tamaño del gráfico

        # Crear el gráfico de barras
        importance_df.sort_values(by="Importance", ascending=False).plot(
            kind="bar",
            x="Feature",
            y="Importance",
            title=f"Feature Importance LR {learning_rate}",
            legend=True,
            figsize=(12, 6)  # Ajustar tamaño directamente aquí también
        )

        # Ajustar etiquetas del eje X
        plt.xticks(rotation=45, ha="right")

        # Añadir márgenes para evitar cortes
        plt.tight_layout()

        # Guardar el gráfico
        plt.savefig(f"plots/feature_importance_lr_{learning_rate}.png")
        mlflow.log_artifact(f"plots/feature_importance_lr_{learning_rate}.png")

    # Guardar las versiones de las librerías
    with open("models/requirements.txt", "w") as f:
        requirements = [f"{pkg.project_name}=={pkg.version}" for pkg in pkg_resources.working_set]
        f.write("\n".join(requirements))

if __name__ == "__main__":
    # Definir los valores fijos de learning_rate
    learning_rate_values = [0.1, 0.2, 0.3]
    
    # Ejecutar la optimización para cada learning_rate
    optimize_model(learning_rate_values)
    
    # Obtener y guardar el mejor modelo global
    best_model, best_f1_score = get_best_model_global()
    
    # Guardar el mejor modelo global con pickle
    with open("models/best_model_global.pkl", "wb") as f:
        pickle.dump(best_model.get_booster(), f)
    print("El mejor modelo global ha sido guardado en 'models/best_model.pkl'")
    
    # Verificar las columnas disponibles en el mejor run
    runs = mlflow.search_runs()
    print("Columnas disponibles en runs:")
    print(runs.columns)

    # Inspeccionar el mejor run
    best_run = runs.loc[runs["metrics.valid_f1"].idxmax()]
    print("Contenido del mejor run:")
    print(best_run)
    
    if best_model is not None:
        # Buscar el mejor run basado en valid_f1
        runs = mlflow.search_runs()
        best_run = runs.loc[runs["metrics.valid_f1"].idxmax()]

        # Recuperar los parámetros relevantes desde el mejor run
        relevant_params = {
            "learning_rate": best_run["params.learning_rate"],
            "max_depth": best_run["params.max_depth"],
            "min_child_weight": best_run["params.min_child_weight"],
            "subsample": best_run["params.subsample"],
            "colsample_bytree": best_run["params.colsample_bytree"],
            "lambda": best_run["params.lambda"],
            "alpha": best_run["params.alpha"],
            "n_estimators": best_run["params.n_estimators"],
        }

        # Convertir a DataFrame
        relevant_params_df = pd.DataFrame(
            list(relevant_params.items()), columns=["Parameter", "Value"]
        )

        # Asegurarse de que los valores sean numéricos
        relevant_params_df["Value"] = pd.to_numeric(relevant_params_df["Value"], errors="coerce")

        # Crear el gráfico con tamaño ajustado
        fig, ax = plt.subplots(figsize=(10, 6))  # Ajustar el tamaño del gráfico

        # Crear gráfico de barras
        bars = ax.bar(relevant_params_df["Parameter"], relevant_params_df["Value"], color="skyblue")

        # Añadir los valores encima de cada barra
        for bar in bars:
            yval = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2, 
                yval + 0.5,  # Desplazar un poco hacia arriba
                round(yval, 2),  # Mostrar valores con 2 decimales
                ha="center",
                va="bottom",
                fontsize=10
            )

        # Ajustar las etiquetas
        ax.set_title("Best Model Configurations", fontsize=16)
        ax.set_ylabel("Value", fontsize=12)
        ax.set_xlabel("Parameter", fontsize=12)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        
        plt.savefig("plots/best_model_configurations.png")
        mlflow.log_artifact("plots/best_model_configurations.png")
        print("Gráfico de configuraciones del mejor modelo guardado en '/plots/best_model_configurations.png'")
    else:
        print("No se encontró ningún modelo para graficar.")