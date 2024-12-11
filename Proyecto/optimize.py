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
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder
from imblearn.over_sampling import SMOTE

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

def optimize_model(n_estimators_values):
    
    # Crear carpetas necesarias
    os.makedirs("plots", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # Cargar los datasets
    X_t1_path = "data/X_t1.parquet"  
    y_t1_path = "data/y_t1.parquet"  
    X_t2_path = "data/X_t2.parquet"

    X_t1 = pd.read_parquet(X_t1_path).reset_index(drop=True)
    y_t1 = pd.read_parquet(y_t1_path).reset_index(drop=True)
    X_t2 = pd.read_parquet(X_t2_path).reset_index(drop=True)
    
    # Preprocesamiento
    df = X_t1.copy()

    # Seleccionar las columnas numéricas
    numerical_cols = X_t1.select_dtypes(include=np.number).columns
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = X_t1.select_dtypes(include='object').columns

    # Feature engineering
    def feature_engineering(df):
        """
        Aplica transformaciones para crear nuevas características en el dataset.
        Args:
            df (DataFrame): DataFrame original preprocesado.
        Returns:
            DataFrame: DataFrame con nuevas características añadidas.
        """
        df = df.copy()  # Evitar modificar el original

        # 1. Crear ratios
        if 'total_balance_eth' in df.columns and 'unique_borrow_protocol_count' in df.columns:
            df['balance_per_protocol'] = df['total_balance_eth'] / (df['unique_borrow_protocol_count'] + 1)

        if 'risk_factor' in df.columns and 'borrow_amount_avg_eth' in df.columns:
            df['weighted_risk'] = df['risk_factor'] * df['borrow_amount_avg_eth']

        # 2. Transformaciones logarítmicas
        if 'total_balance_eth' in df.columns:
            df['log_total_balance'] = np.log1p(df['total_balance_eth'])  # log(1 + x) para evitar log(0)

        # 3. Rankings
        if 'avg_gas_paid_per_tx_eth' in df.columns:
            df['gas_rank'] = df['avg_gas_paid_per_tx_eth'].rank(ascending=False)

        # Transformación logarítmica
        if 'wallet_age' in df.columns:
            df['log_wallet_age'] = np.log1p(df['wallet_age'])  # log(1 + x) para evitar log(0)
            df['log_wallet_age'] = df['log_wallet_age'].fillna(0) 

        # Conversión a años (si wallet_age es tiempo en segundos)
        if 'wallet_age' in df.columns:
            df['wallet_age_years'] = df['wallet_age'] / (60 * 60 * 24 * 365)  # Aproximar a años

        # 5. Interacciones adicionales (ejemplo con timestamps)
        if 'borrow_timestamp_year' in df.columns and 'first_tx_timestamp_year' in df.columns:
            df['time_since_borrow'] = df['borrow_timestamp_year'] - df['first_tx_timestamp_year']

        return df

    # 1. Preprocesamiento
    categorical_columns = []  
    timestamp_columns = ['borrow_timestamp', 'first_tx_timestamp', 'last_tx_timestamp']  # Variables de tiempo
    numerical_columns = [col for col in df.columns if col not in timestamp_columns + ['wallet_address']]  
    df = df.drop(columns=['wallet_address'])

    # 2. Transformador personalizado para fechas
    class DateTransformer(BaseEstimator, TransformerMixin):
        def fit(self, X, y=None):
            return self

        def transform(self, X):
            X = X.copy()
            transformed = pd.DataFrame()  # Crear un DataFrame temporal
            for col in X.columns:
                transformed[f'{col}_year'] = pd.to_datetime(X[col], unit='s').dt.year
                transformed[f'{col}_month'] = pd.to_datetime(X[col], unit='s').dt.month
                transformed[f'{col}_day'] = pd.to_datetime(X[col], unit='s').dt.day
            return transformed.values 

    # 3. Configurar ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline(steps=[
                ('scaler', StandardScaler()),
                # pca
            ]), numerical_columns),  # Escalado de numéricas
            #('cat', OrdinalEncoder(),  categorical_columns),
            ('date', DateTransformer(), timestamp_columns)  # Transformación de fechas
        ],
        remainder='passthrough'  # Passthrough para dejar columnas no especificadas
    )

    numerical_names = numerical_columns

    # Nombres de las columnas transformadas por DateTransformer
    date_names = [f"{col}_{suffix}" for col in timestamp_columns for suffix in ['year', 'month', 'day']]

    # Combinar todos los nombres de columnas
    transformed_column_names = numerical_names+ date_names 

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor)  # Solo preprocesamiento en este ejemplo
    ])

    processed_data = pipeline.fit_transform(df)

    processed_data_df = pd.DataFrame(processed_data, columns=transformed_column_names)

    # Aplicar feature engineering
    df_features = feature_engineering(processed_data_df)

    X = df
    y = y_t1['target']

    # Dividir el conjunto de datos en entrenamiento (80%) y prueba (20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Dividir el conjunto de entrenamiento en entrenamiento (70%) y validación (30%)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=42, stratify=y_train)

    # Aplicar SMOTE en el conjunto de entrenamiento
    smote = SMOTE(random_state=42)

    # Ajustar y transformar el conjunto de entrenamiento (X_train, y_train)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    # Tomar una muestra aleatoria del 10% de los datos balanceados
    X_train_res_sampled = X_train_res.sample(frac=0.1, random_state=42)
    y_train_res_sampled = y_train_res[X_train_res_sampled.index]
    
    
    for n_estimators in n_estimators_values:
        # Configurar experimento de MLflow para este gamma
        experiment_name = f"XGBoost_N_Estimators_{n_estimators}"
        mlflow.set_experiment(experiment_name)

        # Crear un estudio para optimización
        study = optuna.create_study(direction="maximize")

        def objective(trial):
            # Finalizar cualquier run activo
            if mlflow.active_run() is not None:
                mlflow.end_run()
            
            # Definir los hiperparámetros que Optuna optimizará
            param_grid = {
                'n_estimators': n_estimators,
                'max_depth': trial.suggest_int('max_depth', 3, 20),  # Profundidad máxima de los árboles
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 1),  # Tasa de aprendizaje
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),  # Fracción de muestras para cada árbol
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.25, 1.0),  # Fracción de características por árbol
                'gamma': trial.suggest_float('gamma', 0.001, 2),  # Término de regularización para los nodos
                'reg_alpha': trial.suggest_float('reg_alpha', 0.001, 2),  # Regularización L1
                'reg_lambda': trial.suggest_float('reg_lambda', 5, 15),  # Regularización L2
            }

            # Crear el modelo con los hiperparámetros sugeridos
            model = xgb.XGBClassifier(**param_grid, random_state=42)
            
            # Entrenar el modelo
            model.fit(X_train_res_sampled, y_train_res_sampled)
            
            # Realizar predicciones
            y_pred = model.predict(X_test)
            
            # Calcular el F1-Score
            f1 = f1_score(y_test, y_pred)

            # Registrar en un nuevo run de MLflow
            with mlflow.start_run(run_name=f"XGBoost_Run_N_Estimators_{n_estimators:.2f}_Trial_{trial.number}"):
                mlflow.log_params(param_grid)
                mlflow.log_metric("valid_f1", f1)
                mlflow.xgboost.log_model(model, "model")
            
            return f1  # Optuna buscará maximizar este valor

        # Optimizar hiperparámetros para este n_estimators
        study.optimize(objective, n_trials=25)

        # Generar el gráfico de Optuna (Optimization History)
        fig1 = plot_optimization_history(study)

        # Ajustar el tamaño del gráfico
        fig1.figure.set_size_inches(12, 6)

        # Guardar el gráfico como archivo PNG
        fig1.figure.savefig(f"plots/optimization_history_n_estimators_{n_estimators}_.png", bbox_inches="tight")

        # Registrar el archivo en MLflow
        mlflow.log_artifact(f"plots/optimization_history_n_estimators_{n_estimators}_.png")

        # Devolver y guardar el mejor modelo para este experimento
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
        best_model = get_best_model(experiment_id)
        
        # Guardar el modelo con pickle
        with open(f"models/best_model_n_estimators_{n_estimators}.pkl", "wb") as f:
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
            title=f"Feature Importance n estimators {n_estimators}",
            legend=True,
            figsize=(12, 6)  # Ajustar tamaño directamente aquí también
        )

        # Ajustar etiquetas del eje X
        plt.xticks(rotation=45, ha="right")

        # Añadir márgenes para evitar cortes
        plt.tight_layout()

        # Guardar el gráfico
        plt.savefig(f"plots/feature_importance_n_estimators_{n_estimators}.png")
        mlflow.log_artifact(f"plots/feature_importance_n_estimators_{n_estimators}.png")

    # Guardar las versiones de las librerías
    with open("models/requirements.txt", "w") as f:
        requirements = [f"{pkg.project_name}=={pkg.version}" for pkg in pkg_resources.working_set]
        f.write("\n".join(requirements))

if __name__ == "__main__":
    # Definir los valores fijos de n_estimators
    n_estimators_values = [350, 400, 450]
    
    # Ejecutar la optimización para cada n_estimators
    optimize_model(n_estimators_values)
    
    # Obtener y guardar el mejor modelo global
    best_model = get_best_model_global()
    
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
            "n_estimators": best_run["params.n_estimators"],
            "max_depth": best_run["params.max_depth"],
            "learning_rate": best_run["params.learning_rate"],
            "subsample": best_run["params.subsample"],
            "colsample_bytree": best_run["params.colsample_bytree"],
            "gamma": best_run["params.lambda"],
            "reg_alpha": best_run["params.reg_alpha"],
            "reg_lambda": best_run["params.reg_lambda"],           
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