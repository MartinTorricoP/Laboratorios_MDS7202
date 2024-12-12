import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

def retrain_model(reference_data_path, new_data_path, y_true_path, experiment_name):
    """
    Reentrena el mejor modelo registrado en MLFlow con los datos nuevos.

    Args:
        reference_data_path (str): Ruta al dataset de referencia.
        new_data_path (str): Ruta al dataset nuevo.
        y_true_path (str): Ruta a las etiquetas reales de los datos nuevos.
        experiment_name (str): Nombre del experimento en MLFlow.
    """
    # Cargar datos
    reference_data = pd.read_csv(reference_data_path)
    new_data = pd.read_csv(new_data_path)
    y_true = pd.read_csv(y_true_path)["target"]

    # Dividir en train/test
    X_train, X_test, y_train, y_test = train_test_split(
        new_data, y_true, test_size=0.2, random_state=42
    )

    # Cargar el mejor modelo desde MLFlow
    mlflow.set_experiment(experiment_name)
    runs = mlflow.search_runs(filter_string="tags.mlflow.runName = 'Best_Model'")
    if runs.empty:
        raise ValueError("No se encontró un modelo previamente registrado como 'Best_Model'.")
    
    best_run_id = runs.iloc[0]["run_id"]
    model_uri = f"runs:/{best_run_id}/model"
    model = mlflow.sklearn.load_model(model_uri)

    # Reentrenar el modelo
    model.fit(X_train, y_train)

    # Evaluar el modelo
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred, average="weighted")

    # Registrar el modelo reentrenado en MLFlow
    with mlflow.start_run(run_name="Model Retraining"):
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(model, "model")
        print(f"Modelo reentrenado registrado en MLFlow con F1-Score: {f1:.4f}")
