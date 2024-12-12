from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import mlflow

def log_model_performance(y_true, y_pred, experiment_name: str):
    """
    Registra métricas de desempeño del modelo en MLFlow.

    Args:
        y_true (list): Etiquetas reales.
        y_pred (list): Predicciones del modelo.
        experiment_name (str): Nombre del experimento en MLFlow.
    """
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="Model Performance"):
        precision = precision_score(y_true, y_pred, average="weighted")
        recall = recall_score(y_true, y_pred, average="weighted")
        f1 = f1_score(y_true, y_pred, average="weighted")
        accuracy = accuracy_score(y_true, y_pred)

        # Registrar métricas
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("accuracy", accuracy)

        print("Métricas de desempeño registradas en MLFlow.")
