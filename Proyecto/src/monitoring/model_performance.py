from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import mlflow

def monitor_model_performance(y_true, y_pred, run_name="Model Monitoring") -> None:
    """
    Monitorea el desempeño del modelo y registra métricas en MLFlow.

    Args:
        y_true (list): Etiquetas reales.
        y_pred (list): Predicciones generadas por el modelo.
        run_name (str): Nombre del run en MLFlow.
    """
    precision = precision_score(y_true, y_pred, average="weighted")
    recall = recall_score(y_true, y_pred, average="weighted")
    f1 = f1_score(y_true, y_pred, average="weighted")
    accuracy = accuracy_score(y_true, y_pred)

    print(f"Precisión: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Accuracy: {accuracy:.4f}")

    with mlflow.start_run(run_name=run_name):
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("accuracy", accuracy)
        print("Métricas registradas en MLFlow.")
