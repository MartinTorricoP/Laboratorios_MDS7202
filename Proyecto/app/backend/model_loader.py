import mlflow
from mlflow.tracking import MlflowClient

def load_best_model_by_metric(experiment_name, metric_name="f1_score", maximize=True):
    """
    Busca y carga el mejor modelo en un experimento basado en una métrica específica.

    Args:
        experiment_name (str): Nombre del experimento en MLFlow.
        metric_name (str): Nombre de la métrica a evaluar (e.g., "f1_score").
        maximize (bool): True si se desea maximizar la métrica, False para minimizar.

    Returns:
        model: El mejor modelo cargado desde MLFlow.
        run_id: ID del run asociado al mejor modelo.
    """
    # Obtener el ID del experimento
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if not experiment:
        raise ValueError(f"No se encontró el experimento: {experiment_name}")

    experiment_id = experiment.experiment_id

    # Buscar todos los runs del experimento
    runs = mlflow.search_runs(experiment_ids=[experiment_id])

    # Filtrar runs con la métrica especificada
    metric_column = f"metrics.{metric_name}"
    if metric_column not in runs.columns:
        raise ValueError(f"La métrica '{metric_name}' no está registrada en este experimento.")

    valid_runs = runs.dropna(subset=[metric_column])
    if valid_runs.empty:
        raise ValueError(f"No se encontraron runs con la métrica '{metric_name}'.")

    # Encontrar el mejor run
    best_run = valid_runs.sort_values(metric_column, ascending=not maximize).iloc[0]
    best_run_id = best_run["run_id"]

    # Cargar el modelo asociado al mejor run
    model_uri = f"runs:/{best_run_id}/model"
    model = mlflow.sklearn.load_model(model_uri)
    print(f"Mejor modelo cargado desde el run: {best_run_id} con métrica {metric_name} = {best_run[metric_column]}")

    return model, best_run_id
