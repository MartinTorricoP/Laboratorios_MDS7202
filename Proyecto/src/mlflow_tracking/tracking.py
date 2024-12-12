import mlflow

def configure_mlflow(experiment_name):
    """
    Configura el experimento de MLFlow.

    Args:
        experiment_name (str): Nombre del experimento.

    Returns:
        None
    """
    mlflow.set_experiment(experiment_name)
    print(f"Experimento configurado: {experiment_name}")
