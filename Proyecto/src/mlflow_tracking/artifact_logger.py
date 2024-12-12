import os
import mlflow
import pandas as pd

def log_artifact_to_mlflow(file_path, artifact_path=None):
    """
    Registra un archivo como artefacto en MLFlow.

    Args:
        file_path (str): Ruta al archivo local que se desea registrar.
        artifact_path (str, optional): Carpeta donde se almacenará el artefacto en MLFlow.
                                       Si es None, se almacena en la raíz.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"El archivo '{file_path}' no existe.")
    
    with mlflow.start_run(run_name=f"Log Artifact: {os.path.basename(file_path)}"):
        mlflow.log_artifact(file_path, artifact_path)
        print(f"Artefacto registrado en MLFlow: {file_path}")


def log_plot_to_mlflow(fig, artifact_name, artifact_path=None):
    """
    Registra un gráfico como artefacto en MLFlow.

    Args:
        fig: Figura de Matplotlib.
        artifact_name (str): Nombre del archivo para el gráfico (e.g., "plot.png").
        artifact_path (str, optional): Carpeta donde se almacenará el artefacto en MLFlow.
                                       Si es None, se almacena en la raíz.
    """
    temp_file = f"temp_{artifact_name}"
    fig.savefig(temp_file)
    log_artifact_to_mlflow(temp_file, artifact_path)
    os.remove(temp_file)


def log_data_to_mlflow(data, file_name, artifact_path=None):
    """
    Registra datos en un archivo CSV como artefacto en MLFlow.

    Args:
        data: DataFrame o cualquier dato convertible a CSV.
        file_name (str): Nombre del archivo CSV a generar.
        artifact_path (str, optional): Carpeta donde se almacenará el artefacto en MLFlow.
    """

    temp_file = f"temp_{file_name}"
    if isinstance(data, pd.DataFrame):
        data.to_csv(temp_file, index=False)
    else:
        raise ValueError("Solo se soportan objetos DataFrame para esta función.")
    
    log_artifact_to_mlflow(temp_file, artifact_path)
    os.remove(temp_file)
