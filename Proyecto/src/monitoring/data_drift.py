import mlflow
from scipy.stats import ks_2samp
import pandas as pd
import matplotlib.pyplot as plt

def log_data_drift(reference_data: pd.DataFrame, new_data: pd.DataFrame, experiment_name: str):
    """
    Calcula y registra data drift en MLFlow usando Kolmogorov-Smirnov.

    Args:
        reference_data (pd.DataFrame): Datos de referencia.
        new_data (pd.DataFrame): Nuevos datos.
        experiment_name (str): Nombre del experimento en MLFlow.
    """
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="Data Drift Check"):
        for col in reference_data.columns:
            # Cálculo de Kolmogorov-Smirnov
            stat, p_value = ks_2samp(reference_data[col], new_data[col])

            # Registrar métricas
            mlflow.log_metric(f"drift_stat_{col}", stat)
            mlflow.log_metric(f"drift_p_value_{col}", p_value)

            # Generar histogramas para la comparación
            plt.figure(figsize=(8, 5))
            plt.hist(reference_data[col], bins=30, alpha=0.5, label="Reference")
            plt.hist(new_data[col], bins=30, alpha=0.5, label="New Data")
            plt.title(f"Distribution of {col}")
            plt.legend()
            plt.tight_layout()

            # Guardar gráfico como artefacto
            plot_path = f"artifacts/plots/distribution_{col}.png"
            plt.savefig(plot_path)
            mlflow.log_artifact(plot_path)
            plt.close()

        print("Data drift registrado en MLFlow.")
