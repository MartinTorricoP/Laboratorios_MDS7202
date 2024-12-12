import mlflow
import shap
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import is_classifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

def log_shap_interpretation(model_name, dataset):
    """
    Genera y registra explicaciones SHAP para el modelo registrado en MLFlow.

    Args:
        model_name (str): Nombre del modelo registrado en MLFlow.
        dataset_path (str): Ruta al dataset utilizado para interpretabilidad.
    """
    # Cargar el modelo desde MLFlow
    seed = 123
    model = mlflow.sklearn.load_model(f"models:/{model_name}/latest")

    # Inicializar el explainer SHAP basado en el tipo de modelo
    explainer = None
    if isinstance(model, RandomForestClassifier):
        explainer = shap.TreeExplainer(model)
    elif isinstance(model, XGBClassifier) or isinstance(model, LGBMClassifier):
        explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
    elif is_classifier(model):
        explainer = shap.KernelExplainer(model.predict_proba, dataset)
    else:
        raise ValueError("El modelo cargado no es compatible con SHAP interpretability.")

    # Generar valores de SHAP
    shap_values = explainer.shap_values(dataset)

    # TODO: add more plots and feature importance
    # Crear gráficos de SHAP
    shap.summary_plot(shap_values, dataset, show=False)
    plt.tight_layout()

    # Guardar gráfico como artefacto en MLFlow
    plot_path = "shap_summary_plot.png"
    plt.savefig(plot_path)
    plt.close()

    # Registrar el gráfico en MLFlow
    with mlflow.start_run(run_name="SHAP Interpretability"):
        mlflow.log_artifact(plot_path)
        print(f"Gráfico de interpretabilidad SHAP registrado en MLFlow: {plot_path}")
