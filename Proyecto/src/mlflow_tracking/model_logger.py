import mlflow
import mlflow.sklearn
from sklearn.metrics import f1_score, accuracy_score, precision_recall_curve, auc
from mlflow.models import infer_signature
import numpy as np
import time
import json

def log_model_with_mlflow(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    run_name,
    params,
    register_model=False,
):
    """
    Registra el modelo, métricas y parámetros en MLFlow. Opcionalmente, registra el modelo en el Model Registry.

    Args:
        model: Modelo a registrar.
        X_train, y_train: Conjunto de entrenamiento.
        X_test, y_test: Conjunto de prueba.
        run_name (str): Nombre del run en MLFlow.
        params (dict): Hiperparámetros del modelo.
        register_model (bool): Si True, registra el modelo en el Model Registry.
    """
    with mlflow.start_run(run_name=run_name):
        start_time = time.time()
        # Entrenar el modelo
        model.fit(X_train, y_train)
        # Predicciones y métricas
        y_pred = model.predict(X_test)

        elapsed_time = time.time() - start_time
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
        auc_pr = auc(recall, precision)
        metrics = {
            "elapsed_time": float(elapsed_time),
            "accuracy": float(accuracy),
            "precision": float(np.mean(precision)),
            "recall": float(np.mean(recall)),
            "f1_score": float(f1),
            "auc_pr": float(auc_pr)
        }

        # Registrar parámetros y métricas
        mlflow.log_metrics(metrics)
        mlflow.log_params(params)
        signature = infer_signature(X_train, y_pred)

        # Registrar el modelo
        with open("columns.json", "w") as f:
            json.dump(list(X_train.columns), f)
        mlflow.log_artifact("columns.json", artifact_path="model_metadata")

        input_example = X_train.iloc[[0]]
        mlflow.sklearn.log_model(sk_model=model,
                                 artifact_path="model",
                                 input_example=input_example,
                                 registered_model_name=f"{type(model).__name__}_model",
                                 signature=signature,
                                 )

        print(f"Modelo registrado en MLFlow: {mlflow.active_run().info.run_id}")
        model_name = type(model).__name__

        # Registrar en el Model Registry si se especifica
        if register_model:
            if not model_name:
                raise ValueError(
                    "Debe proporcionar 'model_name' para registrar el modelo en el Model Registry."
                )
            model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
            mlflow.register_model(model_uri, model_name)
            print(
                f"Modelo registrado en el Model Registry con el nombre '{model_name}'."
            )
