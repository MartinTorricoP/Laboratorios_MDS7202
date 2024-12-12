import mlflow
import mlflow.sklearn
from sklearn.metrics import f1_score, accuracy_score
from mlflow.models import infer_signature
import time

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

        # Registrar parámetros y métricas
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("elapsed_time", elapsed_time)

        y_pred = model.predict(X_test)
        signature = infer_signature(X_train, y_pred)

        # Registrar el modelo
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
