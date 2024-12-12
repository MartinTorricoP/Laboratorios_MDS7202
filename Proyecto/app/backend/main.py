from fastapi import FastAPI, UploadFile, Form
import pandas as pd
import mlflow
import json
import os
from model_loader import load_best_model_by_metric
from pydantic import create_model

experiment_name = "BCI Riesgo"
def generate_input_model(columns):
    """
    Genera dinámicamente un modelo de entrada para FastAPI basado en las columnas.

    Args:
        columns (list): Lista de nombres de columnas esperadas.

    Returns:
        BaseModel: Clase generada dinámicamente.
    """
    fields = {col: (float, ...) for col in columns}  
    InputData = create_model("InputData", **fields)
    return InputData

# Cargar el mejor modelo desde MLFlow
print("Cargando el modelo desde MLFlow...")
best_model, best_run_id = load_best_model_by_metric(
    experiment_name=experiment_name,
    metric_name="f1_score",
    maximize=True
)

mlflow.artifacts.download_artifacts(run_id=best_run_id, artifact_path="model_metadata/columns.json")

# Cargar las columnas
with open("columns.json", "r") as f:
    feature_names = json.load(f)

print(feature_names)
InputData = generate_input_model(feature_names)

# Crear la app de FastAPI
app = FastAPI()

@app.post("/predict")
async def predict_file(file: UploadFile):
    """
    Realiza predicciones a partir de un archivo .csv.
    """
    data = pd.read_csv(file.file)
    predictions = best_model.predict(data)
    return {"predictions": predictions.tolist()}


@app.post("/predict_manual")
async def predict_manual(data: InputData):
    """
    Realiza predicciones a partir de datos ingresados manualmente.
    """
    # Convertir datos a DataFrame
    input_df = pd.DataFrame([data.dict()])

    # Hacer predicción
    prediction = best_model.predict(input_df)
    return {"prediction": prediction[0]}
