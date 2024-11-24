# Archivo main.py

from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
import pickle
import numpy as np
from xgboost import DMatrix

# Cargar el modelo entrenado
try:
    with open("models/best_model_global.pkl", "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    raise Exception("El archivo 'best_model_global.pkl' no se encontró. Por favor, asegúrate de haber entrenado el modelo correctamente.")

# Crear una instancia de la aplicación FastAPI
app = FastAPI(
    title="API REST para Modelo de Potabilidad de Agua",
    description="API que utiliza un modelo XGBoost para predecir si una medición de agua es potable o no.",
    version="1.0.0",
)

# Definir un esquema para los datos de entrada usando Pydantic
class WaterQualityData(BaseModel):
    ph: float
    Hardness: float
    Solids: float
    Chloramines: float
    Sulfate: float
    Conductivity: float
    Organic_carbon: float
    Trihalomethanes: float
    Turbidity: float

# Ruta de bienvenida
@app.get("/")
def home():
    return {
        "message": "Bienvenido a la API para predicción de potabilidad de agua :)",
        "description": (
            "Este modelo utiliza datos de características químicas y físicas del agua "
            "para predecir si es potable (1) o no potable (0)."
        ),
        "input_example": {
            "ph": 10.316400384553162,
            "Hardness": 217.2668424334475,
            "Solids": 10676.508475429378,
            "Chloramines": 3.445514571005745,
            "Sulfate": 397.7549459751925,
            "Conductivity": 492.20647361771086,
            "Organic_carbon": 12.812732207582542,
            "Trihalomethanes": 72.28192021570328,
            "Turbidity": 3.4073494284238364,
        },
        "output_example": {"potabilidad": 0},
    }

# Ruta para predecir potabilidad
@app.post("/potabilidad/")
def predict_potability(
    data: WaterQualityData = Body(
        example={
            "ph": 10.316400384553162,
            "Hardness": 217.2668424334475,
            "Solids": 10676.508475429378,
            "Chloramines": 3.445514571005745,
            "Sulfate": 397.7549459751925,
            "Conductivity": 492.20647361771086,
            "Organic_carbon": 12.812732207582542,
            "Trihalomethanes": 72.28192021570328,
            "Turbidity": 3.4073494284238364,
        }
    )
):
    try:
        # Crear una matriz con los valores de entrada
        input_values = np.array([
            data.ph,
            data.Hardness,
            data.Solids,
            data.Chloramines,
            data.Sulfate,
            data.Conductivity,
            data.Organic_carbon,
            data.Trihalomethanes,
            data.Turbidity,
        ]).reshape(1, -1)

        # Nombres de las columnas esperadas por el modelo
        feature_names = [
            "ph",
            "Hardness",
            "Solids",
            "Chloramines",
            "Sulfate",
            "Conductivity",
            "Organic_carbon",
            "Trihalomethanes",
            "Turbidity"
        ]

        # Convertir a DMatrix con nombres de características
        dmatrix_input = DMatrix(input_values, feature_names=feature_names)

        # Realizar la predicción de probabilidad
        probability = model.predict(dmatrix_input)[0]

        # Decidir la clase basada en un umbral (aproximamos)
        prediction = 1 if probability >= 0.5 else 0

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al procesar los datos de entrada: {str(e)}")

    return {"potabilidad": int(prediction)}


# Para ejecutar el servidor:
# Ejecuta este archivo con `python main.py` y accede a http://127.0.0.1:8000/docs para probar la API.
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
