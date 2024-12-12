import gradio as gr
import requests
import pandas as pd

# URL del backend
BACKEND_URL = "http://localhost:8000"

# Función para predicción con archivo .csv
def predict_csv(file):
    response = requests.post(f"{BACKEND_URL}/predict", files={"file": file})
    return response.json()["predictions"]

# Función para predicción manual
def predict_manual(feature_1, feature_2):
    
    data = {"feature_1": feature_1, "feature_2": feature_2}
    response = requests.post(f"{BACKEND_URL}/predict_manual", json=data)
    return response.json()["prediction"]

# Crear interfaz de Gradio
with gr.Blocks() as demo:
    gr.Markdown("## Predicción de Crédito - Interfaz")
    
    # Opción 1: Cargar archivo CSV
    with gr.Row():
        file_input = gr.File(label="Subir archivo CSV")
        csv_output = gr.Textbox(label="Predicciones")
        csv_button = gr.Button("Predecir desde archivo")
        csv_button.click(predict_csv, inputs=file_input, outputs=csv_output)
    
    # Opción 2: Ingreso manual
    with gr.Row():
        feature_1_input = gr.Number(label="Feature 1")
        feature_2_input = gr.Number(label="Feature 2")
        manual_output = gr.Textbox(label="Predicción")
        manual_button = gr.Button("Predecir manualmente")
        manual_button.click(
            predict_manual, 
            inputs=[feature_1_input, feature_2_input], 
            outputs=manual_output
        )

# Ejecutar la interfaz
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)

