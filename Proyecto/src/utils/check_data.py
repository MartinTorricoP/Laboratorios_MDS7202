import os
from src.mlflow_tracking.pp_tracking import log_preprocessing

def check_or_create_processed_data(raw_path, processed_path):
    """
    Verifica si existen datos procesados en la carpeta `processed`.
    Si no existen, corre el preprocesamiento para generarlos.
    """

    if not os.path.exists(processed_path):
        print(f"No se encontraron datos procesados en {processed_path}. Ejecutando preprocesamiento...")
        log_preprocessing(data_path=raw_path, output_path=processed_path)
        print("Preprocesamiento completado. Datos procesados guardados.")
    else:
        print(f"Datos procesados encontrados en {processed_path}.")