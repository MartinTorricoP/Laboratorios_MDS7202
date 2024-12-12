from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import os
from src.mlflow_tracking.pp_tracking import log_preprocessing
from src.monitoring.monitoring_pipeline import log_model_performance
from src.retraining.retrain import retrain_model
from src.mlflow_tracking.interpretability import log_shap_interpretation
from airflow_pp.dags.fetch_data import fetch_files_from_gitlab

# Configuración del DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id='credit_risk_pipeline',
    default_args=default_args,
    description='Pipeline productiva para monitorear y reentrenar el modelo de riesgo crediticio',
    schedule_interval=timedelta(days=1),
    start_date=datetime(2023, 1, 1),
    catchup=False,
) as dag:

    # Tarea 1: Extracción de datos
    def extract_data():
        repo_url = "https://gitlab.com/MDS7202/Proyecto-MDS7202"
        branch_name = "main"
        # load env variable
        token = os.getenv("GITLAB_TOKEN")
        target_folder = "data/raw"
        fetch_files_from_gitlab(repo_url, branch_name, token, target_folder)
    
    extract_task = PythonOperator(
        task_id='extract_data',
        python_callable=extract_data,
    )

    # Tarea 2: Preprocesamiento
    def preprocess():
        log_preprocessing(
            raw_data_path="data/raw/new_data.csv",
            processed_data_path="data/processed/new_data.csv"
        )

    preprocess_task = PythonOperator(
        task_id='preprocess_data',
        python_callable=preprocess,
    )

    # Tarea 3: Monitoreo de data drift
    def monitor():
        log_model_performance(
            reference_data_path="data/processed/clean_data.csv",
            new_data_path="data/processed/new_data.csv",
            y_true_path="data/processed/y_true.csv"
        )

    monitor_task = PythonOperator(
        task_id='monitor_data',
        python_callable=monitor,
    )

    # Tarea 4: Reentrenamiento del modelo
    def retrain():
        retrain_model(
            reference_data_path="data/processed/clean_data.csv",
            new_data_path="data/processed/new_data.csv",
            y_true_path="data/processed/y_true.csv",
            experiment_name="Credit Risk Analysis"
        )

    retrain_task = PythonOperator(
        task_id='retrain_model',
        python_callable=retrain,
    )
    # TODO: add predict
    
    # Tarea 5: Interpretabilidad con SHAP
    def interpret():
        log_shap_interpretation(
            model_name="Best_Model",
            dataset_path="data/processed/new_data.csv"
        )

    interpret_task = PythonOperator(
        task_id='log_interpretability',
        python_callable=interpret,
    )

    # flujo de tareas
    extract_task >> preprocess_task >> monitor_task >> retrain_task >> interpret_task
