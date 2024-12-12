# Variables de entorno
MLFLOW_PORT := 5000
AIRFLOW_PORT := 8080
APP_BACKEND_PORT := 8000
APP_FRONTEND_PORT := 7860
DOCKER_COMPOSE_APP := app/docker-compose.yml
DOCKER_COMPOSE_AIRFLOW := docker-compose-airflow.yml
PYTHON := python

.PHONY: help mlflow airflow app-backend app-frontend stop-all

help:
	@echo "Comandos disponibles:"
	@echo "  make mlflow            - Inicia el servidor de MLFlow"
	@echo "  make pipeline_mlflow   - Corre pipeline entrenamiento MLFlow"
	@echo "  make airflow_pp        - Inicia el DAG de Airflow"
	@echo "  make app_web       	- Inicia la aplicación web"
	@echo "  make stop-all          - Detiene todos los servicios"

# Iniciar MLFlow
mlflow:
	@echo "Iniciando MLFlow en el puerto $(MLFLOW_PORT)..."
	mlflow ui 

# Ejecutar el pipeline principal mlflow
pipeline_mlflow:
	@echo "Ejecutando el pipeline principal de MLFlow..."
	$(PYTHON) main_pipeline.py

# Iniciar Airflow
airflow_pp:
	@echo "Iniciando Apache Airflow..."
	docker-compose -f $(DOCKER_COMPOSE_AIRFLOW) up --build

# Iniciar aplicación
app_web:
	@echo "Iniciando el backend de la aplicación..."
	docker-compose -f $(DOCKER_COMPOSE_APP) up --build

# Detener todos los servicios
stop-all:
	@echo "Deteniendo todos los servicios..."
	docker-compose --file $(DOCKER_COMPOSE_AIRFLOW) down
	docker-compose --file $(DOCKER_COMPOSE_APP) down
