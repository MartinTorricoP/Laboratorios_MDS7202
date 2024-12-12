# app/backend/Dockerfile

FROM python:3.9-slim

WORKDIR /app

# Instalar dependencias
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copiar la aplicación al contenedor
ENTRYPOINT mlflow server --default-artifact-root ./mlruns
COPY . .

# Comando para ejecutar FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
