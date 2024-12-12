from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pandas as pd
import mlflow


def log_preprocessing(data_path:str, output_path:str) -> None:
    """
    Rastrea el paso de preprocesamiento en MLFlow.

    Args:
        data_path (str): Ruta al dataset original.
        output_path (str): Ruta al dataset procesado.
    """
    with mlflow.start_run(run_name="Preprocessing"):
        # Cargar y procesar datos
        data = pd.read_parquet(data_path)

        processed_data = preprocessing(data)
        processed_data.to_csv(output_path, index=False)

        # Registrar métricas y artefactos
        mlflow.log_param("rows_before", data.shape[0])
        mlflow.log_param("columns_before", data.shape[1])
        mlflow.log_param("rows_after", processed_data.shape[0])
        mlflow.log_param("columns_after", processed_data.shape[1])
        mlflow.log_artifact(output_path, artifact_path="data/processed")
        print("Preprocesamiento rastreado en MLFlow.")


def preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Realiza el preprocesamiento de los datos

    Args:
        df (pd.DataFrame): Dataset original
        df_pp (pd.DataFrame): Dataset procesado
    """

    # 1. Definir columnas
    df = df.drop(columns=['wallet_address'])
    categorical_columns = ["wallet_address"]
    timestamp_columns = [
        "borrow_timestamp",
        "first_tx_timestamp",
        "last_tx_timestamp",
        "risky_first_tx_timestamp",
        "risky_last_tx_timestamp",
    ]
    numerical_columns = [
        col for col in df.columns if col not in timestamp_columns + categorical_columns
    ]

    # 2. Configurar ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("date", DateTransformer(), timestamp_columns),
            ("num", Pipeline(steps=[("scaler", StandardScaler())]), numerical_columns),
        ],
        remainder="passthrough",  # Dejar columnas no especificadas
    )

    # 3. Nombres de columnas transformadas
    date_names = [
        f"{col}_{suffix}"
        for col in timestamp_columns
        for suffix in ["year", "month", "day"]
    ]
    transformed_column_names = numerical_columns + date_names

    # 4. Configurar el pipeline
    preprocessor_pipeline = Pipeline(steps=[("preprocessor", preprocessor)])

    # 5. Transformar los datos
    processed_data = preprocessor_pipeline.fit_transform(df)
    processed_data_df = pd.DataFrame(processed_data, columns=transformed_column_names)

    # 6. Cambiar el tipo de datos
    processed_data_df[numerical_columns] = processed_data_df[numerical_columns].apply(
        pd.to_numeric, errors="coerce"
    )  # Convertir a float
    processed_data_df[date_names] = processed_data_df[date_names].apply(
        lambda x: x.astype(int)
    )  # Convertir a int

    # 7. Resultados
    print("Forma Data Procesada:", processed_data_df.shape)
    print("Visualizar Data Procesada:")
    processed_data_df.head()

    return processed_data_df


class DateTransformer(BaseEstimator, TransformerMixin):
    """
    Transformador para convertir columnas de timestamp en componentes de fecha (año, mes, día).
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        transformed = pd.DataFrame()  # Crear un dataframe temporal
        for col in X.columns:
            transformed[f"{col}_year"] = pd.to_datetime(
                X[col], unit="s"
            ).dt.year.astype("Int64")
            transformed[f"{col}_month"] = pd.to_datetime(
                X[col], unit="s"
            ).dt.month.astype("Int64")
            transformed[f"{col}_day"] = pd.to_datetime(X[col], unit="s").dt.day.astype(
                "Int64"
            )
        return transformed.values




