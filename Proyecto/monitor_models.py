from src.monitoring.monitoring_pipeline import monitoring_pipeline

monitoring_pipeline(
    reference_data_path="data/processed/clean_data.csv",
    # path de los nuevos datos
    new_data_path="data/processed/new_data.csv",
    # path de las etiquetas de los nuevos datos
    y_true_path="data/processed/y_true.csv",
    experiment_name="Credit Risk Monitoring"
)

