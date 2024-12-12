from src.mlflow_tracking.artifact_logger import log_plot_to_mlflow
import matplotlib.pyplot as plt

def get_predictions(best_model, X_test, y_test):
    print("Realizando predicciones...")
    y_pred = best_model.predict(X_test)

    # Generar y registrar un gráfico de predicción de ejemplo
    fig, ax = plt.subplots()
    ax.plot(y_test.values[:50], label="Real")
    ax.plot(y_pred[:50], label="Predicción", linestyle="--")
    ax.set_title("Predicción vs Real")
    ax.legend()
    log_plot_to_mlflow(fig, f"prediction_vs_real_{type(best_model)}.png", "plots")

    plt.close(fig)
