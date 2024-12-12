from src.mlflow_tracking.artifact_logger import log_plot_to_mlflow
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import mlflow

def get_predictions(best_model, X_test, y_test):
    print("Realizando predicciones...")
    y_pred = best_model.predict(X_test)

    # Generar y registrar un gráfico de predicción de ejemplo
    fig, ax = plt.subplots()
    ax.plot(y_test.values[:50], label="Real")
    ax.plot(y_pred[:50], label="Predicción", linestyle="--")
    ax.set_title("Predicción vs Real")
    ax.legend()
    log_plot_to_mlflow(fig, f"prediction_vs_real_{type(best_model).__name__}.png", "plots")
    plt.close(fig)

    # Grafico curva roc
    fpr, tpr, roc_thresholds = roc_curve(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_pred)
    fig2, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC-ROC = {auc_roc:.2f}")
    ax.plot([0, 1], [0, 1], 'k--', label='Baseline (AUC = 0.5)')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Curva ROC')
    ax.legend()
    ax.grid()
    log_plot_to_mlflow(fig2, f"Curva_ROC_{type(best_model).__name__}.png", "plots")
    plt.close(fig2)

