import optuna
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

def get_param_distributions(model):
    """
    Devuelve las distribuciones de hiperparámetros para el modelo dado.

    Args:
        model: Modelo del cual se quieren optimizar los hiperparámetros.

    Returns:
        dict: Diccionario de distribuciones de parámetros para Optuna.
    """
    if isinstance(model, RandomForestClassifier):
        return {
            "n_estimators": optuna.distributions.IntDistribution(50, 300),
            "max_depth": optuna.distributions.IntDistribution(3, 15),
            "min_samples_split": optuna.distributions.IntDistribution(2, 20),
            "min_samples_leaf": optuna.distributions.IntDistribution(1, 10),
        }
    elif isinstance(model, LGBMClassifier):
        return {
            "num_leaves": optuna.distributions.IntDistribution(10, 100),
            "learning_rate": optuna.distributions.FloatDistribution(0.01, 0.3),
            "n_estimators": optuna.distributions.IntDistribution(50, 300),
            "max_depth": optuna.distributions.IntDistribution(3, 15),
        }
    elif isinstance(model, XGBClassifier):
        return {
            "n_estimators": optuna.distributions.IntDistribution(50, 300),
            "max_depth": optuna.distributions.IntDistribution(3, 15),
            "learning_rate": optuna.distributions.FloatDistribution(0.01, 0.3),
            "subsample": optuna.distributions.FloatDistribution(0.5, 1.0),
            #use label encoder
            "use_label_encoder": optuna.distributions.CategoricalDistribution([False]),
            "eval_metric": optuna.distributions.CategoricalDistribution(["auc"]),
        }
    elif isinstance(model, DecisionTreeClassifier):
        return {
            "max_depth": optuna.distributions.IntDistribution(3, 15),
            "min_samples_split": optuna.distributions.IntDistribution(2, 20),
            "min_samples_leaf": optuna.distributions.IntDistribution(1, 10),
        }
    elif isinstance(model, LogisticRegression):
        return {
            "C": optuna.distributions.FloatDistribution(0.01, 10.0, log=True),
            "penalty": optuna.distributions.CategoricalDistribution(["l1", "l2", "elasticnet", "none"]),
        }
    elif isinstance(model, KNeighborsClassifier):
        return {
            "n_neighbors": optuna.distributions.IntDistribution(3, 20),
            "weights": optuna.distributions.CategoricalDistribution(["uniform", "distance"]),
            "p": optuna.distributions.IntDistribution(1, 2),  # Distancia de Minkowski (Manhattan o Euclidiana)
        }
    else:
        raise ValueError(f"No se han definido distribuciones de parámetros para el modelo: {type(model)}")
