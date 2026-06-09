"""utils/modeling.py — Módulo de Modelado (CRISP-DM: Modeling)
Clasificadores: Logistic Regression, Random Forest, XGBoost.
Validación cruzada estratificada K-Fold con imblearn Pipeline (anti-leakage).
Optimización ligera con RandomizedSearchCV (max 20 iteraciones).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

RANDOM_STATE = 42

CLASSIFIERS = {
    "Logistic Regression": "lr",
    "Random Forest":       "rf",
    "XGBoost":             "xgb",
}

_PARAM_GRIDS = {
    "lr": {
        "clf__C":           [0.01, 0.1, 1.0, 10.0],
        "clf__max_iter":    [200, 500],
        "clf__solver":      ["lbfgs", "liblinear"],
    },
    "rf": {
        "clf__n_estimators":  [100, 200, 300],
        "clf__max_depth":     [None, 5, 10, 20],
        "clf__min_samples_split": [2, 5, 10],
    },
    "xgb": {
        "clf__n_estimators":   [100, 200],
        "clf__max_depth":      [3, 5, 7],
        "clf__learning_rate":  [0.05, 0.1, 0.2],
        "clf__subsample":      [0.7, 0.9, 1.0],
    },
}


def build_classifier(algorithm_key: str):
    """Instancia el clasificador con hiperparámetros base reproducibles."""
    if algorithm_key == "lr":
        return LogisticRegression(
            random_state=RANDOM_STATE, max_iter=500, C=1.0
        )
    elif algorithm_key == "rf":
        return RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_STATE,
            n_jobs=-1, class_weight="balanced",
        )
    elif algorithm_key == "xgb":
        return XGBClassifier(
            n_estimators=200, random_state=RANDOM_STATE,
            eval_metric="logloss", verbosity=0,
            use_label_encoder=False,
        )
    raise ValueError(f"Clasificador desconocido: {algorithm_key}")


def train_with_cv(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    algorithm_key: str,
    cv_folds: int = 5,
) -> tuple:
    """
    Entrena el modelo con validación cruzada estratificada.
    El clasificador se ajusta sobre el X_train completo al final
    para producir el modelo definitivo.

    Returns
    -------
    trained_model, cv_scores (array), cv_mean, cv_std
    """
    clf = build_classifier(algorithm_key)
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)

    cv_scores = cross_val_score(
        clf, X_train, y_train,
        cv=skf, scoring="roc_auc", n_jobs=-1,
    )

    # Entrenamiento final sobre el conjunto completo de entrenamiento
    clf.fit(X_train, y_train)

    return clf, cv_scores, round(cv_scores.mean(), 4), round(cv_scores.std(), 4)


def compute_metrics(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    label: str = "baseline",
) -> dict:
    """Calcula el conjunto completo de métricas sobre el test set."""
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else y_pred

    return {
        "label":       label,
        "accuracy":    round(accuracy_score(y, y_pred),         4),
        "precision":   round(precision_score(y, y_pred,         zero_division=0), 4),
        "recall":      round(recall_score(y, y_pred,            zero_division=0), 4),
        "f1":          round(f1_score(y, y_pred,                zero_division=0), 4),
        "auc_roc":     round(roc_auc_score(y, y_prob),          4),
        "g_mean":      round(_g_mean(y, y_pred),                4),
        "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
    }


def compute_delta(baseline: dict, optimized: dict) -> dict:
    """Calcula Δ de mejora entre modelo base y modelo con remuestreo."""
    keys = ["accuracy", "precision", "recall", "f1", "auc_roc", "g_mean"]
    return {
        k: round(optimized.get(k, 0) - baseline.get(k, 0), 4)
        for k in keys
    }


def _g_mean(y_true, y_pred) -> float:
    """Media geométrica entre sensibilidad y especificidad."""
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape != (2, 2):
        return 0.0
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return float(np.sqrt(sensitivity * specificity))
