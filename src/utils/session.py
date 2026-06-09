"""utils/session.py — Gestión centralizada del estado de sesión (stateless por diseño).
Todas las entidades del modelo lógico se almacenan aquí durante la sesión.
Al cerrar la app, los datos se descartan automáticamente (sin BD persistente).
"""

import streamlit as st

# Claves del modelo lógico en memoria (5 entidades)
_DEFAULTS: dict = {
    # E1 — Dataset
    "dataset_raw":          None,   # pd.DataFrame crudo
    "dataset_processed":    None,   # pd.DataFrame tras preprocesamiento
    "dataset_resampled":    None,   # pd.DataFrame tras remuestreo
    "feature_columns":      [],
    "target_column":        None,
    "class_distribution":   {},
    "filename":             "",
    "done_upload":          False,
    "done_eda":             False,

    # E2 — ResamplingConfig
    "resampling_algorithm": "SMOTE-ENN + Tomek (híbrido)",
    "resampling_ratio":     1.0,
    "random_state":         42,
    "done_resampling":      False,

    # E3 — PredictiveModelConfig
    "model_algorithm":      "Random Forest",
    "cv_folds":             5,
    "model_trained":        None,   # sklearn estimator
    "X_test":               None,
    "y_test":               None,
    "done_modeling":        False,

    # E4 — MetricsEvaluation
    "metrics_baseline":     {},
    "metrics_optimized":    {},
    "cv_scores":            [],
    "confusion_matrices":   {},
    "done_evaluation":      False,

    # E5 — XAIArtifact
    "shap_values":          None,
    "shap_explainer":       None,
    "feature_importance":   {},
    "done_xai":             False,
}


def init_session_state() -> None:
    """Inicializa claves faltantes sin sobreescribir valores existentes."""
    for key, default in _DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = default


def reset_downstream(from_stage: str) -> None:
    """Al cambiar una etapa, invalida todos los resultados posteriores."""
    stages = ["upload", "eda", "resampling", "modeling", "evaluation", "xai"]
    if from_stage not in stages:
        return
    idx = stages.index(from_stage)
    for stage in stages[idx:]:
        st.session_state[f"done_{stage}"] = False
    # Limpia entidades dependientes
    if idx <= 1:
        st.session_state["dataset_processed"] = None
        st.session_state["dataset_resampled"] = None
        st.session_state["model_trained"] = None
        st.session_state["metrics_baseline"] = {}
        st.session_state["metrics_optimized"] = {}
        st.session_state["shap_values"] = None
