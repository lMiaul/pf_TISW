"""modules/page_modeling.py — RF4: Entrenamiento del Modelo (CRISP-DM: Modeling)
HU-04: Como científico de datos, quiero elegir un modelo y entrenarlo con validación cruzada
estratificada sobre los datos balanceados para obtener una estimación robusta del rendimiento.
"""

import time

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from src.utils.modeling import (
    CLASSIFIERS,
    compute_metrics,
    train_with_cv,
)
from src.utils.session import reset_downstream


def render() -> None:
    st.header("RF4 — Entrenamiento del Modelo")
    st.caption("CRISP-DM · Modeling — validación cruzada estratificada K-Fold")

    if not st.session_state.get("done_resampling"):
        st.warning("Primero completa **RF3 — Configurar remuestreo**.")
        return

    X_train_res = st.session_state["X_train_resampled"]
    y_train_res = st.session_state["y_train_resampled"]
    X_train_base = st.session_state["X_train"]
    y_train_base = st.session_state["y_train"]
    X_test      = st.session_state["X_test"]
    y_test      = st.session_state["y_test"]
    resamp_label = st.session_state.get("resampling_label", "Híbrido")

    # ── Configuración ──────────────────────────────────────────────────────
    col_cfg, col_info = st.columns([1.4, 1])

    with col_cfg:
        st.subheader("Selección del clasificador")

        model_label = st.selectbox(
            "Algoritmo de clasificación",
            list(CLASSIFIERS.keys()),
            index=1,   # Random Forest por defecto
            help="LR = baseline clásico · RF = ensamble · XGBoost = boosting.",
        )
        model_key = CLASSIFIERS[model_label]

        cv_folds = st.slider(
            "Folds de validación cruzada estratificada (K)",
            min_value=3, max_value=10, value=5,
            help="StratifiedKFold garantiza la proporción de clases en cada fold.",
        )
        st.caption("🔒 `random_state = 42` · Partición: `stratify=y` · anti-data-leakage activado")

    with col_info:
        st.subheader("Configuración actual")
        st.metric("Técnica de remuestreo", resamp_label)
        st.metric("Muestras entrenamiento (balanceadas)", f"{len(X_train_res):,}")
        st.metric("Muestras test (originales)",           f"{len(X_test):,}")
        st.metric("Variables predictoras",                len(X_test.columns))

    # ── Entrenar ───────────────────────────────────────────────────────────
    st.divider()
    if st.button("▶ Entrenar modelo", type="primary", use_container_width=True):
        _run_training(
            model_key, model_label, cv_folds,
            X_train_base, y_train_base,
            X_train_res,  y_train_res,
            X_test, y_test,
        )

    # ── Mostrar resultados previos si existen ─────────────────────────────
    if st.session_state.get("done_modeling"):
        _show_cv_results()


def _run_training(
    model_key, model_label, cv_folds,
    X_train_base, y_train_base,
    X_train_res,  y_train_res,
    X_test, y_test,
) -> None:

    progress = st.progress(0, text="Entrenando modelo base (sin remuestreo)...")
    t0 = time.time()

    try:
        # Modelo BASE (sin remuestreo) — sirve como baseline comparativo
        model_base, cv_base, cv_mean_base, cv_std_base = train_with_cv(
            X_train_base, y_train_base, model_key, cv_folds
        )
        progress.progress(40, text="Modelo base completado. Entrenando modelo con remuestreo...")

        metrics_base = compute_metrics(model_base, X_test, y_test, label="Base (sin remuestreo)")

        # Modelo OPTIMIZADO (con remuestreo)
        model_opt, cv_opt, cv_mean_opt, cv_std_opt = train_with_cv(
            X_train_res, y_train_res, model_key, cv_folds
        )
        progress.progress(85, text="Calculando métricas finales...")

        metrics_opt = compute_metrics(model_opt, X_test, y_test, label=f"Con {st.session_state['resampling_label']}")

        elapsed = time.time() - t0
        progress.progress(100, text=f"Completado en {elapsed:.1f}s")

        # Advertencia de tiempo (HU-04 criterio de borde)
        if elapsed > 120:
            st.warning(f"El entrenamiento tardó {elapsed:.0f}s. Considera reducir K o el tamaño del dataset.")

        # Guardar en sesión
        reset_downstream("modeling")
        st.session_state.update({
            "model_base":           model_base,
            "model_optimized":      model_opt,
            "model_label":          model_label,
            "cv_scores_base":       cv_base.tolist(),
            "cv_scores_opt":        cv_opt.tolist(),
            "cv_mean_base":         cv_mean_base,
            "cv_mean_opt":          cv_mean_opt,
            "cv_std_base":          cv_std_base,
            "cv_std_opt":           cv_std_opt,
            "metrics_baseline":     metrics_base,
            "metrics_optimized":    metrics_opt,
            "X_test":               X_test,
            "y_test":               y_test,
            "done_modeling":        True,
        })
        st.success(f"Entrenamiento completado en **{elapsed:.1f}s**. Continúa con **RF5 — Evaluar resultados**.")
        st.rerun()

    except Exception as e:
        progress.empty()
        st.error(f"Error durante el entrenamiento: {e}")


def _show_cv_results() -> None:
    cv_base = st.session_state["cv_scores_base"]
    cv_opt  = st.session_state["cv_scores_opt"]
    label   = st.session_state["model_label"]

    st.subheader(f"Resultados CV — {label}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("AUC-ROC CV Base (media)",       f"{st.session_state['cv_mean_base']:.4f}")
    c2.metric("AUC-ROC CV Base (std)",         f"±{st.session_state['cv_std_base']:.4f}")
    c3.metric("AUC-ROC CV Optimizado (media)", f"{st.session_state['cv_mean_opt']:.4f}",
              delta=f"{st.session_state['cv_mean_opt']-st.session_state['cv_mean_base']:+.4f}")
    c4.metric("AUC-ROC CV Optimizado (std)",   f"±{st.session_state['cv_std_opt']:.4f}")

    # Gráfico de distribución de scores por fold
    fig, ax = plt.subplots(figsize=(8, 3.2))
    x = np.arange(len(cv_base))
    w = 0.35
    ax.bar(x - w/2, cv_base, w, label="Base",      color="#185FA5", alpha=0.85)
    ax.bar(x + w/2, cv_opt,  w, label="Optimizado", color="#0F6E56", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Fold {i+1}" for i in x])
    ax.set_ylabel("AUC-ROC")
    ax.set_title("AUC-ROC por fold — Base vs. Optimizado", fontweight="bold")
    ax.legend()
    ax.set_ylim(max(0, min(cv_base + cv_opt) - 0.05), 1.0)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()
