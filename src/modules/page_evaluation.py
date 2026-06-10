"""modules/page_evaluation.py — RF5: Evaluación Comparativa (CRISP-DM: Evaluation)
HU-05: Como científico de datos, quiero un panel comparativo de métricas antes y después
del remuestreo para cuantificar el impacto real de la técnica de balanceo.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import RocCurveDisplay, roc_curve

from src.utils.modeling import compute_delta


def render() -> None:
    st.header("RF5 — Evaluación Comparativa de Métricas")
    st.caption("CRISP-DM · Evaluation — verificación de KPIs del proyecto")

    if not st.session_state.get("done_modeling"):
        st.warning("Primero completa **RF4 — Entrenar modelo**.")
        return

    m_base  = st.session_state["metrics_baseline"]
    m_opt   = st.session_state["metrics_optimized"]
    delta   = compute_delta(m_base, m_opt)
    label   = st.session_state.get("model_label", "Modelo")
    X_test  = st.session_state["X_test"]
    y_test  = st.session_state["y_test"]
    model_b = st.session_state["model_base"]
    model_o = st.session_state["model_optimized"]

    # ── KPI compliance (criterios de Business Understanding) ──────────────
    st.subheader("Verificación de KPIs del proyecto")
    kpis = {
        "AUC-ROC ≥ 0.85":      m_opt.get("auc_roc", 0) >= 0.85,
        "F1-score ≥ 0.75":      m_opt.get("f1",      0) >= 0.75,
        "Recall ≥ 0.80":        m_opt.get("recall",  0) >= 0.80,
    }
    k_cols = st.columns(3)
    for col, (kpi_name, passed) in zip(k_cols, kpis.items()):
        icon = "✅" if passed else "❌"
        col.metric(f"{icon} {kpi_name}", "Cumplido" if passed else "No cumplido")

    # Alerta si Recall base ya es alto (HU-05 borde)
    if m_base.get("recall", 0) >= 0.85:
        st.info(
            "El Recall del modelo base ya supera 0.85. "
            "El remuestreo puede no ser necesario para este dataset."
        )

    st.divider()

    # ── Tabla comparativa de métricas ─────────────────────────────────────
    st.subheader("Tabla comparativa de métricas")
    _show_metrics_table(m_base, m_opt, delta)

    st.divider()

    # ── Matrices de confusión ─────────────────────────────────────────────
    st.subheader("Matrices de confusión")
    col1, col2 = st.columns(2)
    with col1:
        _plot_confusion_matrix(m_base["confusion_matrix"], "Base (sin remuestreo)", "#185FA5")
    with col2:
        _plot_confusion_matrix(m_opt["confusion_matrix"],  f"Con {st.session_state.get('resampling_label','Remuestreo')}", "#0F6E56")

    st.divider()

    # ── Curvas ROC ────────────────────────────────────────────────────────
    st.subheader("Curvas ROC — Base vs. Optimizado")
    _plot_roc_curves(model_b, model_o, X_test, y_test)

    # ── Marcar evaluación completa ────────────────────────────────────────
    if not st.session_state.get("done_evaluation"):
        st.session_state["done_evaluation"] = True

    st.divider()
    if st.button("Continuar → RF6 — Explicabilidad (XAI)", type="primary", use_container_width=True):
        st.rerun()


# ── Helpers ────────────────────────────────────────────────────────────────

def _show_metrics_table(m_base: dict, m_opt: dict, delta: dict) -> None:
    metric_keys   = ["accuracy", "precision", "recall", "f1", "auc_roc", "g_mean"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1-score", "AUC-ROC", "G-mean"]

    rows = []
    for key, label in zip(metric_keys, metric_labels):
        d = delta.get(key, 0)
        rows.append({
            "Métrica":       label,
            "Base":          f"{m_base.get(key, 0):.4f}",
            "Con Remuestreo": f"{m_opt.get(key, 0):.4f}",
            "Δ Mejora":      f"{d:+.4f}",
            "_delta":        d,
        })

    df = pd.DataFrame(rows)

    def _color_delta(val):
        try:
            v = float(val)
            if v > 0:   return "color: #0F6E56; font-weight: 600"
            elif v < 0: return "color: #993C1D; font-weight: 600"
        except Exception:
            pass
        return ""

    styled = (
        df.drop(columns=["_delta"])
        .style.map(_color_delta, subset=["Δ Mejora"])
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)


def _plot_confusion_matrix(cm_list: list, title: str, color: str) -> None:
    cm = np.array(cm_list)
    fig, ax = plt.subplots(figsize=(4, 3.5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    classes = ["Buen crédito (0)", "Impago (1)"]
    tick_marks = np.arange(len(classes))
    ax.set(
        xticks=tick_marks, yticks=tick_marks,
        xticklabels=classes, yticklabels=classes,
        title=title, ylabel="Real", xlabel="Predicho",
    )
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right", fontsize=8)
    plt.setp(ax.get_yticklabels(), fontsize=8)
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]:,}",
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontweight="bold", fontsize=11)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()


def _plot_roc_curves(model_b, model_o, X_test, y_test) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for model, label, color in [
        (model_b, "Base (sin remuestreo)",  "#185FA5"),
        (model_o, "Con remuestreo", "#0F6E56"),
    ]:
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        from sklearn.metrics import auc
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{label} (AUC = {roc_auc:.4f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Clasificador aleatorio")
    ax.set_xlabel("Tasa de Falsos Positivos")
    ax.set_ylabel("Tasa de Verdaderos Positivos (Recall)")
    ax.set_title("Curvas ROC — Base vs. Optimizado", fontweight="bold")
    ax.legend(loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()
