"""modules/page_xai.py — RF6: Explicabilidad XAI (CRISP-DM: Evaluation → Deployment)
HU-06: Como analista de riesgo financiero, quiero ver qué variables determinan la predicción
de riesgo, a nivel global del modelo y para un cliente específico, para fundamentar
decisiones ante auditorías regulatorias.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# SHAP se importa de forma diferida para no bloquear el arranque si no está disponible
try:
    import shap
    _SHAP_AVAILABLE = True
except ImportError:
    _SHAP_AVAILABLE = False

try:
    from lime.lime_tabular import LimeTabularExplainer
    _LIME_AVAILABLE = True
except ImportError:
    _LIME_AVAILABLE = False


def render() -> None:
    st.header("RF6 — Explicabilidad del Modelo (XAI)")
    st.caption("CRISP-DM · Evaluation — SHAP global · LIME local · Decisiones auditables")

    if not st.session_state.get("done_modeling"):
        st.warning("Primero completa **RF4 — Entrenar modelo**.")
        return

    model       = st.session_state["model_optimized"]
    X_test      = st.session_state["X_test"]
    y_test      = st.session_state["y_test"]
    feat_cols   = st.session_state["feature_columns"]
    model_label = st.session_state.get("model_label", "Modelo")

    # Detectar método disponible y compatible
    method = _detect_xai_method(model)

    st.info(
        f"**Modelo explicado:** {model_label} (con remuestreo)  \n"
        f"**Método XAI:** {method}  \n"
        f"**Registros de test:** {len(X_test):,}  \n"
        f"**Variables:** {len(feat_cols)}"
    )

    # ── Generar explicaciones ─────────────────────────────────────────────
    if not st.session_state.get("done_xai"):
        if st.button("▶ Generar explicaciones XAI", type="primary", use_container_width=True):
            _compute_xai(model, X_test, feat_cols, method)
    else:
        st.success("Explicaciones XAI generadas. Explora los paneles a continuación.")

    if not st.session_state.get("done_xai"):
        return

    tab1, tab2, tab3 = st.tabs([
        "🌐 Importancia global",
        "🔍 Explicación local (por cliente)",
        "📋 Tabla de valores SHAP",
    ])

    with tab1:
        _tab_global_importance(feat_cols)

    with tab2:
        _tab_local_explanation(model, X_test, y_test, feat_cols, method)

    with tab3:
        _tab_shap_table(feat_cols)

    # Marcar pipeline completo
    if not st.session_state.get("done_xai"):
        st.session_state["done_xai"] = True


# ── Cómputo XAI ────────────────────────────────────────────────────────────

def _detect_xai_method(model) -> str:
    """Elige SHAP TreeExplainer si el modelo lo soporta, si no usa LIME."""
    tree_models = ("RandomForestClassifier", "XGBClassifier",
                   "GradientBoostingClassifier", "ExtraTreesClassifier")
    model_name = type(model).__name__
    if _SHAP_AVAILABLE and model_name in tree_models:
        return "SHAP (TreeExplainer)"
    elif _LIME_AVAILABLE:
        st.warning(
            f"El modelo **{model_name}** no es compatible con SHAP TreeExplainer. "
            "Se usará **LIME** como método alternativo."
        )
        return "LIME"
    else:
        return "Importancia de características (built-in)"


def _compute_xai(model, X_test: pd.DataFrame, feat_cols: list, method: str) -> None:
    with st.spinner(f"Calculando {method}... (puede tardar hasta 10 segundos)"):
        try:
            if "SHAP" in method and _SHAP_AVAILABLE:
                explainer   = shap.TreeExplainer(model)
                # Limitar a 200 muestras para mantener tiempo < 10s (HU-06)
                X_sample    = X_test.iloc[:min(200, len(X_test))]
                shap_values = explainer.shap_values(X_sample)

                # Para clasificadores binarios, shap_values puede ser lista [clase0, clase1]
                if isinstance(shap_values, list):
                    shap_vals = shap_values[1]   # clase 1 = impago
                else:
                    shap_vals = shap_values

                global_importance = dict(
                    zip(feat_cols, np.abs(shap_vals).mean(axis=0))
                )

                st.session_state.update({
                    "shap_explainer":     explainer,
                    "shap_values":        shap_vals,
                    "shap_X_sample":      X_sample,
                    "feature_importance": global_importance,
                    "xai_method":         method,
                    "done_xai":           True,
                })

            elif "LIME" in method and _LIME_AVAILABLE:
                explainer = LimeTabularExplainer(
                    X_test.values,
                    feature_names=feat_cols,
                    class_names=["Buen crédito", "Impago"],
                    mode="classification",
                    random_state=42,
                )
                # Importancia global aproximada via media de 20 muestras aleatorias
                importances = np.zeros(len(feat_cols))
                sample_idx  = np.random.RandomState(42).choice(len(X_test), min(20, len(X_test)), replace=False)
                for idx in sample_idx:
                    exp = explainer.explain_instance(
                        X_test.values[idx],
                        model.predict_proba,
                        num_features=len(feat_cols),
                    )
                    for feat, weight in exp.as_list():
                        for i, col in enumerate(feat_cols):
                            if col in feat:
                                importances[i] += abs(weight)
                importances /= len(sample_idx)

                st.session_state.update({
                    "lime_explainer":     explainer,
                    "feature_importance": dict(zip(feat_cols, importances)),
                    "xai_method":         method,
                    "done_xai":           True,
                })

            else:
                # Fallback: importancia built-in del modelo (RF / XGB)
                if hasattr(model, "feature_importances_"):
                    importance = dict(zip(feat_cols, model.feature_importances_))
                else:
                    importance = {f: 1/len(feat_cols) for f in feat_cols}

                st.session_state.update({
                    "feature_importance": importance,
                    "xai_method":         "Importancia de características (built-in)",
                    "done_xai":           True,
                })

            st.success("Explicaciones generadas correctamente.")
            st.rerun()

        except Exception as e:
            st.error(f"Error al generar explicaciones XAI: {e}")


# ── Tabs de visualización ──────────────────────────────────────────────────

def _tab_global_importance(feat_cols: list) -> None:
    importance = st.session_state.get("feature_importance", {})
    if not importance:
        st.info("No hay datos de importancia disponibles.")
        return

    # Top-K configurable (HU-06: por defecto top-10)
    top_k = st.slider("Número de variables a mostrar", 5, min(30, len(importance)), 10, key="top_k_global")

    sorted_feats = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:top_k]
    feats, vals  = zip(*sorted_feats)

    fig, ax = plt.subplots(figsize=(8, max(3.5, top_k * 0.38)))
    colors = plt.cm.RdYlGn_r(np.linspace(0.15, 0.85, len(feats)))
    bars = ax.barh(feats[::-1], vals[::-1], color=colors[::-1], edgecolor="white", linewidth=0.8)

    for bar, val in zip(bars, vals[::-1]):
        ax.text(bar.get_width() + max(vals) * 0.01, bar.get_y() + bar.get_height()/2,
                f"{val:.4f}", va="center", fontsize=9)

    method = st.session_state.get("xai_method", "XAI")
    ax.set_title(f"Top-{top_k} variables por importancia global — {method}", fontweight="bold")
    ax.set_xlabel("Importancia media |SHAP|")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()

    # SHAP summary plot si disponible
    if _SHAP_AVAILABLE and st.session_state.get("shap_values") is not None:
        st.subheader("SHAP Summary Plot")
        shap_vals  = st.session_state["shap_values"]
        X_sample   = st.session_state["shap_X_sample"]

        fig2, ax2 = plt.subplots(figsize=(9, max(4, top_k * 0.38)))
        shap.summary_plot(
            shap_vals[:, :top_k],
            X_sample.iloc[:, :top_k],
            feature_names=list(feat_cols[:top_k]),
            show=False, plot_size=None,
        )
        st.pyplot(plt.gcf())
        plt.close("all")


def _tab_local_explanation(model, X_test, y_test, feat_cols, method) -> None:
    st.subheader("Explicación local por registro de cliente")
    st.caption(
        "Selecciona un registro del conjunto de prueba para ver qué variables "
        "influyeron en la predicción de impago para ese cliente específico."
    )

    idx = st.number_input(
        "Índice del registro (0 = primer cliente del test set)",
        min_value=0, max_value=len(X_test) - 1, value=0, step=1,
        key="local_idx",
    )

    row      = X_test.iloc[idx]
    y_true   = int(y_test.iloc[idx])
    y_pred   = int(model.predict(row.values.reshape(1, -1))[0])
    y_prob   = float(model.predict_proba(row.values.reshape(1, -1))[0][1])

    # Ficha del cliente
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Registro #",           idx)
    col2.metric("Clase real",           "Impago (1)" if y_true == 1 else "Buen crédito (0)")
    col3.metric("Predicción",           "Impago (1)" if y_pred == 1 else "Buen crédito (0)",
                delta="✅ Correcta" if y_true == y_pred else "❌ Incorrecta",
                delta_color="normal" if y_true == y_pred else "inverse")
    col4.metric("P(Impago)",            f"{y_prob:.4f}")

    # Visualización local
    if _SHAP_AVAILABLE and st.session_state.get("shap_values") is not None:
        shap_vals  = st.session_state["shap_values"]
        X_sample   = st.session_state["shap_X_sample"]
        explainer  = st.session_state["shap_explainer"]

        if idx < len(shap_vals):
            st.subheader("SHAP Waterfall — contribución de cada variable")
            fig, ax = plt.subplots(figsize=(9, 5))
            shap_exp = shap.Explanation(
                values         = shap_vals[idx],
                base_values    = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value,
                data           = X_sample.iloc[idx].values,
                feature_names  = feat_cols,
            )
            shap.waterfall_plot(shap_exp, max_display=15, show=False)
            st.pyplot(plt.gcf())
            plt.close("all")
        else:
            _local_bar_fallback(row, feat_cols, idx)

    elif "LIME" in method and _LIME_AVAILABLE and st.session_state.get("lime_explainer"):
        st.subheader("LIME — contribución de cada variable")
        lime_exp = st.session_state["lime_explainer"].explain_instance(
            row.values, model.predict_proba, num_features=15,
        )
        fig = lime_exp.as_pyplot_figure()
        fig.set_size_inches(9, 5)
        st.pyplot(fig)
        plt.close()

    else:
        _local_bar_fallback(row, feat_cols, idx)

    # Tabla de valores del cliente — nombres de columna exactos del CSV (HU-06 criterio 3)
    with st.expander("Valores del registro seleccionado"):
        st.dataframe(
            row.reset_index().rename(columns={"index": "Variable", idx: "Valor"}),
            use_container_width=True, hide_index=True,
        )


def _local_bar_fallback(row: pd.Series, feat_cols: list, idx: int) -> None:
    """Gráfico de importancia local basado en valores del modelo si SHAP/LIME no disponibles."""
    importance = st.session_state.get("feature_importance", {})
    if not importance:
        st.info("No hay datos de importancia disponibles para este registro.")
        return

    top = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:15]
    feats, vals = zip(*top)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ["#0F6E56" if v >= 0 else "#993C1D" for v in vals]
    ax.barh(feats[::-1], vals[::-1], color=colors[::-1], edgecolor="white")
    ax.set_title(f"Importancia de variables — Registro #{idx}", fontweight="bold")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()


def _tab_shap_table(feat_cols: list) -> None:
    importance = st.session_state.get("feature_importance", {})
    if not importance:
        st.info("Genera las explicaciones primero.")
        return

    df_imp = pd.DataFrame(
        sorted(importance.items(), key=lambda x: x[1], reverse=True),
        columns=["Variable", "Importancia media |SHAP|"],
    )
    df_imp["Rango"] = range(1, len(df_imp) + 1)
    df_imp["Importancia media |SHAP|"] = df_imp["Importancia media |SHAP|"].round(6)

    st.dataframe(
        df_imp[["Rango", "Variable", "Importancia media |SHAP|"]],
        use_container_width=True, hide_index=True,
    )

    method = st.session_state.get("xai_method", "XAI")
    st.caption(
        f"Método: {method} · Nombres de variable = columnas exactas del CSV cargado (RF6 criterio 3) · "
        f"random_state = 42"
    )
