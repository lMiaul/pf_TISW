"""modules/page_eda.py — RF2: Análisis Exploratorio (CRISP-DM: Data Understanding)
HU-02: Como científico de datos, quiero visualizar automáticamente la distribución
de clases y estadísticas descriptivas para entender el desbalanceo antes de remuestrear.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from src.utils.preprocessing import compute_class_distribution, preprocess
from src.utils.session import reset_downstream

_PALETTE = {0: "#185FA5", 1: "#993C1D"}   # azul = buen crédito · coral = impago


def render() -> None:
    st.header("RF2 — Análisis Exploratorio (EDA)")
    st.caption("CRISP-DM · Data Understanding")

    if not st.session_state.get("done_upload"):
        st.warning("Primero completa **RF1 — Carga de datos**.")
        return

    df         = st.session_state["dataset_raw"]
    target_col = st.session_state["target_column"]
    y_raw      = df[target_col]
    X          = df.drop(columns=[target_col])

    # Binarizar target si no es numérico (e.g. 'good'/'bad')
    unique_vals = sorted(y_raw.dropna().unique(), key=str)
    if len(unique_vals) == 2 and not set(unique_vals).issubset({0, 1}):
        y = y_raw.map({unique_vals[0]: 0, unique_vals[1]: 1}).astype(int)
    elif len(unique_vals) > 2:
        most_common = y_raw.value_counts().idxmax()
        y = (y_raw != most_common).astype(int)
    else:
        y = y_raw.astype(int)

    # ── Separar variables categóricas y numéricas ───────────────────────────
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    # ── KPIs superiores ────────────────────────────────────────────────────
    dist = compute_class_distribution(y)
    ir   = dist["imbalance_ratio"]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total registros",        f"{len(df):,}")
    k2.metric("Variables predictoras",  str(len(X.columns)))
    k3.metric("Imbalance Ratio (IR)",   str(ir),
              delta=f"Desbalanceo {dist['level']}",
              delta_color="inverse")
    k4.metric("Nulos totales",
              f"{df.isnull().sum().sum():,}",
              delta=f"{df.isnull().mean().mean()*100:.1f}% del total",
              delta_color="inverse")

    # ── Resumen de tipos de variables ──────────────────────────────────────
    k5, k6 = st.columns(2)
    k5.metric("Variables numéricas",    str(len(num_cols)))
    k6.metric("Variables categóricas",  str(len(cat_cols)))

    # Advertencia IR severo (HU-02 criterio de borde)
    if ir > 10:
        st.error(
            f"⚠️ Desbalanceo **severo** (IR = {ir}). "
            "La clase minoritaria representa menos del 10% del total. "
            "Se recomienda usar el enfoque híbrido SMOTE-ENN + Tomek."
        )
    elif ir > 3:
        st.warning(f"Desbalanceo **moderado** (IR = {ir}). SMOTE o SMOTE-ENN son opciones adecuadas.")

    st.divider()

    # ── Tabs de exploración ────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Distribución de clases",
        "📈 Variables numéricas",
        "🔥 Correlaciones",
        "🔍 Calidad de datos",
    ])

    with tab1:
        _tab_class_distribution(y, dist)

    with tab2:
        _tab_numeric_features(X, y)

    with tab3:
        _tab_correlations(X, y, target_col)

    with tab4:
        _tab_data_quality(df)

    # ── Preprocesar y guardar para módulos posteriores ────────────────────
    st.divider()
    if st.button("Preprocesar y continuar → RF3", type="primary", use_container_width=True):
        with st.spinner("Preprocesando datos..."):
            X_train, X_test, y_train, y_test, feat_cols, scaler = preprocess(df, target_col)
            reset_downstream("eda")
            st.session_state.update({
                "X_train":          X_train,
                "X_test":           X_test,
                "y_train":          y_train,
                "y_test":           y_test,
                "feature_columns":  feat_cols,
                "scaler":           scaler,
                "class_distribution": dist,
                "done_eda":         True,
            })
        st.success("Preprocesamiento completado. Continúa con **RF3 — Configurar remuestreo**.")
        st.rerun()


# ── Helpers de visualización ───────────────────────────────────────────────

def _tab_class_distribution(y: pd.Series, dist: dict) -> None:
    counts = dist["counts"]
    labels = {0: "Buen crédito (0)", 1: "Impago (1)"}

    col1, col2 = st.columns([1, 1])
    with col1:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        bars = ax.bar(
            [labels.get(k, str(k)) for k in counts],
            counts.values(),
            color=[_PALETTE.get(k, "#666") for k in counts],
            edgecolor="white", linewidth=1.2,
        )
        for bar, val in zip(bars, counts.values()):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f"{val:,}", ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.set_ylabel("Cantidad de registros")
        ax.set_title("Distribución de clases", fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        total = sum(counts.values())
        for cls, cnt in counts.items():
            pct = cnt / total * 100
            st.metric(f"Clase {cls} — {labels.get(cls,'')}", f"{cnt:,}", delta=f"{pct:.1f}%")
        st.metric("Imbalance Ratio", str(dist["imbalance_ratio"]))
        st.metric("Nivel de desbalanceo", str(dist["level"]).capitalize())


def _tab_numeric_features(X: pd.DataFrame, y: pd.Series) -> None:
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols:
        st.info("No se detectaron variables numéricas.")
        return

    st.subheader("Estadísticas descriptivas")
    desc = X[num_cols].describe().T.round(3)
    st.dataframe(desc, use_container_width=True)

    st.subheader("Distribución por variable y clase")
    sel_col = st.selectbox("Selecciona una variable", num_cols, key="eda_feat_sel")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for cls, grp in pd.concat([X[sel_col], y], axis=1).groupby(y.name):
        axes[0].hist(grp[sel_col], alpha=0.6, label=f"Clase {cls}",
                     color=_PALETTE.get(cls, "#666"), bins=30, edgecolor="white")
    axes[0].set_title(f"Histograma: {sel_col}", fontweight="bold")
    axes[0].legend()
    axes[0].spines[["top", "right"]].set_visible(False)

    X_tmp = pd.concat([X[sel_col], y.rename("target")], axis=1)
    X_tmp.boxplot(column=sel_col, by="target", ax=axes[1],
                  patch_artist=True,
                  boxprops=dict(facecolor="#E6F1FB"),
                  medianprops=dict(color="#185FA5", linewidth=2))
    axes[1].set_title(f"Boxplot: {sel_col}", fontweight="bold")
    axes[1].set_xlabel("Clase (0=buen crédito, 1=impago)")
    plt.suptitle("")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()


def _tab_correlations(X: pd.DataFrame, y: pd.Series, target_col: str) -> None:
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) < 2:
        st.info("Se necesitan al menos 2 variables numéricas para la matriz de correlación.")
        return

    df_corr = pd.concat([X[num_cols], y], axis=1)
    corr    = df_corr.corr()

    max_cols = min(len(num_cols), 20)   # limitar a 20 vars para legibilidad
    cols_to_show = num_cols[:max_cols] + [target_col]
    corr_sub = df_corr[cols_to_show].corr()

    fig, ax = plt.subplots(figsize=(max(7, len(cols_to_show)*0.55), max(5, len(cols_to_show)*0.5)))
    mask = np.triu(np.ones_like(corr_sub, dtype=bool))
    sns.heatmap(corr_sub, mask=mask, annot=len(cols_to_show) <= 12,
                fmt=".2f", cmap="RdBu_r", center=0,
                linewidths=.5, ax=ax, annot_kws={"size": 8})
    ax.set_title("Matriz de correlación (triángulo inferior)", fontweight="bold")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Top correlaciones con el target
    target_corr = corr_sub[target_col].drop(target_col).abs().sort_values(ascending=False)
    st.subheader(f"Variables con mayor correlación con '{target_col}'")
    st.dataframe(
        target_corr.reset_index().rename(columns={"index": "Variable", target_col: "|Correlación|"}),
        use_container_width=True, hide_index=True,
    )


def _tab_data_quality(df: pd.DataFrame) -> None:
    st.subheader("Valores nulos por columna")
    null_info = pd.DataFrame({
        "Variable":      df.columns,
        "Nulos":         df.isnull().sum().values,
        "% Nulos":       (df.isnull().mean() * 100).round(2).values,
        "Tipo":          df.dtypes.astype(str).values,
        "Valores únicos": [df[c].nunique() for c in df.columns],
    })
    null_info = null_info.sort_values("% Nulos", ascending=False)
    st.dataframe(
        null_info.style.background_gradient(subset=["% Nulos"], cmap="Oranges"),
        use_container_width=True, hide_index=True,
    )

    st.subheader("Variables con varianza cero (candidatas a eliminar)")
    zero_var = [c for c in df.select_dtypes(include=[np.number]).columns if df[c].std() == 0]
    if zero_var:
        st.warning(f"Columnas con varianza cero: {', '.join(zero_var)}")
    else:
        st.success("No se detectaron variables con varianza cero.")
