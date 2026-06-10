"""modules/page_resampling.py — RF3: Configuración de Remuestreo (CRISP-DM: Data Preparation)
HU-03: Como científico de datos, quiero seleccionar y configurar la técnica de remuestreo
para experimentar con distintos enfoques de balanceo sin introducir ruido.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.utils.resampling import (
    ALGORITHM_DESCRIPTIONS,
    ALGORITHMS,
    apply_resampling,
)
from src.utils.session import reset_downstream

_PALETTE = {0: "#185FA5", 1: "#993C1D"}


def render() -> None:
    st.header("RF3 — Configuración de Remuestreo")
    st.caption("CRISP-DM · Data Preparation — núcleo del proyecto")

    if not st.session_state.get("done_eda"):
        st.warning("Primero completa **RF2 — Análisis exploratorio**.")
        return

    X_train = st.session_state["X_train"]
    y_train = st.session_state["y_train"]
    dist    = st.session_state["class_distribution"]

    # ── Configuración ──────────────────────────────────────────────────────
    col_cfg, col_info = st.columns([1.4, 1])

    with col_cfg:
        st.subheader("Parámetros de remuestreo")

        algo_label = st.selectbox(
            "Técnica de remuestreo",
            list(ALGORITHMS.keys()),
            index=3,          # Híbrido por defecto (propuesta del proyecto)
            help="El enfoque híbrido es la propuesta original de este proyecto.",
        )
        algo_key = ALGORITHMS[algo_label]

        st.info(f"**{algo_label}** — {ALGORITHM_DESCRIPTIONS[algo_key]}")

        sampling_ratio = st.slider(
            "Ratio objetivo (clase minoritaria / clase mayoritaria)",
            min_value=0.3, max_value=1.0, value=1.0, step=0.05,
            help="1.0 = balance perfecto. Valores menores permiten un balance parcial.",
        )

        st.caption(f"🔒 `random_state = 42` — fijo para reproducibilidad (RNF3)")

        # Advertencia SMOTE con clase minoritaria pequeña (HU-03 borde)
        minority_count = dist["counts"].get(dist["minority_class"], 0)
        if algo_key == "smote" and minority_count < 20:
            st.warning(
                f"La clase minoritaria tiene solo **{minority_count}** muestras. "
                "SMOTE puede generar muestras poco representativas. "
                "Considera usar el enfoque **híbrido**."
            )

    with col_info:
        st.subheader("Estado actual del dataset")
        counts_before = dist["counts"]
        total_before  = sum(counts_before.values())
        for cls, cnt in counts_before.items():
            label = "Buen crédito (0)" if cls == 0 else "Impago (1)"
            st.metric(label, f"{cnt:,}", delta=f"{cnt/total_before*100:.1f}%")
        st.metric("Imbalance Ratio actual", str(dist["imbalance_ratio"]))

    # ── Vista previa del balance resultante ───────────────────────────────
    st.divider()
    st.subheader("Vista previa del balance resultante")

    if st.button("▶ Previsualizar distribución post-remuestreo", use_container_width=True):
        with st.spinner("Calculando distribución..."):
            try:
                _, y_res, info = apply_resampling(X_train, y_train, algo_key, sampling_ratio)
                _show_distribution_preview(y_train, y_res, info)
                st.session_state["resampling_preview"] = info
            except Exception as e:
                st.error(f"Error en el remuestreo: {e}")

    elif st.session_state.get("resampling_preview"):
        st.caption("(última vista previa guardada)")
        info = st.session_state["resampling_preview"]
        _show_distribution_preview(y_train, None, info, preview_only=True)

    # ── Confirmar configuración ───────────────────────────────────────────
    st.divider()
    if st.button("Confirmar configuración y continuar → RF4", type="primary", use_container_width=True):
        with st.spinner("Aplicando remuestreo..."):
            try:
                X_res, y_res, info = apply_resampling(X_train, y_train, algo_key, sampling_ratio)
                reset_downstream("resampling")
                st.session_state.update({
                    "X_train_resampled":    X_res,
                    "y_train_resampled":    y_res,
                    "resampling_algorithm": algo_key,
                    "resampling_info":      info,
                    "resampling_label":     algo_label,
                    "done_resampling":      True,
                })
                st.success(
                    f"Remuestreo aplicado. "
                    f"IR antes: **{info['ir_before']}** → IR después: **{info['ir_after']}** · "
                    f"Muestras añadidas/eliminadas: **{info['samples_added']:+,}**"
                )
                st.rerun()
            except Exception as e:
                st.error(f"Error al aplicar remuestreo: {e}")


def _show_distribution_preview(y_before, y_after, info: dict, preview_only: bool = False) -> None:
    before = info["before"]
    after  = info["after"]
    labels = {0: "Buen crédito (0)", 1: "Impago (1)"}

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for ax, data, title in zip(
        axes,
        [before, after],
        ["Antes del remuestreo", f"Después — {info['algorithm'].upper()}"],
    ):
        bars = ax.bar(
            [labels.get(k, k) for k in data],
            data.values(),
            color=[_PALETTE.get(k, "#666") for k in data],
            edgecolor="white", linewidth=1.2,
        )
        for bar, val in zip(bars, data.values()):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f"{val:,}", ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel("Muestras")
        ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close()

    c1, c2, c3 = st.columns(3)
    c1.metric("IR antes",               str(info["ir_before"]))
    c2.metric("IR después",             str(info["ir_after"]),
              delta=f"{info['ir_after']-info['ir_before']:.3f}",
              delta_color="inverse")
    c3.metric("Δ muestras",             f"{info['samples_added']:+,}")
