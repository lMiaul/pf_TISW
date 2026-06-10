"""modules/page_upload.py — RF1: Carga de datos (CRISP-DM: Data Understanding)
HU-01: Como científico de datos, quiero cargar un CSV y que el sistema valide su estructura.
"""

import pandas as pd
import streamlit as st

from src.utils.preprocessing import detect_target_candidates
from src.utils.session import reset_downstream


def render() -> None:
    st.header("RF1 — Carga de datos")
    st.caption("CRISP-DM · Data Understanding")

    uploaded = st.file_uploader(
        "Arrastra o selecciona un archivo CSV con datos crediticios",
        type=["csv"],
        help="Formato requerido: CSV UTF-8 con al menos una columna binaria (0/1) como variable objetivo.",
    )

    if uploaded is None:
        st.info("Sube un archivo CSV para comenzar. Puedes usar el dataset **German Credit** como punto de partida.")
        _show_format_guide()
        return

    # ── Validación y carga ─────────────────────────────────────────────────
    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Error al leer el archivo: {e}")
        return

    if df.empty:
        st.error("El archivo está vacío.")
        return

    # ── Selección de variable objetivo ────────────────────────────────────
    candidates = detect_target_candidates(df)
    all_cols   = df.columns.tolist()

    st.success(f"Archivo cargado correctamente: **{uploaded.name}** — {df.shape[0]:,} filas × {df.shape[1]} columnas")

    col1, col2 = st.columns(2)
    with col1:
        default_idx = 0
        if candidates:
            # Preferir la primera candidata binaria detectada
            default_idx = all_cols.index(candidates[0])
        target_col = st.selectbox(
            "Variable objetivo (columna binaria de impago)",
            options=all_cols,
            index=default_idx,
            help="Debe contener valores 0 (buen crédito) y 1 (impago).",
        )
    with col2:
        st.metric("Columnas detectadas", str(df.shape[1]))
        st.metric("Candidatas binarias", str(len(candidates)))

    # ── Validar que la columna objetivo sea binaria ───────────────────────
    unique_vals = df[target_col].dropna().unique()
    if not set(unique_vals).issubset({0, 1, "0", "1"}):
        st.warning(
            f"La columna **{target_col}** tiene valores: {sorted(unique_vals)}. "
            "Se esperan sólo 0 y 1. Verifica la selección."
        )

    # ── Advertencia de nulos severos ──────────────────────────────────────
    null_pct = df.isnull().mean()
    severe_null_cols = null_pct[null_pct > 0.30].index.tolist()
    if severe_null_cols:
        st.warning(
            f"Las siguientes columnas superan el 30% de valores nulos y serán imputadas con la mediana/moda: "
            f"{', '.join(severe_null_cols)}"
        )

    # ── Vista previa ──────────────────────────────────────────────────────
    with st.expander("Vista previa de los datos (primeras 5 filas)"):
        st.dataframe(df.head(), use_container_width=True)

    # ── Confirmar carga ───────────────────────────────────────────────────
    if st.button("Confirmar carga y continuar →", type="primary", use_container_width=True):
        reset_downstream("upload")
        st.session_state["dataset_raw"]      = df
        st.session_state["target_column"]    = target_col
        st.session_state["feature_columns"]  = [c for c in df.columns if c != target_col]
        st.session_state["filename"]         = uploaded.name
        st.session_state["done_upload"]      = True
        st.success("Datos guardados en sesión. Continúa con **RF2 — Análisis exploratorio**.")
        st.rerun()


def _show_format_guide() -> None:
    with st.expander("Formato esperado del CSV"):
        st.markdown(
            """
| variable_1 | variable_2 | ... | target |
|------------|------------|-----|--------|
| 1500       | 24         | ... | 0      |
| 800        | 36         | ... | 1      |

- `target`: columna binaria donde **1 = impago**, **0 = buen crédito**
- Se aceptan variables numéricas y categóricas (codificación automática)
- Valores nulos permitidos (imputación automática)
            """
        )
