"""
Credit Risk Prediction System — Streamlit App
Arquitectura de Sistemas Predictivos para la Detección de Riesgo Crediticio
Autor: Mauricio Fabian Sandoval Arrieta — USIL 2026
CRISP-DM Pipeline: BU → DU → DP → MOD → EVA → DEP
"""

import streamlit as st

st.set_page_config(
    page_title="Credit Risk Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

from src.modules import (
    page_upload,
    page_eda,
    page_resampling,
    page_modeling,
    page_evaluation,
    page_xai,
)
from src.utils.session import init_session_state

# ── Inicialización de estado de sesión ─────────────────────────────────────
init_session_state()

# ── Sidebar: navegación y configuración global ─────────────────────────────
with st.sidebar:
    st.markdown("## 📊 Credit Risk Predictor")
    st.markdown("*Riesgo crediticio con remuestreo híbrido*")
    st.divider()

    PAGES = {
        "RF1 — Carga de datos":         "upload",
        "RF2 — Análisis exploratorio":  "eda",
        "RF3 — Configurar remuestreo":  "resampling",
        "RF4 — Entrenar modelo":        "modeling",
        "RF5 — Evaluar resultados":     "evaluation",
        "RF6 — Explicabilidad (XAI)":   "xai",
    }

    # Indicadores de progreso por módulo
    def _status_icon(key: str) -> str:
        done = st.session_state.get(f"done_{key}", False)
        return "✅" if done else "⬜"

    page_label = st.radio(
        "Módulo",
        list(PAGES.keys()),
        format_func=lambda x: f"{_status_icon(PAGES[x])}  {x}",
        label_visibility="collapsed",
    )
    page_key = PAGES[page_label]

    st.divider()
    st.caption("CRISP-DM · random_state = 42")
    st.caption("v1.0.0 · USIL 2026")

    if st.button("🔄 Reiniciar sesión", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        init_session_state()
        st.rerun()

# ── Router de páginas ──────────────────────────────────────────────────────
routes = {
    "upload":     page_upload.render,
    "eda":        page_eda.render,
    "resampling": page_resampling.render,
    "modeling":   page_modeling.render,
    "evaluation": page_evaluation.render,
    "xai":        page_xai.render,
}
routes[page_key]()
