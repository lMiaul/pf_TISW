# 📊 Credit Risk Predictor

**Sistema de Predicción de Riesgo Crediticio con Remuestreo Híbrido**  
Proyecto Final · USIL 2026 · Autor: Mauricio Fabian Sandoval Arrieta

[![CI — Lint & Tests](https://github.com/TU_USUARIO/credit-risk-predictor/actions/workflows/ci.yml/badge.svg)](https://github.com/TU_USUARIO/credit-risk-predictor/actions)

---

## Descripción

Aplicación web analítica interactiva desarrollada en **Streamlit** que implementa un pipeline
completo de predicción de riesgo crediticio bajo la metodología **CRISP-DM**, con énfasis en
el tratamiento del desbalance de clases mediante un **enfoque híbrido de remuestreo propuesto**
(SMOTE-ENN + Tomek Links) y explicabilidad mediante **SHAP/LIME**.

> Basado en revisión sistemática PRISMA 2020 · 64 estudios · 2022–2026

---

## Documentación

- [Manual de uso básico](./MANUAL_USO.md) — para usuarios finales (científico de datos / analista de riesgo)
- Este README — para desarrolladores

---

## Pipeline CRISP-DM implementado

| Fase | Módulo | RF |
|------|--------|----|
| Business Understanding | Documentación + KPIs | — |
| Data Understanding | `page_eda.py` | RF2 |
| Data Preparation | `preprocessing.py` + `resampling.py` + `page_resampling.py` | RF1, RF3 |
| Modeling | `modeling.py` + `page_modeling.py` | RF4 |
| Evaluation | `page_evaluation.py` | RF5 |
| Deployment | `app.py` + `page_xai.py` | RF6 |

---

## Técnicas de remuestreo disponibles

| Algoritmo | Clave | Descripción |
|-----------|-------|-------------|
| Random Under-Sampling | `rus` | Elimina muestras de la clase mayoritaria |
| SMOTE | `smote` | Genera sintéticos por interpolación |
| SMOTE-ENN | `smoteenn` | SMOTE + limpieza de ambigüedades |
| **SMOTE-ENN + Tomek (híbrido)** ⭐ | `hybrid` | Propuesta del proyecto: limpia frontera de decisión en ambas clases |

---

## Estructura del proyecto

```
credit-risk-predictor/
├── app.py                        # Entrada principal Streamlit
├── requirements.txt
├── .gitignore
├── README.md
├── .github/
│   └── workflows/
│       └── ci.yml                # CI: lint + tests automáticos
├── src/
│   ├── modules/                  # Páginas de la app (RF1–RF6)
│   │   ├── page_upload.py        # RF1 — Carga de datos
│   │   ├── page_eda.py           # RF2 — Análisis exploratorio
│   │   ├── page_resampling.py    # RF3 — Configurar remuestreo
│   │   ├── page_modeling.py      # RF4 — Entrenar modelo
│   │   ├── page_evaluation.py    # RF5 — Evaluar resultados
│   │   └── page_xai.py           # RF6 — Explicabilidad XAI
│   └── utils/                    # Lógica de negocio
│       ├── session.py            # Estado de sesión (modelo lógico en memoria)
│       ├── preprocessing.py      # Imputación, codificación, partición
│       ├── resampling.py         # 4 técnicas + enfoque híbrido
│       └── modeling.py           # LR, RF, XGBoost + métricas
├── tests/
│   └── test_pipeline.py          # Casos de prueba HU-01 a HU-06
└── data/
    └── sample/                   # Datasets de referencia (German Credit, etc.)
```

---

## Instalación y ejecución local

### 1. Clonar el repositorio

```bash
git clone https://github.com/TU_USUARIO/credit-risk-predictor.git
cd credit-risk-predictor
```

### 2. Crear entorno virtual e instalar dependencias

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/Mac
# .venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

### 3. Ejecutar la aplicación

```bash
streamlit run app.py
```

La app estará disponible en `http://localhost:8501`

### 4. Ejecutar los tests

```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

---

## Flujo de uso de la aplicación

```
1. RF1 — Cargar CSV crediticio
        ↓
2. RF2 — Revisar EDA: distribución, correlaciones, calidad
        ↓
3. RF3 — Seleccionar técnica de remuestreo y configurar parámetros
        ↓
4. RF4 — Elegir clasificador (LR / RF / XGBoost) y entrenar con CV estratificada
        ↓
5. RF5 — Comparar métricas base vs. optimizado (Accuracy, F1, AUC-ROC, Recall, G-mean)
        ↓
6. RF6 — Explorar explicabilidad SHAP/LIME global y por cliente individual
```

---

## KPIs de éxito del proyecto

| KPI | Umbral mínimo |
|-----|--------------|
| AUC-ROC | ≥ 0.85 |
| F1-score (clase minoritaria) | ≥ 0.75 |
| Recall (clase minoritaria) | ≥ 0.80 |

---

## Flujo de trabajo Git

### Ramas principales

| Rama | Propósito |
|------|-----------|
| `main` | Versión estable / producción. Solo merge vía PR aprobado. |
| `develop` | Integración continua. Base para features. |
| `feature/*` | Nueva funcionalidad (ej. `feature/rf3-hibrido`) |
| `fix/*` | Corrección de bugs (ej. `fix/leakage-scaler`) |
| `docs/*` | Solo documentación |

### Flujo estándar

```bash
# Crear rama de feature desde develop
git checkout develop
git pull origin develop
git checkout -b feature/nombre-del-feature

# Desarrollar, añadir y commitear
git add src/modules/page_nueva.py
git commit -m "feat(RF3): implementar vista previa de distribución post-remuestreo"

# Subir y abrir Pull Request hacia develop
git push origin feature/nombre-del-feature
```

### Convención de commits (Conventional Commits)

```
feat(RF4):   nueva funcionalidad en el módulo de modelado
fix(RF3):    corrección en el pipeline de remuestreo híbrido
docs:        actualizar README con instrucciones de instalación
test(HU05):  añadir casos de prueba para métricas comparativas
refactor:    extraer lógica de métricas a utils/modeling.py
ci:          actualizar versión de actions/checkout a v4
```

### Releases con tags

```bash
# Marcar versión estable
git tag -a v1.0.0 -m "Release v1.0.0 — pipeline CRISP-DM completo"
git push origin v1.0.0
```

---

## Consideraciones éticas

- Los modelos **no deben usarse en producción** sin validación con datos reales de la entidad financiera.
- Las predicciones son **probabilísticas**, no determinísticas. Siempre requieren revisión humana.
- Los datasets públicos usados (German Credit, etc.) son de libre uso académico.
- La información de clientes cargada durante la sesión **no se persiste** (stateless por diseño).
- El uso de SHAP/LIME para la explicabilidad facilita el cumplimiento de regulaciones de transparencia en decisiones crediticias automatizadas.

---

## Datasets de referencia

| Dataset | Registros | IR | Fuente |
|---------|-----------|-----|--------|
| German Credit | 1,000 | ~2.3:1 | UCI ML Repository |
| Home Credit Default Risk | 307,511 | ~11:1 | Kaggle |
| Give Me Some Credit | 150,000 | ~14:1 | Kaggle |

---

## Tecnologías utilizadas

| Categoría | Librería |
|-----------|---------|
| App | Streamlit |
| ML | scikit-learn, XGBoost, LightGBM |
| Remuestreo | imbalanced-learn |
| XAI | SHAP, LIME |
| Visualización | Matplotlib, Seaborn |
| Testing | pytest, pytest-cov |
| Linting | Ruff |
| CI/CD | GitHub Actions |
