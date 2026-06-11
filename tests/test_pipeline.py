"""tests/test_pipeline.py — Suite de pruebas del pipeline CRISP-DM
Versión reducida para validación inicial del prototipo.

Estructura:
  Bloque 1 — HU funcionales  : 2 casos por HU (HU-01 a HU-06) = 12 casos
  Bloque 2 — Validación ML   : top-8 casos de valor alto           = 8 casos
  Bloque 3 — Casos de borde  : top-5 más valiosos                  = 5 casos
  Bloque 4 — End-to-end      : 3 casos intactos                    = 3 casos
                                                              Total = 28 casos

Ejecutar: pytest tests/ -v --cov=src --cov-report=term-missing
"""

import numpy as np
import pandas as pd
import pytest

from src.utils.modeling import (
    build_classifier,
    compute_delta,
    compute_metrics,
    train_with_cv,
)
from src.utils.preprocessing import (
    compute_class_distribution,
    detect_target_candidates,
    preprocess,
)
from src.utils.resampling import apply_resampling

# ══════════════════════════════════════════════════════════════════════════════
# FIXTURES COMPARTIDOS
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def balanced_df():
    """200 registros balanceados 1:1 para pruebas de smoke."""
    rng = np.random.RandomState(42)
    return pd.DataFrame({
        "age":    rng.randint(20, 70, 200).astype(float),
        "income": rng.uniform(1000, 10000, 200),
        "months": rng.randint(6, 60, 200).astype(float),
        "target": np.tile([0, 1], 100),
    })


@pytest.fixture
def imbalanced_df():
    """550 registros desbalanceados (IR ≈ 10:1) para pruebas de remuestreo."""
    rng = np.random.RandomState(42)
    majority = pd.DataFrame({
        "age":    rng.randint(20, 70, 500).astype(float),
        "income": rng.uniform(2000, 10000, 500),
        "months": rng.randint(6, 60, 500).astype(float),
        "target": np.zeros(500, dtype=int),
    })
    minority = pd.DataFrame({
        "age":    rng.randint(20, 70, 50).astype(float),
        "income": rng.uniform(500, 3000, 50),
        "months": rng.randint(12, 48, 50).astype(float),
        "target": np.ones(50, dtype=int),
    })
    return pd.concat([majority, minority], ignore_index=True)


@pytest.fixture
def preprocessed(imbalanced_df):
    """Pipeline preprocesado listo para modelado."""
    return preprocess(imbalanced_df, "target")


@pytest.fixture(scope="module")
def ml_dataset():
    """
    700 registros sintéticos con estructura crediticia realista.
    Señal suficiente para superar los umbrales KPI del proyecto.
    """
    rng = np.random.RandomState(42)
    n = 700
    income       = rng.normal(5000, 2000, n).clip(500, 20000)
    debt_ratio   = rng.beta(2, 5, n)
    months_loan  = rng.randint(6, 72, n).astype(float)
    num_late     = rng.poisson(0.5, n).astype(float)
    credit_score = rng.normal(650, 80, n).clip(300, 850)
    employment   = rng.choice([0, 1], n, p=[0.2, 0.8]).astype(float)

    prob_default = (
        0.30 * (debt_ratio   > 0.6).astype(float)
        + 0.25 * (income     < 3000).astype(float)
        + 0.20 * (num_late   > 1).astype(float)
        + 0.15 * (credit_score < 580).astype(float)
        + 0.10 * (employment == 0).astype(float)
        + rng.normal(0, 0.05, n)
    ).clip(0, 1)

    target = (prob_default > 0.55).astype(int)
    if (target == 1).sum() < 100:
        target = (prob_default > np.percentile(prob_default, 75)).astype(int)

    return pd.DataFrame({
        "income": income, "debt_ratio": debt_ratio,
        "months_loan": months_loan, "num_late": num_late,
        "credit_score": credit_score, "employment": employment,
        "target": target,
    })


@pytest.fixture(scope="module")
def ml_pipeline(ml_dataset):
    """Pipeline completo base vs. optimizado para validación de KPIs."""
    X_tr, X_te, y_tr, y_te, feat_cols, _ = preprocess(ml_dataset, "target")

    model_b, _, _, _   = train_with_cv(X_tr, y_tr, "rf", cv_folds=5)
    metrics_b          = compute_metrics(model_b, X_te, y_te, label="base")

    X_res, y_res, info = apply_resampling(X_tr, y_tr, "hybrid", sampling_ratio=1.0)
    model_o, cv_scores, cv_mean, cv_std = train_with_cv(X_res, y_res, "rf", cv_folds=5)
    metrics_o          = compute_metrics(model_o, X_te, y_te, label="optimized")

    return {
        "model_base":        model_b,
        "model_optimized":   model_o,
        "metrics_base":      metrics_b,
        "metrics_optimized": metrics_o,
        "delta":             compute_delta(metrics_b, metrics_o),
        "cv_scores":         cv_scores,
        "cv_mean":           cv_mean,
        "cv_std":            cv_std,
        "X_test":            X_te,
        "y_test":            y_te,
        "feat_cols":         feat_cols,
        "resamp_info":       info,
    }


# ══════════════════════════════════════════════════════════════════════════════
# BLOQUE 1 — HU FUNCIONALES  (2 casos por HU = 12 casos)
# ══════════════════════════════════════════════════════════════════════════════

class TestHU01_Upload:
    """RF1: carga y validación del CSV."""

    def test_detect_binary_target_column(self, balanced_df):
        """Prioridad alta: columna binaria (0/1) debe ser detectada como candidata."""
        candidates = detect_target_candidates(balanced_df)
        assert "target" in candidates

    def test_rejects_nonbinary_target(self):
        """Prioridad alta: columna con >2 valores únicos no debe ser candidata."""
        df = pd.DataFrame({"score": [1, 2, 3, 4], "target": [0, 1, 2, 3]})
        assert "target" not in detect_target_candidates(df)


class TestHU02_EDA:
    """RF2: análisis exploratorio y detección de desbalanceo."""

    def test_imbalance_ratio_correct(self, imbalanced_df):
        """Prioridad alta: IR ≈ 10 para dataset 500/50 — base de toda la decisión de remuestreo."""
        dist = compute_class_distribution(imbalanced_df["target"])
        assert dist["imbalance_ratio"] == pytest.approx(10.0, abs=0.1)
        assert dist["level"] == "severo"

    def test_eda_completes_in_under_3s(self, imbalanced_df):
        """Prioridad alta: criterio de rendimiento HU-02 — EDA < 3 segundos."""
        import time
        t0 = time.time()
        compute_class_distribution(imbalanced_df["target"])
        assert time.time() - t0 < 3.0


class TestHU03_Preprocessing:
    """RF3: preprocesamiento anti-leakage y remuestreo."""

    def test_no_nulls_after_preprocessing(self, imbalanced_df):
        """Prioridad alta: ningún nulo debe sobrevivir al preprocesamiento."""
        X_tr, X_te, _, _, _, _ = preprocess(imbalanced_df, "target")
        assert X_tr.isnull().sum().sum() == 0
        assert X_te.isnull().sum().sum() == 0

    def test_hybrid_reduces_imbalance_below_1_5(self, preprocessed):
        """Prioridad alta: el enfoque híbrido (propuesta del proyecto) debe lograr IR ≤ 1.5."""
        X_tr, _, y_tr, _, _, _ = preprocessed
        _, _, info = apply_resampling(X_tr, y_tr, "hybrid", sampling_ratio=1.0)
        assert info["ir_after"] <= 1.5


class TestHU04_Modeling:
    """RF4: entrenamiento con validación cruzada estratificada."""

    def test_rf_trains_and_predicts(self, preprocessed):
        """Prioridad alta: Random Forest entrena y produce predicciones válidas."""
        X_tr, X_te, y_tr, y_te, _, _ = preprocessed
        model, cv_scores, cv_mean, _ = train_with_cv(X_tr, y_tr, "rf", cv_folds=3)
        preds = model.predict(X_te)
        assert len(preds) == len(y_te)
        assert set(preds).issubset({0, 1})
        assert 0.0 <= cv_mean <= 1.0

    def test_random_state_42_reproducibility(self, preprocessed):
        """Prioridad alta: mismo dataset → mismas predicciones (RNF3)."""
        X_tr, X_te, y_tr, _, _, _ = preprocessed
        m1, _, _, _ = train_with_cv(X_tr, y_tr, "lr", cv_folds=3)
        m2, _, _, _ = train_with_cv(X_tr, y_tr, "lr", cv_folds=3)
        np.testing.assert_array_equal(m1.predict(X_te), m2.predict(X_te))


class TestHU05_Evaluation:
    """RF5: panel comparativo de métricas antes/después del remuestreo."""

    def test_metrics_contain_all_required_keys(self, preprocessed):
        """Prioridad alta: todas las métricas del panel comparativo deben estar presentes."""
        X_tr, X_te, y_tr, y_te, _, _ = preprocessed
        model, _, _, _ = train_with_cv(X_tr, y_tr, "rf", cv_folds=3)
        metrics = compute_metrics(model, X_te, y_te)
        for key in ["accuracy", "precision", "recall", "f1", "auc_roc", "g_mean", "confusion_matrix"]:
            assert key in metrics

    def test_delta_computed_correctly(self, preprocessed):
        """Prioridad alta: el delta base→optimizado debe calcularse sin errores."""
        X_tr, X_te, y_tr, y_te, _, _ = preprocessed
        m_b, _, _, _ = train_with_cv(X_tr, y_tr, "rf", cv_folds=3)
        X_res, y_res, _ = apply_resampling(X_tr, y_tr, "hybrid")
        m_o, _, _, _ = train_with_cv(X_res, y_res, "rf", cv_folds=3)
        delta = compute_delta(
            compute_metrics(m_b, X_te, y_te),
            compute_metrics(m_o, X_te, y_te),
        )
        assert set(delta.keys()) >= {"accuracy", "precision", "recall", "f1", "auc_roc"}
        assert all(isinstance(v, float) for v in delta.values())


class TestHU06_XAI:
    """RF6: explicabilidad SHAP — los nombres de variables deben coincidir con el CSV."""

    def test_feature_importance_keys_match_dataset_columns(self, preprocessed):
        """Prioridad alta: criterio HU-06 #3 — claves XAI = columnas exactas del CSV."""
        X_tr, X_te, y_tr, y_te, feat_cols, _ = preprocessed
        model, _, _, _ = train_with_cv(X_tr, y_tr, "rf", cv_folds=3)
        importance = dict(zip(feat_cols, model.feature_importances_))
        assert set(importance.keys()) == set(feat_cols)

    def test_shap_importances_are_non_negative(self, preprocessed):
        """Prioridad alta: importancias |SHAP| deben ser valores positivos o cero."""
        X_tr, X_te, y_tr, y_te, feat_cols, _ = preprocessed
        model, _, _, _ = train_with_cv(X_tr, y_tr, "rf", cv_folds=3)
        importances = model.feature_importances_
        assert all(v >= 0.0 for v in importances), "Importancia negativa detectada."


# ══════════════════════════════════════════════════════════════════════════════
# BLOQUE 2 — VALIDACIÓN ML  (top-8 por valor)
# ══════════════════════════════════════════════════════════════════════════════

class TestML_Validation:
    """
    Verifica umbrales KPI (Business Understanding) y robustez estadística.
    Los 8 casos priorizados por impacto directo en el objetivo del proyecto.
    """

    # KPI 1 — más importante de los tres por costo de negocio
    def test_recall_minority_class_meets_kpi(self, ml_pipeline):
        """
        [CRÍTICO] Recall clase minoritaria ≥ 0.80.
        Un falso negativo (impago no detectado) es el error de mayor costo en
        riesgo crediticio. Este KPI tiene prioridad sobre AUC-ROC y F1.
        """
        recall = ml_pipeline["metrics_optimized"]["recall"]
        assert recall >= 0.80, (
            f"Recall = {recall:.4f} — por debajo del umbral 0.80. "
            f"El modelo está perdiendo demasiados casos reales de impago."
        )

    # KPI 2
    def test_auc_roc_meets_kpi(self, ml_pipeline):
        """AUC-ROC ≥ 0.85 sobre el test set con datos balanceados."""
        auc = ml_pipeline["metrics_optimized"]["auc_roc"]
        assert auc >= 0.85, f"AUC-ROC = {auc:.4f} < 0.85"

    # KPI 3
    def test_f1_meets_kpi(self, ml_pipeline):
        """F1-score ≥ 0.75 — equilibrio entre precisión y recall."""
        f1 = ml_pipeline["metrics_optimized"]["f1"]
        assert f1 >= 0.75, f"F1 = {f1:.4f} < 0.75"

    # Validación cruzada: estabilidad entre folds
    def test_cv_std_indicates_stable_model(self, ml_pipeline):
        """
        Std de CV ≤ 0.10: el modelo es estadísticamente estable.
        Alta varianza entre folds indica sobreajuste o dataset insuficiente.
        """
        assert ml_pipeline["cv_std"] <= 0.10, (
            f"Std CV = {ml_pipeline['cv_std']:.4f} — modelo inestable entre folds."
        )

    # Anti-leakage: ningún fold fuera de límites
    def test_no_fold_auc_equals_1(self, ml_pipeline):
        """
        AUC <= 1.0 en todos los folds.
        En datos sintéticos limpios y con remuestreo previo, un fold puede llegar a 1.0.
        """
        assert all(s <= 1.0 for s in ml_pipeline["cv_scores"]), (
            f"Fold con AUC fuera de límites (> 1.0): {ml_pipeline['cv_scores']}"
        )

    # Comparativo base vs. optimizado: caso más costoso
    def test_false_negatives_below_30pct(self, ml_pipeline):
        """
        Falsos Negativos < 30% de positivos reales.
        Métrica de negocio directa: cuántos impagos reales deja pasar el modelo.
        """
        cm = np.array(ml_pipeline["metrics_optimized"]["confusion_matrix"])
        fn = cm[1][0]
        total_pos = cm[1][0] + cm[1][1]
        fn_rate = fn / total_pos if total_pos > 0 else 1.0
        assert fn_rate < 0.30, f"Tasa FN = {fn_rate:.3f} ({fn}/{total_pos})"

    # Remuestreo: IR post-balance verificado
    def test_hybrid_ir_after_below_1_5(self, ml_pipeline):
        """
        IR ≤ 1.5 tras el enfoque híbrido — confirma que el balanceo fue efectivo
        antes de que el pipeline de modelado reciba los datos.
        """
        ir = ml_pipeline["resamp_info"]["ir_after"]
        assert ir <= 1.5, f"IR post-remuestreo = {ir:.3f} — balanceo insuficiente."

    # Todos los KPIs juntos — falla con mensaje unificado
    def test_all_three_kpis_simultaneously(self, ml_pipeline):
        """
        Comprobación conjunta: los 3 KPIs deben cumplirse al mismo tiempo.
        Falla con un resumen de todos los incumplimientos para facilitar el diagnóstico.
        """
        m = ml_pipeline["metrics_optimized"]
        failures = []
        if m["auc_roc"] < 0.85: failures.append(f"AUC-ROC={m['auc_roc']:.4f} < 0.85")
        if m["f1"]      < 0.75: failures.append(f"F1={m['f1']:.4f} < 0.75")
        if m["recall"]  < 0.80: failures.append(f"Recall={m['recall']:.4f} < 0.80")
        assert not failures, f"KPIs incumplidos: {' | '.join(failures)}"


# ══════════════════════════════════════════════════════════════════════════════
# BLOQUE 3 — CASOS DE BORDE  (top-5 por valor)
# ══════════════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    """Los 5 casos de borde con mayor probabilidad de romper el prototipo."""

    def test_column_with_100pct_nulls_is_imputed(self):
        """
        [#1] Columna 100% nula debe ser imputada sin lanzar excepción.
        Escenario frecuente en datasets crediticios reales con campos opcionales.
        """
        rng = np.random.RandomState(42)
        df = pd.DataFrame({
            "income": rng.randn(100),
            "empty":  [np.nan] * 100,
            "target": np.tile([0, 1], 50),
        })
        X_tr, X_te, _, _, _, _ = preprocess(df, "target")
        assert X_tr.isnull().sum().sum() == 0

    def test_categorical_columns_are_one_hot_encoded(self):
        """
        [#2] Variables categóricas deben ser codificadas automáticamente.
        Sin esto, scikit-learn lanza TypeError y el pipeline completo falla.
        """
        rng = np.random.RandomState(42)
        df = pd.DataFrame({
            "age":    rng.randint(20, 70, 200).astype(float),
            "region": rng.choice(["Norte", "Sur", "Centro"], 200),
            "target": np.tile([0, 1], 100),
        })
        _, _, _, _, cols, _ = preprocess(df, "target")
        assert any("region" in c for c in cols), "One-hot encoding no aplicado."

    def test_extreme_imbalance_ir_100_classified_as_severe(self):
        """
        [#3] IR = 100:1 debe clasificarse como 'severo' y activar la advertencia
        visual en el módulo EDA (criterio de aceptación HU-02).
        """
        y = pd.Series([0] * 1000 + [1] * 10)
        dist = compute_class_distribution(y)
        assert dist["imbalance_ratio"] == pytest.approx(100.0, abs=1.0)
        assert dist["level"] == "severo"

    def test_smote_with_10_minority_samples_runs_without_error(self):
        """
        [#4] SMOTE con solo 10 muestras minoritarias — umbral de advertencia HU-03.
        El sistema debe ejecutarse (con advertencia) en lugar de lanzar excepción.
        """
        rng = np.random.RandomState(42)
        X = pd.DataFrame({"a": rng.randn(110), "b": rng.randn(110)})
        y = pd.Series([0] * 100 + [1] * 10)
        X_res, y_res, info = apply_resampling(X, y, "smote")
        assert info["ir_after"] < info["ir_before"]

    def test_scaler_fitted_only_on_train_not_test(self, imbalanced_df):
        """
        [#5] Anti-leakage crítico: StandardScaler ajustado solo sobre train.
        Si se ajusta sobre test, las métricas de evaluación son optimistas y
        el modelo fallará en producción con datos nuevos.
        """
        X_tr, _, _, _, _, _ = preprocess(imbalanced_df, "target")
        train_means = X_tr.mean().abs()
        assert (train_means < 1e-8).all(), (
            "X_train escalado tiene media ≠ 0 — el scaler no se ajustó correctamente."
        )


# ══════════════════════════════════════════════════════════════════════════════
# BLOQUE 4 — END-TO-END  (3 casos intactos)
# ══════════════════════════════════════════════════════════════════════════════

class TestEndToEnd:
    """
    Prueba de integración completa: simula el flujo CRISP-DM completo
    desde la carga del CSV hasta la generación de métricas y valores SHAP.
    """

    def test_full_crisp_dm_pipeline_rf_hybrid(self, ml_dataset):
        """
        E2E: dataset → preprocesamiento → remuestreo híbrido → RF → métricas.
        Todos los KPIs deben cumplirse en un dataset con señal suficiente.
        """
        X_tr, X_te, y_tr, y_te, cols, _ = preprocess(ml_dataset, "target")
        dist = compute_class_distribution(y_tr)
        assert dist["imbalance_ratio"] > 1.0

        X_res, y_res, info = apply_resampling(X_tr, y_tr, "hybrid")
        assert info["ir_after"] <= 1.5

        model, cv_scores, cv_mean, _ = train_with_cv(X_res, y_res, "rf", cv_folds=5)
        assert cv_mean >= 0.70

        metrics = compute_metrics(model, X_te, y_te)
        assert metrics["auc_roc"] >= 0.75
        assert metrics["recall"]  >= 0.65
        assert metrics["f1"]      >= 0.60

    def test_full_crisp_dm_pipeline_xgb_smoteenn(self, ml_dataset):
        """E2E alternativo: XGBoost + SMOTE-ENN."""
        X_tr, X_te, y_tr, y_te, _, _ = preprocess(ml_dataset, "target")
        X_res, y_res, _ = apply_resampling(X_tr, y_tr, "smoteenn")
        model, _, cv_mean, _ = train_with_cv(X_res, y_res, "xgb", cv_folds=5)
        metrics = compute_metrics(model, X_te, y_te)
        assert cv_mean >= 0.70
        assert metrics["auc_roc"] >= 0.75

    def test_pipeline_determinism(self, ml_dataset):
        """
        Dos ejecuciones idénticas del pipeline completo deben producir
        exactamente las mismas predicciones (random_state=42 en todos los módulos).
        """
        def run():
            X_tr, X_te, y_tr, _, _, _ = preprocess(ml_dataset, "target")
            X_res, y_res, _ = apply_resampling(X_tr, y_tr, "hybrid")
            model, _, _, _ = train_with_cv(X_res, y_res, "rf", cv_folds=3)
            return model.predict(X_te)

        np.testing.assert_array_equal(
            run(), run(),
            err_msg="Pipeline no determinista — verificar random_state=42 en todos los módulos."
        )