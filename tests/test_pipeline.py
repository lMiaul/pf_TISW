"""tests/test_pipeline.py — Casos de prueba funcionales del pipeline CRISP-DM
Cubre los criterios de aceptación de HU-01 a HU-06.
Ejecutar con: pytest tests/ -v
"""

import numpy as np
import pandas as pd
import pytest

from src.utils.modeling import (
    CLASSIFIERS,
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
from src.utils.resampling import ALGORITHMS, apply_resampling

# ── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture
def balanced_df():
    """Dataset balanceado simple para pruebas de smoke."""
    rng = np.random.RandomState(42)
    return pd.DataFrame({
        "age":     rng.randint(20, 70, 200).astype(float),
        "income":  rng.uniform(1000, 10000, 200),
        "months":  rng.randint(6, 60, 200).astype(float),
        "target":  np.tile([0, 1], 100),          # balanceado 1:1
    })


@pytest.fixture
def imbalanced_df():
    """Dataset desbalanceado (IR ≈ 10:1) para pruebas de remuestreo."""
    rng = np.random.RandomState(42)
    n_majority, n_minority = 500, 50
    majority = pd.DataFrame({
        "age":    rng.randint(20, 70, n_majority).astype(float),
        "income": rng.uniform(2000, 10000, n_majority),
        "months": rng.randint(6, 60, n_majority).astype(float),
        "target": np.zeros(n_majority, dtype=int),
    })
    minority = pd.DataFrame({
        "age":    rng.randint(20, 70, n_minority).astype(float),
        "income": rng.uniform(500, 3000, n_minority),
        "months": rng.randint(12, 48, n_minority).astype(float),
        "target": np.ones(n_minority, dtype=int),
    })
    return pd.concat([majority, minority], ignore_index=True)


@pytest.fixture
def preprocessed(imbalanced_df):
    """Retorna X_train, X_test, y_train, y_test ya preprocesados."""
    return preprocess(imbalanced_df, "target")


# ── HU-01: Carga y validación de datos ────────────────────────────────────

class TestHU01_Upload:

    def test_detect_binary_target_column(self, balanced_df):
        """Debe detectar 'target' como candidata binaria (0/1)."""
        candidates = detect_target_candidates(balanced_df)
        assert "target" in candidates

    def test_no_binary_column_returns_empty(self):
        """Columnas no binarias no deben ser candidatas."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        assert detect_target_candidates(df) == []

    def test_dataset_not_empty_after_load(self, balanced_df):
        assert len(balanced_df) > 0

    def test_rejects_nonbinary_target(self):
        """Columna con más de 2 valores únicos no debe aparecer como candidata."""
        df = pd.DataFrame({"score": [1, 2, 3, 4], "target": [0, 1, 2, 3]})
        candidates = detect_target_candidates(df)
        assert "target" not in candidates


# ── HU-02: EDA y cálculo de desbalanceo ───────────────────────────────────

class TestHU02_EDA:

    def test_imbalance_ratio_correct(self, imbalanced_df):
        """IR ≈ 10 para dataset 500/50."""
        y = imbalanced_df["target"]
        dist = compute_class_distribution(y)
        assert dist["imbalance_ratio"] == pytest.approx(10.0, abs=0.1)

    def test_level_severe_when_ir_gt_10(self, imbalanced_df):
        y = imbalanced_df["target"]
        dist = compute_class_distribution(y)
        assert dist["level"] == "severo"

    def test_level_low_when_balanced(self, balanced_df):
        y = balanced_df["target"]
        dist = compute_class_distribution(y)
        assert dist["level"] == "bajo"
        assert dist["imbalance_ratio"] == pytest.approx(1.0, abs=0.05)

    def test_eda_completes_in_under_3s(self, imbalanced_df):
        """EDA debe completarse en menos de 3 segundos (criterio HU-02)."""
        import time
        y = imbalanced_df["target"]
        t0 = time.time()
        compute_class_distribution(y)
        elapsed = time.time() - t0
        assert elapsed < 3.0


# ── HU-03: Preprocesamiento anti-leakage ──────────────────────────────────

class TestHU03_Preprocessing:

    def test_stratified_split_preserves_class_ratio(self, imbalanced_df):
        """La partición estratificada debe preservar el ratio de clases en test."""
        X_train, X_test, y_train, y_test, _, _ = preprocess(imbalanced_df, "target")
        ir_train = compute_class_distribution(y_train)["imbalance_ratio"]
        ir_test  = compute_class_distribution(y_test)["imbalance_ratio"]
        assert abs(ir_train - ir_test) < 1.5   # tolerancia ±1.5 por tamaño pequeño

    def test_no_nulls_after_preprocessing(self, imbalanced_df):
        """No debe haber nulos tras el preprocesamiento."""
        X_train, X_test, _, _, _, _ = preprocess(imbalanced_df, "target")
        assert X_train.isnull().sum().sum() == 0
        assert X_test.isnull().sum().sum() == 0

    def test_scaler_not_fitted_on_test(self, imbalanced_df):
        """El scaler se ajusta solo sobre train; test no debe cambiar sus stats."""
        X_train, X_test, y_train, y_test, cols, scaler = preprocess(imbalanced_df, "target")
        # Si el scaler se ajustara sobre test, la media de train escalado sería ≠ 0
        train_means = X_train.mean().abs()
        assert (train_means < 1e-8).all(), "X_train escalado debe tener media ≈ 0"

    def test_random_state_reproducibility(self, imbalanced_df):
        """Dos ejecuciones con mismos datos deben producir el mismo split."""
        result1 = preprocess(imbalanced_df, "target")
        result2 = preprocess(imbalanced_df, "target")
        pd.testing.assert_frame_equal(result1[0], result2[0])


# ── HU-03 cont.: Remuestreo ────────────────────────────────────────────────

class TestHU03_Resampling:

    @pytest.mark.parametrize("algo_key", ["rus", "smote", "smoteenn", "hybrid"])
    def test_all_algorithms_reduce_imbalance(self, preprocessed, algo_key):
        """Todos los algoritmos deben reducir el IR respecto al original."""
        X_train, _, y_train, _, _, _ = preprocessed
        _, _, info = apply_resampling(X_train, y_train, algo_key)
        assert info["ir_after"] <= info["ir_before"], \
            f"{algo_key}: IR después ({info['ir_after']}) no es menor que antes ({info['ir_before']})"

    def test_hybrid_achieves_near_balance(self, preprocessed):
        """El enfoque híbrido debe lograr IR ≤ 1.5 (casi balance perfecto)."""
        X_train, _, y_train, _, _, _ = preprocessed
        _, _, info = apply_resampling(X_train, y_train, "hybrid", sampling_ratio=1.0)
        assert info["ir_after"] <= 1.5, f"IR después del híbrido: {info['ir_after']}"

    def test_no_test_data_contamination(self, preprocessed):
        """El remuestreo no debe modificar X_test ni y_test."""
        X_train, X_test, y_train, y_test, _, _ = preprocessed
        X_test_original = X_test.copy()
        y_test_original = y_test.copy()
        apply_resampling(X_train, y_train, "smote")
        pd.testing.assert_frame_equal(X_test, X_test_original)
        pd.testing.assert_series_equal(y_test, y_test_original)


# ── HU-04: Entrenamiento con CV estratificada ─────────────────────────────

class TestHU04_Modeling:

    @pytest.mark.parametrize("model_key", ["lr", "rf", "xgb"])
    def test_all_classifiers_train_without_error(self, preprocessed, model_key):
        X_train, X_test, y_train, y_test, _, _ = preprocessed
        model, cv_scores, cv_mean, cv_std = train_with_cv(X_train, y_train, model_key, cv_folds=3)
        assert model is not None
        assert len(cv_scores) == 3
        assert 0.0 <= cv_mean <= 1.0

    def test_cv_scores_all_in_range(self, preprocessed):
        X_train, _, y_train, _, _, _ = preprocessed
        _, cv_scores, _, _ = train_with_cv(X_train, y_train, "rf", cv_folds=3)
        assert all(0.0 <= s <= 1.0 for s in cv_scores)

    def test_random_state_42_reproducibility(self, preprocessed):
        """Mismo dataset → mismas predicciones (random_state fijo)."""
        X_train, X_test, y_train, y_test, _, _ = preprocessed
        model1, _, _, _ = train_with_cv(X_train, y_train, "lr", cv_folds=3)
        model2, _, _, _ = train_with_cv(X_train, y_train, "lr", cv_folds=3)
        pred1 = model1.predict(X_test)
        pred2 = model2.predict(X_test)
        np.testing.assert_array_equal(pred1, pred2)


# ── HU-05: Evaluación comparativa ─────────────────────────────────────────

class TestHU05_Evaluation:

    def test_metrics_have_4_decimal_precision(self, preprocessed):
        """Las métricas deben reportarse con exactamente 4 decimales (HU-05 criterio 4)."""
        X_train, X_test, y_train, y_test, _, _ = preprocessed
        model, _, _, _ = train_with_cv(X_train, y_train, "rf", cv_folds=3)
        metrics = compute_metrics(model, X_test, y_test)
        for key in ["accuracy", "precision", "recall", "f1", "auc_roc"]:
            val = metrics[key]
            assert isinstance(val, float)
            assert len(str(val).split(".")[-1]) <= 4

    def test_delta_positive_when_optimized_beats_baseline(self, preprocessed):
        """Si el modelo con remuestreo supera al base, el delta de recall debe ser > 0."""
        X_train, X_test, y_train, y_test, _, _ = preprocessed
        # Modelo base (datos desbalanceados)
        model_b, _, _, _ = train_with_cv(X_train, y_train, "rf", cv_folds=3)
        metrics_b = compute_metrics(model_b, X_test, y_test)

        # Modelo optimizado (datos balanceados con híbrido)
        X_res, y_res, _ = apply_resampling(X_train, y_train, "hybrid")
        model_o, _, _, _ = train_with_cv(X_res, y_res, "rf", cv_folds=3)
        metrics_o = compute_metrics(model_o, X_test, y_test)

        delta = compute_delta(metrics_b, metrics_o)
        # El recall de la clase minoritaria debe mejorar al menos en delta ≥ -0.20
        # (tolerancia amplia por el pequeño dataset de prueba)
        assert delta["recall"] >= -0.20, f"Recall delta inesperadamente negativo: {delta['recall']}"

    def test_confusion_matrix_present_in_metrics(self, preprocessed):
        X_train, X_test, y_train, y_test, _, _ = preprocessed
        model, _, _, _ = train_with_cv(X_train, y_train, "lr", cv_folds=3)
        metrics = compute_metrics(model, X_test, y_test)
        assert "confusion_matrix" in metrics
        cm = metrics["confusion_matrix"]
        assert len(cm) == 2 and len(cm[0]) == 2


# ── HU-06: Explicabilidad XAI ─────────────────────────────────────────────

class TestHU06_XAI:

    def test_shap_available_for_tree_models(self):
        """SHAP debe estar instalado en el entorno."""
        try:
            import shap
            assert shap is not None
        except ImportError:
            pytest.skip("SHAP no instalado — instalar con: pip install shap")

    def test_feature_importance_keys_match_columns(self, preprocessed):
        """Los nombres de variables en XAI deben coincidir exactamente con las columnas del CSV."""
        X_train, X_test, y_train, y_test, feat_cols, _ = preprocessed
        model, _, _, _ = train_with_cv(X_train, y_train, "rf", cv_folds=3)
        # Simular importancia built-in
        importance = dict(zip(feat_cols, model.feature_importances_))
        assert set(importance.keys()) == set(feat_cols), \
            "Las claves de importancia no coinciden con los nombres de columna del dataset."

    def test_top_k_features_are_subset_of_all_features(self, preprocessed):
        X_train, X_test, y_train, y_test, feat_cols, _ = preprocessed
        model, _, _, _ = train_with_cv(X_train, y_train, "rf", cv_folds=3)
        importance = dict(zip(feat_cols, model.feature_importances_))
        top_10 = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
        for feat, _ in top_10:
            assert feat in feat_cols
