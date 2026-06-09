"""utils/preprocessing.py — Módulo de Preprocesamiento (CRISP-DM: Data Preparation)
Implementa: imputación, codificación one-hot, estandarización y partición estratificada.
Principio anti-leakage: StandardScaler se ajusta SOLO sobre X_train.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42


def detect_target_candidates(df: pd.DataFrame) -> list[str]:
    """Devuelve columnas binarias candidatas a ser la variable objetivo."""
    candidates = []
    for col in df.columns:
        unique_vals = df[col].dropna().unique()
        if set(unique_vals).issubset({0, 1, "0", "1", True, False}):
            candidates.append(col)
    return candidates


def preprocess(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, list[str], StandardScaler]:
    """
    Pipeline completo de preprocesamiento.

    Returns
    -------
    X_train, X_test, y_train, y_test, feature_names, scaler
    """
    df = df.copy()

    # 1. Separar target
    y = df[target_col].astype(int)
    X = df.drop(columns=[target_col])

    # 2. Imputar nulos
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    for col in num_cols:
        X[col] = X[col].fillna(X[col].median())
    for col in cat_cols:
        X[col] = X[col].fillna(X[col].mode()[0])

    # 3. Codificación one-hot para categóricas
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    feature_names = X.columns.tolist()

    # 4. Partición estratificada (mantiene proporción de clases)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # 5. Estandarización — scaler ajustado SOLO sobre train (anti-leakage)
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=feature_names,
        index=X_train.index,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=feature_names,
        index=X_test.index,
    )

    return X_train_scaled, X_test_scaled, y_train, y_test, feature_names, scaler


def compute_class_distribution(y: pd.Series) -> dict:
    """Calcula distribución, ratio y nivel de desbalanceo."""
    counts = y.value_counts().to_dict()
    majority = max(counts.values())
    minority = min(counts.values())
    ir = round(majority / minority, 2) if minority > 0 else float("inf")

    if ir < 3:
        level = "bajo"
    elif ir < 10:
        level = "moderado"
    else:
        level = "severo"

    return {
        "counts": counts,
        "imbalance_ratio": ir,
        "level": level,
        "majority_class": max(counts, key=counts.get),
        "minority_class": min(counts, key=counts.get),
    }
