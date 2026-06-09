"""utils/resampling.py — Módulo de Remuestreo (CRISP-DM: Data Preparation — núcleo del proyecto)
Implementa las 4 técnicas: RUS, SMOTE, SMOTE-ENN y el ENFOQUE HÍBRIDO propuesto.

Enfoque híbrido (aporte propio):
  SMOTE-ENN  → sobre-muestreo limpio + eliminación de instancias ambiguas de clase mayoritaria
  Tomek Links → limpieza de pares fronterizos ruidosos en AMBAS clases
  Resultado: dataset balanceado sin ruido en la frontera de decisión.
  Aborda las limitaciones identificadas en la revisión PRISMA (64 estudios, 2022–2026).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler, TomekLinks

RANDOM_STATE = 42

ALGORITHMS = {
    "RUS (Random Under-Sampling)":        "rus",
    "SMOTE":                              "smote",
    "SMOTE-ENN":                          "smoteenn",
    "SMOTE-ENN + Tomek (híbrido)":        "hybrid",
}

ALGORITHM_DESCRIPTIONS = {
    "rus":      "Elimina instancias de la clase mayoritaria aleatoriamente. Rápido, puede perder información relevante.",
    "smote":    "Genera muestras sintéticas interpolando vecinos cercanos de la clase minoritaria. Puede introducir ruido.",
    "smoteenn": "SMOTE + Edited Nearest Neighbours. Sobre-muestrea y limpia instancias ambiguas de la clase mayoritaria.",
    "hybrid":   "SMOTE-ENN seguido de Tomek Links. Limpia la frontera de decisión en ambas clases. Propuesta del proyecto.",
}


def apply_resampling(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    algorithm_key: str,
    sampling_ratio: float = 1.0,
) -> tuple[pd.DataFrame, pd.Series, dict]:
    """
    Aplica la técnica de remuestreo seleccionada sobre el conjunto de entrenamiento.

    Parameters
    ----------
    X_train       : Features de entrenamiento (ya escaladas)
    y_train       : Etiquetas de entrenamiento
    algorithm_key : Clave interna del algoritmo (ver ALGORITHMS)
    sampling_ratio: Ratio objetivo (1.0 = balance perfecto)

    Returns
    -------
    X_resampled, y_resampled, resampling_info dict
    """
    feature_cols = X_train.columns.tolist()
    X_arr = X_train.values
    y_arr = y_train.values

    sampler = _build_sampler(algorithm_key, sampling_ratio)
    X_res, y_res = sampler.fit_resample(X_arr, y_arr)

    X_resampled = pd.DataFrame(X_res, columns=feature_cols)
    y_resampled = pd.Series(y_res, name=y_train.name)

    before = dict(zip(*np.unique(y_arr, return_counts=True)))
    after  = dict(zip(*np.unique(y_res,  return_counts=True)))
    ir_before = max(before.values()) / min(before.values())
    ir_after  = max(after.values())  / min(after.values())  if min(after.values()) > 0 else 1.0

    info = {
        "algorithm":    algorithm_key,
        "before":       {int(k): int(v) for k, v in before.items()},
        "after":        {int(k): int(v) for k, v in after.items()},
        "ir_before":    round(ir_before, 3),
        "ir_after":     round(ir_after, 3),
        "samples_added": int(len(X_res) - len(X_arr)),
    }
    return X_resampled, y_resampled, info


def _build_sampler(algorithm_key: str, sampling_ratio: float):
    """Construye el sampler de imbalanced-learn según la clave de algoritmo."""
    ratio = {1: sampling_ratio} if sampling_ratio < 1.0 else "auto"

    if algorithm_key == "rus":
        return RandomUnderSampler(
            sampling_strategy=ratio,
            random_state=RANDOM_STATE,
        )
    elif algorithm_key == "smote":
        return SMOTE(
            sampling_strategy=ratio,
            random_state=RANDOM_STATE,
            k_neighbors=5,
        )
    elif algorithm_key == "smoteenn":
        return SMOTEENN(
            sampling_strategy=ratio,
            random_state=RANDOM_STATE,
        )
    elif algorithm_key == "hybrid":
        # Enfoque híbrido propuesto:
        # Paso 1 — SMOTE-ENN: sobre-muestrea + limpia ambigüedades en mayoritaria
        # Paso 2 — Tomek Links: limpia pares ruidosos en la frontera de ambas clases
        return _HybridSMOTEENNTomek(sampling_ratio=ratio, random_state=RANDOM_STATE)
    else:
        raise ValueError(f"Algoritmo desconocido: {algorithm_key}")


class _HybridSMOTEENNTomek:
    """Implementación del enfoque híbrido SMOTE-ENN → Tomek Links.
    Separa los dos pasos para permitir inspección intermedia si se requiere.
    """

    def __init__(self, sampling_ratio="auto", random_state: int = 42):
        self.sampling_ratio = sampling_ratio
        self.random_state   = random_state
        self._step1 = SMOTEENN(
            sampling_strategy=sampling_ratio,
            random_state=random_state,
        )
        self._step2 = TomekLinks(sampling_strategy="auto")

    def fit_resample(self, X, y):
        X1, y1 = self._step1.fit_resample(X, y)
        X2, y2 = self._step2.fit_resample(X1, y1)
        return X2, y2
