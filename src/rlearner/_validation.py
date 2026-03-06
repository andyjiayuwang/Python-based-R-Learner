"""Validation helpers shared by nuisance-estimation components."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.utils.validation import check_array


def validate_features(X: Any) -> np.ndarray:
    """Validate and coerce a feature matrix to a dense 2D numpy array."""
    return check_array(X, ensure_2d=True, dtype=None, ensure_all_finite="allow-nan")


def validate_vector(values: Any, *, name: str, n_samples: int | None = None) -> np.ndarray:
    """Validate and coerce a one-dimensional target or prediction vector."""
    vector = check_array(values, ensure_2d=False, dtype=None, ensure_all_finite="allow-nan")
    vector = np.asarray(vector).reshape(-1)

    if n_samples is not None and vector.shape[0] != n_samples:
        raise ValueError(f"{name} must have length {n_samples}, got {vector.shape[0]}.")

    return vector


def validate_binary_treatment(d: Any, *, n_samples: int | None = None) -> np.ndarray:
    """Validate that treatment assignments are binary-coded."""
    treatment = validate_vector(d, name="d", n_samples=n_samples)
    unique_values = np.unique(treatment)

    if unique_values.size > 2 or not np.all(np.isin(unique_values, [0, 1])):
        raise ValueError(
            "Binary treatment is required for the built-in nuisance estimators. "
            "Expected values in {0, 1}."
        )

    return treatment.astype(int, copy=False)


def validate_manual_predictions(
    *,
    y: Any,
    d: Any,
    y_hat: Any,
    d_hat: Any,
    propensity_clip: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Validate manual nuisance predictions against observed outcomes and treatment."""
    y_array = validate_vector(y, name="y")
    d_array = validate_binary_treatment(d, n_samples=y_array.shape[0])
    y_hat_array = validate_vector(y_hat, name="y_hat", n_samples=y_array.shape[0])
    d_hat_array = validate_vector(d_hat, name="d_hat", n_samples=y_array.shape[0])

    if np.any(~np.isfinite(y_hat_array)):
        raise ValueError("y_hat must contain only finite values.")
    if np.any(~np.isfinite(d_hat_array)):
        raise ValueError("d_hat must contain only finite values.")

    d_hat_array = np.clip(d_hat_array.astype(float, copy=False), propensity_clip, 1.0 - propensity_clip)

    return y_array, d_array, y_hat_array, d_hat_array
