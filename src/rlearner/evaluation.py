"""Evaluation routines for fitted R-learner models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.stats import norm

from ._validation import validate_features, validate_vector
from .nuisance import NuisanceResult


@dataclass(slots=True)
class CoefficientInference:
    """Point estimate and robust inference for a single coefficient."""

    estimate: float
    std_error: float
    z_value: float
    p_value: float
    confidence_interval: tuple[float, float]


@dataclass(slots=True)
class BLPTestResult:
    """Best linear predictor test output for the residual-on-residual regression."""

    alpha: CoefficientInference
    beta: CoefficientInference
    coefficients: np.ndarray
    std_errors: np.ndarray
    z_values: np.ndarray
    p_values: np.ndarray
    confidence_intervals: np.ndarray
    covariance: np.ndarray
    confidence_level: float
    n_obs: int


@dataclass(slots=True)
class CalibrationBinResult:
    """Bin-level summary for the calibration test."""

    bin_index: int
    size: int
    theta_star: float
    theta_dr: float
    gap: float
    tau_min: float
    tau_max: float


@dataclass(slots=True)
class CalibrationTestResult:
    """Calibration-test output for grouped predicted treatment effects."""

    bins: list[CalibrationBinResult]
    cal_l1: float
    cal_l2: float
    n_bins: int
    n_obs: int


@dataclass(slots=True)
class UpliftCurvePoint:
    """One point on the DR uplift curve."""

    fraction: float
    size: int
    theta_dr: float


@dataclass(slots=True)
class UpliftTestResult:
    """Ranking-based DR uplift curve and its area summary."""

    curve: list[UpliftCurvePoint]
    fractions: np.ndarray
    subgroup_sizes: np.ndarray
    theta_dr: np.ndarray
    auuc: float
    n_obs: int


def _check_confidence_level(confidence_level: float) -> None:
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must lie strictly between 0 and 1.")


def _validate_uplift_fractions(fractions: Any | None) -> np.ndarray:
    if fractions is None:
        fractions_array = np.arange(0.1, 1.01, 0.1)
    else:
        fractions_array = validate_vector(fractions, name="fractions").astype(float, copy=False)

    if fractions_array.size == 0:
        raise ValueError("fractions must contain at least one value.")
    if np.any((fractions_array <= 0.0) | (fractions_array > 1.0)):
        raise ValueError("fractions must lie in the interval (0, 1].")

    return np.sort(fractions_array)


def blp_test(
    *,
    tau_hat: Any,
    nuisance_result: NuisanceResult,
    confidence_level: float = 0.95,
) -> BLPTestResult:
    """Run the no-intercept BLP test with HC2 robust inference."""
    _check_confidence_level(confidence_level)

    y_tilde = validate_vector(
        nuisance_result.y_residual,
        name="y_tilde",
    ).astype(float, copy=False)
    d_tilde = validate_vector(
        nuisance_result.d_residual,
        name="d_tilde",
        n_samples=y_tilde.shape[0],
    ).astype(float, copy=False)
    tau_array = validate_vector(tau_hat, name="tau_hat", n_samples=y_tilde.shape[0]).astype(float, copy=False)

    design = validate_features(np.column_stack([d_tilde, d_tilde * tau_array])).astype(float, copy=False)
    xtx = design.T @ design

    try:
        xtx_inv = np.linalg.inv(xtx)
    except np.linalg.LinAlgError as exc:
        raise np.linalg.LinAlgError("BLP test design matrix is singular and cannot be inverted.") from exc

    coefficients = xtx_inv @ (design.T @ y_tilde)
    residual = y_tilde - design @ coefficients
    leverage = np.einsum("ij,jk,ik->i", design, xtx_inv, design)
    hc2_scale = (residual ** 2) / np.maximum(1.0 - leverage, 1e-12)
    covariance = xtx_inv @ (design.T @ (design * hc2_scale[:, None])) @ xtx_inv

    std_errors = np.sqrt(np.diag(covariance))
    z_values = np.divide(
        coefficients,
        std_errors,
        out=np.full_like(coefficients, np.nan, dtype=float),
        where=std_errors > 0,
    )
    p_values = 2.0 * (1.0 - norm.cdf(np.abs(z_values)))

    alpha_level = 1.0 - confidence_level
    critical_value = norm.ppf(1.0 - alpha_level / 2.0)
    confidence_intervals = np.column_stack(
        [
            coefficients - critical_value * std_errors,
            coefficients + critical_value * std_errors,
        ]
    )

    alpha_result = CoefficientInference(
        estimate=float(coefficients[0]),
        std_error=float(std_errors[0]),
        z_value=float(z_values[0]),
        p_value=float(p_values[0]),
        confidence_interval=(float(confidence_intervals[0, 0]), float(confidence_intervals[0, 1])),
    )
    beta_result = CoefficientInference(
        estimate=float(coefficients[1]),
        std_error=float(std_errors[1]),
        z_value=float(z_values[1]),
        p_value=float(p_values[1]),
        confidence_interval=(float(confidence_intervals[1, 0]), float(confidence_intervals[1, 1])),
    )

    return BLPTestResult(
        alpha=alpha_result,
        beta=beta_result,
        coefficients=coefficients,
        std_errors=std_errors,
        z_values=z_values,
        p_values=p_values,
        confidence_intervals=confidence_intervals,
        covariance=covariance,
        confidence_level=confidence_level,
        n_obs=y_tilde.shape[0],
    )


def calibration_test(
    *,
    tau_hat: Any,
    nuisance_result: NuisanceResult,
    n_bins: int = 5,
    tolerance: float = 1e-12,
) -> CalibrationTestResult:
    """Run grouped L1/L2 calibration tests using residualized outcomes and treatment."""
    y_tilde = validate_vector(nuisance_result.y_residual, name="y_tilde").astype(float, copy=False)
    d_tilde = validate_vector(
        nuisance_result.d_residual,
        name="d_tilde",
        n_samples=y_tilde.shape[0],
    ).astype(float, copy=False)
    tau_array = validate_vector(tau_hat, name="tau_hat", n_samples=y_tilde.shape[0]).astype(float, copy=False)

    if n_bins < 1:
        raise ValueError("n_bins must be at least 1.")
    if n_bins > tau_array.shape[0]:
        raise ValueError("n_bins cannot exceed the number of observations.")

    order = np.argsort(tau_array, kind="mergesort")
    grouped_indices = np.array_split(order, n_bins)

    bin_results: list[CalibrationBinResult] = []
    cal_l1 = 0.0
    cal_l2 = 0.0

    for bin_index, group in enumerate(grouped_indices, start=1):
        if group.size == 0:
            continue

        theta_star = float(np.mean(tau_array[group]))
        denominator = float(np.sum(d_tilde[group] ** 2))
        if denominator <= tolerance:
            raise RuntimeError(
                f"Calibration bin {bin_index} has near-zero residualized treatment variation."
            )
        theta_dr = float(np.sum(d_tilde[group] * y_tilde[group]) / denominator)
        gap = theta_dr - theta_star
        group_size = int(group.size)

        cal_l1 += abs(gap) * group_size
        cal_l2 += (gap ** 2) * group_size
        bin_results.append(
            CalibrationBinResult(
                bin_index=bin_index,
                size=group_size,
                theta_star=theta_star,
                theta_dr=theta_dr,
                gap=float(gap),
                tau_min=float(np.min(tau_array[group])),
                tau_max=float(np.max(tau_array[group])),
            )
        )

    return CalibrationTestResult(
        bins=bin_results,
        cal_l1=float(cal_l1),
        cal_l2=float(cal_l2),
        n_bins=len(bin_results),
        n_obs=y_tilde.shape[0],
    )


def uplift_test(
    *,
    tau_hat: Any,
    nuisance_result: NuisanceResult,
    fractions: Any | None = None,
    tolerance: float = 1e-12,
) -> UpliftTestResult:
    """Run ranking-based validation using a DR uplift curve and AUUC."""
    y_tilde = validate_vector(nuisance_result.y_residual, name="y_tilde").astype(float, copy=False)
    d_tilde = validate_vector(
        nuisance_result.d_residual,
        name="d_tilde",
        n_samples=y_tilde.shape[0],
    ).astype(float, copy=False)
    tau_array = validate_vector(tau_hat, name="tau_hat", n_samples=y_tilde.shape[0]).astype(float, copy=False)
    fractions_array = _validate_uplift_fractions(fractions)

    n_obs = tau_array.shape[0]
    order = np.argsort(-tau_array, kind="mergesort")

    curve: list[UpliftCurvePoint] = []
    subgroup_sizes = []
    theta_values = []

    for fraction in fractions_array:
        subgroup_size = max(1, int(np.ceil(fraction * n_obs)))
        subgroup = order[:subgroup_size]
        denominator = float(np.sum(d_tilde[subgroup] ** 2))
        if denominator <= tolerance:
            raise RuntimeError(
                f"Top-{fraction:.4f} subgroup has near-zero residualized treatment variation."
            )
        theta_dr = float(np.sum(d_tilde[subgroup] * y_tilde[subgroup]) / denominator)
        subgroup_sizes.append(subgroup_size)
        theta_values.append(theta_dr)
        curve.append(
            UpliftCurvePoint(
                fraction=float(fraction),
                size=subgroup_size,
                theta_dr=theta_dr,
            )
        )

    subgroup_sizes_array = np.asarray(subgroup_sizes, dtype=int)
    theta_array = np.asarray(theta_values, dtype=float)
    fractions_unique, unique_indices = np.unique(fractions_array, return_index=True)
    auuc = float(np.trapz(theta_array[unique_indices], fractions_unique))

    return UpliftTestResult(
        curve=curve,
        fractions=fractions_array,
        subgroup_sizes=subgroup_sizes_array,
        theta_dr=theta_array,
        auuc=auuc,
        n_obs=n_obs,
    )
