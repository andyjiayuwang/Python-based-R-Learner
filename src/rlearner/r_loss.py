"""Step 2 R-loss learner components for the R-learner package."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, RegressorMixin, clone

from ._validation import validate_features, validate_vector


class RLossWrapper(BaseEstimator, RegressorMixin):
    """Wrap any sklearn-style regressor for the second-stage R-loss fit."""

    def __init__(self, base_learner: BaseEstimator) -> None:
        self.base_learner = base_learner
        self.model_: BaseEstimator | None = None
        self.w_res_: np.ndarray | None = None

    def fit(self, X: Any, y_res: Any, w_res: Any) -> "RLossWrapper":
        X_array = validate_features(X)
        y_res_array = validate_vector(y_res, name="y_res", n_samples=X_array.shape[0]).astype(float, copy=False)
        w_res_array = validate_vector(w_res, name="w_res", n_samples=X_array.shape[0]).astype(float, copy=False)

        self.w_res_ = w_res_array
        self.model_ = clone(self.base_learner)
        self.model_.fit(X_array * w_res_array[:, None], y_res_array)
        return self

    def predict(self, X: Any) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("The R-loss model is not fitted yet.")
        X_array = validate_features(X)
        return np.asarray(self.model_.predict(X_array), dtype=float)


class RLossStacking(BaseEstimator):
    """Nonnegative stacking for combining second-stage R-loss CATE predictions."""

    def __init__(self, lambda_reg: float = 1.0, tolerance: float = 1e-10, max_iter: int = 1000) -> None:
        if lambda_reg < 0:
            raise ValueError("lambda_reg must be nonnegative.")
        if tolerance <= 0:
            raise ValueError("tolerance must be positive.")
        if max_iter < 1:
            raise ValueError("max_iter must be at least 1.")

        self.lambda_reg = lambda_reg
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.a_hat: float | None = None
        self.b_hat: float | None = None
        self.alpha_hat: np.ndarray | None = None
        self.beta_hat: np.ndarray | None = None
        self.n_learners_: int | None = None
        self.optimization_result_ = None

    def fit(self, tau_matrix: Any, y_res: Any, w_res: Any) -> "RLossStacking":
        tau_array = validate_features(tau_matrix).astype(float, copy=False)
        y_res_array = validate_vector(y_res, name="y_res", n_samples=tau_array.shape[0]).astype(float, copy=False)
        w_res_array = validate_vector(w_res, name="w_res", n_samples=tau_array.shape[0]).astype(float, copy=False)

        n_samples, n_learners = tau_array.shape
        if n_learners == 0:
            raise ValueError("tau_matrix must contain at least one learner column.")

        x_meta = np.column_stack([w_res_array, w_res_array[:, None] * tau_array])

        def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
            residual = y_res_array - x_meta @ theta
            beta = theta[1:]
            value = float(residual @ residual + self.lambda_reg * (beta @ beta))
            gradient = -2.0 * x_meta.T @ residual
            gradient[1:] += 2.0 * self.lambda_reg * beta
            return value, gradient

        initial_theta = np.zeros(n_learners + 1, dtype=float)
        bounds = [(None, None)] + [(0.0, None)] * n_learners

        result = minimize(
            fun=lambda theta: objective(theta)[0],
            x0=initial_theta,
            jac=lambda theta: objective(theta)[1],
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": self.max_iter},
        )

        if not result.success:
            raise RuntimeError(f"RLossStacking optimization failed: {result.message}")

        beta_hat = np.maximum(result.x[1:], 0.0)
        b_hat = float(np.linalg.norm(beta_hat))
        alpha_hat = np.zeros_like(beta_hat)
        if b_hat > self.tolerance:
            alpha_hat = beta_hat / b_hat

        self.a_hat = float(result.x[0])
        self.b_hat = b_hat
        self.alpha_hat = alpha_hat
        self.beta_hat = beta_hat
        self.n_learners_ = n_learners
        self.optimization_result_ = result
        return self

    def predict(self, tau_matrix: Any) -> np.ndarray:
        if self.beta_hat is None or self.a_hat is None or self.n_learners_ is None:
            raise RuntimeError("The R-loss stacking model is not fitted yet.")

        tau_array = validate_features(tau_matrix).astype(float, copy=False)
        if tau_array.shape[1] != self.n_learners_:
            raise ValueError(
                f"tau_matrix must have {self.n_learners_} columns, got {tau_array.shape[1]}."
            )

        return self.a_hat + tau_array @ self.beta_hat
