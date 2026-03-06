"""Step 1 nuisance-estimation building blocks for the R-learner package."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.metrics import log_loss, mean_squared_error, roc_auc_score
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold

from ._validation import (
    validate_binary_treatment,
    validate_features,
    validate_manual_predictions,
    validate_vector,
)


def _coerce_fit_params(fit_params: Mapping[str, Any] | None) -> dict[str, Any]:
    return {} if fit_params is None else dict(fit_params)


def _prepare_estimator(
    estimator: BaseEstimator,
    *,
    param_grid: Mapping[str, Sequence[Any]] | None,
    search_cv: int | KFold | StratifiedKFold,
    scoring: str | None,
    n_jobs: int | None,
) -> BaseEstimator:
    """Optionally wrap an estimator in GridSearchCV."""
    base_estimator = clone(estimator)
    if not param_grid:
        return base_estimator

    return GridSearchCV(
        estimator=base_estimator,
        param_grid=dict(param_grid),
        cv=search_cv,
        scoring=scoring,
        n_jobs=n_jobs,
        refit=True,
    )


def _predict_treatment_probability(model: BaseEstimator, X: np.ndarray) -> np.ndarray:
    """Predict treatment propensity as P(D=1|X) from a fitted classifier."""
    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(X))
        if probabilities.ndim != 2 or probabilities.shape[1] < 2:
            raise ValueError("Treatment model predict_proba must return class probabilities.")
        return probabilities[:, 1]

    raise TypeError(
        "Treatment model must implement predict_proba so the package can compute "
        "binary propensity scores."
    )


def _normalize_weight_vector(weights: np.ndarray, *, normalize_weights: bool, tolerance: float) -> np.ndarray:
    constrained = np.maximum(np.asarray(weights, dtype=float), 0.0)
    if normalize_weights:
        total = float(constrained.sum())
        if total <= tolerance:
            raise RuntimeError("Unable to normalize super learner weights because they sum to zero.")
        constrained = constrained / total
    return constrained


@dataclass(slots=True)
class NuisanceResult:
    """Container for cross-fitted nuisance predictions and residuals."""

    y_hat: np.ndarray
    d_hat: np.ndarray
    y_residual: np.ndarray
    d_residual: np.ndarray
    folds: np.ndarray | None = None
    diagnostics: dict[str, float] = field(default_factory=dict)
    outcome_models: list[BaseEstimator] = field(default_factory=list)
    treatment_models: list[BaseEstimator] = field(default_factory=list)
    outcome_model_full: BaseEstimator | None = None
    treatment_model_full: BaseEstimator | None = None


class ManualNuisanceEstimator(BaseEstimator):
    """Validate user-supplied nuisance predictions and package them consistently."""

    def __init__(self, *, propensity_clip: float = 1e-6) -> None:
        self.propensity_clip = propensity_clip
        self.result_: NuisanceResult | None = None

    def fit(self, *, y: Any, d: Any, y_hat: Any, d_hat: Any) -> "ManualNuisanceEstimator":
        y_array, d_array, y_hat_array, d_hat_array = validate_manual_predictions(
            y=y,
            d=d,
            y_hat=y_hat,
            d_hat=d_hat,
            propensity_clip=self.propensity_clip,
        )

        diagnostics = {
            "outcome_rmse": float(np.sqrt(mean_squared_error(y_array, y_hat_array))),
        }
        try:
            diagnostics["propensity_auc"] = float(roc_auc_score(d_array, d_hat_array))
            diagnostics["propensity_log_loss"] = float(log_loss(d_array, d_hat_array))
        except ValueError:
            pass

        self.result_ = NuisanceResult(
            y_hat=y_hat_array,
            d_hat=d_hat_array,
            y_residual=y_array - y_hat_array,
            d_residual=d_array - d_hat_array,
            folds=None,
            diagnostics=diagnostics,
        )
        return self

    def fit_predict(self, *, y: Any, d: Any, y_hat: Any, d_hat: Any) -> NuisanceResult:
        result = self.fit(y=y, d=d, y_hat=y_hat, d_hat=d_hat).result_
        if result is None:
            raise RuntimeError("Manual nuisance estimation did not produce a result.")
        return result


class _BaseConstrainedSuperLearner(BaseEstimator):
    """Base class for constrained nonnegative super learners."""

    def __init__(
        self,
        estimators: list[tuple[str, BaseEstimator]],
        *,
        estimator_param_grids: Mapping[str, Mapping[str, Sequence[Any]]] | None = None,
        search_cv: int = 5,
        search_scoring: str | None = None,
        search_n_jobs: int | None = None,
        search_shuffle: bool = True,
        random_state: int | None = 42,
        normalize_weights: bool = False,
        tolerance: float = 1e-10,
        max_iter: int = 1000,
    ) -> None:
        if not estimators:
            raise ValueError("estimators must contain at least one base learner.")
        if search_cv < 2:
            raise ValueError("search_cv must be at least 2.")
        if tolerance <= 0:
            raise ValueError("tolerance must be positive.")
        if max_iter < 1:
            raise ValueError("max_iter must be at least 1.")

        self.estimators = estimators
        self.estimator_param_grids = estimator_param_grids
        self.search_cv = search_cv
        self.search_scoring = search_scoring
        self.search_n_jobs = search_n_jobs
        self.search_shuffle = search_shuffle
        self.random_state = random_state
        self.normalize_weights = normalize_weights
        self.tolerance = tolerance
        self.max_iter = max_iter

    def _make_search_splitter(self, y: np.ndarray) -> int | KFold | StratifiedKFold:
        raise NotImplementedError

    def _fit_base_estimators(self, X: np.ndarray, y: np.ndarray) -> tuple[list[str], list[BaseEstimator], dict[str, dict[str, Any] | None]]:
        names: list[str] = []
        fitted_estimators: list[BaseEstimator] = []
        best_params: dict[str, dict[str, Any] | None] = {}

        for index, (name, estimator) in enumerate(self.estimators):
            names.append(name)
            param_grid = None if self.estimator_param_grids is None else self.estimator_param_grids.get(name)
            estimator_for_fit = _prepare_estimator(
                estimator,
                param_grid=param_grid,
                search_cv=self._make_search_splitter(y),
                scoring=self.search_scoring,
                n_jobs=self.search_n_jobs,
            )
            fitted = clone(estimator_for_fit)
            fitted.fit(X, y)
            fitted_estimators.append(fitted)
            if hasattr(fitted, "best_params_"):
                best_params[name] = dict(fitted.best_params_)
            else:
                best_params[name] = None
        return names, fitted_estimators, best_params

    def get_weights(self) -> dict[str, float]:
        if not hasattr(self, "weights_") or not hasattr(self, "estimator_names_"):
            raise RuntimeError("The super learner is not fitted yet.")
        return {name: float(weight) for name, weight in zip(self.estimator_names_, self.weights_)}

    def get_best_params(self) -> dict[str, dict[str, Any] | None]:
        if not hasattr(self, "best_params_"):
            raise RuntimeError("The super learner is not fitted yet.")
        return dict(self.best_params_)


class SuperLearnerRegressor(_BaseConstrainedSuperLearner, RegressorMixin):
    """Constrained super learner for regression with nonnegative base-learner weights."""

    def _make_search_splitter(self, y: np.ndarray) -> KFold:
        return KFold(
            n_splits=self.search_cv,
            shuffle=self.search_shuffle,
            random_state=self.random_state if self.search_shuffle else None,
        )

    def fit(self, X: Any, y: Any) -> "SuperLearnerRegressor":
        X_array = validate_features(X)
        y_array = validate_vector(y, name="y", n_samples=X_array.shape[0]).astype(float, copy=False)

        self.estimator_names_, self.fitted_estimators_, self.best_params_ = self._fit_base_estimators(X_array, y_array)
        prediction_matrix = np.column_stack([
            np.asarray(estimator.predict(X_array), dtype=float) for estimator in self.fitted_estimators_
        ])

        def objective(weights: np.ndarray) -> tuple[float, np.ndarray]:
            residual = y_array - prediction_matrix @ weights
            value = float(residual @ residual)
            gradient = -2.0 * prediction_matrix.T @ residual
            return value, gradient

        initial = np.full(prediction_matrix.shape[1], 1.0 / prediction_matrix.shape[1], dtype=float)
        bounds = [(0.0, None)] * prediction_matrix.shape[1]
        constraints = []
        if self.normalize_weights:
            constraints.append({"type": "eq", "fun": lambda w: np.sum(w) - 1.0, "jac": lambda w: np.ones_like(w)})

        method = "SLSQP" if self.normalize_weights else "L-BFGS-B"
        result = minimize(
            fun=lambda w: objective(w)[0],
            x0=initial,
            jac=lambda w: objective(w)[1],
            method=method,
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": self.max_iter},
        )
        if not result.success:
            raise RuntimeError(f"SuperLearnerRegressor optimization failed: {result.message}")

        self.weights_ = _normalize_weight_vector(result.x, normalize_weights=self.normalize_weights, tolerance=self.tolerance)
        self.optimization_result_ = result
        return self

    def predict(self, X: Any) -> np.ndarray:
        if not hasattr(self, "weights_"):
            raise RuntimeError("The super learner is not fitted yet.")
        X_array = validate_features(X)
        prediction_matrix = np.column_stack([
            np.asarray(estimator.predict(X_array), dtype=float) for estimator in self.fitted_estimators_
        ])
        return prediction_matrix @ self.weights_


class SuperLearnerClassifier(_BaseConstrainedSuperLearner, ClassifierMixin):
    """Constrained super learner for binary classification with nonnegative weights."""

    def __init__(
        self,
        estimators: list[tuple[str, BaseEstimator]],
        *,
        estimator_param_grids: Mapping[str, Mapping[str, Sequence[Any]]] | None = None,
        search_cv: int = 5,
        search_scoring: str | None = "neg_log_loss",
        search_n_jobs: int | None = None,
        search_shuffle: bool = True,
        random_state: int | None = 42,
        normalize_weights: bool = False,
        tolerance: float = 1e-10,
        max_iter: int = 1000,
        probability_clip: float = 1e-6,
    ) -> None:
        super().__init__(
            estimators=estimators,
            estimator_param_grids=estimator_param_grids,
            search_cv=search_cv,
            search_scoring=search_scoring,
            search_n_jobs=search_n_jobs,
            search_shuffle=search_shuffle,
            random_state=random_state,
            normalize_weights=normalize_weights,
            tolerance=tolerance,
            max_iter=max_iter,
        )
        self.probability_clip = probability_clip

    def _make_search_splitter(self, y: np.ndarray) -> StratifiedKFold:
        return StratifiedKFold(
            n_splits=self.search_cv,
            shuffle=self.search_shuffle,
            random_state=self.random_state if self.search_shuffle else None,
        )

    def fit(self, X: Any, y: Any) -> "SuperLearnerClassifier":
        X_array = validate_features(X)
        y_array = validate_binary_treatment(y, n_samples=X_array.shape[0])
        self.classes_ = np.array([0, 1])

        self.estimator_names_, self.fitted_estimators_, self.best_params_ = self._fit_base_estimators(X_array, y_array)
        probability_matrix = np.column_stack([
            np.clip(_predict_treatment_probability(estimator, X_array), self.probability_clip, 1.0 - self.probability_clip)
            for estimator in self.fitted_estimators_
        ])

        def objective(weights: np.ndarray) -> tuple[float, np.ndarray]:
            combined = np.clip(probability_matrix @ weights, self.probability_clip, 1.0 - self.probability_clip)
            value = float(-np.sum(y_array * np.log(combined) + (1 - y_array) * np.log(1.0 - combined)))
            gradient_core = (combined - y_array) / (combined * (1.0 - combined))
            gradient = probability_matrix.T @ gradient_core
            return value, gradient

        initial = np.full(probability_matrix.shape[1], 1.0 / probability_matrix.shape[1], dtype=float)
        bounds = [(0.0, None)] * probability_matrix.shape[1]
        constraints = []
        if self.normalize_weights:
            constraints.append({"type": "eq", "fun": lambda w: np.sum(w) - 1.0, "jac": lambda w: np.ones_like(w)})

        method = "SLSQP" if self.normalize_weights else "L-BFGS-B"
        result = minimize(
            fun=lambda w: objective(w)[0],
            x0=initial,
            jac=lambda w: objective(w)[1],
            method=method,
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": self.max_iter},
        )
        if not result.success:
            raise RuntimeError(f"SuperLearnerClassifier optimization failed: {result.message}")

        self.weights_ = _normalize_weight_vector(result.x, normalize_weights=self.normalize_weights, tolerance=self.tolerance)
        self.optimization_result_ = result
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        if not hasattr(self, "weights_"):
            raise RuntimeError("The super learner is not fitted yet.")
        X_array = validate_features(X)
        probability_matrix = np.column_stack([
            np.clip(_predict_treatment_probability(estimator, X_array), self.probability_clip, 1.0 - self.probability_clip)
            for estimator in self.fitted_estimators_
        ])
        positive = np.clip(probability_matrix @ self.weights_, self.probability_clip, 1.0 - self.probability_clip)
        return np.column_stack([1.0 - positive, positive])

    def predict(self, X: Any) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class CrossFittedNuisanceEstimator(BaseEstimator):
    """Estimate outcome and propensity nuisance functions with shared cross-fitting."""

    def __init__(
        self,
        *,
        outcome_model: BaseEstimator,
        treatment_model: BaseEstimator,
        n_folds: int = 10,
        shuffle: bool = True,
        random_state: int | None = 42,
        propensity_clip: float = 1e-6,
        stratify_treatment: bool = True,
        refit_full: bool = True,
        outcome_param_grid: Mapping[str, Sequence[Any]] | None = None,
        treatment_param_grid: Mapping[str, Sequence[Any]] | None = None,
        outcome_search_cv: int = 5,
        treatment_search_cv: int = 5,
        outcome_scoring: str | None = None,
        treatment_scoring: str | None = "neg_log_loss",
        n_jobs: int | None = None,
        outcome_fit_params: Mapping[str, Any] | None = None,
        treatment_fit_params: Mapping[str, Any] | None = None,
    ) -> None:
        if n_folds < 2:
            raise ValueError("n_folds must be at least 2.")

        self.outcome_model = outcome_model
        self.treatment_model = treatment_model
        self.n_folds = n_folds
        self.shuffle = shuffle
        self.random_state = random_state
        self.propensity_clip = propensity_clip
        self.stratify_treatment = stratify_treatment
        self.refit_full = refit_full
        self.outcome_param_grid = outcome_param_grid
        self.treatment_param_grid = treatment_param_grid
        self.outcome_search_cv = outcome_search_cv
        self.treatment_search_cv = treatment_search_cv
        self.outcome_scoring = outcome_scoring
        self.treatment_scoring = treatment_scoring
        self.n_jobs = n_jobs
        self.outcome_fit_params = outcome_fit_params
        self.treatment_fit_params = treatment_fit_params
        self.result_: NuisanceResult | None = None

    def _make_splitter(self) -> KFold | StratifiedKFold:
        splitter_kwargs = {
            "n_splits": self.n_folds,
            "shuffle": self.shuffle,
            "random_state": self.random_state if self.shuffle else None,
        }
        if self.stratify_treatment:
            return StratifiedKFold(**splitter_kwargs)
        return KFold(**splitter_kwargs)

    def fit(self, X: Any, y: Any, d: Any) -> "CrossFittedNuisanceEstimator":
        X_array = validate_features(X)
        y_array = validate_vector(y, name="y", n_samples=X_array.shape[0])
        d_array = validate_binary_treatment(d, n_samples=X_array.shape[0])

        y_hat = np.zeros(X_array.shape[0], dtype=float)
        d_hat = np.zeros(X_array.shape[0], dtype=float)
        folds = np.full(X_array.shape[0], -1, dtype=int)
        outcome_models: list[BaseEstimator] = []
        treatment_models: list[BaseEstimator] = []

        outcome_fit_params = _coerce_fit_params(self.outcome_fit_params)
        treatment_fit_params = _coerce_fit_params(self.treatment_fit_params)

        splitter = self._make_splitter()
        split_iterator = splitter.split(X_array, d_array if self.stratify_treatment else None)

        for fold_id, (train_idx, test_idx) in enumerate(split_iterator):
            X_train, X_test = X_array[train_idx], X_array[test_idx]
            y_train = y_array[train_idx]
            d_train = d_array[train_idx]

            outcome_model = _prepare_estimator(
                self.outcome_model,
                param_grid=self.outcome_param_grid,
                search_cv=self.outcome_search_cv,
                scoring=self.outcome_scoring,
                n_jobs=self.n_jobs,
            )
            treatment_model = _prepare_estimator(
                self.treatment_model,
                param_grid=self.treatment_param_grid,
                search_cv=self.treatment_search_cv,
                scoring=self.treatment_scoring,
                n_jobs=self.n_jobs,
            )

            outcome_model.fit(X_train, y_train, **outcome_fit_params)
            treatment_model.fit(X_train, d_train, **treatment_fit_params)

            y_hat[test_idx] = np.asarray(outcome_model.predict(X_test), dtype=float)
            d_hat[test_idx] = _predict_treatment_probability(treatment_model, X_test)

            folds[test_idx] = fold_id
            outcome_models.append(outcome_model)
            treatment_models.append(treatment_model)

        d_hat = np.clip(d_hat, self.propensity_clip, 1.0 - self.propensity_clip)

        diagnostics = {
            "outcome_rmse": float(np.sqrt(mean_squared_error(y_array, y_hat))),
        }
        try:
            diagnostics["propensity_auc"] = float(roc_auc_score(d_array, d_hat))
            diagnostics["propensity_log_loss"] = float(log_loss(d_array, d_hat))
        except ValueError:
            pass

        outcome_model_full: BaseEstimator | None = None
        treatment_model_full: BaseEstimator | None = None
        if self.refit_full:
            outcome_model_full = _prepare_estimator(
                self.outcome_model,
                param_grid=self.outcome_param_grid,
                search_cv=self.outcome_search_cv,
                scoring=self.outcome_scoring,
                n_jobs=self.n_jobs,
            )
            treatment_model_full = _prepare_estimator(
                self.treatment_model,
                param_grid=self.treatment_param_grid,
                search_cv=self.treatment_search_cv,
                scoring=self.treatment_scoring,
                n_jobs=self.n_jobs,
            )
            outcome_model_full.fit(X_array, y_array, **outcome_fit_params)
            treatment_model_full.fit(X_array, d_array, **treatment_fit_params)

        self.result_ = NuisanceResult(
            y_hat=y_hat,
            d_hat=d_hat,
            y_residual=y_array - y_hat,
            d_residual=d_array - d_hat,
            folds=folds,
            diagnostics=diagnostics,
            outcome_models=outcome_models,
            treatment_models=treatment_models,
            outcome_model_full=outcome_model_full,
            treatment_model_full=treatment_model_full,
        )
        return self

    def fit_predict(self, X: Any, y: Any, d: Any) -> NuisanceResult:
        result = self.fit(X=X, y=y, d=d).result_
        if result is None:
            raise RuntimeError("Cross-fitted nuisance estimation did not produce a result.")
        return result

    def predict_outcome(self, X: Any) -> np.ndarray:
        if self.result_ is None or self.result_.outcome_model_full is None:
            raise RuntimeError("No full-sample outcome model is available. Fit with refit_full=True first.")
        X_array = validate_features(X)
        return np.asarray(self.result_.outcome_model_full.predict(X_array), dtype=float)

    def predict_propensity(self, X: Any) -> np.ndarray:
        if self.result_ is None or self.result_.treatment_model_full is None:
            raise RuntimeError("No full-sample treatment model is available. Fit with refit_full=True first.")
        X_array = validate_features(X)
        scores = _predict_treatment_probability(self.result_.treatment_model_full, X_array)
        return np.clip(scores, self.propensity_clip, 1.0 - self.propensity_clip)

    def get_outcome_weights(self) -> dict[str, float]:
        if self.result_ is None or self.result_.outcome_model_full is None:
            raise RuntimeError("No full-sample outcome model is available. Fit with refit_full=True first.")
        if not hasattr(self.result_.outcome_model_full, "get_weights"):
            raise TypeError("The outcome model does not expose constrained super learner weights.")
        return self.result_.outcome_model_full.get_weights()

    def get_treatment_weights(self) -> dict[str, float]:
        if self.result_ is None or self.result_.treatment_model_full is None:
            raise RuntimeError("No full-sample treatment model is available. Fit with refit_full=True first.")
        if not hasattr(self.result_.treatment_model_full, "get_weights"):
            raise TypeError("The treatment model does not expose constrained super learner weights.")
        return self.result_.treatment_model_full.get_weights()

    def get_outcome_best_params(self) -> dict[str, dict[str, Any] | None]:
        if self.result_ is None or self.result_.outcome_model_full is None:
            raise RuntimeError("No full-sample outcome model is available. Fit with refit_full=True first.")
        if not hasattr(self.result_.outcome_model_full, "get_best_params"):
            raise TypeError("The outcome model does not expose base-learner search results.")
        return self.result_.outcome_model_full.get_best_params()

    def get_treatment_best_params(self) -> dict[str, dict[str, Any] | None]:
        if self.result_ is None or self.result_.treatment_model_full is None:
            raise RuntimeError("No full-sample treatment model is available. Fit with refit_full=True first.")
        if not hasattr(self.result_.treatment_model_full, "get_best_params"):
            raise TypeError("The treatment model does not expose base-learner search results.")
        return self.result_.treatment_model_full.get_best_params()

