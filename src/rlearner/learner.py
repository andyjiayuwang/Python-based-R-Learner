"""High-level RLearner orchestration API."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, clone

from .evaluation import (
    BLPTestResult,
    CalibrationTestResult,
    UpliftTestResult,
    blp_test,
    calibration_test,
    uplift_test,
)
from .nuisance import CrossFittedNuisanceEstimator, ManualNuisanceEstimator, NuisanceResult
from .r_loss import RLossStacking, RLossWrapper


class RLearner:
    """Orchestrate nuisance estimation and second-stage R-loss learning."""

    def __init__(
        self,
        *,
        nuisance_estimator: CrossFittedNuisanceEstimator | None = None,
        cate_learner: BaseEstimator | None = None,
        cate_learners: Mapping[str, BaseEstimator] | Sequence[tuple[str, BaseEstimator]] | None = None,
        stacking_model: RLossStacking | None = None,
        use_stacking: bool | None = None,
    ) -> None:
        if cate_learner is not None and cate_learners is not None:
            raise ValueError("Specify either cate_learner or cate_learners, not both.")
        if cate_learner is None and cate_learners is None:
            raise ValueError("At least one second-stage learner must be provided.")

        self.nuisance_estimator = nuisance_estimator
        self.cate_learner = cate_learner
        self.cate_learners = cate_learners
        self.stacking_model = stacking_model
        self.use_stacking = use_stacking

        self.nuisance_result_: NuisanceResult | None = None
        self.nuisance_estimator_: CrossFittedNuisanceEstimator | None = None
        self.cate_models_: dict[str, RLossWrapper] = {}
        self.learner_names_: list[str] = []
        self.stacking_model_: RLossStacking | None = None
        self.tau_hat_: np.ndarray | None = None
        self.tau_matrix_: np.ndarray | None = None
        self.nuisance_mode_: str | None = None

    def _normalize_learners(self) -> list[tuple[str, BaseEstimator]]:
        if self.cate_learner is not None:
            return [("cate", self.cate_learner)]

        if isinstance(self.cate_learners, Mapping):
            learners = list(self.cate_learners.items())
        else:
            learners = list(self.cate_learners or [])

        if not learners:
            raise ValueError("At least one second-stage learner must be provided.")

        names = [name for name, _ in learners]
        if len(set(names)) != len(names):
            raise ValueError("Second-stage learner names must be unique.")

        return learners

    def _should_stack(self, n_learners: int) -> bool:
        if self.use_stacking is not None:
            return self.use_stacking
        return n_learners > 1

    def fit(
        self,
        X: Any,
        y: Any,
        d: Any,
        *,
        y_hat: Any | None = None,
        d_hat: Any | None = None,
    ) -> "RLearner":
        if (y_hat is None) ^ (d_hat is None):
            raise ValueError("y_hat and d_hat must be provided together for manual nuisance estimation.")

        if y_hat is not None and d_hat is not None:
            nuisance_estimator = ManualNuisanceEstimator()
            self.nuisance_result_ = nuisance_estimator.fit_predict(y=y, d=d, y_hat=y_hat, d_hat=d_hat)
            self.nuisance_estimator_ = None
            self.nuisance_mode_ = "manual"
        else:
            if self.nuisance_estimator is None:
                raise ValueError(
                    "nuisance_estimator is required when manual nuisance predictions are not provided."
                )
            nuisance_estimator = clone(self.nuisance_estimator)
            self.nuisance_result_ = nuisance_estimator.fit_predict(X=X, y=y, d=d)
            self.nuisance_estimator_ = nuisance_estimator
            self.nuisance_mode_ = "auto"

        learners = self._normalize_learners()
        self.learner_names_ = [name for name, _ in learners]
        self.cate_models_ = {}

        tau_predictions = []
        for name, learner in learners:
            model = RLossWrapper(base_learner=learner)
            model.fit(X, self.nuisance_result_.y_residual, self.nuisance_result_.d_residual)
            self.cate_models_[name] = model
            tau_predictions.append(model.predict(X))

        self.tau_matrix_ = np.column_stack(tau_predictions)

        if self._should_stack(len(learners)):
            stacker = clone(self.stacking_model) if self.stacking_model is not None else RLossStacking()
            stacker.fit(
                tau_matrix=self.tau_matrix_,
                y_res=self.nuisance_result_.y_residual,
                w_res=self.nuisance_result_.d_residual,
            )
            self.stacking_model_ = stacker
            self.tau_hat_ = stacker.predict(self.tau_matrix_)
        else:
            self.stacking_model_ = None
            self.tau_hat_ = self.tau_matrix_[:, 0]

        return self

    def fit_predict(
        self,
        X: Any,
        y: Any,
        d: Any,
        *,
        y_hat: Any | None = None,
        d_hat: Any | None = None,
    ) -> np.ndarray:
        self.fit(X=X, y=y, d=d, y_hat=y_hat, d_hat=d_hat)
        if self.tau_hat_ is None:
            raise RuntimeError("RLearner did not produce fitted CATE predictions.")
        return self.tau_hat_

    def predict(self, X: Any) -> np.ndarray:
        if not self.cate_models_:
            raise RuntimeError("The RLearner instance is not fitted yet.")

        individual = self.predict_individual(X)
        tau_matrix = np.column_stack([individual[name] for name in self.learner_names_])
        if self.stacking_model_ is not None:
            return self.stacking_model_.predict(tau_matrix)
        return tau_matrix[:, 0]

    def predict_individual(self, X: Any) -> dict[str, np.ndarray]:
        if not self.cate_models_:
            raise RuntimeError("The RLearner instance is not fitted yet.")

        return {name: model.predict(X) for name, model in self.cate_models_.items()}

    def blp_test(self, *, tau_hat: Any | None = None, confidence_level: float = 0.95) -> BLPTestResult:
        if self.nuisance_result_ is None:
            raise RuntimeError("The RLearner instance is not fitted yet.")

        tau_values = self.tau_hat_ if tau_hat is None else tau_hat
        if tau_values is None:
            raise RuntimeError("No fitted CATE predictions are available for the BLP test.")

        return blp_test(
            tau_hat=tau_values,
            nuisance_result=self.nuisance_result_,
            confidence_level=confidence_level,
        )

    def calibration_test(
        self,
        *,
        tau_hat: Any | None = None,
        n_bins: int = 5,
    ) -> CalibrationTestResult:
        if self.nuisance_result_ is None:
            raise RuntimeError("The RLearner instance is not fitted yet.")

        tau_values = self.tau_hat_ if tau_hat is None else tau_hat
        if tau_values is None:
            raise RuntimeError("No fitted CATE predictions are available for the calibration test.")

        return calibration_test(
            tau_hat=tau_values,
            nuisance_result=self.nuisance_result_,
            n_bins=n_bins,
        )

    def uplift_test(
        self,
        *,
        tau_hat: Any | None = None,
        fractions: Any | None = None,
    ) -> UpliftTestResult:
        if self.nuisance_result_ is None:
            raise RuntimeError("The RLearner instance is not fitted yet.")

        tau_values = self.tau_hat_ if tau_hat is None else tau_hat
        if tau_values is None:
            raise RuntimeError("No fitted CATE predictions are available for the uplift test.")

        return uplift_test(
            tau_hat=tau_values,
            nuisance_result=self.nuisance_result_,
            fractions=fractions,
        )
