# Python-based R Learner

A Python package named `rlearner` that runs the R learner ([Nie and Wager, 2021](#ref-nie-wager-2021)) for heterogeneous treatment effect estimation and validation with flexible choices.

## Set Up

Install the package via pip:

```bash
pip install "git+https://github.com/andyjiayuwang/Python-based-R-Learner.git"
```

A full example workflow is available in [`demo.ipynb`](demo.ipynb).

A minimal import example is:

```python
from rlearner import (
    CrossFittedNuisanceEstimator,
    RLearner,
    RLossStacking,
    SuperLearnerClassifier,
    SuperLearnerRegressor,
)
```

## Step 1: Nuisance Estimation

The first step estimates the nuisance functions needed by the R learner:

- `m(X) = E[Y | X]`, the outcome regression
- `e(X) = E[W | X]`, the propensity score

These nuisance estimates are used to build the residualized quantities

- `Y_tilde = Y - m_hat(X)`
- `W_tilde = W - e_hat(X)`

which are then passed into the second-stage R-loss optimization.

The package provides two ways to handle step 1.

### Built-in nuisance estimation

Use `CrossFittedNuisanceEstimator` when you want the package to fit nuisance models directly. It supports:

- K-fold cross-fitting for both the outcome model and the treatment model
- Any sklearn-style regressor for the outcome model
- Any sklearn-style binary classifier with `predict_proba` for the treatment model
- Optional grid search on the full nuisance model object through `outcome_param_grid` and `treatment_param_grid`
- Full-sample refitting after cross-fitting so the fitted nuisance models can be reused for prediction

Default settings for `CrossFittedNuisanceEstimator` are:

- `n_folds=10`
- `shuffle=True`
- `random_state=42`
- `propensity_clip=1e-6`
- `stratify_treatment=True`
- `refit_full=True`
- `outcome_search_cv=5`
- `treatment_search_cv=5`
- `treatment_scoring="neg_log_loss"`

### Manual nuisance inputs

Use manual nuisance inputs when you already have trusted out-of-fold nuisance predictions from an external workflow. In that case, pass:

- `y_hat`, the out-of-fold estimate of `m(X)`
- `d_hat`, the out-of-fold estimate of `e(X)`

through `ManualNuisanceEstimator` or directly through `RLearner.fit(..., y_hat=..., d_hat=...)`.

### Constrained super learner for step 1

The package also provides constrained super learners for nuisance prediction:

- `SuperLearnerRegressor`
- `SuperLearnerClassifier`

These models support:

- Multiple base learners
- Nonnegative stacking weights
- Optional normalization of weights to sum to 1 through `normalize_weights=True`
- Separate grid search for each base learner via `estimator_param_grids`
- Stable internal sample splitting for hyperparameter tuning
- Weight inspection through `get_weights()`
- Best-parameter inspection through `get_best_params()`

Default settings for the super learners are:

- `search_cv=5`
- `search_shuffle=True`
- `random_state=42`
- `normalize_weights=False`
- `tolerance=1e-10`
- `max_iter=1000`

For treatment prediction, the built-in step 1 implementation currently assumes a binary treatment indicator.

## Step 2: R-Loss CATE Learning

The second step learns the conditional average treatment effect `tau(X)` using the residualized outcome and treatment from step 1.

The package provides two main components for this stage.

### Single second-stage learner

Use `RLossWrapper` to fit a single sklearn-style regressor under the R-loss construction. This is the simplest way to estimate a single CATE model once `Y_tilde` and `W_tilde` are available.

### Multiple second-stage learners plus stacking

Use `RLearner` with `cate_learners={...}` when you want to fit multiple second-stage learners and combine them. The package then:

- Fits one `RLossWrapper` per learner
- Produces one CATE estimate from each learner
- Optionally combines them with `RLossStacking`

`RLossStacking` follows the positive linear-combination idea used in the R-loss stacking step. The fitted object reports:

- `a_hat`, the constant shift term
- `b_hat`, the scale of the coefficient vector
- `alpha_hat`, the nonnegative relative weights of the second-stage learners

Default settings for `RLossStacking` are:

- `lambda_reg=1.0`
- `tolerance=1e-10`
- `max_iter=1000`

In step 2, the stacking weights are constrained to be nonnegative, but they are not required to sum to 1.

## Step 3: Validation and Diagnostics

The third step validates the fitted treatment-effect model using the out-of-fold nuisance estimates and the fitted CATE predictions. The validation routines implemented here follow the discussions in [Chernozhukov et al. (2024)](#ref-chernozhukov-et-al-2024).

All validation routines are available in two ways:

- As standalone functions in `rlearner`
- As convenience methods on a fitted `RLearner` instance

### BLP test

The BLP test runs the no-intercept regression

- `Y_tilde = alpha * W_tilde + beta * W_tilde * tau_hat(X)`

and reports:

- Point estimates for `alpha` and `beta`
- HC2 standard errors
- Normal-based z statistics
- p-values
- Confidence intervals

Default setting:

- `confidence_level=0.95`

### Calibration test

The calibration test bins observations by predicted treatment effect and compares:

- The average predicted treatment effect within each bin
- The doubly robust bin-level treatment effect estimate

It returns both:

- `CAL_1`, the weighted L1 calibration criterion
- `CAL_2`, the weighted L2 calibration criterion

and also exposes the full bin-level table.

Default setting:

- `n_bins=5`

### Uplift test

The uplift test performs ranking-based validation using a DR uplift curve. Observations are sorted by `tau_hat(X)` from high to low, top-fraction subgroups are formed, and a DR subgroup effect is computed for each fraction.

The output includes:

- The uplift curve table `(fraction, subgroup size, theta_dr)`
- `AUUC`, the area under the uplift curve

Default setting:

- `fractions = 0.1, 0.2, ..., 1.0`

## Notes

- The import name is `rlearner`, even though the GitHub repository is named `Python-based-R-Learner`.
- The package currently declares support for Python `>=3.10`.

## References

- <a id="ref-nie-wager-2021"></a>Nie, X., & Wager, S. (2021). Quasi-oracle estimation of heterogeneous treatment effects. *Biometrika*, 108(2), 299-319.
- <a id="ref-chernozhukov-et-al-2024"></a>Chernozhukov, V., Hansen, C., Kallus, N., Spindler, M., & Syrgkanis, V. (2024). *Applied causal inference powered by ML and AI*. arXiv preprint arXiv:2403.02467.
