import numpy as np
import pandas as pd


def expit(x):
    return 1.0 / (1.0 + np.exp(-x))


def g_func(X, active_idx, beta_coeffs):
    X_sub = X[:, active_idx]
    beta = np.asarray(beta_coeffs)
    base = np.sum(np.tanh(X_sub) * beta, axis=1)
    poly = 0.2 * np.sum((X_sub ** 2) * beta, axis=1)
    inter = 0.3 * X_sub[:, 0] * X_sub[:, 1]
    return base + poly + inter


def dgp_linear_cross_section(n=10000, p=20, seed=42):
    rng = np.random.default_rng(seed)

    X = rng.normal(0.0, 1.0, size=(n, p))

    all_indices = np.arange(p)
    active_d = all_indices[: 4 * p // 5]
    active_y = all_indices[p // 10 : 9 * p // 10]

    beta_t = rng.normal(0.0, 0.3, size=len(active_d))
    beta_y0 = rng.normal(0.0, 0.5, size=len(active_y))

    half = len(active_y) // 2
    beta_y1 = np.concatenate([
        np.repeat(0.15, half),
        np.repeat(-0.15, len(active_y) - half),
    ])
    beta_y1 = rng.permutation(beta_y1)

    eta = g_func(X, active_d, beta_t) + rng.normal(0.0, 0.3, size=n)
    ps = expit(eta)
    t = rng.binomial(1, ps, size=n)

    raw_tau = 2.0 + np.sum(X[:, active_y] * beta_y1, axis=1)
    ite = raw_tau.copy()

    g_x_y0 = g_func(X, active_y, beta_y0)
    y = g_x_y0 + t * raw_tau + rng.normal(0.0, 0.2, size=n)

    actual_ate = np.mean(ite)
    calibration_offset = actual_ate - 2.0

    ite = ite - calibration_offset
    y = y - t * calibration_offset

    columns = [f"X{i}" for i in range(1, p + 1)]
    df = pd.DataFrame(X, columns=columns)
    df.insert(0, "ite", ite)
    df.insert(0, "t", t)
    df.insert(0, "y", y)
    return df


if __name__ == "__main__":
    data = dgp_linear_cross_section(n=10000, p=20, seed=42)
    data.to_csv("data.csv", index=False)
