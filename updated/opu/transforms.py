"""Data transformations matching prepare_missing.m, deseasonal.m, mlags.m."""
import numpy as np


def prepare_missing(x: np.ndarray, tcode: int) -> np.ndarray:
    """Transform a single series based on transformation code.

    Codes: 1=level, 2=first diff, 3=second diff, 4=log,
           5=log first diff, 6=log second diff, 7=first diff of pct change.
    """
    n = len(x)
    y = np.full(n, np.nan)

    if tcode == 1:
        y[:] = x
    elif tcode == 2:
        y[1:] = x[1:] - x[:-1]
    elif tcode == 3:
        y[2:] = x[2:] - 2 * x[1:-1] + x[:-2]
    elif tcode == 4:
        if np.min(x) < 1e-6:
            return y
        y[:] = np.log(x)
    elif tcode == 5:
        if np.min(x) > 1e-6:
            lx = np.log(x)
            y[1:] = lx[1:] - lx[:-1]
    elif tcode == 6:
        if np.min(x) > 1e-6:
            lx = np.log(x)
            y[2:] = lx[2:] - 2 * lx[1:-1] + lx[:-2]
    elif tcode == 7:
        pct = np.full(n, np.nan)
        pct[1:] = (x[1:] - x[:-1]) / x[:-1]
        y[2:] = pct[2:] - pct[1:-1]

    return y


def zscore(x: np.ndarray) -> np.ndarray:
    """Standardize to zero mean, unit std (population std, matching MATLAB)."""
    return (x - np.mean(x)) / np.std(x, ddof=0)


def deseasonalize(y: np.ndarray) -> np.ndarray:
    """Remove monthly seasonal pattern via dummy regression."""
    t = len(y)
    X = np.zeros((t, 12))
    for i in range(t):
        X[i, i % 12] = 1.0
    bhat = np.linalg.lstsq(X, y, rcond=None)[0]
    return y - X @ bhat


def mlags(x: np.ndarray, k: int = 1) -> np.ndarray:
    """Create matrix of k lags. Pads with zeros.

    Input x is (T, nvar). Output is (T, nvar*k).
    """
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    T, nvar = x.shape
    z = np.zeros((T, nvar * k))
    for j in range(1, k + 1):
        z[j:, nvar * (j - 1) : nvar * j] = x[: T - j, :]
    return z
