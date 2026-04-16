"""Forecast error generation. Ports Baseline_Error.m, AR_Error.m, NP_Error.m."""
import numpy as np
from opu.transforms import mlags, zscore, prepare_missing, deseasonalize
from opu.factors import factors_em
from opu.config import PY, PZ, TSTAT_THRESHOLD


def newey_west(y: np.ndarray, x: np.ndarray, nlag: int) -> dict:
    """Newey-West HAC regression. Port of nwest.m."""
    nobs, nvar = x.shape
    xpxi = np.linalg.inv(x.T @ x)
    beta = xpxi @ (x.T @ y)
    yhat = x @ beta
    resid = y - yhat
    sigu = resid @ resid
    sige = sigu / (nobs - nvar)

    emat = np.tile(resid, (nvar, 1))
    hhat = emat * x.T
    G = np.zeros((nvar, nvar))
    a = 0
    while a != nlag + 1:
        w = (nlag + 1 - a) / (nlag + 1)
        za = hhat[:, a:nobs] @ hhat[:, : nobs - a].T
        if a == 0:
            ga = za
        else:
            ga = za + za.T
        G += w * ga
        a += 1

    V = xpxi @ G @ xpxi
    nwerr = np.sqrt(np.diag(V))

    ym = y - np.mean(y)
    rsqr = 1.0 - sigu / (ym @ ym)

    return {
        "beta": beta,
        "tstat": beta / nwerr,
        "resid": resid,
        "yhat": yhat,
        "sige": sige,
        "rsqr": rsqr,
    }


def build_forecast_errors(
    yt: np.ndarray, xt: np.ndarray, py: int = PY, pz: int = PZ
) -> dict:
    """Build forecast errors for oil price series with predictor selection.

    Args:
        yt: (T, N) transformed oil price data (z-scored)
        xt: (T, R) predictor matrix (z-scored)
        py: number of own lags
        pz: number of predictor lags

    Returns dict with keys: vyt, vft, ybetas, fbetas, fmodels, dates_idx, and
    counterfactual forecast errors (vyt_noer, vyt_noy, etc.)
    """
    T, N = yt.shape
    T_x, R = xt.shape
    p = max(py, pz)
    q = int(T ** 0.25)  # Newey-West bandwidth

    ybetas_full = np.zeros((1 + py + pz * R, N))
    vyt = None
    fmodels = None

    # Containers for counterfactual errors
    counterfactuals = {}

    for i in range(N):
        X = np.column_stack([np.ones(T), mlags(yt[:, i : i + 1], py), mlags(xt, pz)])
        y_dep = yt[p:, i]
        X_dep = X[p:, :]

        reg = newey_west(y_dep, X_dep, q)

        # Predictor selection: keep only those with |t| > threshold
        pass_mask = np.abs(reg["tstat"][py + 1 :]) > TSTAT_THRESHOLD
        keep = np.concatenate([np.ones(py + 1, dtype=bool), pass_mask])

        X_new = X_dep[:, keep]
        reg = newey_west(y_dep, X_new, q)

        if vyt is None:
            vyt = np.zeros((len(y_dep), N))
            fmodels = np.zeros((pz * R, N), dtype=bool)

        vyt[:, i] = reg["resid"]
        ybetas_full[keep, i] = reg["beta"]
        fmodels[:, i] = pass_mask

        # Counterfactual: zero out predictor groups
        _build_counterfactuals(counterfactuals, i, yt, X, ybetas_full[:, i], p, py, pz, R)

    # AR errors for predictors
    pf = py
    fbetas = np.zeros((R, pf + 1))
    vft = np.zeros((T - pf, R))
    for i in range(R):
        X_f = np.column_stack([np.ones(T), mlags(xt[:, i : i + 1], pf)])
        reg_f = newey_west(xt[pf:, i], X_f[pf:, :], q)
        vft[:, i] = reg_f["resid"]
        fbetas[i, :] = reg_f["beta"]

    return {
        "vyt": vyt,
        "vft": vft,
        "ybetas": ybetas_full.T,
        "fbetas": fbetas,
        "fmodels": fmodels,
        **counterfactuals,
    }


def _build_counterfactuals(out, i, yt, X, ybetas, p, py, pz, R):
    """Zero out predictor groups to build leave-one-out forecast errors."""
    predictor_groups = {
        "noer": list(range(0, 5)),      # 5 exchange rates
        "noy": [5],                       # REA
        "noq": [6],                       # oil production
        "noinventory": [7],               # inventory
        "nom1": [8],                      # M1
        "nocpi": [9],                     # CPI
        "nocom": [10, 11, 12],            # fuel factor + factor^2 + ghat
    }

    y_dep = yt[p:, i]
    for name, indices in predictor_groups.items():
        key = f"vyt_{name}"
        if key not in out:
            out[key] = np.zeros_like(y_dep).reshape(-1, 1)
        b = ybetas.copy()
        for idx in indices:
            for lag in range(pz):
                coef_idx = py + 1 + lag * R + idx
                if coef_idx < len(b):
                    b[coef_idx] = 0.0
        out[key][:, 0] = y_dep - X[p:, :] @ b


def build_ar_errors(yt: np.ndarray, py: int = PY) -> dict:
    """AR-only forecast errors (no exogenous predictors). Ports AR_Error.m."""
    T, N = yt.shape
    q = int(T ** 0.25)
    ybetas = np.zeros((py + 1, N))
    vyt = np.zeros((T - py, N))

    for i in range(N):
        X = np.column_stack([np.ones(T), mlags(yt[:, i : i + 1], py)])
        reg = newey_west(yt[py:, i], X[py:, :], q)
        vyt[:, i] = reg["resid"]
        ybetas[:, i] = reg["beta"]

    return {"vyt": vyt, "ybetas": ybetas.T}


def build_np_errors(yt: np.ndarray) -> dict:
    """No-predictor forecast errors (raw z-scored series). Ports NP_Error.m."""
    return {"vyt": yt.copy()}
