"""JLN-style uncertainty recursion. Ports compute_uf.m, compute_uy.m."""
import numpy as np
from scipy import sparse
from opu.config import OPU_HORIZON, PY, PZ, SV_BURN, SV_DRAWS, SV_THIN, SEED_SV_Y, SEED_SV_F


def expected_var(a: float, b: float, t2: float, x: np.ndarray, h: int) -> np.ndarray:
    """Compute Et[exp(x_{t+h})] for AR(1) log-vol: Et[v^2_{t+h}].

    a = mu*(1-phi), b = phi, t2 = sigma^2.
    """
    return np.exp(
        a * (1 - b**h) / (1 - b)
        + t2 / 2 * (1 - b ** (2 * h)) / (1 - b**2)
        + b**h * x
    )


def compute_uf(
    xf: np.ndarray, thf: np.ndarray, fb: np.ndarray, h: int
) -> tuple[list, np.ndarray]:
    """Compute expected volatility of predictors up to horizon h.

    Args:
        xf: (T, r) latent log-vol states for each predictor
        thf: (3, r) SV params [alpha; beta; tau2] per predictor
        fb: (r, pf+1) AR coefficients for predictors (intercept + lags)
        h: forecast horizon

    Returns: (evf, phif)
        evf: list of h arrays, each (T, r) -- expected variance at horizon j
        phif: (r*pf, r*pf) companion matrix for predictor AR
    """
    r = xf.shape[1]
    pf = fb.shape[1] - 1

    # Build companion matrix phif
    phif_rows = []
    top_row = np.zeros((r, r * pf))
    for j in range(pf):
        for i in range(r):
            top_row[i, j * r + i] = fb[i, j + 1]
    phif_rows.append(top_row)
    if pf > 1:
        bottom = np.zeros((r * (pf - 1), r * pf))
        bottom[:, : r * (pf - 1)] = np.eye(r * (pf - 1))
        phif_rows.append(bottom)
    phif = np.vstack(phif_rows)

    # Compute expected variances
    evf = []
    for j in range(1, h + 1):
        ev_j = np.zeros((xf.shape[0], r))
        for i in range(r):
            ev_j[:, i] = expected_var(thf[0, i], thf[1, i], thf[2, i], xf[:, i], j)
        evf.append(ev_j)

    return evf, phif


def compute_uy(
    xy: np.ndarray,
    thy: np.ndarray,
    yb: np.ndarray,
    py: int,
    evf: list,
    phif: np.ndarray,
) -> np.ndarray:
    """Compute uncertainty for oil series via JLN recursion.

    Args:
        xy: (T,) latent log-vol for oil forecast errors
        thy: (3,) SV params [alpha, beta, tau2]
        yb: (1+py+pz*r,) regression coefficients (intercept + own lags + predictor lags)
        py: number of own lags
        evf: expected predictor volatilities from compute_uf
        phif: predictor companion matrix

    Returns: (T, h) matrix of squared uncertainty (take sqrt for OPU)
    """
    h = len(evf)
    r = evf[0].shape[1]
    pf = phif.shape[0] // r
    pz = (len(yb) - 1 - py) // r
    T = len(xy)

    # Build composite phi matrix
    # lambda: cross-equation loadings (oil on predictors)
    lambda_top = np.zeros((1, r * pf))
    for j in range(min(pz, pf)):
        lambda_top[0, j * r : (j + 1) * r] = yb[py + 1 + j * r : py + 1 + (j + 1) * r]
    if pf > pz:
        pass  # already zero-padded
    lambda_block = np.zeros((py, r * pf))
    lambda_block[0, :] = lambda_top[0, :]

    # phiy: own-AR companion
    phiy_top = yb[1 : py + 1].reshape(1, -1)
    phiy = np.zeros((py, py))
    phiy[0, :] = phiy_top
    if py > 1:
        phiy[1:, :-1] = np.eye(py - 1)

    # Full phi
    dim = r * pf + py
    phi = np.zeros((dim, dim))
    phi[: r * pf, : r * pf] = phif
    phi[r * pf :, : r * pf] = lambda_block
    phi[r * pf :, r * pf :] = phiy

    # Expected volatility for oil
    evy = []
    for j in range(1, h + 1):
        evy.append(expected_var(thy[0], thy[1], thy[2], xy, j))

    # Recursion
    U = np.zeros((T, h))
    for t in range(T):
        u = None
        for j in range(h):
            ev_diag = np.zeros(dim)
            ev_diag[:r] = evf[j][t, :]  # predictor variances (first r entries)
            ev_diag[r * pf] = evy[j][t]  # oil variance
            ev = np.diag(ev_diag)

            if j == 0:
                u = ev.copy()
            else:
                u = phi @ u @ phi.T + ev
            U[t, j] = u[r * pf, r * pf]

    return U


def build_opu():
    """Full OPU construction pipeline: fetch data -> forecast errors -> SV -> uncertainty."""
    from opu.data import load_raw_data
    from opu.transforms import prepare_missing, zscore, deseasonalize
    from opu.factors import factors_em
    from opu.forecast_errors import build_forecast_errors, build_ar_errors, build_np_errors
    from opu.sv import sv_sample
    from opu.config import OPU_DIR, SV_BURN, SV_DRAWS, SV_THIN, SEED_SV_Y, SEED_SV_F
    import time

    print("=== Building OPU index ===")

    # Step 1: Load and prepare data
    print("Step 1: Loading and transforming data...")
    raw = load_raw_data()
    # This section will be filled during implementation to:
    # 1. Align all series to common monthly date index
    # 2. Apply transformation codes
    # 3. Z-score
    # 4. Extract fuel-group factors
    # 5. Build predictor matrix xt
    # For now, raise NotImplementedError until data fetching is wired up
    raise NotImplementedError("Wire up data loading in Task 9")
