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
    """Full OPU construction pipeline."""
    from opu.data import build_opu_dataset
    from opu.forecast_errors import build_forecast_errors, build_ar_errors, build_np_errors
    from opu.sv import sv_sample
    from opu.config import OPU_DIR, PY, SV_BURN, SV_DRAWS, SV_THIN, SEED_SV_Y, SEED_SV_F, OPU_HORIZON
    import time

    print("=== Building OPU index ===")

    # Load and prepare data
    print("Step 1: Loading data and building predictors...")
    ds = build_opu_dataset()
    yt, xt = ds["yt"], ds["xt"]

    # Baseline forecast errors
    print("Step 2: Computing forecast errors...")
    fe = build_forecast_errors(yt, xt)
    vyt = fe["vyt"]
    vft = fe["vft"]

    # SV estimation on forecast errors
    print("Step 3: Estimating stochastic volatility (this may take a while)...")
    T_fe, N = vyt.shape
    R = vft.shape[1]

    sv_y_params = []
    sv_y_latent = []
    for i in range(N):
        t0 = time.time()
        res = sv_sample(vyt[:, i], draws=SV_DRAWS, burnin=SV_BURN, thin=SV_THIN, seed=SEED_SV_Y + i)
        sv_y_params.append((res["mu"], res["phi"], res["sigma"]))
        sv_y_latent.append(res["latent"])
        print(f"  Oil series {i}: {time.time() - t0:.1f}s")

    sv_f_params = []
    sv_f_latent = []
    for i in range(R):
        t0 = time.time()
        res = sv_sample(vft[:, i], draws=SV_DRAWS, burnin=SV_BURN, thin=SV_THIN, seed=SEED_SV_F + i)
        sv_f_params.append((res["mu"], res["phi"], res["sigma"]))
        sv_f_latent.append(res["latent"])
        print(f"  Predictor {i}: {time.time() - t0:.1f}s")

    # Assemble SV outputs
    thf = np.array([[mu * (1 - phi), phi, sigma ** 2] for mu, phi, sigma in sv_f_params]).T
    xf = np.column_stack(sv_f_latent)
    fb = fe["fbetas"]

    thy = np.array([sv_y_params[0][0] * (1 - sv_y_params[0][1]),
                     sv_y_params[0][1],
                     sv_y_params[0][2] ** 2])
    xy = sv_y_latent[0]

    # Uncertainty recursion
    print("Step 4: Computing uncertainty recursion...")
    h = OPU_HORIZON
    evf, phif = compute_uf(xf, thf, fb, h)
    U = compute_uy(xy, thy, fe["ybetas"][0, :], PY, evf, phif)
    opu_baseline = np.sqrt(U[:, h - 1])

    # Save
    dates_opu = ds["dates"][-len(opu_baseline):]
    np.savez(OPU_DIR / "opu_baseline.npz", opu=opu_baseline, dates=dates_opu)
    print(f"OPU saved to {OPU_DIR / 'opu_baseline.npz'}, shape={opu_baseline.shape}")

    # AR-only variant
    print("Step 5: AR-only OPU...")
    ar_fe = build_ar_errors(yt)
    ar_sv = sv_sample(ar_fe["vyt"][:, 0], draws=SV_DRAWS, burnin=SV_BURN, thin=SV_THIN, seed=SEED_SV_Y + 100)
    ar_thy = np.array([ar_sv["mu"] * (1 - ar_sv["phi"]), ar_sv["phi"], ar_sv["sigma"] ** 2])
    ar_xy = ar_sv["latent"]
    T_ar = len(ar_xy)
    ar_py = ar_fe["ybetas"].shape[1] - 1
    ar_ut = np.zeros((T_ar, h))
    for j in range(h):
        evy_j = expected_var(ar_thy[0], ar_thy[1], ar_thy[2], ar_xy, j + 1)
        phi_ar = np.zeros((ar_py, ar_py))
        phi_ar[0, :] = ar_fe["ybetas"][0, 1:]
        if ar_py > 1:
            phi_ar[1:, :-1] = np.eye(ar_py - 1)
        for t in range(T_ar):
            ev = np.zeros((ar_py, ar_py))
            ev[0, 0] = evy_j[t]
            if j == 0:
                u = ev
            else:
                u = phi_ar @ u @ phi_ar.T + ev
            ar_ut[t, j] = u[0, 0]
    opu_ar = np.sqrt(ar_ut[:, h - 1])
    np.savez(OPU_DIR / "opu_ar.npz", opu=opu_ar)

    # No-predictor variant
    print("Step 6: No-predictor OPU...")
    np_fe = build_np_errors(yt[:, 0:1])
    np_sv = sv_sample(np_fe["vyt"][:, 0], draws=SV_DRAWS, burnin=SV_BURN, thin=SV_THIN, seed=SEED_SV_Y + 200)
    np_thy = np.array([np_sv["mu"] * (1 - np_sv["phi"]), np_sv["phi"], np_sv["sigma"] ** 2])
    np_xy = np_sv["latent"]
    T_np = len(np_xy)
    np_ut = np.zeros((T_np, h))
    for j in range(h):
        np_ut[:, j] = expected_var(np_thy[0], np_thy[1], np_thy[2], np_xy, j + 1)
    opu_np = np.sqrt(np_ut[:, h - 1])
    np.savez(OPU_DIR / "opu_np.npz", opu=opu_np)

    print("=== OPU construction complete ===")
