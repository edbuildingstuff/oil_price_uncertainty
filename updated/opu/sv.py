"""Kim-Shephard-Chib (1998) Gibbs sampler for AR(1) stochastic volatility.

Model:
    y_t = exp(h_t/2) * eps_t,    eps_t ~ N(0,1)
    h_t = mu + phi*(h_{t-1} - mu) + sigma*eta_t,    eta_t ~ N(0,1)
"""
import numpy as np
from numba import njit

# KSC (1998) Table 4: 10-component mixture approximation to log(chi2(1))
KSC_WEIGHTS = np.array([
    0.00609, 0.04775, 0.13057, 0.20674, 0.22715,
    0.18842, 0.12047, 0.05591, 0.01575, 0.00115,
])
KSC_MEANS = np.array([
    1.92677, 1.34744, 0.73504, 0.02266, -0.85173,
    -1.97278, -3.46788, -5.55246, -8.68384, -14.65000,
])
KSC_VARS = np.array([
    0.11265, 0.17788, 0.26768, 0.40611, 0.62699,
    0.98583, 1.57469, 2.54498, 4.16591, 7.33342,
])

_LOG_CHI2_MEAN = -1.2704  # E[log(chi2(1))]


def sv_sample(
    y: np.ndarray,
    draws: int = 50000,
    burnin: int = 50000,
    thin: int = 10,
    seed: int = 0,
) -> dict:
    """Run Gibbs sampler for AR(1) SV model.

    Returns dict with posterior means: mu, phi, sigma, latent (log-vol states).
    """
    rng = np.random.default_rng(seed)
    T = len(y)

    # Handle zeros
    y_safe = y.copy()
    y_safe[y_safe == 0] = 1e-5
    ystar = np.log(y_safe ** 2)

    # Initialize
    h = ystar - _LOG_CHI2_MEAN
    mu = np.mean(h)
    phi = 0.9
    sigma2 = 0.1
    s = np.zeros(T, dtype=int)

    # Storage
    total = burnin + draws
    mu_store = np.zeros(draws // thin)
    phi_store = np.zeros(draws // thin)
    sigma_store = np.zeros(draws // thin)
    h_store = np.zeros((draws // thin, T))
    save_idx = 0

    for iteration in range(total):
        # Block 1: Sample mixture indicators s_t (JIT'd with pre-drawn uniforms)
        u = rng.random(T)
        s = _sample_indicators_jit(ystar, h, u, KSC_WEIGHTS, KSC_MEANS, KSC_VARS)

        # Block 2: Sample latent states h_t via FFBS (JIT'd with pre-drawn normals)
        z = rng.standard_normal(T)
        h = _ffbs_jit(ystar, s, mu, phi, sigma2, z, KSC_MEANS, KSC_VARS)

        # Block 3: Sample parameters (mu, phi, sigma2)
        mu, phi, sigma2 = _sample_params(h, rng)

        # Store
        if iteration >= burnin and (iteration - burnin) % thin == 0:
            mu_store[save_idx] = mu
            phi_store[save_idx] = phi
            sigma_store[save_idx] = np.sqrt(sigma2)
            h_store[save_idx, :] = h
            save_idx += 1

    return {
        "mu": np.mean(mu_store[:save_idx]),
        "phi": np.mean(phi_store[:save_idx]),
        "sigma": np.mean(sigma_store[:save_idx]),
        "latent": np.mean(h_store[:save_idx, :], axis=0),
    }


@njit(cache=True)
def _sample_indicators_jit(ystar, h, uniforms, weights, means, vars_):
    """Sample mixture indicators via inverse CDF. Uniforms pre-drawn by caller."""
    T = ystar.shape[0]
    s = np.zeros(T, dtype=np.int64)
    log_w = np.empty(10)
    log_v = np.empty(10)
    for j in range(10):
        log_w[j] = np.log(weights[j])
        log_v[j] = np.log(vars_[j])
    log_probs = np.empty(10)
    probs = np.empty(10)
    for t in range(T):
        resid_t = ystar[t] - h[t]
        max_lp = -1e300
        for j in range(10):
            m_j = means[j] + _LOG_CHI2_MEAN
            diff = resid_t - m_j
            lp = log_w[j] - 0.5 * log_v[j] - 0.5 * diff * diff / vars_[j]
            log_probs[j] = lp
            if lp > max_lp:
                max_lp = lp
        total = 0.0
        for j in range(10):
            probs[j] = np.exp(log_probs[j] - max_lp)
            total += probs[j]
        u = uniforms[t] * total
        cum = 0.0
        sel = 9
        for j in range(10):
            cum += probs[j]
            if u < cum:
                sel = j
                break
        s[t] = sel
    return s


@njit(cache=True)
def _ffbs_jit(ystar, s, mu, phi, sigma2, z, means, vars_):
    """FFBS with pre-drawn standard normals z (length T)."""
    T = ystar.shape[0]
    h_filt = np.zeros(T)
    P_filt = np.zeros(T)

    h_pred = mu
    if abs(phi) < 0.9999:
        P_pred = sigma2 / (1.0 - phi * phi)
    else:
        P_pred = sigma2 * 100.0

    for t in range(T):
        d_t = means[s[t]] + _LOG_CHI2_MEAN
        R_t = vars_[s[t]]
        v = ystar[t] - d_t - h_pred
        F = P_pred + R_t
        K = P_pred / F
        h_filt[t] = h_pred + K * v
        P_filt[t] = P_pred * (1.0 - K)
        if t < T - 1:
            h_pred = mu + phi * (h_filt[t] - mu)
            P_pred = phi * phi * P_filt[t] + sigma2

    h = np.zeros(T)
    h[T - 1] = h_filt[T - 1] + np.sqrt(P_filt[T - 1]) * z[T - 1]
    for t in range(T - 2, -1, -1):
        h_pred_next = mu + phi * (h_filt[t] - mu)
        P_pred_next = phi * phi * P_filt[t] + sigma2
        J = phi * P_filt[t] / P_pred_next
        h_mean = h_filt[t] + J * (h[t + 1] - h_pred_next)
        h_var = P_filt[t] - J * J * P_pred_next
        if h_var < 1e-12:
            h_var = 1e-12
        h[t] = h_mean + np.sqrt(h_var) * z[t]
    return h


def _sample_params(h: np.ndarray, rng) -> tuple[float, float, float]:
    """Sample (mu, phi, sigma2) from conjugate conditionals."""
    T = len(h)
    y_reg = h[1:]
    x_reg = h[:-1]

    # Regression: h_t = alpha + phi*h_{t-1} + eta_t
    # Use flat prior (OLS posterior)
    X = np.column_stack([np.ones(T - 1), x_reg])
    XtX = X.T @ X
    Xty = X.T @ y_reg

    # Posterior for (alpha, phi) | sigma2
    # First estimate sigma2 from OLS
    b_ols = np.linalg.solve(XtX, Xty)
    resid = y_reg - X @ b_ols
    s2 = resid @ resid

    # sigma2 ~ IG((T-1-2)/2, s2/2) with flat prior
    nu_post = T - 1
    sigma2 = 1.0 / rng.gamma(nu_post / 2.0, 2.0 / s2)

    # (alpha, phi) | sigma2 ~ N(b_ols, sigma2 * inv(XtX))
    cov = sigma2 * np.linalg.inv(XtX)
    cov = (cov + cov.T) / 2.0
    L = np.linalg.cholesky(cov)
    b_draw = b_ols + L @ rng.standard_normal(2)

    alpha = b_draw[0]
    phi = b_draw[1]

    # Enforce stationarity: |phi| < 1
    if abs(phi) >= 0.9999:
        phi = 0.9999 * np.sign(phi)

    mu = alpha / (1.0 - phi) if abs(phi) < 0.9999 else 0.0

    return mu, phi, sigma2
