"""Bayesian SVAR estimation. Ports main_1.m through main_6.m."""
import numpy as np
from pathlib import Path
from opu.config import (
    SVAR_P, SVAR_H, SVAR_ROTATIONS, SVAR_TARGET_DRAWS,
    SVAR_ROOT_BOUND, SVAR_ALPHA, SVAR_ELASTICITY_BOUND,
    SVAR_USE_ELASTICITY_BOUNDS, SVAR_N_WORKERS, SEED_SVAR,
    OPU_DIR, SVAR_DIR, REFERENCE_DIR,
)
from opu.transforms import deseasonalize


def load_svar_data() -> dict:
    """Construct the 5-variable SVAR dataset.

    Variables: (1) oil production growth, (2) REA, (3) log real oil price,
    (4) inventory change, (5) OPU.
    """
    from opu.data import load_raw_data, SAMPLE_START, SAMPLE_END
    import pandas as pd

    raw = load_raw_data()
    start = pd.Timestamp(SAMPLE_START)
    end = pd.Timestamp(SAMPLE_END)
    dates = pd.date_range(start, end, freq="MS")

    def _align(df):
        # Drop duplicates before reindexing (e.g. production series has dupes)
        df = df.drop_duplicates(subset="date", keep="first").sort_values("date")
        return df.set_index("date").reindex(dates).interpolate()["value"].values

    # World oil production (growth rate = diff(log()))
    prod = _align(raw["production"])
    dprod = np.diff(np.log(prod))

    # REA (Kilian index, level)
    rea = _align(raw["IGREA"])[1:]  # trim to match dprod

    # Real oil price: RAC / CPI, rescaled
    rac = _align(raw["rac"])
    cpi = _align(raw["CPIAUCSL"])
    rpoil = rac / cpi
    # Normalize: Nov 2008 = $61.65
    # Find index for Nov 2008
    nov2008 = np.argmin(np.abs(dates - pd.Timestamp("2008-11-01")))
    rpoil = 49.1 * rpoil / rpoil[nov2008]
    lrpoil = np.log(rpoil)[1:]  # trim to match dprod

    # Inventory change
    stocks = _align(raw["stocks"])
    dstocks = np.diff(stocks)

    # OPU (load from artifacts)
    opu_data = np.load(OPU_DIR / "opu_baseline.npz")
    opu = opu_data["opu"]

    # Align all to common sample (OPU loses first 24 obs)
    p = SVAR_P
    n_vars = min(len(dprod), len(rea), len(lrpoil), len(dstocks))
    offset = n_vars - len(opu)

    y = np.column_stack([
        dprod[offset:offset + len(opu)],
        rea[offset:offset + len(opu)],
        lrpoil[offset:offset + len(opu)],
        dstocks[offset:offset + len(opu)],
        opu,
    ])

    # Deseasonalize first 4 variables
    for i in range(4):
        y[:, i] = deseasonalize(y[:, i])

    # Load Q.txt for use-elasticity prior.
    # Q.txt holds world oil production (thousands of barrels/day).
    # The formula converts to a scaled production level used in the
    # use-elasticity bound: (10000 * Q * 30 / 1000).
    Q = np.loadtxt(REFERENCE_DIR / "Q.txt")
    # Q.txt is a single column; handle both 1D (ndim==1) and 2D (ndim==2)
    # cases defensively so this port survives future data updates.
    if Q.ndim == 1:
        # Single-column file: Q is already the production series
        Q_1 = (10000 * Q[1:]) * 30 / 1000
    else:
        # Multi-column file: take the first column (index 0) as in original spec
        Q_1 = (10000 * Q[1:, 0]) * 30 / 1000
    # Trim Q_1 to at most len(y) if Q extends further into the sample than y.
    # Q.txt is a static reference file (original paper ends 2018); when the
    # replication sample extends beyond Q, Q_1 will be shorter than y -- that
    # is acceptable because Q_1 is used only to derive a scalar prior bound,
    # not as an observation-by-observation aligned series.
    Q_1 = Q_1[:len(y)]

    return {
        "y": y,
        "dates": dates[-len(y):],
        "Q_1": Q_1,
    }


def setup_posterior(y: np.ndarray, p: int) -> dict:
    """Compute Normal-Inverse-Wishart posterior parameters.

    Diffuse prior: N0=0, S0=0, nu0=0.
    """
    t, n = y.shape
    T = t - p

    # Build VAR regressors
    Y = y[p:, :]
    lags = np.zeros((T, n * p))
    for i in range(p):
        lags[:, i * n : (i + 1) * n] = y[p - 1 - i : t - 1 - i, :]
    X = np.column_stack([np.ones(T), lags])

    Ydep = y[p:, :]
    Bhat = np.linalg.solve(X.T @ X, X.T @ Ydep)
    Sigmahat = (Ydep - X @ Bhat).T @ (Ydep - X @ Bhat) / T

    # Diffuse prior
    nu0 = 0
    N0 = np.zeros((n * p + 1, n * p + 1))
    S0 = np.zeros((n, n))
    Bbar0 = np.zeros((n * p + 1, n))

    # Posterior
    nuT = T + nu0
    NT = N0 + X.T @ X
    BbarT = np.linalg.solve(NT, N0 @ Bbar0 + X.T @ X @ Bhat)
    ST = (nu0 / nuT) * S0 + (T / nuT) * Sigmahat + \
         (1 / nuT) * (Bhat - Bbar0).T @ N0 @ np.linalg.solve(NT, X.T) @ X @ (Bhat - Bbar0)

    EvecB = BbarT.flatten(order="F")

    return {
        "nuT": nuT, "NT": NT, "BbarT": BbarT, "ST": ST,
        "EvecB": EvecB, "Ydep": Ydep, "X": X,
        "T": T, "n": n, "p": p,
    }


def draw_posterior(post: dict, rng) -> tuple:
    """Draw (B, Sigma) from Normal-Inverse-Wishart posterior."""
    n, p = post["n"], post["p"]
    nuT, NT, ST = post["nuT"], post["NT"], post["ST"]
    EvecB = post["EvecB"]

    # Draw Sigma ~ IW(nuT, ST)
    RANTR = np.linalg.cholesky(np.linalg.inv(ST)) @ rng.standard_normal((n, nuT)) / np.sqrt(nuT)
    Sigma = np.linalg.inv(RANTR @ RANTR.T)

    # Draw B | Sigma ~ MN(BbarT, Sigma x NT^{-1})
    VvecB = np.kron(Sigma, np.linalg.inv(NT))
    VvecB = (VvecB + VvecB.T) / 2
    L = np.linalg.cholesky(VvecB)
    vecB = EvecB + L @ rng.standard_normal(n * (n * p + 1))
    B = vecB.reshape(1 + n * p, n, order="F").T

    return B, Sigma, vecB
