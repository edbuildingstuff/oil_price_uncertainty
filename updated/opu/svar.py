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


import multiprocessing as mp
from opu.identification import (
    check_sign_restrictions, check_elasticity,
    check_dynamic_sign, compute_irf,
)
from opu.narrative import (
    get_narrative_dates, compute_historical_decomposition,
    check_narrative_restrictions,
)


def _worker_rotation(args):
    """Worker: test a batch of QR rotations for one posterior draw."""
    A, B, n, p, H, Q_1, DSbar, Ydep, X, narr_dates, start_rot, end_rot, seed = args
    rng = np.random.default_rng(seed)

    for r in range(start_rot, end_rot):
        # QR rotation
        Z = rng.standard_normal((n, n))
        U, R = np.linalg.qr(Z)
        for j in range(n):
            if R[j, j] < 0:
                U[:, j] = -U[:, j]

        Atilde = A @ U

        # S1: Sign restrictions
        B0inv = check_sign_restrictions(Atilde)
        if B0inv is None:
            continue

        # S2: Elasticity
        if not check_elasticity(B0inv, Q_1, DSbar):
            continue

        # S3: Dynamic sign restrictions
        irf = compute_irf(B, B0inv, n, p, H)
        if not check_dynamic_sign(irf):
            continue

        # S4: Narrative restrictions
        # Normalize supply shock to raise price
        B0inv_n = B0inv.copy()
        B0inv_n[:, 0] = -B0inv_n[:, 0]
        yhat = compute_historical_decomposition(B, B0inv_n, Ydep, X, n, p)
        if not check_narrative_restrictions(yhat, narr_dates, B0inv_n):
            continue

        return B0inv

    return None


def run_svar(resume: bool = False, workers: int = SVAR_N_WORKERS):
    """Run full SVAR estimation with parallel rotation sampling."""
    print("=== SVAR Estimation ===")

    # Load data
    data = load_svar_data()
    y = data["y"]
    Q_1 = data["Q_1"]

    n = y.shape[1]
    p = SVAR_P
    DSbar = np.mean(y[:, 3])

    # Posterior parameters
    post = setup_posterior(y, p)

    # Narrative dates
    T_eff = post["T"]
    sample_start = 1973 + 2 / 12 + (SVAR_P + 24) / 12
    sample_dates = np.arange(T_eff) / 12 + sample_start
    narr_dates = get_narrative_dates(sample_dates)

    # Resume from checkpoint
    accepted = []
    checkpoint_path = SVAR_DIR / "checkpoint.npz"
    if resume and checkpoint_path.exists():
        ckpt = np.load(checkpoint_path, allow_pickle=True)
        accepted = list(ckpt["accepted"])
        print(f"Resumed from checkpoint: {len(accepted)} accepted draws")

    rng = np.random.default_rng(SEED_SVAR + len(accepted))
    total_attempts = 0

    while len(accepted) < SVAR_TARGET_DRAWS:
        total_attempts += 1

        # Draw from posterior
        B, Sigma, vecB = draw_posterior(post, rng)

        # Stationarity check
        A_comp = np.zeros((n * p, n * p))
        A_comp[:n, :] = B[:, 1:]
        if p > 1:
            A_comp[n:, : n * (p - 1)] = np.eye(n * (p - 1))
        max_root = np.max(np.abs(np.linalg.eigvals(A_comp)))
        if max_root >= SVAR_ROOT_BOUND:
            continue

        A = np.linalg.cholesky(Sigma)

        # Parallel rotation search
        rots_per_worker = SVAR_ROTATIONS // workers
        worker_args = []
        for w in range(workers):
            s = w * rots_per_worker
            e = s + rots_per_worker
            seed = rng.integers(0, 2**31)
            worker_args.append((
                A, B, n, p, SVAR_H, Q_1, DSbar,
                post["Ydep"], post["X"], narr_dates,
                s, e, seed,
            ))

        with mp.Pool(workers) as pool:
            results = pool.map(_worker_rotation, worker_args)

        # Check if any worker found an accepted rotation
        for result in results:
            if result is not None:
                accepted.append({
                    "vecB": vecB,
                    "Sigma": Sigma.flatten(),
                    "B0inv": result.flatten(),
                })
                print(f"Draw {len(accepted)}/{SVAR_TARGET_DRAWS} accepted "
                      f"(attempt {total_attempts})")

                # Checkpoint
                np.savez(
                    checkpoint_path,
                    accepted=np.array(accepted, dtype=object),
                    total_attempts=total_attempts,
                )
                break

    # Save final draws
    np.savez(
        SVAR_DIR / "accepted_draws.npz",
        vecB=np.array([d["vecB"] for d in accepted]),
        Sigma=np.array([d["Sigma"] for d in accepted]),
        B0inv=np.array([d["B0inv"] for d in accepted]),
        n=n, p=p,
    )
    print(f"=== SVAR complete: {len(accepted)} draws saved ===")
