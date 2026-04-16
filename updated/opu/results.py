"""Post-estimation: IRFs, FEVDs, HDECs from accepted SVAR draws."""
import numpy as np
from opu.config import SVAR_H, SVAR_ALPHA, SVAR_DIR
from opu.identification import compute_irf


def load_accepted_draws() -> dict:
    path = SVAR_DIR / "accepted_draws.npz"
    data = np.load(path)
    M = data["vecB"].shape[0]
    n = int(data["n"])
    p = int(data["p"])
    draws = []
    for i in range(M):
        vecB = data["vecB"][i]
        B = vecB.reshape(1 + n * p, n, order="F").T
        Sigma = data["Sigma"][i].reshape(n, n)
        B0inv = data["B0inv"][i].reshape(n, n)
        draws.append({"B": B, "Sigma": Sigma, "B0inv": B0inv})
    return {"draws": draws, "n": n, "p": p}


def compute_irfs(draws: list, n: int, p: int, H: int = SVAR_H) -> np.ndarray:
    """Compute IRFs for all accepted draws.

    Returns (M, n, n, H+1) array.
    """
    M = len(draws)
    irfs = np.zeros((M, n, n, H + 1))
    for i, d in enumerate(draws):
        # Normalize supply shock
        B0inv = d["B0inv"].copy()
        B0inv[:, 0] = -B0inv[:, 0]
        irfs[i] = compute_irf(d["B"], B0inv, n, p, H)
    return irfs


def irf_summary(irfs: np.ndarray, alpha: float = SVAR_ALPHA) -> dict:
    """Compute median and credible bands."""
    lo = alpha / 2 * 100
    hi = (1 - alpha / 2) * 100
    return {
        "median": np.median(irfs, axis=0),
        "lower": np.percentile(irfs, lo, axis=0),
        "upper": np.percentile(irfs, hi, axis=0),
    }


def compute_fevd(draws: list, n: int, p: int, H: int = SVAR_H) -> np.ndarray:
    """Forecast error variance decomposition.

    Returns (M, n, n, H+1): fevd[m, i, j, h] = share of var i at horizon h due to shock j.
    """
    M = len(draws)
    fevd = np.zeros((M, n, n, H + 1))

    for m, d in enumerate(draws):
        B0inv = d["B0inv"].copy()
        B0inv[:, 0] = -B0inv[:, 0]
        irf = compute_irf(d["B"], B0inv, n, p, H)

        for h in range(H + 1):
            for i in range(n):
                total = 0.0
                for j in range(n):
                    total += np.sum(irf[i, j, : h + 1] ** 2)
                for j in range(n):
                    if total > 0:
                        fevd[m, i, j, h] = np.sum(irf[i, j, : h + 1] ** 2) / total

    return fevd


def compute_all_results():
    """Load draws and compute all post-estimation results."""
    data = load_accepted_draws()
    draws, n, p = data["draws"], data["n"], data["p"]

    irfs = compute_irfs(draws, n, p)
    irf_stats = irf_summary(irfs)
    fevd = compute_fevd(draws, n, p)
    fevd_stats = {
        "median": np.median(fevd, axis=0),
        "lower": np.percentile(fevd, 16, axis=0),
        "upper": np.percentile(fevd, 84, axis=0),
    }

    np.savez(
        SVAR_DIR / "results.npz",
        irf_median=irf_stats["median"],
        irf_lower=irf_stats["lower"],
        irf_upper=irf_stats["upper"],
        fevd_median=fevd_stats["median"],
        fevd_lower=fevd_stats["lower"],
        fevd_upper=fevd_stats["upper"],
    )
    print(f"Results saved to {SVAR_DIR / 'results.npz'}")
