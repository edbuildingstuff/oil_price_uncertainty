"""Narrative sign restrictions for SVAR. Ports main_1.m lines 79-227."""
import numpy as np


def get_narrative_dates(sample_dates: np.ndarray) -> dict:
    """Compute date indices for narrative restriction episodes.

    sample_dates: array of float years (e.g. 1975.25 for April 1975).
    """
    def _find(year):
        return np.argmin(np.abs(sample_dates - year))

    return {
        "id_90M10": _find(1990 + 10 / 12),
        "id_90M06": _find(1990 + 6 / 12),
        "id_90M07": _find(1990 + 7 / 12),
        "id_79M05": _find(1979 + 5 / 12),
        "id_79M12": _find(1979 + 12 / 12),
        "id_85M12": _find(1985 + 12 / 12),
        "id_86M12": _find(1986 + 12 / 12),
    }


def compute_historical_decomposition(
    B: np.ndarray, B0inv: np.ndarray, Ydep: np.ndarray, X: np.ndarray,
    n: int, p: int
) -> dict:
    """Compute historical decomposition of each shock's contribution to price.

    Returns dict with yhat1 (supply), yhat2 (flow demand),
    yhat3 (speculative), yhat5 (uncertainty).
    """
    T = Ydep.shape[0]

    # Structural shocks
    Uhat = Ydep - X @ B.T
    Ehat = np.linalg.solve(B0inv, Uhat.T)

    # IRFs for full sample
    A_comp = np.zeros((n * p, n * p))
    A_comp[:n, :] = B[:, 1:]
    if p > 1:
        A_comp[n:, : n * (p - 1)] = np.eye(n * (p - 1))

    J = np.zeros((n, n * p))
    J[:n, :n] = np.eye(n)

    # Compute IRF row for price (row 2) for each shock
    IRF = np.zeros((n * n, T))
    for h in range(T):
        Ah = np.linalg.matrix_power(A_comp, h)
        irf_h = J @ Ah @ J.T @ B0inv
        IRF[:, h] = irf_h.flatten(order="F")

    # Historical decomposition for price (row 2)
    # yhat_k(t) = sum_{i=0}^{t-1} IRF_price_shock_k(i) * Ehat_k(t-i)
    price_row = 2
    yhat = {}
    for shock, label in [(0, "supply"), (1, "flow_demand"), (2, "speculative"), (4, "uncertainty")]:
        yh = np.zeros(T)
        irf_idx = price_row + n * shock  # row in flattened IRF
        for t in range(T):
            yh[t] = np.dot(IRF[irf_idx, : t + 1], Ehat[shock, t::-1])
        yhat[label] = yh

    return yhat


def check_narrative_restrictions(
    yhat: dict, dates: dict, B0inv: np.ndarray
) -> bool:
    """Check narrative sign restrictions (S4).

    B0inv column 0 (supply) is normalized to raise oil price, so flip sign.
    """
    yhat1 = yhat["supply"]
    yhat2 = yhat["flow_demand"]
    yhat3 = yhat["speculative"]
    yhat5 = yhat["uncertainty"]

    d = dates

    # NR1: flow supply raised price >0.1 between Jul-Oct 1990
    nr1 = (yhat1[d["id_90M10"]] - yhat1[d["id_90M07"]]) > 0.1

    # NR2: flow demand raised price <0.1 between Jun-Oct 1990
    nr2 = (yhat2[d["id_90M10"]] - yhat2[d["id_90M06"]]) < 0.1

    # NR3: speculative + uncertainty raised price >0.20 between May-Dec 1979
    nr3 = (
        (yhat3[d["id_79M12"]] - yhat3[d["id_79M05"]])
        + (yhat5[d["id_79M12"]] - yhat5[d["id_79M05"]])
    ) > 0.20

    # NR4: speculative + uncertainty raised price >0.1 between Jun-Oct 1990
    nr4 = (
        (yhat3[d["id_90M10"]] - yhat3[d["id_90M06"]])
        + (yhat5[d["id_90M10"]] - yhat5[d["id_90M06"]])
    ) > 0.1

    # NR5: speculative + uncertainty lowered price <-0.15 between Dec 1985-Dec 1986
    nr5 = (
        (yhat3[d["id_86M12"]] - yhat3[d["id_85M12"]])
        + (yhat5[d["id_86M12"]] - yhat5[d["id_85M12"]])
    ) < -0.15

    return nr1 and nr2 and nr3 and nr4 and nr5
